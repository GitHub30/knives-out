// ZONEFALL online server: a Cloudflare Worker that matchmakes players into
// Durable Object rooms. Each Room runs the same Rust/Wasm simulation as the
// offline client, authoritatively, at 30 Hz.

import wasmModule from './core.wasm';

export interface Env {
  ROOMS: DurableObjectNamespace;
  LOBBY: DurableObjectNamespace;
  ALLOWED_ORIGINS: string;
  ASSETS?: Fetcher;
  /** Workers rate-limiting binding: WebSocket join attempts per client IP. */
  JOIN_LIMIT?: RateLimit;
}

interface SrvExports {
  memory: WebAssembly.Memory;
  in_ptr(): number;
  proto_version(): number;
  srv_init(seed: number, slots: number, flags: number): void;
  srv_add(len: number): number;
  srv_remove(pid: number): void;
  srv_msg(pid: number, len: number): void;
  srv_step(): number;
  srv_welcome(pid: number): number;
  srv_snap(pid: number): number;
  srv_out(): number;
  srv_flush(): void;
  srv_phase(): number;
  srv_humans(): number;
}

const S_PONG = 0x83;
const S_ERROR = 0x84;
const ERR_FULL = 1;
const ERR_STARTED = 2;
const ERR_TOO_MANY = 4;
const TICK_MS = 1000 / 30;
const MAX_MSG = 256;
const MAX_ROOM_MS = 25 * 60 * 1000;
const MAX_PER_IP = 3;
const PUBLIC_SLOTS = 30;
const PRIVATE_SLOTS = 30;
const FLAG_PRIVATE = 2;

function originAllowed(req: Request, env: Env): boolean {
  const origin = req.headers.get('Origin');
  if (!origin) return false;
  const self = new URL(req.url).origin;
  if (origin === self) return true;
  return env.ALLOWED_ORIGINS.split(',').map((s) => s.trim()).includes(origin);
}

function securityHeaders(h: Headers) {
  h.set('X-Content-Type-Options', 'nosniff');
  h.set('Referrer-Policy', 'no-referrer');
  return h;
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === '/health') {
      return new Response(JSON.stringify({ ok: true, proto: protoVersion() }), { headers: securityHeaders(new Headers({ 'content-type': 'application/json' })) });
    }
    if (url.pathname === '/ws') {
      if ((req.headers.get('Upgrade') ?? '').toLowerCase() !== 'websocket') return new Response('Expected WebSocket', { status: 426 });
      if (!originAllowed(req, env)) return new Response('Forbidden origin', { status: 403 });
      const ip = req.headers.get('CF-Connecting-IP') ?? 'unknown';
      if (env.JOIN_LIMIT) {
        const { success } = await env.JOIN_LIMIT.limit({ key: ip });
        if (!success) return new Response('Too many join attempts', { status: 429 });
      }
      const name = (url.searchParams.get('name') ?? '').slice(0, 32);
      const code = url.searchParams.get('room');
      let roomName: string;
      let kind: 'pub' | 'priv';
      if (code) {
        if (!/^[A-Z0-9]{3,12}$/.test(code)) return new Response('Bad room code', { status: 400 });
        roomName = `priv-${code}`;
        kind = 'priv';
      } else {
        const lobby = env.LOBBY.get(env.LOBBY.idFromName('main'));
        const r = await lobby.fetch('https://lobby/assign', { method: 'POST' });
        if (!r.ok) return new Response('Lobby unavailable', { status: 503 });
        roomName = ((await r.json()) as { room: string }).room;
        kind = 'pub';
      }
      const stub = env.ROOMS.get(env.ROOMS.idFromName(roomName));
      const fwd = new URL('https://room/ws');
      fwd.searchParams.set('name', name);
      fwd.searchParams.set('kind', kind);
      return stub.fetch(new Request(fwd.toString(), req));
    }
    if (env.ASSETS) return env.ASSETS.fetch(req);
    return new Response('Not found', { status: 404 });
  },
};

let cachedProto = 0;
function protoVersion(): number {
  if (!cachedProto) {
    const inst = new WebAssembly.Instance(wasmModule, {});
    cachedProto = (inst.exports as unknown as SrvExports).proto_version();
  }
  return cachedProto;
}

// ---------------------------------------------------------------------------
// Lobby: hands out the currently open public room.
export class Lobby {
  private current: { name: string; created: number; count: number } | null = null;

  constructor(_state: DurableObjectState, _env: Env) {}

  async fetch(_req: Request): Promise<Response> {
    const now = Date.now();
    const c = this.current;
    if (!c || now - c.created > 18000 || c.count >= PUBLIC_SLOTS - 4) {
      const rnd = crypto.getRandomValues(new Uint32Array(2));
      this.current = { name: `pub-${now.toString(36)}-${rnd[0].toString(36)}${rnd[1].toString(36)}`, created: now, count: 0 };
    }
    this.current!.count++;
    return new Response(JSON.stringify({ room: this.current!.name }), { headers: { 'content-type': 'application/json' } });
  }
}

// ---------------------------------------------------------------------------
interface Conn {
  pid: number;
  ip: string;
  tokens: number;
  lastRefill: number;
  strikes: number;
}

export class Room {
  private ex: SrvExports | null = null;
  private conns = new Map<WebSocket, Conn>();
  private timer: ReturnType<typeof setInterval> | null = null;
  private nextTick = 0;
  private createdAt = 0;
  private emptySince = 0;
  private ended = false;

  constructor(_state: DurableObjectState, _env: Env) {}

  private init(isPrivate: boolean) {
    const inst = new WebAssembly.Instance(wasmModule, {});
    this.ex = inst.exports as unknown as SrvExports;
    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    this.ex.srv_init(seed, isPrivate ? PRIVATE_SLOTS : PUBLIC_SLOTS, isPrivate ? FLAG_PRIVATE : 0);
    this.createdAt = Date.now();
    this.ended = false;
    this.emptySince = 0;
  }

  async fetch(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const isPrivate = url.searchParams.get('kind') === 'priv';
    if (!this.ex || (this.ended && this.conns.size === 0)) this.init(isPrivate);
    const ex = this.ex!;

    const pair = new WebSocketPair();
    const client = pair[0];
    const server = pair[1];
    server.accept();
    const reject = (code: number, reason: string) => {
      server.send(new Uint8Array([S_ERROR, code]));
      server.close(4000 + code, reason);
      return new Response(null, { status: 101, webSocket: client });
    };
    if (this.ended || ex.srv_phase() !== 0) return reject(ERR_STARTED, 'match already started');
    const ip = req.headers.get('CF-Connecting-IP') ?? 'unknown';
    let sameIp = 0;
    for (const c of this.conns.values()) if (c.ip === ip) sameIp++;
    if (sameIp >= MAX_PER_IP) return reject(ERR_TOO_MANY, 'too many connections');

    const name = new TextEncoder().encode(url.searchParams.get('name') ?? '').slice(0, 48);
    new Uint8Array(ex.memory.buffer, ex.in_ptr(), name.length).set(name);
    const pid = ex.srv_add(name.length);
    if (pid < 0) return reject(ERR_FULL, 'room full');
    const conn: Conn = { pid, ip, tokens: 60, lastRefill: Date.now(), strikes: 0 };
    this.conns.set(server, conn);
    this.sendOut(server, ex.srv_welcome(pid));

    server.addEventListener('message', (ev) => this.onMessage(server, ev));
    const close = () => this.onClose(server);
    server.addEventListener('close', close);
    server.addEventListener('error', close);
    this.startLoop();
    return new Response(null, { status: 101, webSocket: client });
  }

  private sendOut(ws: WebSocket, len: number) {
    if (!this.ex || len <= 0) return;
    try {
      ws.send(new Uint8Array(this.ex.memory.buffer, this.ex.srv_out(), len).slice());
    } catch {
      /* socket already closed */
    }
  }

  private onMessage(ws: WebSocket, ev: MessageEvent) {
    const c = this.conns.get(ws);
    const ex = this.ex;
    if (!c || !ex) return;
    if (!(ev.data instanceof ArrayBuffer)) {
      this.strike(ws, c);
      return;
    }
    const data = new Uint8Array(ev.data);
    if (data.length === 0 || data.length > MAX_MSG) {
      this.strike(ws, c);
      return;
    }
    // token bucket: ~45 msgs/s sustained, burst 60
    const now = Date.now();
    c.tokens = Math.min(60, c.tokens + ((now - c.lastRefill) / 1000) * 45);
    c.lastRefill = now;
    if (c.tokens < 1) {
      this.strike(ws, c);
      return;
    }
    c.tokens -= 1;
    if (data[0] === 3 && data.length === 5) {
      const pong = data.slice();
      pong[0] = S_PONG;
      try {
        ws.send(pong);
      } catch {
        /* ignore */
      }
      return;
    }
    new Uint8Array(ex.memory.buffer, ex.in_ptr(), data.length).set(data);
    ex.srv_msg(c.pid, data.length);
  }

  private strike(ws: WebSocket, c: Conn) {
    c.strikes++;
    if (c.strikes > 200) {
      try {
        ws.close(1008, 'rate limit');
      } catch {
        /* ignore */
      }
      this.onClose(ws);
    }
  }

  private onClose(ws: WebSocket) {
    const c = this.conns.get(ws);
    if (!c) return;
    this.conns.delete(ws);
    this.ex?.srv_remove(c.pid);
    if (this.conns.size === 0) this.emptySince = Date.now();
  }

  private startLoop() {
    if (this.timer) return;
    this.nextTick = Date.now() + TICK_MS;
    this.timer = setInterval(() => this.tick(), TICK_MS);
  }

  private stop(reason: string) {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.ended = true;
    for (const ws of this.conns.keys()) {
      try {
        ws.close(1000, reason);
      } catch {
        /* ignore */
      }
    }
    this.conns.clear();
  }

  private tick() {
    const ex = this.ex;
    if (!ex) return;
    const now = Date.now();
    if (this.conns.size === 0 && this.emptySince && now - this.emptySince > 15000) return this.stop('empty');
    if (now - this.createdAt > MAX_ROOM_MS) return this.stop('time limit');
    let steps = 0;
    try {
      while (this.nextTick <= now && steps < 5) {
        const r = ex.srv_step();
        if (r & 1) {
          for (const [ws, c] of this.conns) this.sendOut(ws, ex.srv_snap(c.pid));
          ex.srv_flush();
        }
        if (r & 2) return this.stop('match over');
        this.nextTick += TICK_MS;
        steps++;
      }
    } catch (e) {
      console.error('simulation trapped', e);
      return this.stop('server error');
    }
    if (now - this.nextTick > 500) this.nextTick = now + TICK_MS;
  }
}
