// ZONEFALL online server: a Cloudflare Worker that matchmakes players into
// Durable Object rooms. Each Room runs the same Rust/Wasm simulation as the
// offline client, authoritatively, at 30 Hz.
//
// Abuse limits (all approximate, fail-open if the Lobby object is reset):
// * Workers rate-limit binding: join attempts per IP per minute.
// * Lobby object: concurrent sockets per client network (IPv4 address or IPv6 /64),
//   global concurrent sockets and global concurrent rooms.
// * Room object: sockets per IP per room, message size, message rate (token bucket).

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
  srv_init(seed: number, slots: number, flags: number, secret: number): void;
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
const MAX_PER_IP_ROOM = 3;
const MAX_PER_NET = 4;
const MAX_SOCKETS = 600;
const MAX_ROOMS = 50;
const ENTRY_TTL_MS = 30 * 60 * 1000;
const PUBLIC_SLOTS = 30;
const PRIVATE_SLOTS = 30;
const FLAG_PRIVATE = 2;
const ROOM_CODE = /^[A-Z0-9]{6,12}$/;

function originAllowed(req: Request, env: Env): boolean {
  const origin = req.headers.get('Origin');
  if (!origin) return false;
  const self = new URL(req.url).origin;
  if (origin === self) return true;
  return env.ALLOWED_ORIGINS.split(',').map((s) => s.trim()).includes(origin);
}

/** Group IPv6 clients by /64 so one host cannot rotate through its prefix. */
export function netKey(ip: string): string {
  if (!ip.includes(':')) return ip;
  const [head, tail] = ip.split('::');
  const h = head ? head.split(':') : [];
  const t = tail !== undefined && tail !== '' ? tail.split(':') : [];
  const groups = tail !== undefined ? [...h, ...Array(Math.max(0, 8 - h.length - t.length)).fill('0'), ...t] : h;
  return groups.slice(0, 4).map((g) => g.toLowerCase().replace(/^0+(?=.)/, '')).join(':') + '::/64';
}

function plain(text: string, status: number): Response {
  return new Response(text, { status, headers: { 'content-type': 'text/plain; charset=utf-8', 'X-Content-Type-Options': 'nosniff' } });
}

function lobbyStub(env: Env) {
  return env.LOBBY.get(env.LOBBY.idFromName('main'));
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === '/health') {
      return new Response(JSON.stringify({ ok: true, proto: protoVersion() }), {
        headers: { 'content-type': 'application/json', 'X-Content-Type-Options': 'nosniff' },
      });
    }
    if (url.pathname === '/ws') {
      if ((req.headers.get('Upgrade') ?? '').toLowerCase() !== 'websocket') return plain('Expected WebSocket', 426);
      if (!originAllowed(req, env)) return plain('Forbidden origin', 403);
      const ip = req.headers.get('CF-Connecting-IP') ?? 'unknown';
      const net = netKey(ip);
      if (env.JOIN_LIMIT) {
        const { success } = await env.JOIN_LIMIT.limit({ key: net });
        if (!success) return plain('Too many join attempts', 429);
      }
      const name = (url.searchParams.get('name') ?? '').slice(0, 32);
      const code = url.searchParams.get('room');
      if (code !== null && !ROOM_CODE.test(code)) return plain('Bad room code', 400);
      const r = await lobbyStub(env).fetch('https://lobby/join', { method: 'POST', body: JSON.stringify({ net, code }) });
      if (!r.ok) return plain(r.status === 429 ? 'Too many connections' : 'Server busy', r.status);
      const { room } = (await r.json()) as { room: string };
      const stub = env.ROOMS.get(env.ROOMS.idFromName(room));
      const fwd = new URL('https://room/ws');
      fwd.searchParams.set('name', name);
      fwd.searchParams.set('kind', code ? 'priv' : 'pub');
      fwd.searchParams.set('room', room);
      fwd.searchParams.set('net', net);
      return stub.fetch(new Request(fwd.toString(), req));
    }
    if (env.ASSETS) return env.ASSETS.fetch(req);
    return plain('Not found', 404);
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
// Lobby: assigns the open public room and keeps approximate global accounting.
export class Lobby {
  private current: { name: string; created: number; count: number } | null = null;
  /** net -> timestamps of live sockets */
  private nets = new Map<string, number[]>();
  /** room -> last activity */
  private rooms = new Map<string, number>();

  constructor(_state: DurableObjectState, _env: Env) {}

  private sweep(now: number) {
    for (const [k, list] of this.nets) {
      const live = list.filter((t) => now - t < ENTRY_TTL_MS);
      if (live.length) this.nets.set(k, live);
      else this.nets.delete(k);
    }
    for (const [k, t] of this.rooms) if (now - t > ENTRY_TTL_MS) this.rooms.delete(k);
  }

  private sockets() {
    let n = 0;
    for (const l of this.nets.values()) n += l.length;
    return n;
  }

  async fetch(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const now = Date.now();
    let body: { net?: string; code?: string | null; room?: string } = {};
    try {
      body = (await req.json()) as typeof body;
    } catch {
      return new Response('bad request', { status: 400 });
    }
    const json = (v: unknown, status = 200) => new Response(JSON.stringify(v), { status, headers: { 'content-type': 'application/json' } });
    if (url.pathname === '/join') {
      this.sweep(now);
      const net = String(body.net ?? 'unknown');
      const mine = this.nets.get(net) ?? [];
      if (mine.length >= MAX_PER_NET) return json({ error: 'net' }, 429);
      if (this.sockets() >= MAX_SOCKETS) return json({ error: 'busy' }, 503);
      let room: string;
      if (body.code) {
        if (!ROOM_CODE.test(body.code)) return json({ error: 'code' }, 400);
        room = `priv-${body.code}`;
      } else {
        const c = this.current;
        if (!c || now - c.created > 18000 || c.count >= PUBLIC_SLOTS - 4) {
          const rnd = crypto.getRandomValues(new Uint32Array(2));
          this.current = { name: `pub-${now.toString(36)}-${rnd[0].toString(36)}${rnd[1].toString(36)}`, created: now, count: 0 };
        }
        this.current!.count++;
        room = this.current!.name;
      }
      if (!this.rooms.has(room) && this.rooms.size >= MAX_ROOMS) return json({ error: 'busy' }, 503);
      this.rooms.set(room, now);
      mine.push(now);
      this.nets.set(net, mine);
      return json({ room });
    }
    if (url.pathname === '/leave') {
      const net = String(body.net ?? 'unknown');
      const list = this.nets.get(net);
      if (list) {
        list.shift();
        if (!list.length) this.nets.delete(net);
      }
      if (body.room && this.rooms.has(body.room)) this.rooms.set(body.room, now);
      return json({ ok: true });
    }
    if (url.pathname === '/closed') {
      if (body.room) this.rooms.delete(body.room);
      return json({ ok: true });
    }
    return json({ error: 'not found' }, 404);
  }
}

// ---------------------------------------------------------------------------
interface Conn {
  pid: number;
  ip: string;
  net: string;
  tokens: number;
  lastRefill: number;
  strikes: number;
  strikeWindow: number;
}

export class Room {
  private state: DurableObjectState;
  private env: Env;
  private ex: SrvExports | null = null;
  private conns = new Map<WebSocket, Conn>();
  private timer: ReturnType<typeof setInterval> | null = null;
  private nextTick = 0;
  private createdAt = 0;
  private emptySince = 0;
  private ended = false;
  private roomName = '';

  constructor(state: DurableObjectState, env: Env) {
    this.state = state;
    this.env = env;
  }

  private notifyLobby(path: '/leave' | '/closed', body: Record<string, string>) {
    this.state.waitUntil(
      lobbyStub(this.env)
        .fetch(`https://lobby${path}`, { method: 'POST', body: JSON.stringify(body) })
        .then(() => undefined)
        .catch(() => undefined),
    );
  }

  private init(isPrivate: boolean) {
    const inst = new WebAssembly.Instance(wasmModule, {});
    this.ex = inst.exports as unknown as SrvExports;
    const [seed, secret] = crypto.getRandomValues(new Uint32Array(2));
    this.ex.srv_init(seed, isPrivate ? PRIVATE_SLOTS : PUBLIC_SLOTS, isPrivate ? FLAG_PRIVATE : 0, secret);
    this.createdAt = Date.now();
    this.ended = false;
    this.emptySince = 0;
  }

  async fetch(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const isPrivate = url.searchParams.get('kind') === 'priv';
    this.roomName = url.searchParams.get('room') ?? this.roomName;
    const net = url.searchParams.get('net') ?? 'unknown';
    if (!this.ex || (this.ended && this.conns.size === 0)) this.init(isPrivate);
    const ex = this.ex!;

    const pair = new WebSocketPair();
    const client = pair[0];
    const server = pair[1];
    server.accept();
    const reject = (code: number, reason: string) => {
      // The Lobby already counted this socket.
      this.notifyLobby('/leave', { net, room: this.roomName });
      try {
        server.send(new Uint8Array([S_ERROR, code]));
        server.close(4000 + code, reason);
      } catch {
        /* ignore */
      }
      return new Response(null, { status: 101, webSocket: client });
    };
    if (this.ended || ex.srv_phase() !== 0) return reject(ERR_STARTED, 'match already started');
    const ip = req.headers.get('CF-Connecting-IP') ?? 'unknown';
    let sameIp = 0;
    for (const c of this.conns.values()) if (c.ip === ip) sameIp++;
    if (sameIp >= MAX_PER_IP_ROOM) return reject(ERR_TOO_MANY, 'too many connections');

    const name = new TextEncoder().encode(url.searchParams.get('name') ?? '').slice(0, 48);
    new Uint8Array(ex.memory.buffer, ex.in_ptr(), name.length).set(name);
    const pid = ex.srv_add(name.length);
    if (pid < 0) return reject(ERR_FULL, 'room full');
    const now = Date.now();
    const conn: Conn = { pid, ip, net, tokens: 60, lastRefill: now, strikes: 0, strikeWindow: now };
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
    // token bucket: 45 msgs/s sustained, burst 60 (clients send ~15-30/s)
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

  /** Strikes decay: only sustained abuse (not a network hiccup) gets a socket closed. */
  private strike(ws: WebSocket, c: Conn) {
    const now = Date.now();
    if (now - c.strikeWindow > 10000) {
      c.strikeWindow = now;
      c.strikes = 0;
    }
    c.strikes++;
    if (c.strikes > 300) {
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
    this.notifyLobby('/leave', { net: c.net, room: this.roomName });
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
    for (const [ws, c] of this.conns) {
      this.notifyLobby('/leave', { net: c.net, room: this.roomName });
      try {
        ws.close(1000, reason);
      } catch {
        /* ignore */
      }
    }
    this.conns.clear();
    this.notifyLobby('/closed', { room: this.roomName });
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
