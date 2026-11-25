// ZONEFALL online server: a Cloudflare Worker that matchmakes players into
// Durable Object rooms. Each Room runs the same Rust/Wasm simulation as the
// offline client, authoritatively, at 30 Hz.
//
// Abuse limits (all approximate, fail-open if the Lobby object is reset):
// * Workers rate-limit bindings: join attempts per IP / IPv6 /64, and (a larger
//   budget, checked after it) per IPv6 /48 so one allocation cannot rotate
//   through its /64s while one /64 cannot use up its neighbours' joins.
// * Lobby object: concurrent sockets per client network (IPv4 address or IPv6 /64,
//   plus an aggregate per IPv6 /48), global concurrent sockets and rooms. Private
//   rooms have their own, smaller cap (public matchmaking always keeps capacity)
//   and each network may only hold two private rooms it created. Joins that
//   leave again do not advance the public room's fill count.
// * Refusals are sent in-band (S_ERROR + reason code) so the player sees why;
//   a network that keeps retrying past the attempt limit gets plain HTTP 429s
//   (an in-band refusal is a socket that stays open until the client closes it).
// * Room object: sockets silent for 2 minutes are closed; a private room closes
//   as soon as its last player leaves.
// * Room object: sockets per IP per room, message size, message rate (token bucket).

import wasmModule from './core.wasm';
import { isBoard, MAX_PLACE, MAX_SURVIVAL_S, MIN_WIN_S, type RankEntry } from './ranking';

export { Ranking } from './ranking';

export interface Env {
  ROOMS: DurableObjectNamespace;
  LOBBY: DurableObjectNamespace;
  ALLOWED_ORIGINS: string;
  ASSETS?: Fetcher;
  /** Workers rate-limiting binding: WebSocket join attempts per client IP / IPv6 /64. */
  JOIN_LIMIT?: RateLimit;
  /** Join attempts per IPv6 /48: several /64s' worth (like MAX_PER_NET48 vs MAX_PER_NET). */
  JOIN_LIMIT48?: RateLimit;
  /** In-band "too many attempts" refusals per network (the rest get plain 429s). */
  REFUSE_LIMIT?: RateLimit;
  /** ... and per IPv6 /48 (four /64s' worth). */
  REFUSE_LIMIT48?: RateLimit;
  /** The global rankings object (see ranking.ts). */
  RANKING?: DurableObjectNamespace;
  /** Self-reported solo results per network. */
  RANK_POST_LIMIT?: RateLimit;
  /** Ranking reads per network. */
  RANK_READ_LIMIT?: RateLimit;
  /** The same two per IPv6 /48 (a host with a routed /48 has 65536 /64s). */
  RANK_POST_LIMIT48?: RateLimit;
  RANK_READ_LIMIT48?: RateLimit;
}

interface SrvExports {
  memory: WebAssembly.Memory;
  in_ptr(): number;
  proto_version(): number;
  srv_init(seed: number, slots: number, flags: number, s0: number, s1: number, s2: number, s3: number): void;
  srv_add(len: number): number;
  srv_remove(pid: number): void;
  srv_msg(pid: number, len: number): void;
  /** behind = 1 when this step catches up on ticks the Room fell behind. */
  srv_step(behind: number): number;
  srv_welcome(pid: number): number;
  srv_snap(pid: number): number;
  srv_out(): number;
  srv_flush(): void;
  srv_phase(): number;
  srv_humans(): number;
  /** Final placement of a player, 0 while alive. */
  srv_place(pid: number): number;
  srv_winner(): number;
  /** Writes the player's in-match name to the input buffer; returns its length. */
  srv_name(pid: number): number;
  /** Sanitises the name in the input buffer in place (the join rules); returns its length. */
  name_clean(len: number): number;
}

const S_PONG = 0x83;
const S_ERROR = 0x84;
const ERR_FULL = 1;
const ERR_STARTED = 2;
const ERR_TOO_MANY = 4;
const ERR_BUSY = 5;
const ERR_PRIV_NET = 6;
const ERR_RATE = 7;
const TICK_MS = 1000 / 30;
const MAX_MSG = 256;
/** Token bucket burst (messages), and its size right after a long silence. */
const BURST = 130;
const MAX_BURST = 2000;
const MAX_ROOM_MS = 25 * 60 * 1000;
const MAX_PER_IP_ROOM = 3;
const MAX_PER_NET = 4;
const MAX_PER_NET48 = 16;
const MAX_SOCKETS = 600;
const MAX_ROOMS = 50;
/** Private rooms may use at most this many of MAX_ROOMS. */
const MAX_PRIVATE_ROOMS = 30;
const MAX_PRIVATE_PER_NET = 2;
const MAX_PRIVATE_PER_NET48 = 4;
const SILENT_MS = 120_000;
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

/** IPv6 /48 aggregate of a /64 net key (IPv4 addresses are their own group). */
export function net48(net: string): string {
  if (!net.endsWith('::/64')) return net;
  return net.split(':').slice(0, 3).join(':') + '::/48';
}

/** Refuse a WebSocket join in-band: a browser cannot read an HTTP error on an
 * upgrade, so the reason travels as S_ERROR on an accepted socket. */
function refuse(code: number, reason: string): Response {
  const pair = new WebSocketPair();
  const server = pair[1];
  server.accept();
  try {
    server.send(new Uint8Array([S_ERROR, code]));
    server.close(4000 + code, reason);
  } catch {
    /* ignore */
  }
  return new Response(null, { status: 101, webSocket: pair[0] });
}

/** Refuse a join, whatever the reason: in-band (so the player sees why) a
 * few times a minute per network and per IPv6 /48 (so rotating /64s does not
 * multiply them), then with a plain HTTP status, which the runtime closes by
 * itself (an in-band refusal is an accepted socket that lingers until the
 * client answers its close frame). */
async function refuseJoin(env: Env, net: string, code: number, reason: string, status = 429): Promise<Response> {
  if (env.REFUSE_LIMIT) {
    const { success } = await env.REFUSE_LIMIT.limit({ key: net });
    if (!success) return plain(reason, status);
  }
  const g48 = net48(net);
  if (env.REFUSE_LIMIT48 && g48 !== net) {
    const { success } = await env.REFUSE_LIMIT48.limit({ key: g48 });
    if (!success) return plain(reason, status);
  }
  return refuse(code, reason);
}

function plain(text: string, status: number): Response {
  return new Response(text, { status, headers: { 'content-type': 'text/plain; charset=utf-8', 'X-Content-Type-Options': 'nosniff' } });
}

function lobbyStub(env: Env) {
  return env.LOBBY.get(env.LOBBY.idFromName('main'));
}

// ---------------------------------------------------------------------------
// Rankings API (storage: the Ranking object, ranking.ts)
//   GET  /api/rank?board=online|solo -> { survival: [...], champions: [...] }
//   POST /api/rank/solo {name, survived, place, won} -> the solo board (+ name as stored)
// The online board is written by Room objects only, never over HTTP.
const RANK_BODY_MAX = 256;
/** Leaving a match alive earlier than this (ticks of play) records nothing. */
const LEAVE_MIN_TICKS = 30 * 30;

function rankStub(env: Env) {
  return env.RANKING ? env.RANKING.get(env.RANKING.idFromName('global')) : null;
}

function rankJson(req: Request, env: Env, body: string, status = 200, extra: Record<string, string> = {}): Response {
  const headers: Record<string, string> = { 'content-type': 'application/json; charset=utf-8', 'X-Content-Type-Options': 'nosniff', Vary: 'Origin', ...extra };
  if (originAllowed(req, env)) headers['Access-Control-Allow-Origin'] = req.headers.get('Origin')!;
  return new Response(body, { status, headers });
}

/** The request body as text, or null when it is larger than `max` bytes. */
async function smallBody(req: Request, max: number): Promise<string | null> {
  if (Number(req.headers.get('content-length') ?? '0') > max) return null;
  if (!req.body) return '';
  const reader = req.body.getReader();
  const buf = new Uint8Array(max);
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (n + value.length > max) {
      await reader.cancel().catch(() => undefined);
      return null;
    }
    buf.set(value, n);
    n += value.length;
  }
  return new TextDecoder().decode(buf.subarray(0, n));
}

/** A player name by the rules joining a match uses ('' = nothing usable). */
function cleanName(raw: string): string {
  const ex = sharedCore();
  const bytes = new TextEncoder().encode(raw).slice(0, 64);
  new Uint8Array(ex.memory.buffer, ex.in_ptr(), bytes.length).set(bytes);
  const n = ex.name_clean(bytes.length);
  return new TextDecoder().decode(new Uint8Array(ex.memory.buffer, ex.in_ptr(), n).slice());
}

async function rankApi(req: Request, env: Env, url: URL): Promise<Response> {
  const err = (status: number, error: string) => rankJson(req, env, JSON.stringify({ error }), status, { 'Cache-Control': 'no-store' });
  const solo = url.pathname === '/api/rank/solo';
  if (!solo && url.pathname !== '/api/rank') return err(404, 'not found');
  if (req.method === 'OPTIONS') {
    if (!originAllowed(req, env)) return plain('Forbidden origin', 403);
    return new Response(null, {
      status: 204,
      headers: {
        'Access-Control-Allow-Origin': req.headers.get('Origin')!,
        'Access-Control-Allow-Methods': 'GET, POST',
        'Access-Control-Allow-Headers': 'content-type',
        'Access-Control-Max-Age': '86400',
        Vary: 'Origin',
      },
    });
  }
  const stub = rankStub(env);
  if (!stub) return err(503, 'unavailable');
  const net = netKey(req.headers.get('CF-Connecting-IP') ?? 'unknown');
  const g48 = net48(net);
  if (!solo) {
    if (req.method !== 'GET') return err(405, 'method');
    const board = url.searchParams.get('board') ?? 'online';
    if (!isBoard(board)) return err(400, 'board');
    if (env.RANK_READ_LIMIT) {
      const { success } = await env.RANK_READ_LIMIT.limit({ key: net });
      if (!success) return err(429, 'rate');
    }
    if (env.RANK_READ_LIMIT48 && g48 !== net) {
      const { success } = await env.RANK_READ_LIMIT48.limit({ key: g48 });
      if (!success) return err(429, 'rate');
    }
    const r = await stub.fetch(`https://rank/top?board=${board}`);
    if (!r.ok) return err(503, 'unavailable');
    return rankJson(req, env, await r.text(), 200, { 'Cache-Control': 'public, max-age=5' });
  }
  // self-reported solo result: browsers only (Origin), a few per minute per network
  if (req.method !== 'POST') return err(405, 'method');
  if (!originAllowed(req, env)) return err(403, 'origin');
  if (env.RANK_POST_LIMIT) {
    const { success } = await env.RANK_POST_LIMIT.limit({ key: net });
    if (!success) return err(429, 'rate');
  }
  if (env.RANK_POST_LIMIT48 && g48 !== net) {
    const { success } = await env.RANK_POST_LIMIT48.limit({ key: g48 });
    if (!success) return err(429, 'rate');
  }
  const text = await smallBody(req, RANK_BODY_MAX);
  if (text === null) return err(413, 'too large');
  let b: Record<string, unknown>;
  try {
    const v: unknown = JSON.parse(text);
    if (!v || typeof v !== 'object' || Array.isArray(v)) return err(400, 'body');
    b = v as Record<string, unknown>;
  } catch {
    return err(400, 'body');
  }
  const { name, survived, place, won } = b;
  if (typeof name !== 'string' || name.length === 0 || name.length > 64) return err(400, 'name');
  if (typeof survived !== 'number' || !Number.isFinite(survived) || survived < 0 || survived > MAX_SURVIVAL_S) return err(400, 'survived');
  if (typeof place !== 'number' || !Number.isInteger(place) || place < 1 || place > MAX_PLACE) return err(400, 'place');
  if (typeof won !== 'boolean' || won !== (place === 1)) return err(400, 'won');
  // nobody outlives a whole lobby while the bots are still parachuting
  if (won && survived < MIN_WIN_S) return err(400, 'survived');
  const clean = cleanName(name);
  if (!clean) return err(400, 'name');
  const entry: RankEntry = { name: clean, seconds: survived, place, won };
  const r = await stub.fetch('https://rank/report?board=solo', { method: 'POST', body: JSON.stringify({ entries: [entry] }) });
  if (!r.ok) return err(503, 'unavailable');
  const top = (await r.json()) as Record<string, unknown>;
  return rankJson(req, env, JSON.stringify({ name: clean, ...top }), 200, { 'Cache-Control': 'no-store' });
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
        // per address (IPv4, or IPv6 /64)
        const { success } = await env.JOIN_LIMIT.limit({ key: net });
        if (!success) return refuseJoin(env, net, ERR_RATE, 'too many join attempts');
      }
      const g48 = net48(net);
      if (env.JOIN_LIMIT48 && g48 !== net) {
        // per IPv6 /48, so one allocation cannot rotate through its /64s. Its
        // own, larger budget, checked after the /64 key: a single /64 can
        // spend at most its own allowance of it, not lock its /48 out.
        const { success } = await env.JOIN_LIMIT48.limit({ key: g48 });
        if (!success) return refuseJoin(env, net, ERR_RATE, 'too many join attempts');
      }
      const name = (url.searchParams.get('name') ?? '').slice(0, 32);
      const code = url.searchParams.get('room');
      // The client checks codes before connecting, so a malformed one is no
      // player's mistake: a plain 400 (no lingering in-band refusal).
      if (code !== null && !ROOM_CODE.test(code)) return plain('Bad room code', 400);
      const r = await lobbyStub(env).fetch('https://lobby/join', { method: 'POST', body: JSON.stringify({ net, code }) });
      if (!r.ok) {
        const err = ((await r.json().catch(() => ({}))) as { error?: string }).error;
        if (err === 'privnet') return refuseJoin(env, net, ERR_PRIV_NET, 'too many private rooms');
        if (r.status === 429) return refuseJoin(env, net, ERR_TOO_MANY, 'too many connections');
        return refuseJoin(env, net, ERR_BUSY, 'server busy', 503);
      }
      const { room } = (await r.json()) as { room: string };
      const stub = env.ROOMS.get(env.ROOMS.idFromName(room));
      const fwd = new URL('https://room/ws');
      fwd.searchParams.set('name', name);
      fwd.searchParams.set('kind', code ? 'priv' : 'pub');
      fwd.searchParams.set('room', room);
      fwd.searchParams.set('net', net);
      return stub.fetch(new Request(fwd.toString(), req));
    }
    if (url.pathname === '/api/rank' || url.pathname.startsWith('/api/rank/')) return rankApi(req, env, url);
    if (env.ASSETS) return env.ASSETS.fetch(req);
    return plain('Not found', 404);
  },
};

/** A core instance for stateless helpers (protocol version, name rules). */
let shared: SrvExports | null = null;
function sharedCore(): SrvExports {
  if (!shared) shared = new WebAssembly.Instance(wasmModule, {}).exports as unknown as SrvExports;
  return shared;
}

function protoVersion(): number {
  return sharedCore().proto_version();
}

// ---------------------------------------------------------------------------
// Lobby: assigns the open public room and keeps approximate global accounting.
export class Lobby {
  private current: { name: string; created: number; count: number } | null = null;
  /** net -> timestamps of live sockets */
  private nets = new Map<string, number[]>();
  /** room -> last activity */
  private rooms = new Map<string, number>();
  /** private room -> network that created it */
  private privOwner = new Map<string, string>();

  constructor(_state: DurableObjectState, _env: Env) {}

  private sweep(now: number) {
    for (const [k, list] of this.nets) {
      const live = list.filter((t) => now - t < ENTRY_TTL_MS);
      if (live.length) this.nets.set(k, live);
      else this.nets.delete(k);
    }
    for (const [k, t] of this.rooms) {
      if (now - t > ENTRY_TTL_MS) {
        this.rooms.delete(k);
        this.privOwner.delete(k);
      }
    }
  }

  private sockets() {
    let n = 0;
    for (const l of this.nets.values()) n += l.length;
    return n;
  }

  private sockets48(g: string) {
    let n = 0;
    for (const [k, l] of this.nets) if (net48(k) === g) n += l.length;
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
      const g48 = net48(net);
      const mine = this.nets.get(net) ?? [];
      if (mine.length >= MAX_PER_NET) return json({ error: 'net' }, 429);
      if (g48 !== net && this.sockets48(g48) >= MAX_PER_NET48) return json({ error: 'net' }, 429);
      if (this.sockets() >= MAX_SOCKETS) return json({ error: 'busy' }, 503);
      let room: string;
      if (body.code) {
        if (!ROOM_CODE.test(body.code)) return json({ error: 'code' }, 400);
        room = `priv-${body.code}`;
        if (!this.rooms.has(room)) {
          // creating a private room: capped globally (public matchmaking keeps
          // its share) and per network
          let total = 0;
          let byNet = 0;
          let by48 = 0;
          for (const [k, owner] of this.privOwner) {
            if (!this.rooms.has(k)) continue;
            total++;
            if (owner === net) byNet++;
            if (net48(owner) === g48) by48++;
          }
          if (total >= MAX_PRIVATE_ROOMS) return json({ error: 'busy' }, 503);
          if (byNet >= MAX_PRIVATE_PER_NET || by48 >= MAX_PRIVATE_PER_NET48) return json({ error: 'privnet' }, 429);
        }
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
      if (body.code && !this.rooms.has(room)) this.privOwner.set(room, net);
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
      // a join that left again (or never got in) must not fill the open public
      // room: otherwise a stream of short joins forces new rooms
      const c = this.current;
      if (c && body.room === c.name && c.count > 0) c.count--;
      return json({ ok: true });
    }
    if (url.pathname === '/closed') {
      if (body.room) {
        this.rooms.delete(body.room);
        this.privOwner.delete(body.room);
      }
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
  /** Token bucket size: BURST, or more right after a silence (see onMessage). */
  cap: number;
  lastRefill: number;
  strikes: number;
  strikeWindow: number;
  lastMsg: number;
  /** In-match name (as the core deduplicated it), for the rankings. */
  name: string;
  /** This player's result has been handed to the rankings (once per match). */
  ranked: boolean;
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
  private isPrivate = false;
  /** Simulated ticks of the play phase so far (survival time = ticks / 30). */
  private playTicks = 0;
  /** Results not yet sent to the Ranking object. */
  private rankQueue: RankEntry[] = [];

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
    this.isPrivate = isPrivate;
    const inst = new WebAssembly.Instance(wasmModule, {});
    this.ex = inst.exports as unknown as SrvExports;
    // public island seed + 128 bits of server-only entropy (match and combat RNG)
    const [seed, s0, s1, s2, s3] = crypto.getRandomValues(new Uint32Array(5));
    this.ex.srv_init(seed, isPrivate ? PRIVATE_SLOTS : PUBLIC_SLOTS, isPrivate ? FLAG_PRIVATE : 0, s0, s1, s2, s3);
    this.createdAt = Date.now();
    this.ended = false;
    this.emptySince = 0;
    this.playTicks = 0;
    this.rankQueue = [];
  }

  /** Queue a human player's result for the online board: when it is placed
   * (killed, or the winner), or when it leaves or the room stops while it is
   * still alive (`force`; survival time only). Bots have no connection, so
   * they are never recorded. */
  private settle(c: Conn, force: boolean) {
    const ex = this.ex;
    if (!ex || c.ranked || !c.name) return;
    try {
      if (ex.srv_phase() === 0) return;
      const place = ex.srv_place(c.pid);
      if (!place && !force) return;
      c.ranked = true;
      // (left while alive within the first moments: nothing worth a row)
      if (!place && this.playTicks < LEAVE_MIN_TICKS) return;
      this.rankQueue.push({ name: c.name, seconds: this.playTicks / 30, place, won: place === 1 && ex.srv_winner() === c.pid });
    } catch {
      /* the simulation trapped: nothing to report */
    }
  }

  private flushRanks() {
    if (!this.rankQueue.length) return;
    const entries = this.rankQueue;
    this.rankQueue = [];
    const stub = rankStub(this.env);
    if (!stub) return;
    this.state.waitUntil(
      stub
        .fetch('https://rank/report?board=online', { method: 'POST', body: JSON.stringify({ entries }) })
        .then(() => undefined)
        .catch(() => undefined),
    );
  }

  async fetch(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const isPrivate = url.searchParams.get('kind') === 'priv';
    this.roomName = url.searchParams.get('room') ?? this.roomName;
    const net = url.searchParams.get('net') ?? 'unknown';
    if (!this.ex || (this.ended && this.conns.size === 0)) this.init(isPrivate);
    const ex = this.ex!;

    const reject = (code: number, reason: string) => {
      // The Lobby already counted this socket.
      this.notifyLobby('/leave', { net, room: this.roomName });
      return refuseJoin(this.env, net, code, reason);
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
    const pair = new WebSocketPair();
    const client = pair[0];
    const server = pair[1];
    server.accept();
    const now = Date.now();
    const playerName = new TextDecoder().decode(new Uint8Array(ex.memory.buffer, ex.in_ptr(), ex.srv_name(pid)).slice());
    const conn: Conn = { pid, ip, net, tokens: BURST, cap: BURST, lastRefill: now, strikes: 0, strikeWindow: now, lastMsg: now, name: playerName, ranked: false };
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
    // token bucket: 45 msgs/s sustained, burst BURST. The first message after
    // a silence (an uplink stall: everything queued arrives at once) opens
    // the bucket to what the stall queued, ~16 messages a second, so the
    // commands reach the core, whose lag accounting drops the ones its idle
    // steps covered (keeping their taps). CMDQ_CAP still bounds the queue.
    const now = Date.now();
    const silent = now - c.lastMsg;
    c.lastMsg = now;
    if (silent > 2000) {
      c.cap = Math.min(MAX_BURST, Math.max(BURST, (silent / 1000) * 16));
      c.tokens = c.cap;
    } else if (c.tokens < BURST) {
      c.cap = BURST;
    }
    c.tokens = Math.min(c.cap, c.tokens + ((now - c.lastRefill) / 1000) * 45);
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
    if (c.strikes > 300) this.serverClose(ws, 1008, 'rate limit');
  }

  /** A close the Room starts itself (strike limit, idle): the player leaves the
   * match at once, but its network keeps the Lobby slot until the client
   * answers the close frame (or 60 s), so a socket the client never lets go
   * of cannot be traded for a new one straight away. */
  private serverClose(ws: WebSocket, code: number, reason: string) {
    const c = this.conns.get(ws);
    if (!c) return;
    try {
      ws.close(code, reason);
    } catch {
      /* ignore */
    }
    this.conns.delete(ws);
    this.settle(c, true);
    this.flushRanks();
    this.ex?.srv_remove(c.pid);
    let done = false;
    const release = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      this.notifyLobby('/leave', { net: c.net, room: this.roomName });
    };
    const timer = setTimeout(release, 60_000);
    ws.addEventListener('close', release);
    ws.addEventListener('error', release);
    if (this.conns.size === 0) {
      this.emptySince = Date.now();
      if (this.isPrivate && !this.ended) this.stop('empty');
    }
  }

  private onClose(ws: WebSocket) {
    const c = this.conns.get(ws);
    if (!c) return;
    this.conns.delete(ws);
    this.settle(c, true);
    this.flushRanks();
    this.ex?.srv_remove(c.pid);
    this.notifyLobby('/leave', { net: c.net, room: this.roomName });
    if (this.conns.size === 0) {
      this.emptySince = Date.now();
      // nobody is coming back to a private room (its code makes a fresh one):
      // free it at once, so its creator's network can open another
      if (this.isPrivate && !this.ended) this.stop('empty');
    }
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
    for (const c of this.conns.values()) this.settle(c, true);
    this.flushRanks();
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
    // dead connections (clients keep a 10 s keep-alive even in background tabs)
    for (const [ws, c] of this.conns) {
      if (now - c.lastMsg > SILENT_MS) this.serverClose(ws, 1001, 'idle');
    }
    let steps = 0;
    try {
      while (this.nextTick <= now && steps < 5) {
        // a catch-up step: at least one more tick is already due after it
        const r = ex.srv_step(now - this.nextTick >= TICK_MS ? 1 : 0);
        if (r & 1) {
          for (const [ws, c] of this.conns) this.sendOut(ws, ex.srv_snap(c.pid));
          ex.srv_flush();
        }
        const phase = ex.srv_phase();
        if (phase === 1) this.playTicks++;
        if (phase !== 0) for (const c of this.conns.values()) if (!c.ranked) this.settle(c, false);
        if (r & 2) return this.stop('match over');
        this.nextTick += TICK_MS;
        steps++;
      }
    } catch (e) {
      console.error('simulation trapped', e);
      return this.stop('server error');
    }
    this.flushRanks();
    if (now - this.nextTick > 500) this.nextTick = now + TICK_MS;
  }
}
