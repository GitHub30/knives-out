// Client game: connects to an authoritative server (Worker or Durable Object),
// predicts the local player with the shared Rust movement code, interpolates
// everyone else, and renders/plays the result.

import type { Core } from './core';
import { CAR_STRIDE, PUFF_FIREBALL, PUFF_FLAME, PUFF_SMOKE, Puffs, type Renderer } from './render/renderer';
import { Instances, rgba, surf } from './render/instances';
import { drawPlane, emitCharacter, emitItem, line } from './render/models';
import { HELI_HULL_HALF, HELI_ROTOR_SPEED, HELI_STRIDE } from './render/heli';
import {
  add, clamp, dirFrom, invert, lerp, lerpAngle, lookAt, mul, norm, perspectiveRZ, qAxis, qIdent, qMul, qRot, qY, rightOf, scale, sub, transformPoint, wrapAngle, type M4, type Q, type V3,
} from './render/math';
import * as P from './protocol';
import type { Transport } from './net';
import { Hud, itemName, TOWN_NAMES, WEAPON_SHORT, type NoiseKind } from './hud';
import type { Input, Edges } from './input';
import type { TouchControls } from './touch';
import type { Sound } from './audio';

export interface Settings {
  sens: number;
  adsSens: number;
  touchSens: number;
  volume: number;
  quality: 'auto' | 'low' | 'medium' | 'high';
  autoFire: boolean;
  showFps: boolean;
  invertY: boolean;
  adsToggle: boolean;
  bots: number;
  touchUi: 'auto' | 'on' | 'off';
}

export interface MatchResult {
  place: number;
  total: number;
  kills: number;
  damage: number;
  won: boolean;
  /** Seconds survived, in simulation time (what the rankings record). */
  time: number;
  /** The player's own in-match name. */
  name: string;
  killerName: string | null;
  /** Set when the whole match is over. */
  winnerName?: string | null;
  final?: boolean;
}

interface Anim {
  phase: number;
  x: number;
  z: number;
  speed: number;
  deadAt: number;
  foot: number;
  wasAlive: boolean;
  seen: number;
  /** When it last touched down (knees give for a moment) and last splashed. */
  landAt: number;
  splash: number;
}

interface Fx {
  k: 'tracer' | 'flash' | 'spark' | 'smoke' | 'fire' | 'debris';
  t: number;
  life: number;
  p: V3;
  v: V3;
  d: V3;
  len: number;
  col: number;
  size: number;
}

interface Pred {
  pos: V3;
  vel: V3;
  mode: number;
  onGround: boolean;
  crouch: boolean;
  inWater: boolean;
  /** While driving or flying: the vehicle's heading. */
  airtime: number;
  /** Index of the car being driven or the helicopter being flown (255 = none). */
  car: number;
}

/** Timed samples of the last few snapshots, oldest first: what is drawn lies between two of them. */
const TRAIL = 8;
/**
 * Add a sample to a trail. Something that sat still (not sent for a
 * while) starts moving from where it stood one snapshot ago.
 */
function trailPush<T extends { t: number }>(h: T[], p: T, step: number) {
  const last = h[h.length - 1];
  if (last && p.t - last.t > 8) {
    h.length = 0;
    h.push({ ...last, t: p.t - step });
  } else if (last && p.t <= last.t) h.length = 0;
  h.push(p);
  if (h.length > TRAIL) h.shift();
}
/** The two samples of a trail around tick `t` and how far between them it lies. */
function trailAt<T extends { t: number }>(h: T[], t: number): { a: T; b: T; f: number } {
  for (let i = h.length - 1; i > 0; i--) {
    if (h[i - 1].t <= t) {
      const span = h[i].t - h[i - 1].t;
      return { a: h[i - 1], b: h[i], f: span > 0 ? clamp((t - h[i - 1].t) / span, 0, 1) : 1 };
    }
  }
  return { a: h[0], b: h[0], f: 0 };
}

/** A car as the client draws it: timed samples to interpolate between (`b` is the latest, `a` the one before). */
interface CarC {
  a: { x: number; y: number; z: number; yaw: number; t: number };
  b: { x: number; y: number; z: number; yaw: number; t: number };
  h: { x: number; y: number; z: number; yaw: number; t: number }[];
  driver: number;
  /** 0 police car, 1 fire truck. */
  kind: number;
  /** Bodywork left in 255ths (0 = wreck) and when it blew up (-1 = intact). */
  hp: number;
  deadAt: number;
  /** Drawn state: wheel roll and steering angles, the pose of the last frame. */
  spin: number;
  steer: number;
  px: number;
  pz: number;
  pyaw: number;
  speed: number;
  /** Distance since the last tyre mark, the rear wheel spots it ended at, time to the next smoke puff. */
  trackAcc: number;
  trackAt: (V3 | null)[];
  smokeT: number;
}

interface HeliPose {
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  roll: number;
  t: number;
}

/** A helicopter as the client draws it: two timed samples to interpolate between. */
interface HeliC {
  a: HeliPose;
  b: HeliPose;
  h: HeliPose[];
  /** From the server: rotor speed 0..1, pilot id (255 nobody, 254 unseen), hull in 255ths, missiles left. */
  rotor: number;
  pilot: number;
  hp: number;
  ammo: number;
  airborne: boolean;
  deadAt: number;
  /** Drawn state: rotor speed and angle, landing gear (0 on its wheels, 1 struts extended, 2 stowed). */
  spin: number;
  angle: number;
  gear: number;
  smokeT: number;
}

/** A loose crate or container as the client draws it. */
interface PropC {
  hx: number;
  hy: number;
  hz: number;
  crate: boolean;
  col: number;
  a: { x: number; y: number; z: number; t: number };
  b: { x: number; y: number; z: number; t: number };
  h: { x: number; y: number; z: number; t: number }[];
  moving: boolean;
  /** Tumbling while it flies; it rights itself when it comes to rest. */
  rot: Q;
  axis: V3;
}

/** A stretch of tyre mark on the ground. */
interface Track {
  p: V3;
  q: Q;
  len: number;
  w: number;
  a: number;
  t: number;
}
/** Half width, half length, top speed (m/s) and name per vehicle kind. */
const CAR_KINDS = [
  { hw: 0.9, hl: 2.45, vmax: 24, name: 'パトカー', wheelR: 0.334, base: 2.92, rear: 1.37, track: 0.77, tyre: 0.24, h: 1.45 },
  { hw: 1.25, hl: 3.05, vmax: 18, name: '消防車', wheelR: 0.385, base: 2.82, rear: 1.22, track: 0.7, tyre: 0.5, h: 3.0 },
];

const DT = 1 / 30;
/** As in core/src/sim.rs (m/s). */
const MISSILE_SPEED = 95;
/** A unit cube with sharp edges (pos + normal, 36 vertices), for crates and containers. */
function unitCube(): Float32Array {
  const out: number[] = [];
  for (let axis = 0; axis < 3; axis++) {
    for (const s of [-1, 1]) {
      const n = [0, 0, 0];
      n[axis] = s;
      const u = (axis + 1) % 3, v = (axis + 2) % 3;
      const corner = (a: number, b: number) => {
        const p = [0, 0, 0];
        p[axis] = 0.5 * s;
        p[u] = a;
        p[v] = b;
        out.push(p[0], p[1], p[2], n[0], n[1], n[2]);
      };
      // counter-clockwise seen from outside
      const q: [number, number][] = s > 0 ? [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]] : [[-0.5, -0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]];
      for (const [a, b] of q) corner(a, b);
    }
  }
  return new Float32Array(out);
}
/** The cube is registered with a renderer once, however many matches are played. */
const cubeShapes = new WeakMap<object, number>();

/** A colour (0xRRGGBB) scaled in brightness. */
function tintHex(hex: number, k: number): number {
  return (Math.min(255, ((hex >> 16) & 255) * k) << 16) | (Math.min(255, ((hex >> 8) & 255) * k) << 8) | Math.min(255, (hex & 255) * k);
}
const DEBUG = typeof location !== 'undefined' && new URLSearchParams(location.search).has('debug');
/** Seconds a tyre mark stays on the ground. */
const TRACK_LIFE = 50;
type Snap = P.Snapshot & { byId: Map<number, P.RemotePlayer> };
const HEAL_TIME = [3, 6, 2];
const EYE = (c: boolean) => (c ? 1.12 : 1.62);

export interface GameDeps {
  core: Core;
  renderer: Renderer;
  hud: Hud;
  input: Input;
  touch: TouchControls;
  sound: Sound;
  settings: Settings;
  transport: Transport;
  onResult: (r: MatchResult) => void;
  onLobby: (info: { count: number; timeLeft: number; isPrivate: boolean; slots: number } | null) => void;
  onPause: (paused: boolean) => void;
  onError: (msg: string) => void;
}

export class Game {
  private d: GameDeps;
  welcome: P.Welcome | null = null;
  myId = -1;
  private players = new Map<number, P.PlayerInfo>();
  private items = new Map<number, P.ItemInfo>();
  private snaps: Snap[] = [];
  latest: P.Snapshot | null = null;
  private lastSnapAt = 0;
  private renderTick = 0;
  private interpDelay = 3;
  private pred: Pred | null = null;
  private predPrev: V3 = [0, 0, 0];
  private smooth: V3 = [0, 0, 0];
  private cars: CarC[] = [];
  private myCar = -1;
  /** Free car within reach (-1 = none) and whether an item is, too. */
  private nearCar = -1;
  private nearItem = false;
  private helis: HeliC[] = [];
  private myHeli = -1;
  /** Free helicopter within reach (-1 = none). */
  private nearHeli = -1;
  private nearDoor = -1;
  /** Doors as generated, whether each stands open and how far it has swung (0 shut .. 1 open). */
  private doors: { x: number; y: number; z: number; w: number; h: number; shut: number; open: number; col: number; isOpen: boolean; a: number; hold: number }[] = [];
  /** Things heard lately, for the direction ring. */
  private noises: { kind: NoiseKind; x: number; z: number; t: number; life: number }[] = [];
  private vehNoiseT = 0;
  private heliPoses = new Float32Array(0);
  /** Missiles already heard leaving (by id), and when a smoke puff was last left behind them. */
  private missilesSeen = new Set<number>();
  private missileSmokeT = 0;
  private predPrevYaw = 0;
  private smoothYaw = 0;
  private lastLookAt = 0;
  private carPoses = new Float32Array(0);
  /** Per car while I drive: its predicted pose one command ago, the offset that hides a correction, and how far it is drawn from the prediction (0..1). */
  private carPrev = new Float32Array(0);
  private carSm = new Float32Array(0);
  private carPw = new Float32Array(0);
  private carsWere: Float32Array | null = null;
  private myLandAt = -9;
  private glassPanes: Float32Array = new Float32Array(0);
  private glass = new Instances(256);
  private tracks: Track[] = [];
  private props: PropC[] = [];
  private pending: { cmd: P.CmdQ; shots: number; at: number }[] = [];
  private outbox: P.CmdQ[] = [];
  private seq = 0;
  private cmdAcc = 0;
  private edges: Edges | null = null;
  yaw = 0;
  pitch = 0;
  private recoil = 0;
  private fireCd = 0;
  private slotReq = -1;
  private slotReqAt = 0;
  private displayCur = 2;
  private anims = new Map<number, Anim>();
  private fx: Fx[] = [];
  private markers: { x: number; z: number; t: number }[] = [];
  private boxes = new Instances(4096);
  private glow = new Instances(1024);
  private smoke = new Instances(512);
  private puffs = new Puffs();
  private time = 0;
  private shake = 0;
  private phase = -1;
  private zoneShrinking = false;
  private zoneStage = -1;
  private ended = false;
  private resultSent = false;
  private paused = false;
  private running = false;
  private raf = 0;
  private lastFrame = 0;
  private pingSent = 0;
  private rtt = 0;
  /** Recent ping samples for the path's rtt (their median: a real change of
   * route is taken within two pings). A ping that was in flight during an
   * uplink stall measured the stall, not the path, and is left out. */
  private rttSamples: number[] = [];
  /** When a snapshot last reported a debt of 4 ticks or more while this
   * client was producing commands (an uplink stall; a route change or jitter
   * leaves a tick or three, and a hidden tab or a long task leaves a debt
   * of the client's own making). */
  private stallSeenAt = -1;
  /** End of the last gap between frames of more than 100 ms (no commands were
   * made during it). */
  private lastGapEnd = -1e9;
  private fps = 60;
  private lastHealth = 100;
  private myFoot = 0;
  private lastZoneBeep = 0;
  private viewProj: M4 = new Float32Array(16);
  private camEye: V3 = [0, 0, 0];
  private camYaw = 0;
  private matchStartTime = 0;
  /** Play-phase time (s) the player has survived: up to the snapshot that reports its death. */
  private aliveT = -1;
  private deathSeen = false;
  private welcomeWait: { resolve: () => void; reject: (e: Error) => void } | null = null;
  private visHandler = () => this.onVisibility();
  /** Keep-alive that also runs in background tabs (rAF stops there). */
  private keepAlive: ReturnType<typeof setInterval> | null = null;

  constructor(deps: GameDeps) {
    this.d = deps;
  }

  // ------------------------------------------------------------------ lifecycle
  start(): Promise<void> {
    const t = this.d.transport;
    t.onMessage = (b) => this.onMessage(b);
    t.onClose = (reason) => {
      if (reason === 'ended') return;
      if (!this.ended) this.d.onError(reason);
      this.welcomeWait?.reject(new Error(reason));
    };
    document.addEventListener('visibilitychange', this.visHandler);
    this.keepAlive = setInterval(() => {
      if (this.running && performance.now() - this.pingSent > 9000) {
        this.pingSent = performance.now();
        this.send(P.encodePing(this.pingSent >>> 0));
      }
    }, 10000);
    return new Promise<void>((resolve, reject) => {
      this.welcomeWait = { resolve, reject };
      setTimeout(() => reject(new Error('サーバーからの応答がありません')), 15000);
    });
  }

  destroy() {
    this.running = false;
    cancelAnimationFrame(this.raf);
    if (this.keepAlive) clearInterval(this.keepAlive);
    this.keepAlive = null;
    document.removeEventListener('visibilitychange', this.visHandler);
    this.d.transport.close();
    this.d.sound.loop('plane', 0);
    this.d.sound.loop('wind', 0);
    this.d.sound.loop('engine', 0);
    this.d.sound.rotor(0);
    this.d.input.enabled = false;
    this.d.input.unlock();
    this.d.touch.releaseAll();
  }

  private onVisibility() {
    if (document.hidden && this.d.transport.kind === 'offline' && !this.paused && !this.ended) this.setPaused(true);
  }

  setPaused(p: boolean) {
    if (this.d.transport.kind !== 'offline') p = false;
    this.paused = p;
    this.d.transport.setPaused?.(p);
    this.d.onPause(p);
    if (p) {
      this.d.input.unlock();
      this.d.touch.releaseAll();
    }
  }

  get isPaused() {
    return this.paused;
  }

  // ------------------------------------------------------------------ network
  private onMessage(buf: ArrayBuffer) {
    const type = new Uint8Array(buf)[0];
    if (type === P.S_WELCOME) this.onWelcome(buf);
    else if (type === P.S_SNAP) {
      const s = P.decodeSnapshot(buf);
      if (s) this.onSnapshot(s);
    } else if (type === P.S_PONG) {
      const sent = new DataView(buf).getUint32(1, true);
      const now = performance.now() >>> 0;
      const sample = (now - sent) >>> 0;
      this.rtt = lerp(this.rtt || sample, sample, 0.3);
      if (this.stallSeenAt < 0 || (((this.stallSeenAt >>> 0) - sent) | 0) < 0) {
        this.rttSamples.push(sample);
        if (this.rttSamples.length > 3) this.rttSamples.shift();
      }
    } else if (type === P.S_ERROR) {
      const code = new Uint8Array(buf)[1];
      const msg =
        [
          '不明なエラー',
          '部屋が満員です',
          '試合はすでに開始しています',
          'バージョンが一致しません',
          '同じネットワークからの接続数が多すぎます',
          'サーバーが混雑しています。しばらくしてから再試行してください',
          '同じネットワークから作成できる非公開ルームの数を超えています',
          '接続の試行が多すぎます。1分ほど待ってから再試行してください',
          'ルームコードが正しくありません',
        ][code] ?? 'エラー';
      this.d.onError(msg);
      this.welcomeWait?.reject(new Error(msg));
    }
  }

  private onWelcome(buf: ArrayBuffer) {
    const w = P.decodeWelcome(buf);
    if (!w) return this.welcomeWait?.reject(new Error('不正なデータを受信しました'));
    if (w.proto !== P.PROTO || w.proto !== this.d.core.proto) {
      return this.welcomeWait?.reject(new Error('サーバーとクライアントのバージョンが一致しません。ページを再読み込みしてください。'));
    }
    this.welcome = w;
    this.myId = w.pid;
    const core = this.d.core;
    const checksum = core.init(w.seed);
    if (checksum !== w.checksum) return this.welcomeWait?.reject(new Error('ワールド生成が一致しません（バージョン不一致）'));
    this.d.renderer.uploadWorld({
      terrainVerts: core.bytes(0),
      terrainIdx: core.bytes(1),
      terrainChunks: core.f32(2),
      propVerts: core.bytes(3),
      propChunks: core.f32(4),
      heights: this.heightsFromCore(),
      mapSize: 1024,
      waterLevel: 0,
      cars: core.f32(8),
      carKinds: core.bytes(9),
      grass: core.bytes(11),
      trees: core.f32(12),
    });
    this.d.hud.setMap(core.bytes(5), 512);
    this.glassPanes = core.f32(10);
    const dr = core.f32(15);
    this.doors = [];
    this.noises = [];
    for (let i = 0; i + 8 <= dr.length; i += 8) this.doors.push({ x: dr[i], y: dr[i + 1], z: dr[i + 2], w: dr[i + 3], h: dr[i + 4], shut: dr[i + 5], open: dr[i + 6], col: dr[i + 7], isOpen: false, a: 0, hold: 0 });
    for (let k = 0; k < this.doors.length; k++) core.setDoor(k, false);
    const pr = core.f32(13);
    this.props = [];
    for (let i = 0; i + 8 <= pr.length; i += 8) {
      const at = { x: (pr[i] + pr[i + 3]) / 2, y: pr[i + 1], z: (pr[i + 2] + pr[i + 5]) / 2, t: 0 };
      this.props.push({ hx: (pr[i + 3] - pr[i]) / 2, hy: pr[i + 4] - pr[i + 1], hz: (pr[i + 5] - pr[i + 2]) / 2, crate: pr[i + 6] > 0.5, col: pr[i + 7], a: at, b: { ...at }, h: [{ ...at }], moving: false, rot: qIdent(), axis: [1, 0, 0] });
    }
    const c0 = core.f32(8);
    const kinds = core.bytes(9);
    this.cars = [];
    for (let i = 0; i + 4 <= c0.length; i += 4) {
      const at = { x: c0[i], y: c0[i + 1], z: c0[i + 2], yaw: c0[i + 3], t: 0 };
      this.cars.push({
        a: at, b: { ...at }, h: [{ ...at }], driver: 255, kind: Math.min(kinds[i >> 2] ?? 0, CAR_KINDS.length - 1), hp: 255, deadAt: -1,
        spin: 0, steer: 0, px: at.x, pz: at.z, pyaw: at.yaw, speed: 0, trackAcc: 0, trackAt: [null, null], smokeT: 0,
      });
    }
    this.myCar = -1;
    this.carPrev = new Float32Array(this.cars.length * 4);
    this.carSm = new Float32Array(this.cars.length * 4);
    this.carPw = new Float32Array(this.cars.length);
    const h0 = core.helis();
    this.helis = [];
    for (let i = 0; i + 4 <= h0.length; i += 4) {
      const at: HeliPose = { x: h0[i], y: h0[i + 1], z: h0[i + 2], yaw: h0[i + 3], pitch: 0, roll: 0, t: 0 };
      this.helis.push({ a: at, b: { ...at }, h: [{ ...at }], rotor: 0, pilot: 255, hp: 255, ammo: 0, airborne: false, deadAt: -1, spin: 0, angle: i, gear: 0, smokeT: 0 });
    }
    this.myHeli = -1;
    this.missilesSeen.clear();
    this.tracks = [];
    core.freeMeshes();
    for (const p of w.players) this.players.set(p.id, p);
    for (const it of w.items) this.items.set(it.id, it);
    this.interpDelay = w.snapEvery * 1.5 + 1.5;
    this.running = true;
    this.lastFrame = performance.now();
    this.d.input.enabled = true;
    this.d.input.reset();
    this.raf = requestAnimationFrame((t) => this.frame(t));
    this.welcomeWait?.resolve();
    this.welcomeWait = null;
  }

  private heightsFromCore(): Float32Array {
    const h = new Float32Array(257 * 257);
    for (let j = 0; j < 257; j++) for (let i = 0; i < 257; i++) h[j * 257 + i] = this.d.core.height(i * 4, j * 4);
    return h;
  }

  private onSnapshot(s: P.Snapshot) {
    if (this.latest && s.tick <= this.latest.tick) return;
    const byId = new Map<number, P.RemotePlayer>();
    for (const p of s.players) byId.set(p.id, p);
    this.snaps.push({ ...s, byId });
    while (this.snaps.length > 48) this.snaps.shift();
    const first = !this.latest;
    this.latest = s;
    this.lastSnapAt = performance.now();
    if (first) {
      this.renderTick = s.tick - this.interpDelay;
      this.displayCur = s.self.cur;
    }
    for (const e of s.events) this.onEvent(e);
    this.onPhase(s);
    // (where my prediction had the cars, before the server's word replaces it)
    this.carsWere = this.pred && this.pred.mode === P.MODE_DRIVE ? this.d.core.carPoses().slice() : null;
    this.onCars(s);
    this.onHelis(s);
    this.onProps(s);
    this.onDoors(s);
    this.reconcile(s);
    if (this.slotReq < 0 || performance.now() - this.slotReqAt > 1200 || s.self.cur === this.slotReq) {
      this.displayCur = s.self.cur;
      this.slotReq = -1;
    }
    // took damage (zone included)
    if (s.self.health < this.lastHealth - 0.5 && s.self.alive) {
      this.d.hud.hurtFlash(this.lastHealth - s.self.health);
    }
    this.lastHealth = s.self.health;
  }

  /** Seconds survived: simulation time (pauses and lag do not count), else the wall clock. */
  private survived(): number {
    return this.aliveT >= 0 ? this.aliveT : (performance.now() - this.matchStartTime) / 1000;
  }

  private onPhase(s: P.Snapshot) {
    const hud = this.d.hud;
    if (s.phase === P.PH_PLAY && !this.deathSeen) {
      this.aliveT = s.phaseT;
      this.deathSeen = !s.self.alive;
    }
    if (s.phase === P.PH_LOBBY) {
      this.d.onLobby({ count: s.total, timeLeft: s.phaseT, isPrivate: ((this.welcome?.flags ?? 0) & 2) !== 0, slots: this.welcome?.slots ?? 30 });
    }
    if (s.phase !== this.phase) {
      const prev = this.phase;
      this.phase = s.phase;
      if (s.phase === P.PH_PLAY) {
        this.d.onLobby(null);
        this.matchStartTime = performance.now();
        const touch = matchMedia('(pointer: coarse)').matches;
        hud.showBanner(touch ? '「降下」ボタンで飛行機から飛び降りよう' : 'SPACE で飛行機から降下', 4);
        this.yaw = Math.atan2(s.plane.dx, s.plane.dz);
        this.pitch = -0.25;
      }
      if (s.phase === P.PH_END && prev === P.PH_PLAY) this.finish(s);
    }
    if (s.phase === P.PH_PLAY) {
      if (s.zone.stage !== this.zoneStage || s.zone.shrinking !== this.zoneShrinking) {
        if (this.zoneStage >= 0 && s.zone.shrinking && !this.zoneShrinking) hud.showBanner('⚠ 安全地帯の縮小が始まりました', 3, 'warn');
        else if (this.zoneStage >= 0 && !s.zone.shrinking && s.zone.stage > this.zoneStage) hud.showBanner('次の安全地帯が表示されました', 2.5);
        this.zoneStage = s.zone.stage;
        this.zoneShrinking = s.zone.shrinking;
      }
    }
  }

  private nameOf(id: number) {
    if (id === P.KILLER_ZONE) return '安全地帯';
    return this.players.get(id)?.name ?? `#${id}`;
  }

  private onEvent(e: P.GameEvent) {
    const hud = this.d.hud;
    const snd = this.d.sound;
    switch (e.t) {
      case 'pinfo':
        if (e.info.left) this.players.delete(e.info.id);
        else this.players.set(e.info.id, e.info);
        break;
      case 'item':
        this.items.set(e.item.id, e.item);
        break;
      case 'idel':
        this.items.delete(e.id);
        break;
      case 'ichunk':
        for (const [id, it] of this.items) if (P.itemChunk(it.x, it.z) === e.chunk) this.items.delete(id);
        for (const it of e.items) this.items.set(it.id, it);
        break;
      case 'shot': {
        const mine = e.shooter === this.myId;
        if (!mine) {
          if (e.shooter === 255) {
            // a shooter we cannot see: the server only sends a coarse origin,
            // so draw just the last part of the tracer and no muzzle flash
            const k = 0.6;
            const mid: V3 = [lerp(e.from[0], e.to[0], k), lerp(e.from[1], e.to[1], k), lerp(e.from[2], e.to[2], k)];
            this.spawnTracer(mid, e.to, e.weapon);
          } else {
            this.spawnTracer(e.from, e.to, e.weapon);
            this.spawnFlash(e.from);
          }
          const d = Math.hypot(e.from[0] - this.camEye[0], e.from[1] - this.camEye[1], e.from[2] - this.camEye[2]);
          snd.shot(e.weapon, d, this.panOf(e.from));
          if (d < 260) this.markers.push({ x: e.from[0], z: e.from[2], t: this.time });
          if (d < 260) this.heard('shot', e.from[0], e.from[2], 1.8);
          if (this.markers.length > 30) this.markers.shift();
        }
        if (e.hit === 3 || e.hit === 0) this.spawnImpact(e.to, 0x9a8b70);
        else this.spawnImpact(e.to, e.hit === 2 ? 0xff5040 : 0xd23a2a);
        break;
      }
      case 'hit': {
        hud.hitmarker(e.head, e.kill);
        snd.hit(e.head, e.kill);
        if (e.armor) snd.armorHit();
        const p = this.interpPlayer(e.victim);
        if (p) {
          const sp = this.project([p.x, p.y + (p.crouch ? 1.3 : 1.9), p.z]);
          if (sp) hud.damageNumber(sp[0], sp[1], e.dmg, e.head, e.armor);
        }
        break;
      }
      case 'hurt': {
        const me = this.myPos();
        if (e.attacker !== P.KILLER_ZONE && me) {
          const srcYaw = Math.atan2(e.sx - me[0], e.sz - me[2]);
          hud.damageDir(-wrapAngle(srcYaw - this.yaw));
          snd.hurt();
        } else if (this.time - this.lastZoneBeep > 1) {
          this.lastZoneBeep = this.time;
          snd.ui('zone');
        }
        break;
      }
      case 'kill': {
        const killer = this.nameOf(e.killer);
        const victim = this.nameOf(e.victim);
        const weapon = e.killer === P.KILLER_ZONE ? '圏外' : e.weapon === P.WK_NADE ? 'グレネード' : e.weapon === P.WK_CAR ? 'パトカー' : e.weapon === P.WK_TRUCK ? '消防車' : e.weapon === P.WK_BOOM ? '車両爆発' : e.weapon === P.WK_MISSILE ? 'ミサイル' : e.weapon === P.WK_HELI ? 'CH-47J' : WEAPON_SHORT[e.weapon] ?? '?';
        const mine = e.killer === this.myId ? 'me' : e.victim === this.myId ? 'victim' : '';
        hud.killfeed(killer, victim, (e.head ? '◎ ' : '') + weapon, mine);
        if (e.killer === this.myId && e.victim !== this.myId) hud.showBanner(`キル！ ${victim}${e.head ? '（ヘッドショット）' : ''}`, 2, 'kill');
        if (e.victim === this.myId) {
          hud.showBanner(e.killer === P.KILLER_ZONE ? '安全地帯の外で力尽きた…' : `${killer} に倒された`, 3, 'warn');
          snd.ui('lose');
          setTimeout(() => {
            if (this.running) this.finishPersonal();
          }, 1800);
        }
        break;
      }
      case 'boom': {
        const p: V3 = [e.x, e.y, e.z];
        this.spawnExplosion(p);
        const d = Math.hypot(p[0] - this.camEye[0], p[1] - this.camEye[1], p[2] - this.camEye[2]);
        snd.explosion(d, this.panOf(p));
        if (d < 220 && d > 3) this.heard('boom', p[0], p[2], 2.2);
        this.shake = Math.max(this.shake, clamp(1.2 - d / 40, 0, 1));
        break;
      }
      case 'pick':
        hud.showToast(`${itemName(e.kind, e.amount)} を入手`);
        snd.ui('pickup');
        break;
      case 'throw':
        if (e.id === this.myId) snd.ui('throw');
        break;
      case 'land':
        if (e.id === this.myId) snd.ui('land');
        else {
          // someone else touching down: dust, and their knees give
          const p = this.interpPlayer(e.id);
          if (p && Math.hypot(p.x - this.camEye[0], p.z - this.camEye[2]) < 150) {
            this.spawnDust([p.x, p.y, p.z]);
            const a = this.anims.get(e.id);
            if (a) a.landAt = this.time;
          }
        }
        break;
      case 'end':
        break;
    }
  }

  private finishPersonal() {
    const s = this.latest;
    if (!s || this.resultSent || s.self.alive) return;
    this.resultSent = true;
    this.d.onResult({
      place: s.self.place || s.alive + 1,
      total: s.total,
      kills: s.self.kills,
      damage: s.self.dmg,
      won: false,
      time: this.survived(),
      name: this.nameOf(this.myId),
      killerName: s.self.killer === 255 ? null : this.nameOf(s.self.killer),
    });
  }

  private finish(s: P.Snapshot) {
    this.ended = true;
    // the server names a dead winner only when the last players died together
    const won = s.winner === this.myId;
    if (won) {
      this.d.hud.showBanner('🏆 チャンピオン！ 最後の生存者', 6, 'win');
      this.d.sound.ui('win');
    }
    // Always show the final card, also for players who were already
    // eliminated and chose to spectate.
    this.resultSent = true;
    const winnerName = s.winner !== 255 ? this.nameOf(s.winner) : null;
    setTimeout(
      () =>
        this.running &&
        this.d.onResult({
          place: won ? 1 : s.self.place || 2,
          total: s.total,
          kills: s.self.kills,
          damage: s.self.dmg,
          won,
          time: this.survived(),
          name: this.nameOf(this.myId),
          killerName: s.self.killer === 255 || won ? null : this.nameOf(s.self.killer),
          winnerName,
          final: true,
        }),
      won ? 2500 : 900,
    );
  }

  private send(data: Uint8Array<ArrayBuffer>) {
    this.d.transport.send(data);
  }

  /** Private rooms: ask the server to start early (only honoured for the host). */
  requestStart() {
    this.send(new Uint8Array([P.C_START]));
  }

  // ------------------------------------------------------------------ prediction
  private speedMult(s: P.SelfState) {
    const w = s.cur < 2 ? s.slots[s.cur].kind : 0;
    let m = this.d.core.wdefs[w]?.speed ?? 1;
    if (s.healT > 0) m *= 0.55;
    if (s.boost > 0) m *= 1.06;
    return m;
  }

  private step(q: P.CmdQ, mult: number, fresh: boolean) {
    const p = this.pred;
    if (!p) return;
    const io = this.d.core.pred();
    io[0] = p.pos[0]; io[1] = p.pos[1]; io[2] = p.pos[2];
    io[3] = p.vel[0]; io[4] = p.vel[1]; io[5] = p.vel[2];
    io[6] = p.mode; io[7] = p.onGround ? 1 : 0; io[8] = p.crouch ? 1 : 0; io[9] = p.inWater ? 1 : 0; io[10] = p.airtime; io[11] = mult;
    io[12] = q.mx; io[13] = q.mz; io[14] = q.yaw; io[15] = q.pitch; io[16] = q.buttons; io[18] = p.car;
    this.d.core.predStep();
    const o = this.d.core.pred();
    p.pos = [o[0], o[1], o[2]];
    p.vel = [o[3], o[4], o[5]];
    p.mode = o[6];
    p.onGround = o[7] !== 0;
    p.crouch = o[8] !== 0;
    p.inWater = o[9] !== 0;
    p.airtime = o[10];
    if (fresh) {
      const ev = o[17];
      if (ev & 2) this.d.sound.ui('jump');
      if (ev & 4) this.d.sound.ui('chute');
      if (ev & 1) {
        this.d.sound.ui('land');
        this.spawnDust(p.pos);
        this.myLandAt = this.time;
      }
    }
  }

  /** Car poses from the server: for drawing, and for collisions in the core. */
  private onCars(s: P.Snapshot) {
    const core = this.d.core;
    let mine = -1;
    for (const c of s.cars) {
      const k = this.cars[c.id];
      if (!k) continue;
      // a car that sat still starts moving from where it stood one snapshot ago
      k.a = s.tick - k.b.t > 8 ? { ...k.b, t: s.tick - (this.welcome?.snapEvery ?? 2) } : k.b;
      k.b = { x: c.x, y: c.y, z: c.z, yaw: c.yaw, t: s.tick };
      trailPush(k.h, k.b, this.welcome?.snapEvery ?? 2);
      k.driver = c.driver;
      if (c.hp === 0 && k.hp > 0) {
        // it just blew up (unless this is old news from far away)
        const near = Math.hypot(c.x - this.camEye[0], c.z - this.camEye[2]) < 320;
        k.deadAt = near ? this.time : this.time - 1000;
        if (near) this.spawnCarExplosion([c.x, c.y + 1, c.z], k.kind);
      } else if (c.hp > 0 && k.hp === 0) k.deadAt = -1;
      k.hp = c.hp;
      if (c.driver === this.myId) mine = c.id;
      core.setCar(c.id, c.x, c.y, c.z, c.yaw);
      core.setCarMotion(c.id, c.vx, c.vz, c.spin, c.driver === 255);
    }
    const driving = s.self.alive && s.self.mode === P.MODE_DRIVE;
    this.myCar = driving ? (mine >= 0 ? mine : this.myCar) : -1;
    // my own car follows my prediction: keep the server's copy of it out of
    // the way of the camera and of the rays
    if (this.myCar >= 0) {
      core.setCar(this.myCar, 0, -1000, 0, 0);
      core.setCarMotion(this.myCar, 0, 0, 0, false);
    }
  }

  /** Helicopter poses from the server: for drawing, and solid in the core where they are now. */
  private onHelis(s: P.Snapshot) {
    const core = this.d.core;
    const step = this.welcome?.snapEvery ?? 2;
    let mine = -1;
    for (const h of s.helis) {
      const k = this.helis[h.id];
      if (!k) continue;
      k.a = s.tick - k.b.t > 8 ? { ...k.b, t: s.tick - step } : k.b;
      k.b = { x: h.x, y: h.y, z: h.z, yaw: h.yaw, pitch: h.pitch, roll: h.roll, t: s.tick };
      trailPush(k.h, k.b, step);
      if (h.hp === 0 && k.hp > 0) {
        // it just blew up (unless this is old news from far away); the blast itself comes as an event
        k.deadAt = Math.hypot(h.x - this.camEye[0], h.z - this.camEye[2]) < 320 ? this.time : this.time - 1000;
      } else if (h.hp > 0 && k.hp === 0) k.deadAt = -1;
      k.rotor = h.rotor;
      k.pilot = h.pilot;
      k.hp = h.hp;
      k.ammo = h.ammo;
      k.airborne = h.airborne;
      if (h.pilot === this.myId) mine = h.id;
      core.setHeli(h.id, h.x, h.y, h.z, h.yaw);
    }
    const flying = s.self.alive && s.self.mode === P.MODE_FLY;
    this.myHeli = flying ? (mine >= 0 ? mine : this.myHeli) : -1;
    // the one I fly follows my prediction: keep the server's copy of it out
    // of the way of the camera and of the aiming ray
    if (this.myHeli >= 0) core.setHeli(this.myHeli, 0, -1000, 0, 0);
    // a missile seen for the first time has just left its tube
    const live = new Set<number>();
    for (const m of s.missiles) {
      live.add(m.id);
      if (this.missilesSeen.has(m.id)) continue;
      const d = dirFrom(m.yaw, m.pitch);
      // back to the pod it came from (it is a tick or two old by now)
      const at: V3 = [m.x - d[0] * 4, m.y - d[1] * 4, m.z - d[2] * 4];
      this.spawnFlash(at);
      for (let i = 0; i < 5; i++) {
        this.fx.push({ k: 'smoke', t: 0, life: 0.7 + Math.random() * 0.5, p: [...at], v: [-d[0] * (4 + i * 3) + (Math.random() - 0.5) * 2, -d[1] * (4 + i * 3) + 0.6, -d[2] * (4 + i * 3) + (Math.random() - 0.5) * 2], d: [0, 0, 0], len: 0, col: 0xcfcbc4, size: 0.5 + Math.random() * 0.3 });
      }
      this.d.sound.missile(Math.hypot(at[0] - this.camEye[0], at[1] - this.camEye[1], at[2] - this.camEye[2]), this.panOf(at));
    }
    this.missilesSeen = live;
  }

  private heliPose(k: number): HeliPose {
    const h = trailAt(this.helis[k].h, this.renderTick);
    const f = h.f;
    const pitch = lerp(h.a.pitch, h.b.pitch, f), roll = lerp(h.a.roll, h.b.roll, f);
    if (k === this.myHeli && this.pred && this.pred.mode === P.MODE_FLY) {
      const p = this.renderPos();
      const a = clamp(this.cmdAcc / DT, 0, 1);
      return { x: p[0], y: p[1], z: p[2], yaw: lerpAngle(this.predPrevYaw, this.pred.airtime, a) + this.smoothYaw, pitch, roll, t: 0 };
    }
    return { x: lerp(h.a.x, h.b.x, f), y: lerp(h.a.y, h.b.y, f), z: lerp(h.a.z, h.b.z, f), yaw: lerpAngle(h.a.yaw, h.b.yaw, f), pitch, roll, t: 0 };
  }

  /** The free, landed helicopter a player at `me` could board, or -1 (the server allows 1.8 m from the hull). */
  private heliInReach(me: V3): number {
    for (let k = 0; k < this.helis.length; k++) {
      const h = this.helis[k];
      if (h.pilot !== 255 || h.hp === 0 || h.airborne || Math.abs(me[1] - h.b.y) > 1.6) continue;
      const dx = me[0] - h.b.x, dz = me[2] - h.b.z;
      if (dx * dx + dz * dz > 144) continue;
      const sn = Math.sin(h.b.yaw), cs = Math.cos(h.b.yaw);
      const lx = -dx * cs + dz * sn, lz = dx * sn + dz * cs;
      if (Math.hypot(Math.max(0, Math.abs(lx) - HELI_HULL_HALF[0]), Math.max(0, Math.abs(lz) - HELI_HULL_HALF[2])) < 1.6) return k;
    }
    return -1;
  }

  /** Where the missiles of the helicopter I fly would go: the server aims both pods at this point. */
  private missileAim(): V3 | null {
    const p = this.pred;
    if (!p || p.mode !== P.MODE_FLY) return null;
    const pos = this.renderPos();
    const off = clamp(wrapAngle(this.yaw - p.airtime), -1.05, 1.05);
    // (as the server does: flatter than a downward look, so they land ahead of the nose)
    const d = dirFrom(p.airtime + off, this.pitch < 0 ? Math.max(this.pitch * 0.72, -0.9) : Math.min(this.pitch, 0.6));
    const mid: V3 = [pos[0], pos[1] + 2, pos[2]];
    const reach = 380;
    const t = this.d.core.ray(mid, d, reach);
    return add(mid, scale(d, Math.max(30, t >= 0 ? t : reach)));
  }

  /** Doors: the ones listed stand open, every other one is shut (also in the core, where a shut door is solid). */
  private onDoors(s: P.Snapshot) {
    const open = new Set(s.doors);
    for (let k = 0; k < this.doors.length; k++) {
      const d = this.doors[k];
      const o = open.has(k);
      // (one we just walked into opens at once here; the server follows)
      if (o === d.isOpen || (!o && this.time < d.hold)) continue;
      d.isOpen = o;
      this.d.core.setDoor(k, o);
      const at: V3 = [d.x + Math.sin(d.shut) * d.w * 0.5, d.y + 1, d.z + Math.cos(d.shut) * d.w * 0.5];
      const dist = Math.hypot(at[0] - this.camEye[0], at[1] - this.camEye[1], at[2] - this.camEye[2]);
      this.d.sound.door(dist, this.panOf(at), o);
      if (dist < 30) this.heard('door', at[0], at[2], 1.2);
    }
  }

  /** The nearest door within reach (as the server measures it), or -1. */
  private doorInReach(me: V3): number {
    let best = -1;
    let bd = 2.2;
    for (let k = 0; k < this.doors.length; k++) {
      const d = this.doors[k];
      if (Math.abs(me[1] - d.y) > 1.5) continue;
      const dd = Math.hypot(me[0] - (d.x + Math.sin(d.shut) * d.w * 0.5), me[2] - (d.z + Math.cos(d.shut) * d.w * 0.5));
      if (dd < bd) {
        bd = dd;
        best = k;
      }
    }
    return best;
  }

  /** Door leaves, swinging on their hinges. */
  private drawDoors(eye: V3, dt: number) {
    let cube = cubeShapes.get(this.d.renderer);
    if (cube === undefined) {
      cube = this.d.renderer.addShapes([unitCube()]);
      cubeShapes.set(this.d.renderer, cube);
    }
    const wood = surf(0.7, 0), metal = surf(0.35, 0.9);
    for (const d of this.doors) {
      const t = d.isOpen ? 1 : 0;
      if (d.a !== t) d.a = Math.abs(t - d.a) < 0.01 ? t : d.a + (t - d.a) * Math.min(1, dt * 9);
      if ((d.x - eye[0]) ** 2 + (d.z - eye[2]) ** 2 > 220 * 220) continue;
      let sw = d.open - d.shut;
      sw = Math.atan2(Math.sin(sw), Math.cos(sw));
      const e = d.a * d.a * (3 - 2 * d.a);
      const ang = d.shut + sw * e;
      const sn = Math.sin(ang), cs = Math.cos(ang);
      const q = qY(ang);
      const c: V3 = [d.x + sn * d.w * 0.5, d.y + d.h * 0.5, d.z + cs * d.w * 0.5];
      this.boxes.push(c, q, [0.07, d.h - 0.04, d.w - 0.04], rgba(d.col), cube, wood);
      // rails, a panel line and the handles
      const dark = rgba(tintHex(d.col, 0.72));
      for (const yy of [-d.h * 0.5 + 0.12, 0.05, d.h * 0.5 - 0.12]) this.boxes.push([c[0], c[1] + yy, c[2]], q, [0.09, 0.1, d.w - 0.04], dark, cube, wood);
      const hx = d.x + sn * (d.w - 0.16), hz = d.z + cs * (d.w - 0.16);
      this.boxes.push([hx, d.y + 1.02, hz], q, [0.2, 0.05, 0.12], rgba(0xb9b4a6), cube, metal);
    }
  }

  /** Remember a sound for the direction ring (a repeat from the same place refreshes it). */
  private heard(kind: NoiseKind, x: number, z: number, life: number) {
    for (const n of this.noises) {
      if (n.kind === kind && Math.hypot(n.x - x, n.z - z) < 6) {
        n.x = x;
        n.z = z;
        n.t = this.time;
        n.life = life;
        return;
      }
    }
    this.noises.push({ kind, x, z, t: this.time, life });
    if (this.noises.length > 24) this.noises.shift();
  }

  /** Crates and containers that were knocked about: for drawing, and solid in the core where they are now. */
  private onProps(s: P.Snapshot) {
    const step = this.welcome?.snapEvery ?? 2;
    for (const p of s.props) {
      const k = this.props[p.id];
      if (!k) continue;
      k.a = s.tick - k.b.t > 8 ? { ...k.b, t: s.tick - step } : k.b;
      k.b = { x: p.x, y: p.y, z: p.z, t: s.tick };
      trailPush(k.h, k.b, step);
      if (p.moving && !k.moving) k.axis = norm([Math.random() - 0.5, (Math.random() - 0.5) * 0.4, Math.random() - 0.5]);
      k.moving = p.moving;
      this.d.core.setProp(p.id, p.x, p.y, p.z);
    }
  }

  /** Loose props as boxes: crates with their frame planks, containers with their ribs. */
  private drawProps(eye: V3, dt: number) {
    const B = this.boxes;
    let cube = cubeShapes.get(this.d.renderer);
    if (cube === undefined) {
      cube = this.d.renderer.addShapes([unitCube()]);
      cubeShapes.set(this.d.renderer, cube);
    }
    const wood = surf(0.82, 0), steel = surf(0.5, 0.55);
    for (const k of this.props) {
      const span = k.b.t - k.a.t;
      const tr = trailAt(k.h, this.renderTick);
      const x = lerp(tr.a.x, tr.b.x, tr.f), y = lerp(tr.a.y, tr.b.y, tr.f), z = lerp(tr.a.z, tr.b.z, tr.f);
      if ((x - eye[0]) ** 2 + (z - eye[2]) ** 2 > 300 * 300) continue;
      if (k.moving) {
        // tumbling with its speed (crates much more than containers)
        const v = span > 0 ? Math.hypot(k.b.x - k.a.x, k.b.y - k.a.y, k.b.z - k.a.z) / (span / 30) : 0;
        k.rot = qMul(qAxis(k.axis[0], k.axis[1], k.axis[2], v * dt * (k.crate ? 0.9 : 0.05)), k.rot);
      } else if (k.rot[3] < 0.9999) {
        // settling flat again
        const e = Math.min(1, dt * 9);
        const q: Q = [k.rot[0] * (1 - e), k.rot[1] * (1 - e), k.rot[2] * (1 - e), k.rot[3] * (1 - e) + e * Math.sign(k.rot[3] || 1)];
        const l = Math.hypot(...q) || 1;
        k.rot = [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
      }
      const c: V3 = [x, y + k.hy / 2, z];
      const q = k.rot;
      const at = (o: V3): V3 => add(c, qRot(q, o));
      const dark = rgba(tintHex(k.col, k.crate ? 0.7 : 0.72));
      B.push(c, q, [k.hx * 2, k.hy, k.hz * 2], rgba(k.col), cube, k.crate ? wood : steel);
      if (k.crate) {
        // frame planks around the top and the bottom, a brace on each side
        const e = 0.07;
        B.push(at([0, k.hy / 2 - e / 2, 0]), q, [k.hx * 2 + 0.04, e, k.hz * 2 + 0.04], dark, cube, wood);
        B.push(at([0, -k.hy / 2 + e / 2, 0]), q, [k.hx * 2 + 0.04, e, k.hz * 2 + 0.04], dark, cube, wood);
        B.push(c, q, [e, k.hy, k.hz * 2 + 0.04], dark, cube, wood);
        B.push(c, q, [k.hx * 2 + 0.04, k.hy, e], dark, cube, wood);
      } else {
        // corrugation ribs along the long sides
        const alongX = k.hx > k.hz;
        for (let r = 1; r < 10; r++) {
          const t = (r / 10 - 0.5) * 2;
          if (alongX) B.push(at([t * k.hx, 0, 0]), q, [0.1, k.hy - 0.2, k.hz * 2 + 0.07], dark, cube, steel);
          else B.push(at([0, 0, t * k.hz]), q, [k.hx * 2 + 0.07, k.hy - 0.2, 0.1], dark, cube, steel);
        }
      }
    }
  }

  private carPose(k: number): { x: number; y: number; z: number; yaw: number } {
    const c = this.cars[k];
    if (k === this.myCar && this.pred && this.pred.mode === P.MODE_DRIVE) {
      const p = this.renderPos();
      const a = clamp(this.cmdAcc / DT, 0, 1);
      return { x: p[0], y: p[1], z: p[2], yaw: lerpAngle(this.predPrevYaw, this.pred.airtime, a) + this.smoothYaw };
    }
    const tr = trailAt(c.h, this.renderTick);
    const at = { x: lerp(tr.a.x, tr.b.x, tr.f), y: lerp(tr.a.y, tr.b.y, tr.f), z: lerp(tr.a.z, tr.b.z, tr.f), yaw: lerpAngle(tr.a.yaw, tr.b.yaw, tr.f) };
    const pw = this.carPw[k] ?? 0;
    if (pw > 0.002 && this.pred && this.pred.mode === P.MODE_DRIVE) {
      // a car my own (predicted) car shoves along: where my prediction
      // has it, not where the server had it a moment ago
      const now = this.d.core.carPoses();
      const i = k * 4, a = clamp(this.cmdAcc / DT, 0, 1), pv = this.carPrev, sm = this.carSm;
      at.x = lerp(at.x, lerp(pv[i], now[i], a) + sm[i], pw);
      at.y = lerp(at.y, lerp(pv[i + 1], now[i + 1], a) + sm[i + 1], pw);
      at.z = lerp(at.z, lerp(pv[i + 2], now[i + 2], a) + sm[i + 2], pw);
      at.yaw = lerpAngle(at.yaw, lerpAngle(pv[i + 3], now[i + 3], a) + sm[i + 3], pw);
    }
    return at;
  }

  /**
   * Which cars are drawn where my prediction has them: the driverless
   * ones it has moved off the server's last word while I drive (shoved by
   * my car, or sliding on). The change-over is blended in and out.
   */
  private blendPredictedCars(dt: number) {
    const pw = this.carPw;
    const driving = !!this.pred && this.pred.mode === P.MODE_DRIVE;
    const now = driving ? this.d.core.carPoses() : null;
    const e = Math.exp(-dt * 10), up = Math.min(1, dt * 14), down = Math.min(1, dt * 5);
    for (let k = 0; k < pw.length; k++) {
      const c = this.cars[k];
      let on = false;
      if (now && k !== this.myCar && c.driver === 255 && c.hp > 0) {
        const i = k * 4;
        on = Math.abs(now[i] - c.b.x) + Math.abs(now[i + 2] - c.b.z) > 0.01 || Math.abs(wrapAngle(now[i + 3] - c.b.yaw)) > 0.003;
      }
      pw[k] = !driving ? 0 : on ? pw[k] + (1 - pw[k]) * up : pw[k] * (1 - down);
    }
    const sm = this.carSm;
    for (let i = 0; i < sm.length; i++) sm[i] *= e;
  }

  /** The free car a player at `me` could get into, or -1. */
  private carInReach(me: V3): number {
    let best = -1;
    let bd = 1e9;
    for (let k = 0; k < this.cars.length; k++) {
      const c = this.cars[k];
      if (c.driver !== 255 || c.hp === 0) continue;
      const dx = me[0] - c.b.x, dz = me[2] - c.b.z;
      const d2 = dx * dx + dz * dz;
      if (d2 > 36 || Math.abs(me[1] - c.b.y) > 1.5 || d2 >= bd) continue;
      const sn = Math.sin(c.b.yaw), cs = Math.cos(c.b.yaw);
      // into the car's frame (x = right, z = forward), distance to its outline
      const lx = -dx * cs + dz * sn, lz = dx * sn + dz * cs;
      const dim = CAR_KINDS[c.kind];
      const ex = Math.max(0, Math.abs(lx) - dim.hw), ez = Math.max(0, Math.abs(lz) - dim.hl);
      if (Math.hypot(ex, ez) > 1.3) continue;
      bd = d2;
      best = k;
    }
    return best;
  }

  private reconcile(s: P.Snapshot) {
    const me = s.self;
    const sorted = [...this.rttSamples].sort((x, y) => x - y);
    const pathRtt = sorted.length ? sorted[(sorted.length - 1) >> 1] : this.rtt;
    const tnow = performance.now();
    // An uplink stall, unless the client itself made no commands for a while
    // (its debt then says nothing about the network)
    if (me.debt >= 4 && !document.hidden && tnow - this.lastFrame < 150 && tnow - this.lastGapEnd > pathRtt + 600) this.stallSeenAt = tnow;
    this.pending = this.pending.filter((p) => p.cmd.seq > s.ack);
    const predictable = me.alive && (me.mode === P.MODE_FALL || me.mode === P.MODE_PARA || me.mode === P.MODE_GROUND || (me.mode === P.MODE_DRIVE && this.myCar >= 0) || (me.mode === P.MODE_FLY && this.myHeli >= 0));
    // the cars go back to where the server last said; replaying my commands
    // at the wheel then moves the ones I shove on from there again
    const carsWere = me.alive && me.mode === P.MODE_DRIVE ? this.carsWere : null;
    this.carsWere = null;
    this.d.core.predReset();
    if (!predictable) {
      this.pred = null;
      return;
    }
    const old = this.pred ? ([...this.pred.pos] as V3) : null;
    const oldYaw = this.pred && this.pred.mode === me.mode && (me.mode === P.MODE_DRIVE || me.mode === P.MODE_FLY) ? this.pred.airtime : null;
    this.pred = { pos: [...me.pos], vel: [...me.vel], mode: me.mode, onGround: me.onGround, crouch: me.crouch, inWater: me.inWater, airtime: me.airtime, car: me.mode === P.MODE_DRIVE ? this.myCar : me.mode === P.MODE_FLY ? this.myHeli : 255 };
    const mult = this.speedMult(me);
    // The server stood in for late commands with idle steps and drops that many of
    // the oldest when they arrive, so they must not be replayed. Only commands old
    // enough to be late count: a client that skipped time (frame hitch, hidden tab)
    // has a debt on the server but no late commands.
    const late = tnow - (pathRtt + 100);
    let skip = 0;
    while (skip < me.debt && skip < this.pending.length && this.pending[skip].at < late) skip++;
    for (let k = skip; k < this.pending.length; k++) this.step(this.pending[k].cmd, mult, false);
    if (this.pred.mode === P.MODE_DRIVE) {
      const now = this.d.core.carPoses(), pv = this.carPrev, sm = this.carSm;
      for (let i = 0; i + 4 <= pv.length; i += 4) {
        const dx = carsWere ? carsWere[i] - now[i] : 0, dy = carsWere ? carsWere[i + 1] - now[i + 1] : 0, dz = carsWere ? carsWere[i + 2] - now[i + 2] : 0;
        const dw = carsWere ? wrapAngle(carsWere[i + 3] - now[i + 3]) : 0;
        if (carsWere && this.carPw[i >> 2] > 0.002 && Math.hypot(dx, dy, dz) < 3 && Math.abs(dw) < 0.5) {
          // what the snapshot corrected is hidden and then let go, as for my own car
          sm[i] += dx; sm[i + 1] += dy; sm[i + 2] += dz; sm[i + 3] += dw;
          pv[i] -= dx; pv[i + 1] -= dy; pv[i + 2] -= dz; pv[i + 3] -= dw;
        } else {
          sm[i] = sm[i + 1] = sm[i + 2] = sm[i + 3] = 0;
          pv[i] = now[i]; pv[i + 1] = now[i + 1]; pv[i + 2] = now[i + 2]; pv[i + 3] = now[i + 3];
        }
      }
    }
    if (oldYaw !== null) {
      const dy = wrapAngle(oldYaw - this.pred.airtime);
      if (Math.abs(dy) < 0.5) {
        this.smoothYaw += dy;
        this.predPrevYaw -= dy;
      } else {
        this.smoothYaw = 0;
        this.predPrevYaw = this.pred.airtime;
      }
    } else {
      this.smoothYaw = 0;
      this.predPrevYaw = this.pred.airtime;
    }
    if (old) {
      const dlt = sub(old, this.pred.pos);
      if (Math.hypot(dlt[0], dlt[1], dlt[2]) < 3) {
        this.smooth = add(this.smooth, dlt);
        this.predPrev = sub(this.predPrev, dlt);
      } else {
        this.smooth = [0, 0, 0];
        this.predPrev = [...this.pred.pos];
      }
    } else {
      this.smooth = [0, 0, 0];
      this.predPrev = [...this.pred.pos];
    }
  }

  private predictedMag(): number {
    const s = this.latest?.self;
    if (!s || this.displayCur >= 2) return 0;
    const unacked = this.pending.reduce((a, p) => a + p.shots, 0);
    return Math.max(0, s.slots[this.displayCur].mag - unacked);
  }

  private mergeEdges(e: Edges) {
    if (!this.edges) {
      this.edges = e;
      return;
    }
    const a = this.edges;
    a.jump ||= e.jump;
    a.reload ||= e.reload;
    a.pickup ||= e.pickup;
    a.spect ||= e.spect;
    a.nade ||= e.nade;
    if (e.heal) a.heal = e.heal;
    if (e.slot >= 0) a.slot = e.slot;
  }

  private bestHeal(s: P.SelfState): number {
    if (s.health < 50 && s.heals[1] > 0) return 2;
    if (s.health < 75 && s.heals[0] > 0) return 1;
    if (s.health < 100 && s.heals[1] > 0) return 2;
    if (s.heals[2] > 0 && s.boost < 60) return 3;
    if (s.health < 75 && s.heals[1] > 0) return 2;
    return 0;
  }

  private makeCmd() {
    const s = this.latest;
    if (!s) return;
    const inp = this.d.input;
    const e = this.edges;
    this.edges = null;
    const me = s.self;
    let [mx, mz] = this.paused || this.d.hud.mapOpen ? [0, 0] : inp.move();
    let b = 0;
    const firing = inp.fire && !this.d.hud.mapOpen;
    const weapon = this.displayCur < 2 ? me.slots[this.displayCur].kind : 0;
    let wantFire = firing;
    if (!wantFire && this.d.settings.autoFire && matchMedia('(pointer: coarse)').matches && weapon > 0) wantFire = this.autoFireTarget(weapon);
    if (wantFire) b |= P.B_FIRE;
    if (inp.ads && weapon > 0) b |= P.B_ADS;
    if (inp.crouch) b |= P.B_CROUCH;
    if (inp.sprint && !wantFire && !(b & P.B_ADS)) b |= P.B_SPRINT;
    const driving = this.pred?.mode === P.MODE_DRIVE;
    if (driving) {
      // at the wheel: throttle and steering only, Space is the brake
      b = inp.keys.has('Space') ? P.B_JUMP : 0;
      if (e) {
        e.jump = e.reload = e.nade = false;
        e.heal = 0;
        e.slot = -1;
      }
    }
    const flying = this.pred?.mode === P.MODE_FLY;
    if (flying) {
      // at the controls: Space climbs and C / Ctrl descends while held, the trigger launches missiles
      b = firing ? P.B_FIRE : 0;
      if (inp.keys.has('Space') || inp.touchUp) b |= P.B_JUMP;
      if (inp.keys.has('KeyC') || inp.keys.has('ControlLeft') || inp.touchDown) b |= P.B_CROUCH;
      inp.crouch = false;
      if (e) {
        e.jump = e.reload = e.nade = false;
        e.heal = 0;
        e.slot = -1;
      }
    }
    if (e) {
      if (e.jump) {
        b |= P.B_JUMP;
        if (inp.crouch && this.pred?.mode === P.MODE_GROUND) inp.crouch = false;
      }
      if (e.reload) b |= P.B_RELOAD;
      // one key: leave the car, get into one, or pick up what lies here
      if (e.pickup) b |= driving || flying || ((this.nearCar >= 0 || this.nearHeli >= 0 || this.nearDoor >= 0) && !this.nearItem) ? P.B_USE : P.B_PICKUP;
      if (e.spect) b |= P.B_SPECT;
      if (e.nade) b |= P.B_NADE;
      let heal = e.heal;
      if (heal === 4) heal = this.bestHeal(me);
      if (heal > 0 && heal < 4) b |= heal << 10;
      if (e.slot >= 0) {
        let slot = e.slot;
        if (slot >= 8) {
          const avail = [0, 1, 2].filter((i) => i === 2 || me.slots[i].kind > 0);
          const idx = avail.indexOf(this.displayCur);
          slot = avail[(idx + (slot === 9 ? 1 : avail.length - 1)) % avail.length];
        }
        if (slot === 2 || me.slots[slot]?.kind > 0) {
          b |= (slot + 1) << 8;
          this.slotReq = slot;
          this.slotReqAt = performance.now();
          this.displayCur = slot;
          this.fireCd = Math.max(this.fireCd, 0.35);
        }
      }
    }
    if (this.paused) {
      mx = mz = 0;
      b = 0;
    }
    const view = Math.max(0, Math.floor(this.renderTick * 256));
    const q = P.quantizeCmd(++this.seq, view, mx, mz, this.yaw, this.pitch, b);
    const shots = this.localFire(q);
    this.pending.push({ cmd: q, shots, at: performance.now() });
    if (this.pending.length > 90) this.pending.shift();
    this.outbox.push(q);
    if (this.pred) {
      this.predPrev = [...this.pred.pos];
      this.predPrevYaw = this.pred.airtime;
      if (this.pred.mode === P.MODE_DRIVE) this.carPrev.set(this.d.core.carPoses().subarray(0, this.carPrev.length));
      this.step(q, this.speedMult(me), true);
    }
  }

  private autoFireTarget(weapon: number): boolean {
    const def = this.d.core.wdefs[weapon];
    const o = this.camEye;
    const d = dirFrom(this.yaw, this.pitch);
    const wall = this.d.core.ray(o, d, def.range);
    const lim = wall >= 0 ? wall : def.range;
    for (const p of this.interpAll()) {
      if (!p.alive || p.id === this.myId || p.mode === P.MODE_PLANE || p.mode === P.MODE_FLY) continue;
      const c: V3 = [p.x, p.y + (p.crouch ? 0.8 : 1.15), p.z];
      const to = sub(c, o);
      const along = to[0] * d[0] + to[1] * d[1] + to[2] * d[2];
      if (along < 0 || along > lim) continue;
      const perp = Math.hypot(to[0] - d[0] * along, to[1] - d[1] * along, to[2] - d[2] * along);
      if (perp < 0.45) return true;
    }
    return false;
  }

  private localFire(q: P.CmdQ): number {
    this.fireCd = Math.max(this.fireCd - DT, -DT);
    const s = this.latest?.self;
    const p = this.pred;
    if (!s || !p || !s.alive || p.mode !== P.MODE_GROUND || p.inWater) return 0;
    if (!(q.buttons & P.B_FIRE)) return 0;
    if (s.reloadT > 0 || s.switchT > 0) return 0;
    const weapon = this.displayCur < 2 ? s.slots[this.displayCur].kind : 0;
    if (weapon === 0) {
      if (this.fireCd <= 0) this.fireCd += 0.5;
      return 0;
    }
    const def = this.d.core.wdefs[weapon];
    const mag = this.predictedMag();
    if (mag <= 0) {
      if (this.fireCd <= 0) {
        this.d.sound.ui('empty');
        this.fireCd = 0.3;
      }
      return 0;
    }
    let shots = 0;
    while (this.fireCd <= 0 && shots < mag && shots < 3) {
      this.fireCd += def.interval;
      shots++;
      this.localShot(weapon, (q.buttons & P.B_ADS) !== 0);
    }
    return shots;
  }

  private localShot(weapon: number, ads: boolean) {
    const p = this.pred!;
    const def = this.d.core.wdefs[weapon];
    const pos = this.renderPos();
    const crouch = p.crouch;
    const head: V3 = [pos[0], pos[1] + EYE(crouch), pos[2]];
    const d0 = dirFrom(this.yaw, this.pitch);
    const r = rightOf(this.yaw);
    const side = ads ? 0.38 : 0.55, back = ads ? 1.3 : 2.7;
    const cam: V3 = [head[0] + r[0] * side - d0[0] * back, head[1] + 0.28 - d0[1] * back, head[2] + r[2] * side - d0[2] * back];
    const t0 = Math.max(0, (head[0] - cam[0]) * d0[0] + (head[1] - cam[1]) * d0[1] + (head[2] - cam[2]) * d0[2]);
    // scoped weapons fire from the eye, matching the first-person scope view (and the server)
    const scoped = ads && def.zoom >= 3.5;
    const o: V3 = scoped ? head : add(cam, scale(d0, t0));
    const muz: V3 = add(add([pos[0], pos[1] + EYE(crouch) - 0.12, pos[2]], scale(r, 0.22)), scale(d0, 0.55));
    const up = norm([r[1] * d0[2] - r[2] * d0[1], r[2] * d0[0] - r[0] * d0[2], r[0] * d0[1] - r[1] * d0[0]]);
    let spread = ads ? def.ads : def.hip;
    if (Math.hypot(p.vel[0], p.vel[2]) > 1) spread *= 1.7;
    if (!p.onGround) spread *= 2.5;
    if (crouch) spread *= 0.7;
    for (let k = 0; k < def.pellets; k++) {
      const a = Math.random() * Math.PI * 2;
      const rad = spread * Math.sqrt(Math.random());
      const d = norm([d0[0] + r[0] * Math.cos(a) * rad + up[0] * Math.sin(a) * rad, d0[1] + r[1] * Math.cos(a) * rad + up[1] * Math.sin(a) * rad, d0[2] + r[2] * Math.cos(a) * rad + up[2] * Math.sin(a) * rad]);
      const tw = this.d.core.ray(o, d, def.range);
      let end = tw >= 0 ? tw : def.range;
      for (const q of this.interpAll()) {
        if (q.id === this.myId || !q.alive || q.mode === P.MODE_PLANE || q.mode === P.MODE_FLY) continue;
        const c: V3 = [q.x, q.y + (q.crouch ? 0.8 : 1.1), q.z];
        const to = sub(c, o);
        const along = to[0] * d[0] + to[1] * d[1] + to[2] * d[2];
        if (along < 0 || along > end) continue;
        const perp = Math.hypot(to[0] - d[0] * along, to[1] - d[1] * along, to[2] - d[2] * along);
        if (perp < 0.4) end = along;
      }
      this.spawnTracer(muz, add(o, scale(d, end)), weapon);
    }
    this.spawnFlash(muz);
    this.d.sound.shot(weapon, 0, 0);
    const kick = def.recoil * (ads ? 0.6 : 1) * (crouch ? 0.8 : 1);
    this.recoil += kick;
    this.pitch = clamp(this.pitch + kick, -1.45, 1.45);
    this.yaw += (Math.random() - 0.5) * kick * 0.5;
  }

  // ------------------------------------------------------------------ interpolation
  private bracket(): { a: Snap; b: Snap; f: number } | null {
    const n = this.snaps.length;
    if (!n) return null;
    const rt = this.renderTick;
    if (rt >= this.snaps[n - 1].tick) return { a: this.snaps[n - 1], b: this.snaps[n - 1], f: 0 };
    for (let i = n - 2; i >= 0; i--) {
      const a = this.snaps[i];
      if (a.tick <= rt) {
        const b = this.snaps[i + 1];
        return { a, b, f: (rt - a.tick) / Math.max(1, b.tick - a.tick) };
      }
    }
    return { a: this.snaps[0], b: this.snaps[0], f: 0 };
  }

  private interpPlayer(id: number): P.RemotePlayer | null {
    const br = this.bracket();
    if (!br) return null;
    const pb = br.b.byId.get(id);
    const pa = br.a.byId.get(id);
    if (!pb) return pa ?? null;
    if (!pa) return pb;
    return {
      ...pb,
      x: lerp(pa.x, pb.x, br.f),
      y: lerp(pa.y, pb.y, br.f),
      z: lerp(pa.z, pb.z, br.f),
      yaw: lerpAngle(pa.yaw, pb.yaw, br.f),
      pitch: lerp(pa.pitch, pb.pitch, br.f),
      alive: br.f < 0.5 ? pa.alive : pb.alive,
    };
  }

  private interpCache: P.RemotePlayer[] = [];
  private interpCacheT = -1;
  private interpAll(): P.RemotePlayer[] {
    if (this.interpCacheT === this.time) return this.interpCache;
    const br = this.bracket();
    const out: P.RemotePlayer[] = [];
    if (br) {
      for (const pb of br.b.players) {
        const pa = br.a.byId.get(pb.id);
        if (!pa) {
          out.push(pb);
          continue;
        }
        out.push({
          ...pb,
          x: lerp(pa.x, pb.x, br.f),
          y: lerp(pa.y, pb.y, br.f),
          z: lerp(pa.z, pb.z, br.f),
          yaw: lerpAngle(pa.yaw, pb.yaw, br.f),
          pitch: lerp(pa.pitch, pb.pitch, br.f),
          alive: br.f < 0.5 ? pa.alive : pb.alive,
        });
      }
    }
    this.interpCache = out;
    this.interpCacheT = this.time;
    return out;
  }

  private planeT(): number {
    const br = this.bracket();
    if (!br) return 0;
    return lerp(br.a.plane.t, br.b.plane.t, br.f);
  }

  private renderPos(): V3 {
    const p = this.pred;
    if (!p) return this.latest ? [...this.latest.self.pos] : [512, 50, 512];
    const a = clamp(this.cmdAcc / DT, 0, 1);
    return [lerp(this.predPrev[0], p.pos[0], a) + this.smooth[0], lerp(this.predPrev[1], p.pos[1], a) + this.smooth[1], lerp(this.predPrev[2], p.pos[2], a) + this.smooth[2]];
  }

  private myPos(): V3 | null {
    const s = this.latest;
    if (!s) return null;
    if (this.pred) return this.renderPos();
    if (s.self.mode === P.MODE_PLANE && s.plane.active) {
      const t = this.planeT();
      return [s.plane.sx + s.plane.dx * t, 330, s.plane.sz + s.plane.dz * t];
    }
    return [...s.self.pos];
  }

  // ------------------------------------------------------------------ effects
  private spawnTracer(a: V3, b: V3, weapon: number) {
    const d = sub(b, a);
    const l = Math.hypot(d[0], d[1], d[2]);
    if (l < 0.5) return;
    const speed = 650;
    this.fx.push({ k: 'tracer', t: 0, life: l / speed + 0.02, p: a, v: scale(d, 1 / l), d: b, len: l, col: weapon === 6 ? 0xffe0a0 : 0xffd27a, size: weapon >= 5 ? 0.05 : 0.035 });
  }
  private spawnFlash(p: V3) {
    this.fx.push({ k: 'flash', t: 0, life: 0.05, p, v: [0, 0, 0], d: [0, 0, 0], len: 0, col: 0xffc860, size: 0.35 });
  }
  private spawnImpact(p: V3, col: number) {
    for (let i = 0; i < 5; i++) {
      this.fx.push({ k: 'spark', t: 0, life: 0.35 + Math.random() * 0.3, p: [...p], v: [(Math.random() - 0.5) * 4, Math.random() * 3.5, (Math.random() - 0.5) * 4], d: [0, 0, 0], len: 0, col, size: 0.06 + Math.random() * 0.05 });
    }
  }
  /** Touching down: a ring of dust rolls out along the ground and a little hangs in the air. */
  private spawnDust(p: V3) {
    if (this.d.core.height(p[0], p[2]) < -0.3) return; // into water: no dust
    const n = 16;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 + Math.random() * 0.4;
      const v = 2.2 + Math.random() * 1.8;
      this.fx.push({ k: 'smoke', t: 0, life: 0.9 + Math.random() * 0.6, p: [p[0] + Math.cos(a) * 0.25, p[1] + 0.12, p[2] + Math.sin(a) * 0.25], v: [Math.cos(a) * v, 0.25 + Math.random() * 0.3, Math.sin(a) * v], d: [0, 0, 0], len: 0, col: i % 3 ? 0xb9a98c : 0xcfc3ab, size: 0.32 + Math.random() * 0.22 });
    }
    for (let i = 0; i < 5; i++) {
      this.fx.push({ k: 'smoke', t: 0, life: 1.3 + Math.random() * 0.7, p: [p[0] + (Math.random() - 0.5) * 0.5, p[1] + 0.2, p[2] + (Math.random() - 0.5) * 0.5], v: [(Math.random() - 0.5) * 0.8, 0.9 + Math.random() * 0.8, (Math.random() - 0.5) * 0.8], d: [0, 0, 0], len: 0, col: 0xc7baa0, size: 0.4 + Math.random() * 0.3 });
    }
  }
  private spawnExplosion(p: V3) {
    this.fx.push({ k: 'fire', t: 0, life: 0.45, p, v: [0, 0, 0], d: [0, 0, 0], len: 0, col: 0xffa040, size: 5 });
    for (let i = 0; i < 18; i++) {
      this.fx.push({ k: 'smoke', t: 0, life: 1.6 + Math.random(), p: [...p], v: [(Math.random() - 0.5) * 7, 1.5 + Math.random() * 4, (Math.random() - 0.5) * 7], d: [0, 0, 0], len: 0, col: 0x4a4744, size: 1.2 + Math.random() });
      this.fx.push({ k: 'debris', t: 0, life: 0.8 + Math.random() * 0.5, p: [...p], v: [(Math.random() - 0.5) * 16, 4 + Math.random() * 9, (Math.random() - 0.5) * 16], d: [0, 0, 0], len: 0, col: 0xffb060, size: 0.12 });
    }
  }

  /** A vehicle going up: a bigger, longer fireball than a grenade, with wreckage thrown out. */
  private spawnCarExplosion(p: V3, kind: number) {
    const big = kind ? 1.4 : 1;
    this.fx.push({ k: 'fire', t: 0, life: 0.7, p, v: [0, 0, 0], d: [0, 0, 0], len: 0, col: 0xff8a30, size: 7.5 * big });
    this.fx.push({ k: 'fire', t: 0, life: 1.1, p: [p[0], p[1] + 1.5, p[2]], v: [0, 0, 0], d: [0, 0, 0], len: 0, col: 0xff5a18, size: 4.5 * big });
    for (let i = 0; i < 34; i++) {
      const a = Math.random() * Math.PI * 2, r = Math.random();
      this.fx.push({ k: 'smoke', t: 0, life: 2.4 + Math.random() * 1.8, p: [p[0] + Math.cos(a) * r, p[1] + Math.random(), p[2] + Math.sin(a) * r], v: [Math.cos(a) * 5 * r, 3 + Math.random() * 6, Math.sin(a) * 5 * r], d: [0, 0, 0], len: 0, col: i % 3 ? 0x1c1a19 : 0x5a3a24, size: 1.6 + Math.random() * 1.4 });
      this.fx.push({ k: 'debris', t: 0, life: 1.1 + Math.random() * 0.9, p: [...p], v: [Math.cos(a) * (6 + Math.random() * 14), 6 + Math.random() * 13, Math.sin(a) * (6 + Math.random() * 14)], d: [0, 0, 0], len: 0, col: i % 2 ? 0xffb060 : 0xff6a20, size: 0.14 + Math.random() * 0.12 });
    }
    for (let i = 0; i < 24; i++) {
      this.fx.push({ k: 'spark', t: 0, life: 0.6 + Math.random() * 0.7, p: [...p], v: [(Math.random() - 0.5) * 18, 5 + Math.random() * 10, (Math.random() - 0.5) * 18], d: [0, 0, 0], len: 0, col: 0x2a2a2c, size: 0.12 + Math.random() * 0.16 });
    }
    const d = Math.hypot(p[0] - this.camEye[0], p[1] - this.camEye[1], p[2] - this.camEye[2]);
    this.shake = Math.max(this.shake, clamp(1.6 - d / 45, 0, 1.2));
  }

  private updateFx(dt: number) {
    const out: Fx[] = [];
    const B = this.boxes, G = this.glow;
    for (const f of this.fx) {
      f.t += dt;
      if (f.t >= f.life) continue;
      const k = f.t / f.life;
      switch (f.k) {
        case 'tracer': {
          const head = Math.min(f.len, f.t * 650);
          const tail = Math.max(0, head - 7);
          line(G, add(f.p, scale(f.v, tail)), add(f.p, scale(f.v, head)), f.size, rgba(f.col, 220));
          break;
        }
        case 'flash':
          G.push(f.p, qY(Math.random() * 6), [f.size, f.size, f.size * 1.6], rgba(f.col, 255));
          break;
        case 'spark':
          f.v[1] -= 9.8 * dt;
          f.p = add(f.p, scale(f.v, dt));
          B.push(f.p, qIdent(), [f.size, f.size, f.size], rgba(f.col, 200));
          break;
        case 'debris':
          f.v[1] -= 14 * dt;
          f.p = add(f.p, scale(f.v, dt));
          G.push(f.p, qIdent(), [f.size, f.size, f.size], rgba(f.col, Math.round(255 * (1 - k))));
          break;
        case 'smoke': {
          f.p = add(f.p, scale(f.v, dt));
          f.v = scale(f.v, 1 - dt * 1.5);
          // a soft billowing puff: grows, turns slowly and thins out
          const s = f.size * (0.8 + k * 2.2);
          const seed = (f.life * 13.37 + f.size * 7.1) % 10;
          this.puffs.push(f.p, s, rgba(f.col, Math.round(215 * Math.min(1, f.t * 12) * (1 - k) ** 1.3)), seed * 3 + f.t * 0.35, seed, PUFF_SMOKE);
          break;
        }
        case 'fire': {
          // a roiling fireball that swells and cools
          const s = f.size * (0.35 + 0.75 * Math.sqrt(k));
          const seed = (f.size * 3.3 + f.life * 5.7) % 10;
          this.puffs.push(f.p, s, rgba(0xffffff, Math.round(255 * (1 - k) ** 0.8)), seed, seed, PUFF_FIREBALL);
          this.puffs.push([f.p[0], f.p[1] + s * 0.25, f.p[2]], s * 0.7, rgba(0xffffff, Math.round(255 * (1 - k))), seed + 2, seed + 4.2, PUFF_FIREBALL);
          break;
        }
      }
      out.push(f);
    }
    this.fx = out.length > 1500 ? out.slice(out.length - 1500) : out;
  }

  private panOf(p: V3): number {
    const r = rightOf(this.camYaw);
    const d = norm(sub(p, this.camEye));
    return d[0] * r[0] + d[2] * r[2];
  }

  private project(p: V3): [number, number] | null {
    const c = transformPoint(this.viewProj, p);
    if (c[3] <= 0.01) return null;
    const x = c[0] / c[3], y = c[1] / c[3];
    if (Math.abs(x) > 1.1 || Math.abs(y) > 1.1) return null;
    return [(x * 0.5 + 0.5) * window.innerWidth, (0.5 - y * 0.5) * window.innerHeight];
  }

  // ------------------------------------------------------------------ frame
  private frame(now: number) {
    if (!this.running) return;
    this.raf = requestAnimationFrame((t) => this.frame(t));
    if (now - this.lastFrame > 100) this.lastGapEnd = now;
    const dt = Math.min(0.1, Math.max(0, (now - this.lastFrame) / 1000));
    this.lastFrame = now;
    this.fps = lerp(this.fps, 1 / Math.max(dt, 1e-3), 0.05);
    try {
      this.update(dt);
      this.draw(dt);
    } catch (err) {
      console.error(err);
    }
  }

  private update(dt: number) {
    const inp = this.d.input;
    const s = this.latest;
    this.time += dt;
    if (!s) return;
    // menus / map
    const e = inp.takeEdges();
    if (e.menu) {
      if (this.d.hud.mapOpen) this.toggleMap(false);
      else if (this.d.transport.kind === 'offline') this.setPaused(!this.paused);
      else {
        this.d.input.unlock();
        this.d.touch.releaseAll();
        this.d.onPause(true);
      }
    } else if (e.lockLost && !this.paused && !this.d.hud.mapOpen && !this.ended) {
      // The browser released the mouse (Esc, tab switch): open the menu. A
      // stale edge from a hidden tab must not un-pause the game on return.
      if (this.d.transport.kind === 'offline') this.setPaused(true);
      else {
        this.d.touch.releaseAll();
        this.d.onPause(true);
      }
    }
    if (e.map) this.toggleMap(!this.d.hud.mapOpen);
    if (!this.paused) this.mergeEdges(e);

    // look
    const [lx, ly] = inp.takeLook();
    if (!this.paused) {
      const me = s.self;
      const weapon = this.displayCur < 2 ? me.slots[this.displayCur].kind : 0;
      const zoom = inp.ads && weapon > 0 ? this.d.core.wdefs[weapon].zoom : 1;
      const sens = (this.d.settings.sens * 0.0022 * (zoom > 1 ? this.d.settings.adsSens : 1)) / zoom;
      this.yaw = wrapAngle(this.yaw - lx * sens);
      this.pitch = clamp(this.pitch - ly * sens * (this.d.settings.invertY ? -1 : 1), -1.45, 1.45);
      if (lx !== 0 || ly !== 0) this.lastLookAt = this.time;
      // at the wheel the camera swings back behind the car when left alone
      const p = this.pred;
      if (p && p.mode === P.MODE_DRIVE && this.time - this.lastLookAt > 1.2) {
        const v = Math.hypot(p.vel[0], p.vel[2]);
        if (v > 2) {
          const fwd = p.vel[0] * Math.sin(p.airtime) + p.vel[2] * Math.cos(p.airtime) >= 0;
          const k = 1 - Math.exp(-dt * 2.5);
          this.yaw = wrapAngle(this.yaw + wrapAngle(p.airtime + (fwd ? 0 : Math.PI) - this.yaw) * k);
          this.pitch += (-0.2 - this.pitch) * k;
        }
      }
    }
    // recoil recovery
    if (this.recoil > 0) {
      const r = Math.min(this.recoil, this.recoil * dt * 6);
      this.recoil -= r;
      this.pitch = clamp(this.pitch - r * 0.55, -1.45, 1.45);
    }

    // commands at 30 Hz (none while the offline match is paused: the local
    // server is frozen, so predicting ahead would only snap back later)
    if (!this.paused) this.cmdAcc += dt;
    let n = 0;
    while (this.cmdAcc >= DT && n < 4) {
      this.cmdAcc -= DT;
      n++;
      this.makeCmd();
    }
    if (this.cmdAcc > DT * 4) this.cmdAcc = 0;
    // Online: batch two commands per message (15 msg/s) to halve per-message cost.
    const batch = this.d.transport.kind === 'online' ? 2 : 1;
    if (this.outbox.length >= batch && !this.paused) {
      this.send(P.encodeInputs(this.outbox));
      this.outbox = [];
    } else if (this.paused) this.outbox = [];
    if (performance.now() - this.pingSent > 2000) {
      this.pingSent = performance.now();
      this.send(P.encodePing(this.pingSent >>> 0));
    }

    // clock
    const est = s.tick + ((performance.now() - this.lastSnapAt) / 1000) * 30 - this.interpDelay;
    const err = est - this.renderTick;
    if (Math.abs(err) > 10) this.renderTick = est;
    else this.renderTick += dt * 30 + err * Math.min(1, dt * 4);
    this.smooth = scale(this.smooth, Math.exp(-dt * 10));
    this.smoothYaw *= Math.exp(-dt * 10);
    this.blendPredictedCars(dt);
    this.shake = Math.max(0, this.shake - dt * 2);
  }

  private toggleMap(open: boolean) {
    this.d.hud.openMap(open);
    if (open) this.d.input.unlock();
  }

  private camera(): { eye: V3; dir: V3; vfov: number; scoped: boolean; pivot: V3 } {
    const s = this.latest!;
    const core = this.d.core;
    const base = Math.min(75, Math.max(55, (2 * Math.atan(Math.tan((100 * Math.PI) / 360) / this.d.renderer.aspect) * 180) / Math.PI)) * (Math.PI / 180);
    // debug (?debug=1): a fixed camera [x, y, z, yaw, pitch] for looking at the world up close
    const dbg = DEBUG ? (window as unknown as { __zfCam?: number[] }).__zfCam : undefined;
    if (dbg) {
      const eye: V3 = [dbg[0], dbg[1], dbg[2]];
      return { eye, dir: dirFrom(dbg[3], dbg[4]), vfov: base, scoped: true, pivot: eye };
    }
    const d = dirFrom(this.yaw, this.pitch);
    const back = (pivot: V3, dist: number): V3 => {
      const t = core.ray(pivot, scale(d, -1), dist + 0.3);
      const k = t >= 0 ? Math.max(0.3, t - 0.3) : dist;
      return sub(pivot, scale(d, k));
    };
    if (this.pred && s.self.alive) {
      const pos = this.renderPos();
      if (this.pred.mode === P.MODE_GROUND) {
        const weapon = this.displayCur < 2 ? s.self.slots[this.displayCur].kind : 0;
        const ads = this.d.input.ads && weapon > 0;
        const zoom = ads ? core.wdefs[weapon].zoom : 1;
        const vfov = 2 * Math.atan(Math.tan(base / 2) / zoom);
        const head: V3 = [pos[0], pos[1] + EYE(this.pred.crouch), pos[2]];
        if (ads && zoom >= 3.5) return { eye: add(head, scale(d, 0.25)), dir: d, vfov, scoped: true, pivot: head };
        const r = rightOf(this.yaw);
        const side = ads ? 0.38 : 0.55;
        const hs = core.ray(head, r, side + 0.2);
        const sideK = hs >= 0 ? Math.max(0, hs - 0.2) : side;
        const sh: V3 = [head[0] + r[0] * sideK, head[1] + 0.28, head[2] + r[2] * sideK];
        return { eye: back(sh, ads ? 1.3 : 2.7), dir: d, vfov, scoped: false, pivot: head };
      }
      if (this.pred.mode === P.MODE_DRIVE) {
        // chase camera: wider with speed, further back for the tall truck
        const kind = this.cars[this.myCar]?.kind ?? 0;
        const k = Math.min(1, Math.hypot(this.pred.vel[0], this.pred.vel[2]) / CAR_KINDS[kind].vmax);
        const pivot: V3 = [pos[0], pos[1] + (kind ? 3.7 : 2.0), pos[2]];
        return { eye: back(pivot, (kind ? 10.5 : 7.2) + 1.2 * k), dir: d, vfov: base * (1 + 0.1 * k), scoped: false, pivot };
      }
      if (this.pred.mode === P.MODE_FLY) {
        // chase camera looking over the rotors, further back with speed
        const k = Math.min(1, Math.hypot(this.pred.vel[0], this.pred.vel[2]) / 45);
        const pivot: V3 = [pos[0], pos[1] + 7, pos[2]];
        return { eye: back(pivot, 21 + 4 * k), dir: d, vfov: base * (1 + 0.08 * k), scoped: false, pivot };
      }
      const pivot: V3 = [pos[0], pos[1] + 1.2, pos[2]];
      return { eye: back(pivot, this.pred.mode === P.MODE_PARA ? 8 : 6.5), dir: d, vfov: base, scoped: false, pivot };
    }
    if (s.self.alive && s.self.mode === P.MODE_PLANE && s.plane.active) {
      const t = this.planeT();
      // far enough back to take in a 74 m airliner
      const pivot: V3 = [s.plane.sx + s.plane.dx * t, 336, s.plane.sz + s.plane.dz * t];
      return { eye: sub(pivot, scale(d, this.d.renderer.hasPlaneModel ? 78 : 34)), dir: d, vfov: base, scoped: false, pivot };
    }
    // spectating / dead
    const target = s.self.spect !== 255 ? this.interpPlayer(s.self.spect) : null;
    // (a pilot is watched from behind the helicopter, not from inside it)
    const fly = !!target && target.alive && target.mode === P.MODE_FLY;
    const tp: V3 = target ? [target.x, target.y + (fly ? 7 : 1.6), target.z] : [s.self.pos[0], s.self.pos[1] + 1.6, s.self.pos[2]];
    return { eye: back(tp, fly ? 22 : 4.5), dir: d, vfov: base, scoped: false, pivot: tp };
  }

  private draw(dt: number) {
    const s = this.latest;
    if (!s) return;
    const hud = this.d.hud;
    const core = this.d.core;
    this.boxes.reset();
    this.glow.reset();
    this.smoke.reset();
    this.puffs.reset();
    const cam = this.camera();
    let eye = cam.eye;
    if (this.shake > 0) eye = add(eye, [(Math.random() - 0.5) * this.shake * 0.3, (Math.random() - 0.5) * this.shake * 0.3, (Math.random() - 0.5) * this.shake * 0.3]);
    this.camEye = eye;
    this.camYaw = this.yaw;
    const proj = perspectiveRZ(cam.vfov, this.d.renderer.aspect, 0.08);
    const view = lookAt(eye, add(eye, cam.dir), [0, 1, 0]);
    this.viewProj = mul(proj, view);

    // ---- players
    const tsec = this.time;
    const seen = new Set<number>();
    for (const p of this.interpAll()) {
      // a driver sits inside the car, a pilot in the helicopter
      if (p.id === this.myId || p.mode === P.MODE_PLANE || ((p.mode === P.MODE_DRIVE || p.mode === P.MODE_FLY) && p.alive)) continue;
      const dx = p.x - eye[0], dz = p.z - eye[2];
      if (dx * dx + dz * dz > 420 * 420) continue;
      seen.add(p.id);
      const a = this.anim(p.id, p.x, p.z, p.alive, dt, p.onGround && !p.inWater && !p.crouch, Math.hypot(dx, p.y - eye[1], dz));
      emitCharacter(this.boxes, {
        x: p.x, y: p.y, z: p.z, yaw: p.yaw, pitch: p.pitch, mode: p.alive ? p.mode : 4, crouch: p.crouch || this.time - a.landAt < 0.22, speed: a.speed, phase: a.phase,
        weapon: p.weapon, ads: p.ads, firing: p.firing, deadT: p.alive ? -1 : tsec - a.deadAt, color: this.players.get(p.id)?.color ?? p.id,
        vest: p.vest, helm: p.helm, healing: p.healing, reloading: p.reloading, inWater: p.inWater, onGround: p.onGround, t: tsec,
      });
    }
    for (const id of this.anims.keys()) if (!seen.has(id) && id !== this.myId) this.anims.get(id)!.seen += dt;

    // ---- me
    if (this.pred && s.self.alive && !cam.scoped && this.pred.mode !== P.MODE_DRIVE && this.pred.mode !== P.MODE_FLY) {
      const pos = this.renderPos();
      const a = this.anim(this.myId, pos[0], pos[2], true, dt, this.pred.onGround && !this.pred.inWater && !this.pred.crouch, 0);
      const weapon = this.displayCur < 2 ? s.self.slots[this.displayCur].kind : 0;
      emitCharacter(this.boxes, {
        x: pos[0], y: pos[1], z: pos[2], yaw: this.yaw, pitch: this.pitch, mode: this.pred.mode, crouch: this.pred.crouch || this.time - this.myLandAt < 0.22, speed: a.speed, phase: a.phase,
        weapon, ads: this.d.input.ads && weapon > 0, firing: this.d.input.fire, deadT: -1, color: this.players.get(this.myId)?.color ?? 0,
        vest: s.self.vest, helm: s.self.helm, healing: s.self.healT > 0, reloading: s.self.reloadT > 0, inWater: this.pred.inWater, onGround: this.pred.onGround, t: tsec,
      });
    }

    // ---- items
    for (const it of this.items.values()) {
      const dx = it.x - eye[0], dz = it.z - eye[2];
      if (dx * dx + dz * dz > 75 * 75) continue;
      emitItem(this.boxes, this.glow, it.kind, it.x, it.y, it.z, tsec, it.id);
    }
    // ---- grenades
    const br = this.bracket();
    if (br) {
      for (const n of br.b.nades) {
        const na = br.a.nades.find((x) => x.id === n.id) ?? n;
        const p: V3 = [lerp(na.x, n.x, br.f), lerp(na.y, n.y, br.f), lerp(na.z, n.z, br.f)];
        this.boxes.push(p, qY(tsec * 8), [0.13, 0.16, 0.13], rgba(0x3f4a2e));
        this.glow.push(add(p, [0, 0.12, 0]), qIdent(), [0.05, 0.05, 0.05], rgba(0xff3020, Math.round(128 + 127 * Math.sin(tsec * 20))));
      }
    }
    // ---- plane
    if (s.plane.active || (br && br.a.plane.active)) {
      const t = this.planeT();
      drawPlane(this.d.renderer, this.boxes, [s.plane.sx + s.plane.dx * t, 330, s.plane.sz + s.plane.dz * t], Math.atan2(s.plane.dx, s.plane.dz), tsec);
    }
    this.updateFx(dt);
    this.drawCars(eye, dt);
    this.drawHelis(eye, dt);
    this.drawMissiles(br, dt);
    this.drawProps(eye, dt);
    this.drawDoors(eye, dt);
    // window glass near the camera
    this.glass.reset();
    const gp = this.glassPanes;
    for (let i = 0; i + 6 <= gp.length; i += 6) {
      const gx = (gp[i] + gp[i + 3]) * 0.5, gz = (gp[i + 2] + gp[i + 5]) * 0.5;
      if ((gx - eye[0]) ** 2 + (gz - eye[2]) ** 2 > 170 * 170) continue;
      this.glass.push([gx, (gp[i + 1] + gp[i + 4]) * 0.5, gz], qIdent(), [gp[i + 3] - gp[i], gp[i + 4] - gp[i + 1], gp[i + 5] - gp[i + 2]], rgba(0x9cc0d2, 255));
    }
    // swimmers churn up the water
    for (const p of this.interpAll()) {
      if (!p.alive || !p.inWater || p.mode !== P.MODE_GROUND) continue;
      const me = p.id === this.myId;
      const a = this.anims.get(p.id);
      if (!a || a.speed < 0.6) continue;
      a.splash -= dt;
      if (a.splash > 0) continue;
      a.splash = 0.16;
      const pos: V3 = me && this.pred ? this.renderPos() : [p.x, p.y, p.z];
      const yaw = me ? this.yaw : p.yaw;
      const side = Math.random() < 0.5 ? 0.45 : -0.45;
      this.fx.push({ k: 'smoke', t: 0, life: 0.45 + Math.random() * 0.25, p: [pos[0] + Math.sin(yaw) * 0.6 + Math.cos(yaw) * side, 0.08, pos[2] + Math.cos(yaw) * 0.6 - Math.sin(yaw) * side], v: [(Math.random() - 0.5) * 0.8, 0.9 + Math.random() * 0.6, (Math.random() - 0.5) * 0.8], d: [0, 0, 0], len: 0, col: 0xeef6fb, size: 0.16 + Math.random() * 0.12 });
    }

    const me = this.myPos();
    // (a debug camera takes the shadows with it)
    const shadowCenter: V3 = this.pred && !(DEBUG && (window as unknown as { __zfCam?: number[] }).__zfCam) ? this.renderPos() : cam.pivot;
    this.d.renderer.render({
      viewProj: this.viewProj,
      invViewProj: invert(this.viewProj),
      camPos: eye,
      time: tsec,
      shadowCenter,
      zone: { cx: s.zone.cx, cz: s.zone.cz, r: s.zone.r, visible: s.phase === P.PH_PLAY },
      hurt: 0,
      focus: cam.scoped ? null : cam.pivot,
      boxes: this.boxes,
      glow: this.glow,
      smoke: this.smoke,
      puffs: this.puffs,
      glass: this.glass,
    });

    // ---- audio loops
    const inPlane = s.self.alive && s.self.mode === P.MODE_PLANE && s.plane.active;
    this.d.sound.loop('plane', inPlane ? 1 : 0);
    const fallSpeed = this.pred && (this.pred.mode === P.MODE_FALL || this.pred.mode === P.MODE_PARA) ? Math.hypot(...this.pred.vel) : 0;
    this.d.sound.loop('wind', clamp(fallSpeed / 50, 0, 1));
    // own footsteps
    if (this.pred && this.pred.mode === P.MODE_GROUND && this.pred.onGround) {
      const sp = Math.hypot(this.pred.vel[0], this.pred.vel[2]);
      this.myFoot += sp * dt;
      if (this.myFoot > (sp > 6 ? 2.1 : 1.6) && sp > 0.5 && !this.pred.crouch) {
        this.myFoot = 0;
        this.d.sound.footstep(2, 0, this.pred.inWater);
      }
    }

    // ---- HUD
    hud.tick(dt);
    hud.updateTop(s.alive, s.self.kills);
    hud.updateCompass(this.yaw);
    hud.updateZone(s.zone, me ? { x: me[0], z: me[2] } : null, s.phase);
    const wd = core.wdefs;
    hud.updateSelf(s.self, wd.map((w) => w.mag), this.displayCur, this.predictedMag(), wd.map((w) => w.ammo));
    const weapon = this.displayCur < 2 ? s.self.slots[this.displayCur].kind : 0;
    const alive = s.self.alive;
    const ground = alive && this.pred?.mode === P.MODE_GROUND;
    if (weapon > 0 && ground) {
      const def = wd[weapon];
      let spread = this.d.input.ads ? def.ads : def.hip;
      if (this.pred && Math.hypot(this.pred.vel[0], this.pred.vel[2]) > 1) spread *= 1.7;
      if (this.pred && !this.pred.onGround) spread *= 2.5;
      if (this.pred?.crouch) spread *= 0.7;
      const px = (Math.tan(spread) / Math.tan(cam.vfov / 2)) * (window.innerHeight / 2);
      hud.crosshair(px + 4, !cam.scoped);
    } else hud.crosshair(4, ground && !cam.scoped);
    hud.setScope(cam.scoped);
    if (s.self.reloadT > 0 && weapon > 0) hud.setProgress('リロード', 1 - s.self.reloadT / wd[weapon].reload);
    else if (s.self.healT > 0 && s.self.healKind > 0) hud.setProgress('回復中', 1 - s.self.healT / HEAL_TIME[s.self.healKind - 1]);
    else hud.setProgress(null, 0);
    const item = ground && me ? this.nearestPickup(me) : null;
    const driving = alive && this.pred?.mode === P.MODE_DRIVE;
    const flying = alive && this.pred?.mode === P.MODE_FLY;
    const heli = flying ? this.helis[this.myHeli] : undefined;
    this.nearItem = !!item;
    this.nearCar = ground && me ? this.carInReach(me) : -1;
    this.nearHeli = ground && me && this.nearCar < 0 ? this.heliInReach(me) : -1;
    this.nearDoor = ground && me && this.nearCar < 0 && this.nearHeli < 0 ? this.doorInReach(me) : -1;
    // doors open by themselves for a walker: do it here too, ahead of the server's word
    if (ground && me && this.pred && Math.hypot(this.pred.vel[0], this.pred.vel[2]) > 0.5) {
      for (let k = 0; k < this.doors.length; k++) {
        const d = this.doors[k];
        if (d.isOpen || Math.abs(me[1] - d.y) > 1.5) continue;
        const mx = d.x + Math.sin(d.shut) * d.w * 0.5 - me[0], mz = d.z + Math.cos(d.shut) * d.w * 0.5 - me[2];
        if (Math.hypot(mx, mz) >= 1.5 || mx * this.pred.vel[0] + mz * this.pred.vel[2] <= 0) continue;
        d.isOpen = true;
        d.hold = this.time + 0.6;
        core.setDoor(k, true);
        this.d.sound.door(1, 0, true);
      }
    }
    const doorText = this.nearDoor >= 0 ? (this.doors[this.nearDoor].isOpen ? 'ドアを閉める' : 'ドアを開ける') : null;
    const landed = flying && !!this.pred?.onGround && Math.hypot(...this.pred.vel) < 1.5;
    let flyText = '';
    if (flying && this.pred && heli) {
      const pp = this.pred.pos;
      const alt = Math.max(0, Math.round(pp[1] - Math.max(0, core.height(pp[0], pp[2]))));
      flyText = `${Math.round(Math.hypot(this.pred.vel[0], this.pred.vel[2]) * 3.6)} km/h　高度 ${alt} m　機体 ${Math.ceil((heli.hp / 255) * 100)}%　ミサイル ${heli.ammo}`;
    }
    // where the missiles would go
    const aim = flying ? this.missileAim() : null;
    {
      // where they will land, on the screen: a ring with the range
      const sp = aim && heli ? this.project(aim) : null;
      const range = aim ? Math.hypot(aim[0] - this.renderPos()[0], aim[1] - this.renderPos()[1] - 2, aim[2] - this.renderPos()[2]) : 0;
      hud.setImpact(sp && heli ? { x: sp[0], y: sp[1], text: heli.ammo === 0 ? '弾切れ' : range > 375 ? '射程外' : `着弾 ${Math.round(range)} m`, live: heli.ammo > 0 && range <= 375 } : null);
    }
    if (aim && heli) {
      const sz = Math.max(0.12, Math.hypot(aim[0] - eye[0], aim[1] - eye[1], aim[2] - eye[2]) * 0.004);
      const col = rgba(heli.ammo > 0 ? 0xff5a28 : 0x9a9a9a, 230);
      const q = qY(this.yaw);
      for (const sgn of [-1, 1]) {
        this.glow.push(add(aim, qRot(q, [sgn * sz * 3.2, 0, 0])), q, [sz * 3, sz * 0.6, sz * 0.6], col);
        this.glow.push(add(aim, [0, sgn * sz * 3.2, 0]), q, [sz * 0.6, sz * 3, sz * 0.6], col);
      }
    }
    const kmh = driving && this.pred ? Math.round(Math.hypot(this.pred.vel[0], this.pred.vel[2]) * 3.6) : 0;
    const carName = CAR_KINDS[this.cars[driving ? this.myCar : this.nearCar]?.kind ?? 0].name;
    const pickup = driving ? '降りる' : flying ? (landed ? '降りる' : null) : item ?? (this.nearCar >= 0 ? `${carName}に乗る` : this.nearHeli >= 0 ? 'CH-47Jに乗る' : doorText);
    const touch = matchMedia('(pointer: coarse)').matches;
    const needLock = !touch && !this.d.input.pointerLocked && !hud.mapOpen && !this.paused && s.phase !== P.PH_LOBBY;
    hud.setPrompt(
      needLock ? 'クリックしてマウス操作を開始'
        : driving && (this.cars[this.myCar]?.hp ?? 255) < 50 ? '🔥 炎上中 — まもなく爆発！' + (touch ? ' 降りて離れろ' : ' [F] で降りて離れろ')
        : driving ? `${kmh} km/h　車体 ${Math.ceil(((this.cars[this.myCar]?.hp ?? 255) / 255) * 100)}%` + (touch ? '' : '　[F] 降りる　[Space] ブレーキ')
        : flying ? flyText + (touch ? '' : landed ? '　[Space] 上昇　[F] 降りる' : '　[Space] 上昇　[C] 下降　[クリック] ミサイル')
        : touch ? null
        : item ? `[F] 拾う: ${item}`
        : this.nearCar >= 0 ? `[F] ${carName}に乗る`
        : this.nearHeli >= 0 ? '[F] CH-47Jに乗る'
        : doorText ? `[F] ${doorText}` : null,
    );
    // engines and rotors that are not ours
    this.vehNoiseT -= dt;
    if (this.vehNoiseT <= 0 && me) {
      this.vehNoiseT = 0.4;
      for (let k = 0; k < this.cars.length; k++) {
        const c = this.cars[k];
        if (c.driver === 255 || k === this.myCar || c.hp === 0) continue;
        if (Math.hypot(c.b.x - me[0], c.b.z - me[2]) < 110) this.heard('car', c.b.x, c.b.z, 0.9);
      }
      for (let k = 0; k < this.helis.length; k++) {
        const h = this.helis[k];
        if (k === this.myHeli || h.rotor < 0.3 || h.hp === 0) continue;
        if (Math.hypot(h.b.x - me[0], h.b.z - me[2]) < 260) this.heard('heli', h.b.x, h.b.z, 0.9);
      }
    }
    this.noises = this.noises.filter((n) => this.time - n.t < n.life);
    {
      const r = rightOf(this.camYaw);
      const fx = Math.sin(this.camYaw), fz = Math.cos(this.camYaw);
      hud.setNoises(s.phase === P.PH_PLAY ? this.noises.map((n) => {
        const dx = n.x - this.camEye[0], dz = n.z - this.camEye[2];
        return { kind: n.kind, ang: Math.atan2(dx * r[0] + dz * r[2], dx * fx + dz * fz), alpha: Math.min(1, (1 - (this.time - n.t) / n.life) * 2.2), dist: Math.hypot(dx, dz) };
      }) : []);
    }
    this.d.sound.loop('engine', driving ? 1 : 0, kmh / (CAR_KINDS[this.cars[this.myCar]?.kind ?? 0].vmax * 3.6));
    const spectating = !alive && s.phase === P.PH_PLAY;
    hud.setSpectate(spectating ? (s.self.spect !== 255 ? this.nameOf(s.self.spect) : '—') : null);
    // helicopters that still fly, where they were last seen
    const heliMarks = this.helis.filter((h) => h.hp > 0).map((h) => ({ x: h.b.x, z: h.b.z }));
    if (me) hud.drawMinimap(me[0], me[2], this.yaw, s.phase === P.PH_PLAY ? s.zone : null, s.phase === P.PH_PLAY ? { ...s.plane, t: this.planeT() } : null, this.markers, this.time, 1024, heliMarks);
    if (hud.mapOpen) hud.drawFullMap(me ? me[0] : null, me ? me[2] : 0, this.yaw, s.phase === P.PH_PLAY ? s.zone : null, s.phase === P.PH_PLAY ? { ...s.plane, t: this.planeT() } : null, core.towns, 1024, heliMarks);
    if (this.d.settings.showFps) hud.setNet(`${Math.round(this.fps)} fps${this.d.transport.kind === 'online' ? ` · ${Math.round(this.rtt)} ms` : ''}`);
    else hud.setNet(this.d.transport.kind === 'online' && this.rtt > 180 ? `⚠ ${Math.round(this.rtt)} ms` : '');
    this.d.touch.setState({
      plane: inPlane,
      air: !!this.pred && this.pred.mode === P.MODE_FALL,
      canJumpPlane: s.plane.canJump,
      pickup: pickup,
      ads: this.d.input.ads,
      crouch: this.d.input.crouch,
      nades: s.self.nades,
      heals: s.self.heals[0] + s.self.heals[1] + s.self.heals[2],
      dead: !alive,
      driving,
      flying,
    });
  }

  /**
   * Hand the renderer every helicopter: rotors turning as fast as the server
   * says, struts extending as the weight comes off and the gear stowed once
   * well clear of the ground, wrecks charred. Also the rotor sound of the
   * loudest one, and smoke from damaged or burning ones.
   */
  private drawHelis(eye: V3, dt: number) {
    const core = this.d.core;
    const n = this.helis.length;
    if (this.heliPoses.length !== n * HELI_STRIDE) this.heliPoses = new Float32Array(n * HELI_STRIDE);
    const out = this.heliPoses;
    let loud = 0, loudSpin = 0, loudPan = 0;
    for (let k = 0; k < n; k++) {
      const h = this.helis[k];
      const c = this.heliPose(k);
      h.spin += ((h.hp > 0 ? h.rotor : Math.min(h.rotor, h.spin)) - h.spin) * Math.min(1, dt * 5);
      if (h.spin < 0.003) h.spin = 0;
      h.angle = (h.angle + h.spin * HELI_ROTOR_SPEED * dt) % (Math.PI * 2);
      if (h.hp > 0) {
        const alt = c.y - Math.max(0, core.height(c.x, c.z));
        // up at 9 m, down again below 6 m (the wheels are out well before touching down)
        const want = !h.airborne ? 0 : alt > 9 || (h.gear > 1.02 && alt > 6) ? 2 : 1;
        const rate = (h.gear > 1 || want > 1 ? 0.45 : 2.5) * dt;
        h.gear += clamp(want - h.gear, -rate, rate);
      }
      const q = qMul(qAxis(0, 1, 0, c.yaw), qMul(qAxis(1, 0, 0, c.pitch), qAxis(0, 0, 1, c.roll)));
      const tint = h.hp === 0 ? 0.25 : 0.72 + 0.28 * Math.min(1, h.hp / 128);
      out.set([c.x, c.y, c.z, q[0], q[1], q[2], q[3], tint, h.angle, h.gear, h.spin], k * HELI_STRIDE);
      const mine = k === this.myHeli;
      const dist = Math.hypot(c.x - eye[0], c.y + 3 - eye[1], c.z - eye[2]);
      const level = h.hp > 0 ? Math.min(1, h.spin * 1.3) * (mine ? 0.8 : 1 / (1 + (dist / 45) ** 2)) : 0;
      if (level > loud) {
        loud = level;
        loudSpin = h.spin;
        loudPan = mine ? 0 : this.panOf([c.x, c.y + 3, c.z]) * 0.7;
      }
      if (dist < 320) this.heliSmoke(h, c, q, dt);
    }
    this.d.renderer.setHelis(out);
    this.d.sound.rotor(this.paused || this.ended ? 0 : loud, loudSpin, loudPan);
  }

  /** Smoke from the engines of a badly hit helicopter, fire and black smoke from a wreck. */
  private heliSmoke(h: HeliC, c: HeliPose, q: Q, dt: number) {
    const hurt = h.hp > 0 && h.hp < 100;
    const burning = h.hp === 0 && this.time - h.deadAt < 90;
    if (!hurt && !burning) return;
    h.smokeT -= dt;
    const base: V3 = [c.x, c.y, c.z];
    if (h.smokeT <= 0) {
      h.smokeT = burning ? 0.09 : h.hp < 45 ? 0.1 : 0.2;
      // the engines beside the aft pylon; all along a wreck
      const at = add(base, qRot(q, burning ? [(Math.random() - 0.5) * 2.4, 2.4, (Math.random() - 0.5) * 12] : [Math.random() < 0.5 ? 1.4 : -1.4, 3.5, -5.2]));
      this.fx.push({
        k: 'smoke', t: 0, life: burning ? 3 + Math.random() * 2 : 1.5 + Math.random() * 0.8, p: at,
        v: [(Math.random() - 0.5) * 1.2, 1.8 + Math.random() * 1.8, (Math.random() - 0.5) * 1.2],
        d: [0, 0, 0], len: 0, col: burning ? 0x121110 : h.hp < 45 ? 0x2a2826 : 0x8d8a86, size: burning ? 1.3 + Math.random() : 0.5 + Math.random() * 0.3,
      });
    }
    if (burning && this.time - h.deadAt < 50) {
      const fade = Math.min(1, (50 - (this.time - h.deadAt)) / 8);
      const id = h.b.x * 0.37 + h.b.z * 0.11;
      for (let i = 0; i < 9; i++) {
        const h1 = Math.abs(Math.sin(id + i * 12.9898) * 43758.5453) % 1, h2 = Math.abs(Math.sin(id + i * 78.233) * 12543.123) % 1;
        const sz = (1.0 + 0.8 * h1) * (0.6 + 0.4 * fade);
        const at = add(base, qRot(q, [(h2 - 0.5) * 3, 1.2 + 2 * h2 + sz, (h1 - 0.5) * 13]));
        this.puffs.push(at, sz, rgba(0xffffff, Math.round(255 * fade)), 0, (id + i * 3.1) % 10, PUFF_FLAME);
      }
      this.glow.push([c.x, c.y + 0.4, c.z], qIdent(), [0.05, 0.05, 0.05], rgba(0xff8030, 60));
    }
  }

  /** Missiles in flight: the round, its motor flame and the smoke it leaves behind. */
  private drawMissiles(br: { a: Snap; b: Snap; f: number } | null, dt: number) {
    if (!br || !br.b.missiles.length) return;
    this.missileSmokeT -= dt;
    const puff = this.missileSmokeT <= 0;
    if (puff) this.missileSmokeT = 0.022;
    const span = Math.max(1, br.b.tick - br.a.tick) / 30;
    for (const m of br.b.missiles) {
      const d = dirFrom(m.yaw, m.pitch);
      const was = br.a.missiles.find((x) => x.id === m.id);
      // not in the older snapshot yet: it flew straight to where it is
      const from: V3 = was && br.a !== br.b ? [was.x, was.y, was.z] : [m.x - d[0] * MISSILE_SPEED * span, m.y - d[1] * MISSILE_SPEED * span, m.z - d[2] * MISSILE_SPEED * span];
      const p: V3 = [lerp(from[0], m.x, br.f), lerp(from[1], m.y, br.f), lerp(from[2], m.z, br.f)];
      const q = qMul(qAxis(0, 1, 0, m.yaw), qAxis(1, 0, 0, -m.pitch));
      this.boxes.push(p, q, [0.14, 0.14, 1.5], rgba(0x4a5136), undefined, surf(0.5, 0.3));
      this.boxes.push(add(p, scale(d, 0.85)), q, [0.1, 0.1, 0.25], rgba(0x1c1c1c));
      for (const [fx, fy] of [[0.16, 0], [-0.16, 0], [0, 0.16], [0, -0.16]]) this.boxes.push(add(p, qRot(q, [fx, fy, -0.62])), q, [fx ? 0.18 : 0.02, fy ? 0.18 : 0.02, 0.26], rgba(0x3a4030));
      // motor flame
      this.glow.push(add(p, scale(d, -1.1)), q, [0.2, 0.2, 0.9], rgba(0xffc070, 255));
      this.glow.push(add(p, scale(d, -1.9)), q, [0.12, 0.12, 1.2], rgba(0xff7a30, 200));
      if (puff) {
        const at = add(p, scale(d, -1.6 - Math.random() * 1.5));
        this.fx.push({ k: 'smoke', t: 0, life: 1.1 + Math.random() * 0.7, p: at, v: [(Math.random() - 0.5) * 0.9 - d[0] * 3, (Math.random() - 0.5) * 0.9 + 0.3, (Math.random() - 0.5) * 0.9 - d[2] * 3], d: [0, 0, 0], len: 0, col: 0xd8d5cf, size: 0.38 + Math.random() * 0.22 });
      }
    }
  }

  /**
   * Hand the renderer every car: settled on the ground it stands on, its
   * wheels rolled by the distance covered, wrecks charred. Also lays tyre
   * marks behind moving cars and smoke over damaged ones.
   */
  private drawCars(eye: V3, dt: number) {
    const core = this.d.core;
    const n = this.cars.length;
    if (this.carPoses.length !== n * CAR_STRIDE) this.carPoses = new Float32Array(n * CAR_STRIDE);
    const out = this.carPoses;
    for (let k = 0; k < n; k++) {
      const car = this.cars[k], dim = CAR_KINDS[car.kind];
      const c = this.carPose(k);
      const o = k * CAR_STRIDE;
      const sn = Math.sin(c.yaw), cs = Math.cos(c.yaw);
      const near = (c.x - eye[0]) ** 2 + (c.z - eye[2]) ** 2 < 260 * 260;
      // how it moved since the last frame
      const mx = c.x - car.px, mz = c.z - car.pz;
      const moved = Math.hypot(mx, mz);
      const along = mx * sn + mz * cs;
      const turned = wrapAngle(c.yaw - car.pyaw);
      const prevSpeed = car.speed;
      if (moved < 4 && dt > 0) {
        car.spin = (car.spin + along / dim.wheelR) % (Math.PI * 2);
        car.speed = lerp(car.speed, along / dt, Math.min(1, dt * 12));
        const want = Math.abs(car.speed) > 0.6 ? clamp(Math.atan2((dim.base * turned) / dt, Math.abs(car.speed)) * Math.sign(car.speed), -0.6, 0.6) : car.steer;
        car.steer = lerp(car.steer, want, Math.min(1, dt * 10));
      } else if (moved >= 4) {
        car.trackAt = [null, null];
        car.speed = 0;
      }
      car.px = c.x;
      car.pz = c.z;
      car.pyaw = c.yaw;

      let q: Q = qY(c.yaw);
      let y = c.y;
      let tint = 1;
      if (near) {
        // heights under the axles and the sides: nose up / down and lean
        const ax = car.kind ? 1.9 : 1.4, tr = car.kind ? 1.0 : 0.8;
        const hf = core.height(c.x + sn * ax, c.z + cs * ax), hb = core.height(c.x - sn * ax, c.z - cs * ax);
        const hl = core.height(c.x + cs * tr, c.z - sn * tr), hr = core.height(c.x - cs * tr, c.z + sn * tr);
        let pitch = clamp(Math.atan2(hf - hb, ax * 2), -0.6, 0.6), roll = clamp(Math.atan2(hl - hr, tr * 2), -0.5, 0.5);
        y = Math.max(c.y - 0.4, (hf + hb + hl + hr) / 4) + 0.02;
        if (car.hp === 0) {
          // thrown up by the blast, then it settles askew
          const t = this.time - car.deadAt;
          const side = k % 2 ? 1 : -1;
          if (t < 1.0) {
            const u = t / 1.0;
            y += (car.kind ? 0.9 : 1.7) * 4 * u * (1 - u);
            roll += side * 0.55 * Math.sin(u * Math.PI);
            pitch += 0.35 * Math.sin(u * Math.PI * 2);
          }
          roll += side * 0.05;
          y -= 0.06;
          tint = 0.13;
        } else tint = 0.72 + 0.28 * Math.min(1, car.hp / 128);
        q = qMul(qMul(q, qAxis(1, 0, 0, -pitch)), qAxis(0, 0, 1, roll));
        this.carSmoke(car, c, y, dt, sn, cs);
        if (car.hp > 0 && moved > 1e-4 && moved < 4 && c.y > 0.2) this.layTracks(car, c, moved, sn, cs, (prevSpeed - Math.abs(car.speed)) / Math.max(dt, 1e-3));
      }
      out.set([c.x, y, c.z, q[0], q[1], q[2], q[3], tint, car.spin, car.hp === 0 ? 0.3 * (k % 2 ? 1 : -1) : car.steer], o);
    }
    this.d.renderer.setCars(out);
    // tyre marks: thin dark strips that fade away
    const S = this.smoke;
    const keep: Track[] = [];
    for (const t of this.tracks) {
      const age = this.time - t.t;
      if (age > TRACK_LIFE) continue;
      keep.push(t);
      if ((t.p[0] - eye[0]) ** 2 + (t.p[2] - eye[2]) ** 2 > 140 * 140) continue;
      S.push(t.p, t.q, [t.w, 0.02, t.len], rgba(0x0b0b0c, Math.round(255 * t.a * Math.min(1, (TRACK_LIFE - age) / 12))));
    }
    this.tracks = keep.length > 1600 ? keep.slice(keep.length - 1600) : keep;
    // contact shade under whatever stands on the ground (the sun's shadow alone leaves it floating)
    const BLOB = surf(0, 1);
    const blob = (x: number, y: number, z: number, yaw: number, w: number, l: number, a: number) => {
      if ((x - eye[0]) ** 2 + (z - eye[2]) ** 2 > 100 * 100) return;
      S.push([x, y + 0.035, z], qY(yaw), [w, 0.02, l], rgba(0, Math.round(255 * a)), undefined, BLOB);
    };
    for (let k = 0; k < this.cars.length; k++) {
      const c = this.cars[k];
      const dim = CAR_KINDS[c.kind];
      const tr = trailAt(c.h, this.renderTick);
      blob(lerp(tr.a.x, tr.b.x, tr.f), lerp(tr.a.y, tr.b.y, tr.f), lerp(tr.a.z, tr.b.z, tr.f), tr.b.yaw, dim.hw * 2 + 0.9, dim.hl * 2 + 0.7, 0.62);
    }
    for (const k of this.props) {
      if (k.moving) continue;
      blob(k.b.x, k.b.y, k.b.z, 0, k.hx * 2 + 0.7, k.hz * 2 + 0.7, 0.5);
    }
    for (const h of this.helis) {
      if (h.b.y - Math.max(0, this.d.core.height(h.b.x, h.b.z)) < 1.2) blob(h.b.x, h.b.y, h.b.z, h.b.yaw, 6.5, 19, 0.5);
    }
    for (const q of this.interpAll()) {
      if (!q.alive || q.mode !== P.MODE_GROUND || q.inWater) continue;
      blob(q.x, q.y, q.z, 0, 1.25, 1.25, 0.5);
    }
  }

  /** Smoke from a damaged engine, fire and black smoke from a wreck. */
  private carSmoke(car: CarC, c: { x: number; y: number; z: number }, y: number, dt: number, sn: number, cs: number) {
    const hurt = car.hp > 0 && car.hp < 110;
    const burning = car.hp === 0 && this.time - car.deadAt < 75;
    if (!hurt && !burning) return;
    const dim = CAR_KINDS[car.kind];
    car.smokeT -= dt;
    // the bonnet of a running car, the middle of a wreck
    const f = hurt ? dim.hl * 0.72 : 0;
    const p: V3 = [c.x + sn * f, y + (hurt ? Math.min(1.0, dim.h * 0.7) : dim.h * 0.6), c.z + cs * f];
    if (car.smokeT <= 0) {
      const bad = car.hp < 50;
      car.smokeT = burning ? 0.11 : bad ? 0.1 : 0.22;
      this.fx.push({
        k: 'smoke', t: 0, life: burning ? 2.6 + Math.random() * 1.6 : 1.3 + Math.random() * 0.7,
        p: [p[0] + (Math.random() - 0.5) * (burning ? 1.6 : 0.4), p[1], p[2] + (Math.random() - 0.5) * (burning ? 1.6 : 0.4)],
        v: [(Math.random() - 0.5) * 0.8 + car.speed * sn * 0.3, 1.6 + Math.random() * 1.6, (Math.random() - 0.5) * 0.8 + car.speed * cs * 0.3],
        d: [0, 0, 0], len: 0, col: burning ? 0x121110 : bad ? 0x2a2826 : 0x8d8a86, size: burning ? 0.9 + Math.random() * 0.7 : 0.35 + Math.random() * 0.25,
      });
    }
    // flames: a wreck burns for a while, a car about to go licks at the bonnet
    if ((burning && this.time - car.deadAt < 40) || (hurt && car.hp < 50)) {
      // tongues of flame at fixed spots on the body (the shader makes them flicker)
      const id = car.px * 0.37 + car.pz * 0.11;
      const fade = burning ? Math.min(1, (40 - (this.time - car.deadAt)) / 8) : 1;
      for (let i = 0; i < (burning ? 6 : 2); i++) {
        const h1 = Math.abs(Math.sin(id + i * 12.9898) * 43758.5453) % 1, h2 = Math.abs(Math.sin(id + i * 78.233) * 12543.123) % 1;
        const s = (burning ? 0.75 + 0.5 * h1 : 0.32 + 0.14 * h1) * (0.6 + 0.4 * fade);
        const along = burning ? (h1 - 0.5) * dim.hl * 1.4 : (h1 - 0.5) * 0.5, across = (h2 - 0.5) * (burning ? dim.hw * 1.5 : 0.7);
        const at: V3 = [p[0] + sn * along + cs * across, (burning ? y + dim.h * (0.25 + 0.5 * h2) : p[1]) + s * 1.2, p[2] + cs * along - sn * across];
        this.puffs.push(at, s, rgba(0xffffff, Math.round(255 * fade)), 0, (id + i * 3.1) % 10, PUFF_FLAME);
      }
      // the light of the fire on what is around it
      this.glow.push([p[0], y + 0.3, p[2]], qIdent(), [0.05, 0.05, 0.05], rgba(0xff8030, 60));
    }
  }

  /** Marks behind the rear wheels: faint on a steady run, black when the tyres slide. */
  private layTracks(car: CarC, c: { x: number; y: number; z: number; yaw: number }, moved: number, sn: number, cs: number, decel: number) {
    const dim = CAR_KINDS[car.kind];
    car.trackAcc += moved;
    if (car.trackAcc < 0.55) return;
    car.trackAcc = 0;
    const core = this.d.core;
    const sliding = decel > 7 || (Math.abs(car.steer) > 0.2 && Math.abs(car.speed) > 9);
    const a = sliding ? 0.62 : 0.2;
    for (let side = 0; side < 2; side++) {
      const s = side ? 1 : -1;
      const wx = c.x - sn * dim.rear + cs * dim.track * s, wz = c.z - cs * dim.rear - sn * dim.track * s;
      const now: V3 = [wx, core.height(wx, wz) + 0.045, wz];
      const was = car.trackAt[side];
      car.trackAt[side] = now;
      if (!was) continue;
      const dx = now[0] - was[0], dy = now[1] - was[1], dz = now[2] - was[2];
      const len = Math.hypot(dx, dz);
      if (len < 0.05 || len > 3) continue;
      const q = qMul(qY(Math.atan2(dx, dz)), qAxis(1, 0, 0, -Math.atan2(dy, len)));
      this.tracks.push({ p: [(now[0] + was[0]) / 2, (now[1] + was[1]) / 2, (now[2] + was[2]) / 2], q, len: len * 1.08, w: dim.tyre, a, t: this.time });
    }
  }

  private nearestPickup(me: V3): string | null {
    let best: P.ItemInfo | null = null;
    let bd = 2.6;
    for (const it of this.items.values()) {
      const d = Math.hypot(it.x - me[0], it.z - me[2]);
      if (d < bd && Math.abs(it.y - me[1]) < 1.8) {
        bd = d;
        best = it;
      }
    }
    return best ? itemName(best.kind, best.amount) : null;
  }

  private anim(id: number, x: number, z: number, alive: boolean, dt: number, stepping: boolean, dist: number): Anim {
    let a = this.anims.get(id);
    if (!a || a.seen > 2) {
      a = { phase: Math.random() * 6, x, z, speed: 0, deadAt: alive ? -1 : this.time - 5, foot: 0, wasAlive: alive, seen: 0, landAt: -9, splash: 0 };
      this.anims.set(id, a);
    }
    a.seen = 0;
    const moved = Math.hypot(x - a.x, z - a.z);
    const inst = dt > 0 ? moved / dt : 0;
    a.speed = lerp(a.speed, Math.min(inst, 12), Math.min(1, dt * 10));
    a.phase += a.speed * dt * 1.9;
    a.x = x;
    a.z = z;
    if (a.wasAlive && !alive) a.deadAt = this.time;
    a.wasAlive = alive;
    if (id !== this.myId && stepping && alive) {
      a.foot += moved;
      if (a.foot > (a.speed > 6 ? 2.1 : 1.6)) {
        a.foot = 0;
        if (dist < 40) this.d.sound.footstep(dist, this.panOf([x, 0, z]), false);
        if (dist < 30) this.heard('step', x, z, 1.1);
      }
    }
    return a;
  }

  townName(i: number) {
    return TOWN_NAMES[i % TOWN_NAMES.length];
  }
}
