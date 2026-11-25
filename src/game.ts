// Client game: connects to an authoritative server (Worker or Durable Object),
// predicts the local player with the shared Rust movement code, interpolates
// everyone else, and renders/plays the result.

import type { Core } from './core';
import { CAR_STRIDE, PUFF_FIREBALL, PUFF_FLAME, PUFF_SMOKE, Puffs, type Renderer } from './render/renderer';
import { Instances, rgba } from './render/instances';
import { drawPlane, emitCharacter, emitItem, line } from './render/models';
import {
  add, clamp, dirFrom, invert, lerp, lerpAngle, lookAt, mul, norm, perspectiveRZ, qAxis, qIdent, qMul, qY, rightOf, scale, sub, transformPoint, wrapAngle, type M4, type Q, type V3,
} from './render/math';
import * as P from './protocol';
import type { Transport } from './net';
import { Hud, itemName, TOWN_NAMES, WEAPON_SHORT } from './hud';
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
  time: number;
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
  /** While driving: the car's heading. */
  airtime: number;
  /** Index of the car being driven (255 = none). */
  car: number;
}

/** A car as the client draws it: two timed samples to interpolate between. */
interface CarC {
  a: { x: number; y: number; z: number; yaw: number; t: number };
  b: { x: number; y: number; z: number; yaw: number; t: number };
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
  private predPrevYaw = 0;
  private smoothYaw = 0;
  private lastLookAt = 0;
  private carPoses = new Float32Array(0);
  private myLandAt = -9;
  private glassPanes: Float32Array = new Float32Array(0);
  private glass = new Instances(256);
  private tracks: Track[] = [];
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
    });
    this.d.hud.setMap(core.bytes(5), 512);
    this.glassPanes = core.f32(10);
    const c0 = core.f32(8);
    const kinds = core.bytes(9);
    this.cars = [];
    for (let i = 0; i + 4 <= c0.length; i += 4) {
      const at = { x: c0[i], y: c0[i + 1], z: c0[i + 2], yaw: c0[i + 3], t: 0 };
      this.cars.push({
        a: at, b: { ...at }, driver: 255, kind: Math.min(kinds[i >> 2] ?? 0, CAR_KINDS.length - 1), hp: 255, deadAt: -1,
        spin: 0, steer: 0, px: at.x, pz: at.z, pyaw: at.yaw, speed: 0, trackAcc: 0, trackAt: [null, null], smokeT: 0,
      });
    }
    this.myCar = -1;
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
    this.onCars(s);
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

  private onPhase(s: P.Snapshot) {
    const hud = this.d.hud;
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
        const weapon = e.killer === P.KILLER_ZONE ? '圏外' : e.weapon === P.WK_NADE ? 'グレネード' : e.weapon === P.WK_CAR ? 'パトカー' : e.weapon === P.WK_TRUCK ? '消防車' : e.weapon === P.WK_BOOM ? '車両爆発' : WEAPON_SHORT[e.weapon] ?? '?';
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
      time: (performance.now() - this.matchStartTime) / 1000,
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
          time: (performance.now() - this.matchStartTime) / 1000,
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
    }
    const driving = s.self.alive && s.self.mode === P.MODE_DRIVE;
    this.myCar = driving ? (mine >= 0 ? mine : this.myCar) : -1;
    // my own car follows my prediction: keep the server's copy of it out of
    // the way of the camera and of the rays
    if (this.myCar >= 0) core.setCar(this.myCar, 0, -1000, 0, 0);
  }

  private carPose(k: number): { x: number; y: number; z: number; yaw: number } {
    const c = this.cars[k];
    if (k === this.myCar && this.pred && this.pred.mode === P.MODE_DRIVE) {
      const p = this.renderPos();
      const a = clamp(this.cmdAcc / DT, 0, 1);
      return { x: p[0], y: p[1], z: p[2], yaw: lerpAngle(this.predPrevYaw, this.pred.airtime, a) + this.smoothYaw };
    }
    const span = c.b.t - c.a.t;
    const f = span > 0 ? clamp((this.renderTick - c.a.t) / span, 0, 1) : 1;
    return { x: lerp(c.a.x, c.b.x, f), y: lerp(c.a.y, c.b.y, f), z: lerp(c.a.z, c.b.z, f), yaw: lerpAngle(c.a.yaw, c.b.yaw, f) };
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
    const predictable = me.alive && (me.mode === P.MODE_FALL || me.mode === P.MODE_PARA || me.mode === P.MODE_GROUND || (me.mode === P.MODE_DRIVE && this.myCar >= 0));
    if (!predictable) {
      this.pred = null;
      return;
    }
    const old = this.pred ? ([...this.pred.pos] as V3) : null;
    const oldYaw = this.pred && this.pred.mode === P.MODE_DRIVE && me.mode === P.MODE_DRIVE ? this.pred.airtime : null;
    this.pred = { pos: [...me.pos], vel: [...me.vel], mode: me.mode, onGround: me.onGround, crouch: me.crouch, inWater: me.inWater, airtime: me.airtime, car: me.mode === P.MODE_DRIVE ? this.myCar : 255 };
    const mult = this.speedMult(me);
    // The server stood in for late commands with idle steps and drops that many of
    // the oldest when they arrive, so they must not be replayed. Only commands old
    // enough to be late count: a client that skipped time (frame hitch, hidden tab)
    // has a debt on the server but no late commands.
    const late = tnow - (pathRtt + 100);
    let skip = 0;
    while (skip < me.debt && skip < this.pending.length && this.pending[skip].at < late) skip++;
    for (let k = skip; k < this.pending.length; k++) this.step(this.pending[k].cmd, mult, false);
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
    if (e) {
      if (e.jump) {
        b |= P.B_JUMP;
        if (inp.crouch && this.pred?.mode === P.MODE_GROUND) inp.crouch = false;
      }
      if (e.reload) b |= P.B_RELOAD;
      // one key: leave the car, get into one, or pick up what lies here
      if (e.pickup) b |= driving || (this.nearCar >= 0 && !this.nearItem) ? P.B_USE : P.B_PICKUP;
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
      if (!p.alive || p.id === this.myId || p.mode === P.MODE_PLANE) continue;
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
        if (q.id === this.myId || !q.alive || q.mode === P.MODE_PLANE) continue;
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
    const tp: V3 = target ? [target.x, target.y + 1.6, target.z] : [s.self.pos[0], s.self.pos[1] + 1.6, s.self.pos[2]];
    return { eye: back(tp, 4.5), dir: d, vfov: base, scoped: false, pivot: tp };
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
      // a driver sits inside the car
      if (p.id === this.myId || p.mode === P.MODE_PLANE || (p.mode === P.MODE_DRIVE && p.alive)) continue;
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
    if (this.pred && s.self.alive && !cam.scoped && this.pred.mode !== P.MODE_DRIVE) {
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
    const shadowCenter: V3 = this.pred ? this.renderPos() : cam.pivot;
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
    this.nearItem = !!item;
    this.nearCar = ground && me ? this.carInReach(me) : -1;
    const kmh = driving && this.pred ? Math.round(Math.hypot(this.pred.vel[0], this.pred.vel[2]) * 3.6) : 0;
    const carName = CAR_KINDS[this.cars[driving ? this.myCar : this.nearCar]?.kind ?? 0].name;
    const pickup = driving ? '降りる' : item ?? (this.nearCar >= 0 ? `${carName}に乗る` : null);
    const touch = matchMedia('(pointer: coarse)').matches;
    const needLock = !touch && !this.d.input.pointerLocked && !hud.mapOpen && !this.paused && s.phase !== P.PH_LOBBY;
    hud.setPrompt(
      needLock ? 'クリックしてマウス操作を開始'
        : driving ? `${kmh} km/h　車体 ${Math.ceil(((this.cars[this.myCar]?.hp ?? 255) / 255) * 100)}%` + (touch ? '' : '　[F] 降りる　[Space] ブレーキ')
        : touch ? null
        : item ? `[F] 拾う: ${item}`
        : this.nearCar >= 0 ? `[F] ${carName}に乗る` : null,
    );
    this.d.sound.loop('engine', driving ? 1 : 0, kmh / (CAR_KINDS[this.cars[this.myCar]?.kind ?? 0].vmax * 3.6));
    const spectating = !alive && s.phase === P.PH_PLAY;
    hud.setSpectate(spectating ? (s.self.spect !== 255 ? this.nameOf(s.self.spect) : '—') : null);
    if (me) hud.drawMinimap(me[0], me[2], this.yaw, s.phase === P.PH_PLAY ? s.zone : null, s.phase === P.PH_PLAY ? { ...s.plane, t: this.planeT() } : null, this.markers, this.time, 1024);
    if (hud.mapOpen) hud.drawFullMap(me ? me[0] : null, me ? me[2] : 0, this.yaw, s.phase === P.PH_PLAY ? s.zone : null, s.phase === P.PH_PLAY ? { ...s.plane, t: this.planeT() } : null, core.towns, 1024);
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
    });
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
      }
    }
    return a;
  }

  townName(i: number) {
    return TOWN_NAMES[i % TOWN_NAMES.length];
  }
}
