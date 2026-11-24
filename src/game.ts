// Client game: connects to an authoritative server (Worker or Durable Object),
// predicts the local player with the shared Rust movement code, interpolates
// everyone else, and renders/plays the result.

import type { Core } from './core';
import type { Renderer } from './render/renderer';
import { Instances, rgba } from './render/instances';
import { emitCharacter, emitItem, emitPlane, line } from './render/models';
import {
  add, clamp, dirFrom, invert, lerp, lerpAngle, lookAt, mul, norm, perspectiveRZ, qIdent, qY, rightOf, scale, sub, transformPoint, wrapAngle, type M4, type V3,
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
  airtime: number;
}

const DT = 1 / 30;
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
  private pending: { cmd: P.CmdQ; shots: number }[] = [];
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
    return new Promise<void>((resolve, reject) => {
      this.welcomeWait = { resolve, reject };
      setTimeout(() => reject(new Error('サーバーからの応答がありません')), 15000);
    });
  }

  destroy() {
    this.running = false;
    cancelAnimationFrame(this.raf);
    document.removeEventListener('visibilitychange', this.visHandler);
    this.d.transport.close();
    this.d.sound.loop('plane', 0);
    this.d.sound.loop('wind', 0);
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
      this.rtt = lerp(this.rtt || (now - sent), (now - sent) >>> 0, 0.3);
    } else if (type === P.S_ERROR) {
      const code = new Uint8Array(buf)[1];
      const msg = ['不明なエラー', '部屋が満員です', '試合はすでに開始しています', 'バージョンが一致しません', '接続数が多すぎます'][code] ?? 'エラー';
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
    });
    this.d.hud.setMap(core.bytes(5), 512);
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
      case 'shot': {
        const mine = e.shooter === this.myId;
        if (!mine) {
          this.spawnTracer(e.from, e.to, e.weapon);
          this.spawnFlash(e.from);
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
        const weapon = e.killer === P.KILLER_ZONE ? '圏外' : e.weapon === 200 ? 'グレネード' : WEAPON_SHORT[e.weapon] ?? '?';
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
    const won = s.winner === this.myId && s.self.alive;
    if (won) {
      this.d.hud.showBanner('🏆 チャンピオン！ 最後の生存者', 6, 'win');
      this.d.sound.ui('win');
    }
    if (!this.resultSent) {
      this.resultSent = true;
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
          }),
        won ? 2500 : 500,
      );
    }
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
    io[12] = q.mx; io[13] = q.mz; io[14] = q.yaw; io[15] = q.pitch; io[16] = q.buttons;
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
      }
    }
  }

  private reconcile(s: P.Snapshot) {
    const me = s.self;
    this.pending = this.pending.filter((p) => p.cmd.seq > s.ack);
    const predictable = me.alive && (me.mode === P.MODE_FALL || me.mode === P.MODE_PARA || me.mode === P.MODE_GROUND);
    if (!predictable) {
      this.pred = null;
      return;
    }
    const old = this.pred ? ([...this.pred.pos] as V3) : null;
    this.pred = { pos: [...me.pos], vel: [...me.vel], mode: me.mode, onGround: me.onGround, crouch: me.crouch, inWater: me.inWater, airtime: me.airtime };
    const mult = this.speedMult(me);
    for (const p of this.pending) this.step(p.cmd, mult, false);
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
    if (e) {
      if (e.jump) {
        b |= P.B_JUMP;
        if (inp.crouch && this.pred?.mode === P.MODE_GROUND) inp.crouch = false;
      }
      if (e.reload) b |= P.B_RELOAD;
      if (e.pickup) b |= P.B_PICKUP;
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
    this.pending.push({ cmd: q, shots });
    if (this.pending.length > 90) this.pending.shift();
    this.outbox.push(q);
    if (this.pred) {
      this.predPrev = [...this.pred.pos];
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
    const o: V3 = add(cam, scale(d0, t0));
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
  private spawnDust(p: V3) {
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      this.fx.push({ k: 'smoke', t: 0, life: 0.8 + Math.random() * 0.4, p: [p[0], p[1] + 0.1, p[2]], v: [Math.cos(a) * 2, 0.4, Math.sin(a) * 2], d: [0, 0, 0], len: 0, col: 0xb8a888, size: 0.5 });
    }
  }
  private spawnExplosion(p: V3) {
    this.fx.push({ k: 'fire', t: 0, life: 0.45, p, v: [0, 0, 0], d: [0, 0, 0], len: 0, col: 0xffa040, size: 5 });
    for (let i = 0; i < 18; i++) {
      this.fx.push({ k: 'smoke', t: 0, life: 1.6 + Math.random(), p: [...p], v: [(Math.random() - 0.5) * 7, 1.5 + Math.random() * 4, (Math.random() - 0.5) * 7], d: [0, 0, 0], len: 0, col: 0x4a4744, size: 1.2 + Math.random() });
      this.fx.push({ k: 'debris', t: 0, life: 0.8 + Math.random() * 0.5, p: [...p], v: [(Math.random() - 0.5) * 16, 4 + Math.random() * 9, (Math.random() - 0.5) * 16], d: [0, 0, 0], len: 0, col: 0xffb060, size: 0.12 });
    }
  }

  private updateFx(dt: number) {
    const out: Fx[] = [];
    const B = this.boxes, G = this.glow, S = this.smoke;
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
          const s = f.size * (1 + k * 2.5);
          S.push(f.p, qY(f.t + f.size), [s, s, s], rgba(f.col, Math.round(180 * (1 - k))));
          break;
        }
        case 'fire': {
          const s = f.size * (0.4 + k);
          G.push(f.p, qY(k * 3), [s, s, s], rgba(f.col, Math.round(255 * (1 - k))));
          G.push(f.p, qY(-k * 2), [s * 0.6, s * 0.6, s * 0.6], rgba(0xfff0c0, Math.round(255 * (1 - k))));
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
      else this.d.onPause(true);
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
    }
    // recoil recovery
    if (this.recoil > 0) {
      const r = Math.min(this.recoil, this.recoil * dt * 6);
      this.recoil -= r;
      this.pitch = clamp(this.pitch - r * 0.55, -1.45, 1.45);
    }

    // commands at 30 Hz
    this.cmdAcc += dt;
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
      const pivot: V3 = [pos[0], pos[1] + 1.2, pos[2]];
      return { eye: back(pivot, this.pred.mode === P.MODE_PARA ? 8 : 6.5), dir: d, vfov: base, scoped: false, pivot };
    }
    if (s.self.alive && s.self.mode === P.MODE_PLANE && s.plane.active) {
      const t = this.planeT();
      const pivot: V3 = [s.plane.sx + s.plane.dx * t, 332, s.plane.sz + s.plane.dz * t];
      return { eye: sub(pivot, scale(d, 34)), dir: d, vfov: base, scoped: false, pivot };
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
      if (p.id === this.myId || p.mode === P.MODE_PLANE) continue;
      const dx = p.x - eye[0], dz = p.z - eye[2];
      if (dx * dx + dz * dz > 420 * 420) continue;
      seen.add(p.id);
      const a = this.anim(p.id, p.x, p.z, p.alive, dt, p.onGround && !p.inWater && !p.crouch, Math.hypot(dx, p.y - eye[1], dz));
      emitCharacter(this.boxes, {
        x: p.x, y: p.y, z: p.z, yaw: p.yaw, pitch: p.pitch, mode: p.alive ? p.mode : 4, crouch: p.crouch, speed: a.speed, phase: a.phase,
        weapon: p.weapon, ads: p.ads, firing: p.firing, deadT: p.alive ? -1 : tsec - a.deadAt, color: this.players.get(p.id)?.color ?? p.id,
        vest: p.vest, helm: p.helm, healing: p.healing, reloading: p.reloading, inWater: p.inWater, onGround: p.onGround, t: tsec,
      });
    }
    for (const id of this.anims.keys()) if (!seen.has(id) && id !== this.myId) this.anims.get(id)!.seen += dt;

    // ---- me
    if (this.pred && s.self.alive && !cam.scoped) {
      const pos = this.renderPos();
      const a = this.anim(this.myId, pos[0], pos[2], true, dt, this.pred.onGround && !this.pred.inWater && !this.pred.crouch, 0);
      const weapon = this.displayCur < 2 ? s.self.slots[this.displayCur].kind : 0;
      emitCharacter(this.boxes, {
        x: pos[0], y: pos[1], z: pos[2], yaw: this.yaw, pitch: this.pitch, mode: this.pred.mode, crouch: this.pred.crouch, speed: a.speed, phase: a.phase,
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
      emitPlane(this.boxes, [s.plane.sx + s.plane.dx * t, 330, s.plane.sz + s.plane.dz * t], Math.atan2(s.plane.dx, s.plane.dz), tsec);
    }
    this.updateFx(dt);

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
    const pickup = ground && me ? this.nearestPickup(me) : null;
    const touch = matchMedia('(pointer: coarse)').matches;
    const needLock = !touch && !this.d.input.pointerLocked && !hud.mapOpen && !this.paused && s.phase !== P.PH_LOBBY;
    hud.setPrompt(needLock ? 'クリックしてマウス操作を開始' : pickup && !touch ? `[F] 拾う: ${pickup}` : null);
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
    });
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
      a = { phase: Math.random() * 6, x, z, speed: 0, deadAt: alive ? -1 : this.time - 5, foot: 0, wasAlive: alive, seen: 0 };
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
