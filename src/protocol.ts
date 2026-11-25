// Binary wire protocol. Must mirror core/src/sim.rs exactly.

export const PROTO = 16;
export const C_INPUT = 2;
export const C_PING = 3;
export const C_START = 4;
export const S_WELCOME = 0x81;
export const S_SNAP = 0x82;
export const S_PONG = 0x83;
export const S_ERROR = 0x84;

export const PH_LOBBY = 0;
export const PH_PLAY = 1;
export const PH_END = 2;

export const MODE_PLANE = 0;
export const MODE_FALL = 1;
export const MODE_PARA = 2;
export const MODE_GROUND = 3;
export const MODE_DEAD = 4;
/** Driving a car: the self state's pos/vel are the car's and airtime is its heading. */
export const MODE_DRIVE = 5;
/** Flying a helicopter: the self state's position / velocity are the helicopter's (its model origin), `airtime` its heading, `onGround` = landed. */
export const MODE_FLY = 6;

export const B_FIRE = 1 << 0;
export const B_ADS = 1 << 1;
export const B_CROUCH = 1 << 2;
export const B_SPRINT = 1 << 3;
export const B_JUMP = 1 << 4;
export const B_RELOAD = 1 << 5;
export const B_PICKUP = 1 << 6;
export const B_SPECT = 1 << 7;
export const B_NADE = 1 << 12;
export const B_USE = 1 << 13;
/** Kill-feed weapon ids that are not guns. */
export const WK_NADE = 200;
export const WK_CAR = 201;
export const WK_TRUCK = 202;
export const WK_BOOM = 203;
/** Kill-feed ids: a helicopter's missile, and being hit by its hull. */
export const WK_MISSILE = 204;
export const WK_HELI = 205;

export const KILLER_ZONE = 255;
export const ITEM_CHUNKS = 8;
export const MAP_SIZE = 1024;
export function itemChunk(x: number, z: number): number {
  const w = MAP_SIZE / ITEM_CHUNKS;
  const cx = Math.min(ITEM_CHUNKS - 1, Math.max(0, Math.floor(x / w)));
  const cz = Math.min(ITEM_CHUNKS - 1, Math.max(0, Math.floor(z / w)));
  return cz * ITEM_CHUNKS + cx;
}

class Reader {
  private v: DataView;
  private i = 0;
  bad = false;
  constructor(buf: ArrayBuffer) {
    this.v = new DataView(buf);
  }
  private need(n: number) {
    if (this.i + n > this.v.byteLength) {
      this.bad = true;
      return false;
    }
    return true;
  }
  u8() {
    if (!this.need(1)) return 0;
    return this.v.getUint8(this.i++);
  }
  u16() {
    if (!this.need(2)) return 0;
    const r = this.v.getUint16(this.i, true);
    this.i += 2;
    return r;
  }
  i16() {
    if (!this.need(2)) return 0;
    const r = this.v.getInt16(this.i, true);
    this.i += 2;
    return r;
  }
  u32() {
    if (!this.need(4)) return 0;
    const r = this.v.getUint32(this.i, true);
    this.i += 4;
    return r;
  }
  f32() {
    if (!this.need(4)) return 0;
    const r = this.v.getFloat32(this.i, true);
    this.i += 4;
    return Number.isFinite(r) ? r : 0;
  }
  str(n: number) {
    if (!this.need(n)) return '';
    const s = new TextDecoder().decode(new Uint8Array(this.v.buffer, this.v.byteOffset + this.i, n));
    this.i += n;
    return s;
  }
}

export interface PlayerInfo {
  id: number;
  bot: boolean;
  color: number;
  name: string;
  left?: boolean;
}

export interface ItemInfo {
  id: number;
  kind: number;
  amount: number;
  x: number;
  y: number;
  z: number;
}

export interface Welcome {
  proto: number;
  pid: number;
  seed: number;
  checksum: number;
  tickHz: number;
  snapEvery: number;
  slots: number;
  flags: number;
  lobbyT: number;
  players: PlayerInfo[];
  items: ItemInfo[];
}

export interface SelfState {
  mode: number;
  alive: boolean;
  onGround: boolean;
  crouch: boolean;
  inWater: boolean;
  ads: boolean;
  pos: [number, number, number];
  vel: [number, number, number];
  airtime: number;
  health: number;
  vest: number;
  vestDur: number;
  helm: number;
  helmDur: number;
  cur: number;
  slots: { kind: number; mag: number }[];
  ammo: number[];
  heals: number[];
  nades: number;
  reloadT: number;
  healT: number;
  healKind: number;
  boost: number;
  switchT: number;
  kills: number;
  dmg: number;
  spect: number;
  place: number;
  killer: number;
  /** Late commands the server will drop instead of simulating (see Game.reconcile). */
  debt: number;
}

export interface ZoneState {
  stage: number;
  shrinking: boolean;
  done: boolean;
  timer: number;
  cx: number;
  cz: number;
  r: number;
  nx: number;
  nz: number;
  nr: number;
  dps: number;
}

export interface PlaneState {
  active: boolean;
  canJump: boolean;
  sx: number;
  sz: number;
  dx: number;
  dz: number;
  t: number;
  enter: number;
  exit: number;
}

export interface RemotePlayer {
  id: number;
  mode: number;
  alive: boolean;
  crouch: boolean;
  ads: boolean;
  firing: boolean;
  healing: boolean;
  reloading: boolean;
  inWater: boolean;
  onGround: boolean;
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  weapon: number;
  vest: number;
  helm: number;
  health: number;
}

export type GameEvent =
  | { t: 'shot'; shooter: number; weapon: number; from: [number, number, number]; to: [number, number, number]; hit: number }
  | { t: 'hit'; victim: number; dmg: number; head: boolean; kill: boolean; armor: boolean }
  | { t: 'hurt'; attacker: number; sx: number; sz: number; dmg: number }
  | { t: 'kill'; killer: number; victim: number; weapon: number; head: boolean }
  | { t: 'item'; item: ItemInfo }
  | { t: 'idel'; id: number }
  | { t: 'pinfo'; info: PlayerInfo }
  | { t: 'boom'; x: number; y: number; z: number }
  | { t: 'end'; winner: number }
  | { t: 'pick'; kind: number; amount: number }
  | { t: 'throw'; id: number }
  | { t: 'land'; id: number }
  | { t: 'ichunk'; chunk: number; items: ItemInfo[] };

export interface Snapshot {
  tick: number;
  ack: number;
  phase: number;
  phaseT: number;
  alive: number;
  total: number;
  winner: number;
  self: SelfState;
  zone: ZoneState;
  plane: PlaneState;
  players: RemotePlayer[];
  nades: { id: number; x: number; y: number; z: number }[];
  /** Cars that moved lately (plus a rotating few): driver 255 = empty, 254 = someone unseen. */
  /** hp: bodywork left in 255ths, 0 = a burnt-out wreck. */
  /** `vx`/`vz` (m/s) and `spin` (rad/s): how the car moves right now. */
  cars: { id: number; x: number; y: number; z: number; yaw: number; driver: number; hp: number; vx: number; vz: number; spin: number }[];
  /** Loose crates and containers that moved lately: the middle of the underside. */
  props: { id: number; x: number; y: number; z: number; moving: boolean }[];
  /**
   * Helicopters in range. Angles in radians: pitch > 0 is nose down, roll > 0 is right side down
   * (right-handed turns about the model's +X (left) and +Z (nose) axes). rotor: speed 0..1.
   * pilot: 255 = nobody, 254 = someone unseen. hp: hull left in 255ths, 0 = a wreck. ammo: missiles left.
   */
  helis: { id: number; x: number; y: number; z: number; yaw: number; pitch: number; roll: number; rotor: number; pilot: number; hp: number; ammo: number; flags: number; airborne: boolean }[];
  /** Missiles in flight, with the direction they fly in. */
  missiles: { id: number; x: number; y: number; z: number; yaw: number; pitch: number }[];
  /** Doors standing open within range (every other door is shut). */
  doors: number[];
  events: GameEvent[];
}

const TAU = Math.PI * 2;

function readItem(r: Reader): ItemInfo {
  const id = r.u16();
  const kind = r.u8();
  const amount = r.u16();
  return { id, kind, amount, x: r.f32(), y: r.f32(), z: r.f32() };
}

export function decodeWelcome(buf: ArrayBuffer): Welcome | null {
  const r = new Reader(buf);
  if (r.u8() !== S_WELCOME) return null;
  const w: Welcome = {
    proto: r.u16(),
    pid: r.u8(),
    seed: r.u32(),
    checksum: r.u32(),
    tickHz: r.u8(),
    snapEvery: r.u8(),
    slots: r.u8(),
    flags: r.u8(),
    lobbyT: r.f32(),
    players: [],
    items: [],
  };
  const np = r.u8();
  for (let i = 0; i < np && !r.bad; i++) {
    const id = r.u8();
    const bot = r.u8() !== 0;
    const color = r.u8();
    const name = r.str(r.u8());
    w.players.push({ id, bot, color, name });
  }
  const ni = r.u16();
  for (let i = 0; i < ni && !r.bad; i++) w.items.push(readItem(r));
  return r.bad ? null : w;
}

export function decodeSnapshot(buf: ArrayBuffer): Snapshot | null {
  const r = new Reader(buf);
  if (r.u8() !== S_SNAP) return null;
  const tick = r.u32();
  const ack = r.u32();
  const phase = r.u8();
  const phaseT = r.f32();
  const alive = r.u8();
  const total = r.u8();
  const winner = r.u8();
  const mode = r.u8();
  const fl = r.u8();
  const self: SelfState = {
    mode,
    alive: (fl & 1) !== 0,
    onGround: (fl & 2) !== 0,
    crouch: (fl & 4) !== 0,
    inWater: (fl & 8) !== 0,
    ads: (fl & 16) !== 0,
    pos: [r.f32(), r.f32(), r.f32()],
    vel: [r.f32(), r.f32(), r.f32()],
    airtime: r.f32(),
    health: r.f32(),
    vest: r.u8(),
    vestDur: r.u8(),
    helm: r.u8(),
    helmDur: r.u8(),
    cur: r.u8(),
    slots: [],
    ammo: [],
    heals: [],
    nades: 0,
    reloadT: 0,
    healT: 0,
    healKind: 0,
    boost: 0,
    switchT: 0,
    kills: 0,
    dmg: 0,
    spect: 255,
    place: 0,
    killer: 255,
    debt: 0,
  };
  for (let s = 0; s < 2; s++) self.slots.push({ kind: r.u8(), mag: r.u16() });
  for (let a = 0; a < 4; a++) self.ammo.push(r.u16());
  for (let h = 0; h < 3; h++) self.heals.push(r.u8());
  self.nades = r.u8();
  self.reloadT = r.f32();
  self.healT = r.f32();
  self.healKind = r.u8();
  self.boost = r.f32();
  self.switchT = r.f32();
  self.kills = r.u16();
  self.dmg = r.u16();
  self.spect = r.u8();
  self.place = r.u8();
  self.killer = r.u8();
  self.debt = r.u8();
  const zstage = r.u8();
  const zf = r.u8();
  const zone: ZoneState = {
    stage: zstage,
    shrinking: (zf & 1) !== 0,
    done: (zf & 2) !== 0,
    timer: r.f32(),
    cx: r.f32(),
    cz: r.f32(),
    r: r.f32(),
    nx: r.f32(),
    nz: r.f32(),
    nr: r.f32(),
    dps: r.f32(),
  };
  const pf = r.u8();
  const plane: PlaneState = {
    active: (pf & 1) !== 0,
    canJump: (pf & 2) !== 0,
    sx: r.f32(),
    sz: r.f32(),
    dx: r.f32(),
    dz: r.f32(),
    t: r.f32(),
    enter: r.f32(),
    exit: r.f32(),
  };
  const players: RemotePlayer[] = [];
  const np = r.u8();
  for (let i = 0; i < np && !r.bad; i++) {
    const id = r.u8();
    const pm = r.u8();
    const f = r.u8();
    const x = r.f32(), y = r.f32(), z = r.f32();
    const yaw = (r.u16() * TAU) / 65536;
    const pitch = (r.i16() * 1.5) / 32767;
    const weapon = r.u8();
    const gear = r.u8();
    const health = r.u8();
    players.push({
      id, mode: pm, alive: (f & 1) !== 0, crouch: (f & 2) !== 0, ads: (f & 4) !== 0, firing: (f & 8) !== 0,
      healing: (f & 16) !== 0, reloading: (f & 32) !== 0, inWater: (f & 64) !== 0, onGround: (f & 128) !== 0,
      x, y, z, yaw, pitch, weapon, vest: gear & 3, helm: (gear >> 2) & 3, health,
    });
  }
  const nades: Snapshot['nades'] = [];
  const nn = r.u8();
  for (let i = 0; i < nn && !r.bad; i++) nades.push({ id: r.u16(), x: r.f32(), y: r.f32(), z: r.f32() });
  const cars: Snapshot['cars'] = [];
  const nc = r.u8();
  for (let i = 0; i < nc && !r.bad; i++) cars.push({ id: r.u8(), x: r.f32(), y: r.f32(), z: r.f32(), yaw: (r.u16() * TAU) / 65536, driver: r.u8(), hp: r.u8(), vx: ((r.u16() << 16) >> 16) / 100, vz: ((r.u16() << 16) >> 16) / 100, spin: ((r.u8() << 24) >> 24) / 50 });
  const props: Snapshot['props'] = [];
  const nProps = r.u8();
  for (let i = 0; i < nProps && !r.bad; i++) props.push({ id: r.u16(), x: r.f32(), y: r.f32(), z: r.f32(), moving: r.u8() !== 0 });
  const helis: Snapshot['helis'] = [];
  const nHelis = r.u8();
  for (let i = 0; i < nHelis && !r.bad; i++) {
    const id = r.u8();
    const x = r.f32();
    const y = r.f32();
    const z = r.f32();
    const yaw = (r.u16() * TAU) / 65536;
    const pitch = ((r.u8() << 24) >> 24) / 100;
    const roll = ((r.u8() << 24) >> 24) / 100;
    const rotor = r.u8() / 255;
    const pilot = r.u8();
    const hp = r.u8();
    const ammo = r.u8();
    const flags = r.u8();
    helis.push({ id, x, y, z, yaw, pitch, roll, rotor, pilot, hp, ammo, flags, airborne: (flags & 1) !== 0 });
  }
  const missiles: Snapshot['missiles'] = [];
  const nMissiles = r.u8();
  for (let i = 0; i < nMissiles && !r.bad; i++) missiles.push({ id: r.u16(), x: r.f32(), y: r.f32(), z: r.f32(), yaw: (r.u16() * TAU) / 65536, pitch: (r.i16() * 1.5) / 32767 });
  const doors: number[] = [];
  const nDoors = r.u8();
  for (let i = 0; i < nDoors && !r.bad; i++) doors.push(r.u16());
  const events: GameEvent[] = [];
  const ne = r.u16();
  for (let i = 0; i < ne && !r.bad; i++) {
    const t = r.u8();
    switch (t) {
      case 1: {
        const shooter = r.u8(), weapon = r.u8();
        const from: [number, number, number] = [r.f32(), r.f32(), r.f32()];
        const to: [number, number, number] = [r.f32(), r.f32(), r.f32()];
        events.push({ t: 'shot', shooter, weapon, from, to, hit: r.u8() });
        break;
      }
      case 2: {
        const victim = r.u8(), dmg = r.u8(), fl2 = r.u8();
        events.push({ t: 'hit', victim, dmg, head: (fl2 & 1) !== 0, kill: (fl2 & 2) !== 0, armor: (fl2 & 4) !== 0 });
        break;
      }
      case 3: {
        const attacker = r.u8(), sx = r.f32(), sz = r.f32(), dmg = r.u8();
        events.push({ t: 'hurt', attacker, sx, sz, dmg });
        break;
      }
      case 4: {
        const killer = r.u8(), victim = r.u8(), weapon = r.u8(), head = r.u8() !== 0;
        events.push({ t: 'kill', killer, victim, weapon, head });
        break;
      }
      case 5:
        events.push({ t: 'item', item: readItem(r) });
        break;
      case 6:
        events.push({ t: 'idel', id: r.u16() });
        break;
      case 7: {
        const id = r.u8(), f2 = r.u8(), color = r.u8();
        const name = r.str(r.u8());
        events.push({ t: 'pinfo', info: { id, bot: (f2 & 1) !== 0, left: (f2 & 2) !== 0, color, name } });
        break;
      }
      case 8:
        events.push({ t: 'boom', x: r.f32(), y: r.f32(), z: r.f32() });
        break;
      case 9:
        events.push({ t: 'end', winner: r.u8() });
        break;
      case 10: {
        const kind = r.u8();
        events.push({ t: 'pick', kind, amount: r.u16() });
        break;
      }
      case 11:
        events.push({ t: 'throw', id: r.u8() });
        break;
      case 12:
        events.push({ t: 'land', id: r.u8() });
        break;
      case 13: {
        const chunk = r.u8();
        const n = r.u16();
        const items: ItemInfo[] = [];
        for (let k = 0; k < n && !r.bad; k++) items.push(readItem(r));
        events.push({ t: 'ichunk', chunk, items });
        break;
      }
      default:
        r.bad = true;
    }
  }
  if (r.bad) return null;
  return { tick, ack, phase, phaseT, alive, total, winner, self, zone, plane, players, nades, cars, props, helis, missiles, doors, events };
}

export interface CmdQ {
  seq: number;
  view: number;
  mx: number; // -127..127
  mz: number;
  yaw: number; // u16
  pitch: number; // i16
  buttons: number;
}

export function quantizeCmd(seq: number, view: number, mx: number, mz: number, yaw: number, pitch: number, buttons: number): CmdQ {
  const q8 = (v: number) => Math.max(-127, Math.min(127, Math.round(v * 127)));
  let t = yaw / TAU;
  t -= Math.floor(t);
  const yq = Math.floor(t * 65536) & 0xffff;
  const pq = Math.trunc((Math.max(-1.5, Math.min(1.5, pitch)) * 32767) / 1.5);
  return { seq: seq >>> 0, view: view >>> 0, mx: q8(mx), mz: q8(mz), yaw: yq, pitch: pq, buttons: buttons & 0x3fff };
}

export function encodeInputs(cmds: CmdQ[]): Uint8Array<ArrayBuffer> {
  const n = Math.min(cmds.length, 8);
  const out = new Uint8Array(2 + n * 16);
  const v = new DataView(out.buffer);
  out[0] = C_INPUT;
  out[1] = n;
  let o = 2;
  for (let i = cmds.length - n; i < cmds.length; i++) {
    const c = cmds[i];
    v.setUint32(o, c.seq, true);
    v.setUint32(o + 4, c.view, true);
    v.setInt8(o + 8, c.mx);
    v.setInt8(o + 9, c.mz);
    v.setUint16(o + 10, c.yaw, true);
    v.setInt16(o + 12, c.pitch, true);
    v.setUint16(o + 14, c.buttons, true);
    o += 16;
  }
  return out;
}

export function encodePing(t: number): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(5);
  out[0] = C_PING;
  new DataView(out.buffer).setUint32(1, t >>> 0, true);
  return out;
}
