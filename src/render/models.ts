// Procedural models from smooth instanced shapes (bevelled boxes, ellipsoids,
// capsules, rods): characters, guns, loot, the drop plane. The character keeps
// the joint frames and part sizes the server's hit volumes are built from.

import { Instances, rgba, surf, SHAPE_BOX, SHAPE_PILL, SHAPE_ROD, SHAPE_SPHERE } from './instances';
import { add, clamp, qIdent, qMul, qRot, qX, qY, qZ, type Q, type V3 } from './math';

type T = { p: V3; q: Q };
const child = (t: T, pivot: V3, q: Q): T => ({ p: add(t.p, qRot(t.q, pivot)), q: qMul(t.q, q) });
const box = (I: Instances, t: T, c: V3, s: V3, col: number, shape = SHAPE_BOX, mat = CLOTH) => I.push(add(t.p, qRot(t.q, c)), t.q, s, col, shape, mat);

const CLOTH = surf(0.9, 0), SKINM = surf(0.55, 0), GEAR = surf(0.6, 0.05), GUNMETAL = surf(0.38, 0.85), POLY = surf(0.55, 0.1), PAINT = surf(0.45, 0.4);

const METAL = rgba(0x2b2e33), METAL2 = rgba(0x40464d), WOOD = rgba(0x7a4e2c), TAN = rgba(0xb39a6c), OLIVE = rgba(0x4b5a3b);
const GLASS = rgba(0x1d3b52);

// [cx, cy, cz, sx, sy, sz, color]
type Part = [number, number, number, number, number, number, number];
export const GUNS: Part[][] = [
  [],
  [
    [0, 0.05, 0.07, 0.05, 0.075, 0.2, METAL],
    [0, -0.04, -0.01, 0.045, 0.12, 0.06, METAL2],
  ],
  [
    [0, 0.04, 0.1, 0.06, 0.1, 0.34, METAL],
    [0, -0.09, 0.13, 0.04, 0.17, 0.05, METAL2],
    [0, 0.03, -0.15, 0.04, 0.07, 0.17, METAL2],
    [0, 0.05, 0.32, 0.03, 0.03, 0.12, METAL],
  ],
  [
    [0, 0.04, 0.12, 0.06, 0.1, 0.5, METAL],
    [0, -0.1, 0.17, 0.045, 0.18, 0.07, METAL2],
    [0, 0.02, -0.22, 0.05, 0.11, 0.25, TAN],
    [0, 0.06, 0.48, 0.03, 0.03, 0.24, METAL],
    [0, 0.04, 0.3, 0.07, 0.08, 0.16, TAN],
  ],
  [
    [0, 0.03, 0.17, 0.06, 0.09, 0.62, METAL],
    [0, -0.03, 0.33, 0.07, 0.06, 0.18, WOOD],
    [0, 0.0, -0.2, 0.05, 0.12, 0.28, WOOD],
  ],
  [
    [0, 0.04, 0.15, 0.06, 0.1, 0.6, OLIVE],
    [0, 0.13, 0.1, 0.05, 0.05, 0.22, METAL],
    [0, -0.08, 0.14, 0.04, 0.13, 0.06, METAL2],
    [0, 0.02, -0.22, 0.05, 0.11, 0.24, OLIVE],
    [0, 0.06, 0.55, 0.03, 0.03, 0.22, METAL],
  ],
  [
    [0, 0.04, 0.2, 0.06, 0.1, 0.8, rgba(0x55603f)],
    [0, 0.14, 0.12, 0.06, 0.06, 0.32, METAL],
    [0, 0.14, 0.29, 0.07, 0.07, 0.03, GLASS],
    [0, -0.07, 0.12, 0.04, 0.1, 0.06, METAL2],
    [0, 0.01, -0.27, 0.06, 0.13, 0.3, rgba(0x55603f)],
    [0, 0.06, 0.74, 0.035, 0.035, 0.36, METAL],
  ],
];

export const RARITY = [0xdddddd, 0xe8e8e8, 0x6fd35a, 0x4aa3ff, 0x6fd35a, 0x4aa3ff, 0xffc33a];
export const LEVEL_COL = [0x888888, 0xe0e0e0, 0x4aa3ff, 0xb06bff];
export const AMMO_COL = [0xe0c040, 0x5fbf4a, 0xd04a3a, 0x3f7fd0];

const SHIRTS = [0x3d5a80, 0x9c3d3d, 0x4f7942, 0x6b4f9c, 0xc28f2c, 0x2f6f73, 0x8a5a44, 0x556270, 0xb5523b, 0x3e7cb1, 0x7a8b3a, 0x5b5b66];
const PANTS = [0x2b2d42, 0x3a3a3a, 0x4a3b2a, 0x2f3e46, 0x5a4a3a, 0x283845];
const SKIN = [0xe8b896, 0xc68e64, 0x8d5a3b, 0xf1c9a5, 0xa86f4c];
const CANOPY = [0xe05a3a, 0x3a8fe0, 0xf0c040, 0x5ac06a, 0xb05ad0, 0xf08a3a];

export interface CharState {
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  mode: number; // 1 fall, 2 para, 3 ground, 4 dead
  crouch: boolean;
  speed: number;
  phase: number;
  weapon: number;
  ads: boolean;
  firing: boolean;
  deadT: number; // seconds since death, <0 if alive
  color: number;
  vest: number;
  helm: number;
  healing: boolean;
  reloading: boolean;
  inWater: boolean;
  onGround: boolean;
  t: number;
}

export function emitGun(I: Instances, t: T, wk: number, s = 1) {
  for (const [cx, cy, cz, sx, sy, sz, col] of GUNS[wk] ?? []) {
    // long thin parts along the bore are barrels
    const barrel = sz > 2.6 * Math.max(sx, sy) && sx < 0.05;
    const metal = col === METAL || col === METAL2;
    box(I, t, [cx * s, cy * s, cz * s], [sx * s, sy * s, sz * s], col, barrel ? SHAPE_ROD : SHAPE_BOX, metal ? GUNMETAL : POLY);
  }
}

export function emitCharacter(I: Instances, c: CharState) {
  const shirt = rgba(SHIRTS[c.color % SHIRTS.length]);
  const pants = rgba(PANTS[c.color % PANTS.length]);
  const skin = rgba(SKIN[(c.color * 7) % SKIN.length]);
  const boots = rgba(0x2a2420);
  const pitch = clamp(c.pitch, -1.2, 1.2);
  const dead = c.mode === 4 || c.deadT >= 0;
  let rootQ = qY(c.yaw);
  let root: V3 = [c.x, c.y, c.z];
  const freefall = c.mode === 1;
  const para = c.mode === 2;
  const crouch = c.crouch && !dead && !freefall && !para;
  const hipH = crouch ? 0.5 : 0.95;

  if (dead) {
    const k = clamp(Math.max(0, c.deadT) / 0.45, 0, 1);
    rootQ = qMul(rootQ, qX(-Math.PI * 0.5 * k));
    root = [c.x, c.y + 0.16 * k, c.z];
  }
  let T0: T = { p: root, q: rootQ };
  if (freefall) {
    // rotate around the hips: face down
    const hip: V3 = [c.x, c.y + 0.95, c.z];
    const q = qMul(rootQ, qX(1.25));
    T0 = { p: add(hip, qRot(q, [0, -0.95, 0])), q };
  }
  const moving = c.speed > 0.4 && c.onGround && !dead;
  const amp = moving ? Math.min(c.speed / 5, 1.35) * 0.62 : 0;
  const sprint = c.speed > 6.2;
  // crouched: knees bent, back nearly upright so the head stays over the
  // body axis where the server's head hitbox is
  let lean = crouch ? 0.15 : sprint ? 0.22 : 0.05;
  if (dead || freefall || para) lean = 0;

  const hips = child(T0, [0, hipH, 0], qIdent());
  const torso = child(hips, [0, 0.02, 0], qX(lean));
  box(I, torso, [0, 0.3, 0], [0.46, 0.56, 0.26], shirt);
  // collar and belt (inside the torso volume)
  box(I, torso, [0, 0.03, 0], [0.47, 0.07, 0.27], rgba(0x2a2622), SHAPE_BOX, GEAR);
  if (c.vest > 0) {
    const vc = c.vest === 3 ? rgba(0x3d3550) : c.vest === 2 ? rgba(0x2f4a5f) : rgba(0x6b6650);
    box(I, torso, [0, 0.27, 0], [0.5, 0.42, 0.31], vc, SHAPE_BOX, GEAR);
    box(I, torso, [0, 0.33, 0.14], [0.3, 0.2, 0.03], rgba(0x1c1f22), SHAPE_BOX, GEAR);
  }
  box(I, torso, [0, 0.32, -0.2], [0.34, 0.42, 0.14], rgba(0x4d4a3a), SHAPE_BOX, GEAR);
  // head
  const headPitch = dead ? 0 : -pitch * 0.55 - lean * 0.6;
  const head = child(torso, [0, 0.6, 0], qX(headPitch));
  box(I, head, [0, 0.14, 0.01], [0.25, 0.27, 0.27], skin, SHAPE_SPHERE, SKINM);
  // neck
  box(I, head, [0, -0.01, 0], [0.12, 0.1, 0.12], skin, SHAPE_PILL, SKINM);
  box(I, head, [0.065, 0.16, 0.125], [0.045, 0.03, 0.02], rgba(0x111111), SHAPE_SPHERE, surf(0.2, 0));
  box(I, head, [-0.065, 0.16, 0.125], [0.045, 0.03, 0.02], rgba(0x111111), SHAPE_SPHERE, surf(0.2, 0));
  if (c.helm > 0) {
    const hc = c.helm === 3 ? rgba(0x2b2f2a) : c.helm === 2 ? rgba(0x45503a) : rgba(0x6a6a5a);
    box(I, head, [0, 0.22, 0], [0.3, 0.19, 0.31], hc, SHAPE_SPHERE, surf(0.45, 0.25));
  } else {
    box(I, head, [0, 0.235, -0.015], [0.26, 0.12, 0.27], rgba(0x2a1d14), SHAPE_SPHERE, surf(0.8, 0));
  }

  // legs
  for (const s of [1, -1]) {
    let thigh = 0, knee = 0;
    if (crouch) {
      thigh = -1.45;
      knee = 2.1;
      if (moving) thigh += Math.sin(c.phase + (s > 0 ? 0 : Math.PI)) * 0.25;
    } else if (freefall) {
      thigh = 0.25;
      knee = 0.6;
    } else if (para) {
      thigh = -0.25 + Math.sin(c.t * 2 + s) * 0.1;
      knee = 0.35;
    } else if (!c.onGround && !dead) {
      thigh = -0.5;
      knee = 0.8;
    } else {
      const ph = c.phase + (s > 0 ? 0 : Math.PI);
      thigh = -Math.sin(ph) * amp;
      knee = Math.max(0, -Math.cos(ph)) * amp * 1.4;
    }
    const L = child(hips, [0.12 * s, 0, 0], qX(thigh));
    box(I, L, [0, -0.23, 0], [0.17, 0.5, 0.19], pants, SHAPE_PILL);
    const K = child(L, [0, -0.46, 0], qX(knee));
    box(I, K, [0, -0.23, 0], [0.15, 0.5, 0.17], pants, SHAPE_PILL);
    // knee pad
    box(I, K, [0, -0.02, 0.06], [0.13, 0.12, 0.08], rgba(0x26282b), SHAPE_SPHERE, GEAR);
    box(I, K, [0, -0.44, 0.045], [0.15, 0.1, 0.27], boots, SHAPE_BOX, surf(0.6, 0));
  }

  // arms + weapon
  const hasGun = c.weapon > 0 && !dead;
  const armPitch = -Math.PI / 2 - pitch * 0.85 - lean;
  for (const s of [1, -1]) {
    let q: Q;
    let elbow = -0.25;
    if (dead) {
      q = qZ(s * 0.5);
    } else if (freefall) {
      q = qMul(qZ(s * 1.25), qX(-0.3));
      elbow = -0.2;
    } else if (para) {
      q = qMul(qZ(s * 0.25), qX(-Math.PI + 0.25));
      elbow = -0.3;
    } else if (c.healing) {
      q = qMul(qY(s * -0.4), qX(-1.1));
      elbow = -0.9;
    } else if (hasGun) {
      if (s > 0) {
        q = qMul(qY(-0.55), qX(armPitch + (c.reloading ? 0.7 : 0.05)));
        elbow = c.reloading ? -0.9 : -0.35;
      } else {
        q = qMul(qY(-0.1), qX(armPitch + 0.25));
        elbow = -0.75;
      }
    } else if (c.firing && s < 0) {
      q = qX(armPitch);
      elbow = 0;
    } else {
      const ph = c.phase + (s > 0 ? Math.PI : 0);
      q = qMul(qZ(s * 0.08), qX(-Math.sin(ph) * amp * 0.8));
      elbow = -0.35 - (moving ? 0.25 : 0);
    }
    const A = child(torso, [0.29 * s, 0.5, 0], q);
    box(I, A, [0, -0.15, 0], [0.13, 0.36, 0.13], shirt, SHAPE_PILL);
    const E = child(A, [0, -0.3, 0], qX(elbow));
    box(I, E, [0, -0.14, 0], [0.11, 0.32, 0.11], shirt, SHAPE_PILL);
    box(I, E, [0, -0.31, 0], [0.1, 0.11, 0.1], skin, SHAPE_SPHERE, SKINM);
  }
  if (hasGun && !freefall && !para) {
    const gp: V3 = c.ads ? [-0.06, 0.52, 0.34] : [-0.13, 0.42, 0.36];
    const gq = c.reloading || c.healing ? qMul(qX(0.65), qZ(0.3)) : qX(-pitch - lean);
    const G = child(torso, gp, gq);
    emitGun(I, G, c.weapon);
  }
  if (c.healing && !dead) {
    box(I, child(torso, [0, 0.25, 0.42], qIdent()), [0, 0, 0], [0.16, 0.1, 0.1], rgba(0xf4f4f4));
  }
  if (para) {
    const col = CANOPY[c.color % CANOPY.length];
    const top: T = { p: [c.x, c.y, c.z], q: qY(c.yaw) };
    for (let k = -3; k <= 3; k++) {
      const a = k * 0.24;
      const seg = child(top, [Math.sin(a) * 3.2, 4.4 + Math.cos(a) * 1.1, 0.3], qZ(-a));
      box(I, seg, [0, 0, 0], [1.02, 0.12, 2.1], rgba(k % 2 === 0 ? col : 0xf2f2f2));
    }
    for (const s of [1, -1]) {
      const from: V3 = add([c.x, c.y, c.z], qRot(top.q, [0.25 * s, 1.55, 0]));
      const to: V3 = add([c.x, c.y, c.z], qRot(top.q, [2.6 * s, 4.9, 0.3]));
      line(I, from, to, 0.02, rgba(0xdddddd));
    }
  }
}

/** Thin box between two points. */
export function line(I: Instances, a: V3, b: V3, w: number, col: number) {
  const d: V3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const l = Math.hypot(d[0], d[1], d[2]);
  if (l < 1e-4) return;
  const yaw = Math.atan2(d[0], d[2]);
  const pitch = Math.asin(clamp(d[1] / l, -1, 1));
  I.push([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2], qMul(qY(yaw), qX(-pitch)), [w, w, l], col);
}

export function emitItem(I: Instances, glow: Instances, kind: number, x: number, y: number, z: number, t: number, id: number) {
  const rot = t * 0.9 + id * 1.7;
  const bob = 0.12 + Math.sin(t * 2.2 + id) * 0.04;
  const T0: T = { p: [x, y + bob, z], q: qY(rot) };
  let glowCol = 0xffffff;
  if (kind >= 1 && kind <= 6) {
    emitGun(I, child(T0, [0, 0.05, -0.15], qIdent()), kind, 1.15);
    glowCol = RARITY[kind];
  } else if (kind >= 10 && kind <= 13) {
    const col = AMMO_COL[kind - 10];
    box(I, T0, [0, 0.07, 0], [0.24, 0.14, 0.17], rgba(col));
    box(I, T0, [0, 0.145, 0], [0.2, 0.02, 0.13], rgba(0x333333));
    glowCol = col;
  } else if (kind === 20) {
    box(I, T0, [0, 0.05, 0], [0.18, 0.1, 0.12], rgba(0xf2efe6));
    box(I, T0, [0, 0.05, 0], [0.19, 0.04, 0.13], rgba(0xd0c8b0));
    glowCol = 0x9fe08f;
  } else if (kind === 21) {
    box(I, T0, [0, 0.1, 0], [0.34, 0.2, 0.22], rgba(0xf4f4f4));
    box(I, T0, [0, 0.2, 0], [0.18, 0.012, 0.05], rgba(0xd03030));
    box(I, T0, [0, 0.2, 0], [0.05, 0.012, 0.16], rgba(0xd03030));
    glowCol = 0x9fe08f;
  } else if (kind === 22) {
    box(I, T0, [0, 0.11, 0], [0.09, 0.22, 0.09], rgba(0xf08a2a));
    box(I, T0, [0, 0.23, 0], [0.07, 0.02, 0.07], rgba(0xcccccc));
    glowCol = 0x9fe08f;
  } else if (kind >= 30 && kind <= 32) {
    const lv = kind - 29;
    box(I, T0, [0, 0.22, 0], [0.42, 0.42, 0.14], rgba(lv === 3 ? 0x3d3550 : lv === 2 ? 0x2f4a5f : 0x6b6650));
    box(I, T0, [0, 0.36, 0.075], [0.3, 0.05, 0.02], rgba(LEVEL_COL[lv]));
    glowCol = LEVEL_COL[lv];
  } else if (kind >= 33 && kind <= 35) {
    const lv = kind - 32;
    box(I, T0, [0, 0.08, 0], [0.3, 0.16, 0.32], rgba(lv === 3 ? 0x2b2f2a : lv === 2 ? 0x45503a : 0x6a6a5a));
    box(I, T0, [0, 0.12, 0.165], [0.2, 0.04, 0.01], rgba(LEVEL_COL[lv]));
    glowCol = LEVEL_COL[lv];
  } else if (kind === 40) {
    box(I, T0, [0, 0.08, 0], [0.11, 0.15, 0.11], rgba(0x3f4a2e));
    box(I, T0, [0, 0.17, 0], [0.04, 0.04, 0.04], rgba(0x999999));
    glowCol = 0xff8040;
  }
  glow.push([x, y + 0.02, z], qIdent(), [0.7, 0.01, 0.7], rgba(glowCol, 70));
  if (kind === 6 || kind === 32 || kind === 35) glow.push([x, y + 1.5, z], qIdent(), [0.05, 3, 0.05], rgba(glowCol, 60));
}

export function emitPlane(I: Instances, p: V3, yaw: number, t: number) {
  const T0: T = { p, q: qY(yaw) };
  const body = rgba(0x9aa3ab), dark = rgba(0x3b434a), stripe = rgba(0xd86a2a);
  box(I, T0, [0, 0, 0], [3.2, 3.2, 24], body, SHAPE_ROD, PAINT);
  box(I, T0, [0, -0.2, 12.6], [2.9, 2.9, 4.5], body, SHAPE_SPHERE, PAINT);
  box(I, T0, [0, 0.7, 13.6], [2.2, 0.8, 1.2], dark);
  box(I, T0, [0, 0.3, 0], [3.25, 0.5, 20], stripe);
  box(I, T0, [0, -0.5, 1.5], [34, 0.45, 4.6], body);
  box(I, T0, [0, 2.6, -10.5], [0.4, 4.2, 3.2], body);
  box(I, T0, [0, 1.1, -11], [10, 0.35, 2.6], body);
  for (const s of [1, -1]) {
    const eng = child(T0, [7 * s, -1.0, 3.5], qIdent());
    box(I, eng, [0, 0, 0], [1.7, 1.7, 3.6], dark, SHAPE_ROD, PAINT);
    const prop = child(eng, [0, 0, 1.95], qZ(t * 35 * s));
    box(I, prop, [0, 0, 0], [0.3, 4.2, 0.12], rgba(0x222222));
  }
}
