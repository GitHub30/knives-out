// Vegetation: broadleaf and conifer trees grown as a branch skeleton (opaque
// tubes) carrying leaf cards (alpha cut-out), with procedural PBR textures,
// instancing from a storage buffer, three levels of detail and their own
// alpha-tested shadow pass.
//
// Fits the engine as it is: group 0 is the frame uniform + shadow cascades
// (main pass) or the frame uniform alone (shadow passes); the scene target is
// HDR without MSAA, reverse-Z depth; shadows are two cascades in a depth array.

import { COMMON, FRAME } from './shaders';
import { aabbVisible, type V3 } from './math';

/** The scene pass has no MSAA: leaves are alpha-tested with dithering. With MSAA they would use alpha-to-coverage. */
const MSAA_SAMPLES = 1;
const ALPHA_CUTOFF = 0.45;
const TEX = 1024;
/** Heights the two species are modelled at (m); instances scale from this. */
const H_BROAD = 10;
const H_CONIFER = 13;
const VARIANTS = 3;
const T_STRIDE = 8; // trunk vertex: pos, normal, uv
const L_STRIDE = 18; // leaf vertex: position, normal, canopyNormal, uv, aoAndWind, pivot

// ------------------------------------------------------------------ shaders
const TREE_IO = /* wgsl */ `
struct TreeInst {
  pos: vec3f,
  yaw: f32,
  scale: f32,
  seed: f32,       // 0..1, colour of the individual
  cardScale: f32,  // leaf cards grow as they are thinned out with distance
  species: f32,    // 0 conifer, 1 broadleaf
};
@group(1) @binding(0) var<storage, read> trees: array<TreeInst>;
@group(1) @binding(1) var tSamp: sampler;
@group(1) @binding(2) var tAlbedo: texture_2d<f32>;
@group(1) @binding(3) var tNormal: texture_2d<f32>;
@group(1) @binding(4) var tRTA: texture_2d<f32>;   // roughness, translucency, ao

fn rotY(v: vec3f, a: f32) -> vec3f {
  let c = cos(a);
  let s = sin(a);
  return vec3f(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

// Wind in two layers, the same for the main pass and the shadow passes so a
// leaf and its shadow never part:
//  1. the whole tree leans and sways slowly, more the higher up and the
//     further out along the branches;
//  2. the leaf cards shiver quickly (flutter weight and phase per card).
// p is the vertex in the tree's own frame (metres), relative to its foot.
fn applyTreeWind(p: vec3f, treePos: vec3f, seed: f32, flutter: f32, phase: f32, time: f32) -> vec3f {
  let gust = 0.55 + 0.45 * sin(time * 0.31 + treePos.x * 0.023 + treePos.z * 0.019);
  let t1 = time * 0.85 + seed * 6.2831 + treePos.x * 0.047 + treePos.z * 0.041;
  let lean = sin(t1) + 0.35 * sin(t1 * 2.3 + 1.7);
  let cross_ = 0.5 * cos(t1 * 0.77 + 0.6);
  let hw = clamp(p.y / 12.0, 0.0, 1.3);
  let spread = length(p.xz);
  let bend = hw * hw * 0.16 + spread * hw * 0.012;
  var q = p;
  q.x += (0.86 * lean - 0.5 * cross_) * bend * gust;
  q.z += (0.5 * lean + 0.86 * cross_) * bend * gust;
  // boughs nod
  q.y += sin(t1 * 1.6 + spread * 0.9 + seed * 11.0) * spread * spread * 0.006 * gust;
  let t2 = time * 4.4 + phase * 6.2831 + p.x * 1.7 + p.z * 1.3;
  q += vec3f(sin(t2), 0.6 * sin(t2 * 1.31 + phase * 9.0), cos(t2 * 0.83)) * flutter * 0.03 * (0.4 + gust);
  return q;
}

struct TrunkIn {
  @builtin(instance_index) ii: u32,
  @location(0) pos: vec3f,
  @location(1) nrm: vec3f,
  @location(2) uv: vec2f,
};
struct LeafIn {
  @builtin(instance_index) ii: u32,
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) canopyNormal: vec3f,
  @location(3) uv: vec2f,
  @location(4) aoAndWind: vec4f,   // canopy ao, sway weight, flutter phase, colour variation
  @location(5) pivot: vec3f,
};
fn trunkWorld(v: TrunkIn, time: f32) -> vec3f {
  let t = trees[v.ii];
  let p = applyTreeWind(v.pos * t.scale, t.pos, t.seed, 0.0, 0.0, time);
  return t.pos + rotY(p, t.yaw);
}
fn leafWorld(v: LeafIn, time: f32) -> vec3f {
  let t = trees[v.ii];
  let local = (v.pivot + (v.position - v.pivot) * t.cardScale) * t.scale;
  let p = applyTreeWind(local, t.pos, t.seed, v.aoAndWind.y, v.aoAndWind.z, time);
  return t.pos + rotY(p, t.yaw);
}
`;

const TREE_WGSL = /* wgsl */ `
${COMMON}
${TREE_IO}
struct TrunkOut {
  @builtin(position) clip: vec4f,
  @location(0) wp: vec3f,
  @location(1) n: vec3f,
  @location(2) uv: vec2f,
  @location(3) tint: vec2f,   // individual seed, height above the foot
};
@vertex fn vs_trunk(v: TrunkIn) -> TrunkOut {
  let t = trees[v.ii];
  var o: TrunkOut;
  o.wp = trunkWorld(v, F.camPos.w);
  o.clip = F.viewProj * vec4f(o.wp, 1.0);
  o.n = rotY(v.nrm, t.yaw);
  o.uv = v.uv;
  o.tint = vec2f(t.seed + t.species * 2.0, v.pos.y * t.scale);
  return o;
}
// tangent frame from screen-space derivatives (no tangents are stored)
fn perturb(n: vec3f, wp: vec3f, uv: vec2f, tn: vec3f) -> vec3f {
  let dp1 = dpdx(wp);
  let dp2 = dpdy(wp);
  let du1 = dpdx(uv);
  let du2 = dpdy(uv);
  let dp2p = cross(dp2, n);
  let dp1p = cross(n, dp1);
  let T = dp2p * du1.x + dp1p * du2.x;
  let B = dp2p * du1.y + dp1p * du2.y;
  let inv = inverseSqrt(max(max(dot(T, T), dot(B, B)), 1e-12));
  return normalize(T * inv * tn.x + B * inv * tn.y + n * tn.z);
}
@fragment fn fs_trunk(i: TrunkOut) -> @location(0) vec4f {
  let n0 = normalize(i.n);
  let species = floor(i.tint.x * 0.5);
  let seed = i.tint.x - species * 2.0;
  var albedo = textureSample(tAlbedo, tSamp, i.uv).rgb;
  // conifers: redder, flakier bark; every tree a little its own shade
  albedo *= mix(vec3f(1.12, 0.9, 0.78), vec3f(0.95, 1.0, 1.0), species) * (0.8 + 0.4 * seed);
  let tn = textureSample(tNormal, tSamp, i.uv).xyz * 2.0 - 1.0;
  let rta = textureSample(tRTA, tSamp, i.uv);
  var s: Surface;
  s.n = perturb(n0, i.wp, i.uv, tn);
  s.rough = clamp(rta.r, 0.85, 0.95);
  s.metal = 0.0;
  // damp and mossy at the foot, shaded up in the crown
  let foot = 1.0 - smoothstep(0.0, 1.4, i.tint.y);
  albedo = mix(albedo, albedo * vec3f(0.62, 0.78, 0.5), foot * 0.6);
  s.albedo = albedo;
  s.ao = rta.b * mix(1.0, 0.65, smoothstep(2.5, 7.0, i.tint.y)) * (1.0 - 0.25 * foot);
  s.trans = 0.0;
  var c = shade(s, i.wp);
  c = applyFog(c, i.wp);
  return vec4f(c, 1.0);
}

struct LeafOut {
  @builtin(position) clip: vec4f,
  @location(0) wp: vec3f,
  @location(1) n: vec3f,
  @location(2) cn: vec3f,
  @location(3) uv: vec2f,
  @location(4) aw: vec4f,
  @location(5) seed: f32,
};
@vertex fn vs_leaf(v: LeafIn) -> LeafOut {
  let t = trees[v.ii];
  var o: LeafOut;
  o.wp = leafWorld(v, F.camPos.w);
  o.clip = F.viewProj * vec4f(o.wp, 1.0);
  o.n = rotY(v.normal, t.yaw);
  o.cn = rotY(v.canopyNormal, t.yaw);
  o.uv = v.uv;
  o.aw = v.aoAndWind;
  o.seed = t.seed;
  return o;
}
fn ign(p: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}
fn leafShade(i: LeafOut, front: bool, tex: vec4f) -> vec3f {
  // 1. two-sided normal, then blended towards the canopy's spherical normal
  var Ng = normalize(i.n);
  if (!front) { Ng = -Ng; }
  let tn = textureSample(tNormal, tSamp, i.uv).xyz * 2.0 - 1.0;
  let normalMapN = perturb(Ng, i.wp, i.uv, tn);
  let canopyNormal = normalize(i.cn);
  let N = normalize(mix(normalMapN, canopyNormal, 0.65));
  let rta = textureSample(tRTA, tSamp, i.uv);
  let rough = clamp(rta.r, 0.55, 0.8);
  let translucency = rta.g;
  let canopyAO = i.aw.x;
  // colour: each card and each tree a little different (fresh to yellowing)
  var albedo = tex.rgb * 0.8;
  albedo = mix(albedo * vec3f(0.82, 0.92, 0.8), albedo * vec3f(1.18, 1.08, 0.72), i.aw.w);
  albedo *= mix(vec3f(0.86, 0.96, 0.9), vec3f(1.12, 1.04, 0.86), i.seed);

  let L = F.sunDir.xyz;
  let V = normalize(F.camPos.xyz - i.wp);
  let sunColor = F.sunColor.rgb;
  let sh = shadowFactor(i.wp, N);
  // 3. energy-conserving GGX + Lambert, F0 = 0.04
  let ndl = max(dot(N, L), 0.0);
  let ndv = max(dot(N, V), 1e-3);
  let H = normalize(L + V);
  let ndh = max(dot(N, H), 0.0);
  let vdh = max(dot(V, H), 0.0);
  let a = rough * rough;
  let a2 = a * a;
  let dd = ndh * ndh * (a2 - 1.0) + 1.0;
  let D = a2 / (PI * dd * dd);
  let k = (rough + 1.0) * (rough + 1.0) / 8.0;
  let G = (ndl / (ndl * (1.0 - k) + k)) * (ndv / (ndv * (1.0 - k) + k));
  let Fr = 0.04 + 0.96 * pow(1.0 - vdh, 5.0);
  let spec = Fr * D * G / (4.0 * ndl * ndv + 1e-3);
  var c = (albedo / PI * (1.0 - Fr) + vec3f(spec)) * ndl * sunColor * sh * PI * (0.45 + 0.55 * canopyAO);
  // 2. light through the leaf when the sun is behind it
  let backLightDir = normalize(L + N * 0.35);
  let sssDot = pow(clamp(dot(V, -backLightDir), 0.0, 1.0), 3.0);
  let sss = sssDot * translucency * sunColor * vec3f(0.45, 0.72, 0.12) * (0.3 + 0.7 * canopyAO);
  c += sss * (0.3 + 0.7 * sh) * 0.45;
  // ambient, cut deep inside the crown
  let up = N.y * 0.5 + 0.5;
  let ground = sunColor * vec3f(0.11, 0.1, 0.08);
  let irr = mix(ground, mix(F.skyHorizon.rgb, F.skyTop.rgb, 0.5) * 0.75 + sunColor * 0.03, up);
  let amb = canopyAO * canopyAO * rta.b;
  c += albedo * irr * F.sunColor.w * amb;
  let fa = 0.04 + (max(1.0 - rough, 0.04) - 0.04) * pow(1.0 - ndv, 5.0);
  c += irr * fa * amb * F.sunColor.w * 0.5;
  return applyFog(c, i.wp);
}
// no MSAA: alpha test, the threshold dithered to soften the cut edge
@fragment fn fs_leaf(i: LeafOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let tex = textureSample(tAlbedo, tSamp, i.uv);
  if (tex.a < ${ALPHA_CUTOFF} + (ign(i.clip.xy) - 0.5) * 0.16) { discard; }
  return vec4f(leafShade(i, front, tex), 1.0);
}
// MSAA: alpha-to-coverage on a sharpened alpha
@fragment fn fs_leaf_a2c(i: LeafOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let tex = textureSample(tAlbedo, tSamp, i.uv);
  let a = clamp((tex.a - ${ALPHA_CUTOFF}) / max(fwidth(tex.a), 1e-4) + 0.5, 0.0, 1.0);
  return vec4f(leafShade(i, front, tex), a);
}
`;

const TREE_SHADOW_WGSL = /* wgsl */ `
${FRAME}
@group(0) @binding(0) var<uniform> F: Frame;
${TREE_IO}
@vertex fn vs_trunk(v: TrunkIn) -> @builtin(position) vec4f {
  return F.lightVP0 * vec4f(trunkWorld(v, F.camPos.w), 1.0);
}
struct LeafSh {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
};
@vertex fn vs_leaf(v: LeafIn) -> LeafSh {
  var o: LeafSh;
  o.clip = F.lightVP0 * vec4f(leafWorld(v, F.camPos.w), 1.0);
  o.uv = v.uv;
  return o;
}
// depth only, but through the leaf's alpha: sunlight falls between the leaves
@fragment fn fs_leaf(i: LeafSh) {
  let alpha = textureSample(tAlbedo, tSamp, i.uv).a;
  if (alpha < ${ALPHA_CUTOFF}) { discard; }
}
`;

// ------------------------------------------------------------------ helpers
type Rng = () => number;
function rng(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const v3 = {
  add: (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  mul: (a: V3, k: number): V3 => [a[0] * k, a[1] * k, a[2] * k],
  dot: (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: (a: V3) => Math.hypot(a[0], a[1], a[2]),
  norm: (a: V3): V3 => {
    const l = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / l, a[1] / l, a[2] / l];
  },
};
/** Some unit vector at right angles to `d`. */
function perp(d: V3, r: Rng): V3 {
  const a: V3 = Math.abs(d[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const u = v3.norm(v3.cross(d, a)), w = v3.cross(d, u);
  const ang = r() * Math.PI * 2;
  return v3.norm(v3.add(v3.mul(u, Math.cos(ang)), v3.mul(w, Math.sin(ang))));
}

// ------------------------------------------------------------------ geometry
interface Branch {
  pts: V3[];
  rad: number[];
  level: number;
}
interface TreeMesh {
  trunk: Float32Array;
  trunkIdx: Uint32Array;
  /** Index count of the trunk and main limbs alone (the distant levels of detail). */
  trunkLod: number;
  leaf: Float32Array;
  leafIdx: Uint32Array;
  branches: Branch[];
  cards: { c: V3; a: V3; w: V3; len: number; wid: number; cell: number; ao: number }[];
  radius: number;
  height: number;
}

/**
 * A branch as a wandering path: every step the heading drifts a little
 * (`wobble`), is pulled by `tropism` (up towards the light, down under its own
 * weight), and the radius tapers from r0 to r1. The points are then smoothed
 * with a Catmull-Rom spline so the branch curves instead of kinking.
 */
function grow(r: Rng, start: V3, dir: V3, len: number, r0: number, r1: number, steps: number, wobble: number, tropism: (t: number) => V3, level: number): Branch {
  const ctrl: V3[] = [start];
  let d = v3.norm(dir);
  let p = start;
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    d = v3.norm(v3.add(v3.add(d, v3.mul([r() - 0.5, r() - 0.5, r() - 0.5], wobble * 2)), tropism(t)));
    p = v3.add(p, v3.mul(d, len / steps));
    ctrl.push(p);
  }
  const pts: V3[] = [], rad: number[] = [];
  const sub = level === 0 ? 3 : level === 1 ? 2 : 1;
  const at = (i: number) => ctrl[Math.max(0, Math.min(ctrl.length - 1, i))];
  for (let i = 0; i < steps; i++) {
    for (let k = 0; k < sub; k++) {
      const u = k / sub;
      const [p0, p1, p2, p3] = [at(i - 1), at(i), at(i + 1), at(i + 2)];
      const q: V3 = [0, 0, 0];
      for (let c = 0; c < 3; c++) {
        q[c] = 0.5 * (2 * p1[c] + (-p0[c] + p2[c]) * u + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * u * u + (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * u * u * u);
      }
      pts.push(q);
      const t = (i + u) / steps;
      rad.push(r0 + (r1 - r0) * Math.pow(t, 0.8));
    }
  }
  pts.push(ctrl[steps]);
  rad.push(r1);
  return { pts, rad, level };
}
/** Point and heading at parameter t (0..1) along a branch, and its radius there. */
function along(b: Branch, t: number): { p: V3; d: V3; r: number } {
  const f = Math.max(0, Math.min(0.9999, t)) * (b.pts.length - 1);
  const i = Math.floor(f), u = f - i;
  const p: V3 = v3.add(v3.mul(b.pts[i], 1 - u), v3.mul(b.pts[i + 1], u));
  return { p, d: v3.norm(v3.sub(b.pts[i + 1], b.pts[i])), r: b.rad[i] * (1 - u) + b.rad[i + 1] * u };
}

/** Tube around a branch. `flare` widens (and lobes) the foot of a trunk into roots. */
function tube(V: number[], I: number[], b: Branch, seg: number, flare: ((y: number, ang: number) => number) | null) {
  const base = V.length / T_STRIDE;
  const n = b.pts.length;
  let nrm: V3 = perp(v3.norm(v3.sub(b.pts[1], b.pts[0])), () => 0.13);
  let v = 0;
  const uRep = b.level === 0 ? 2 : 1;
  for (let i = 0; i < n; i++) {
    const t = v3.norm(v3.sub(b.pts[Math.min(n - 1, i + 1)], b.pts[Math.max(0, i - 1)]));
    // carry the frame along without twisting it
    nrm = v3.norm(v3.sub(nrm, v3.mul(t, v3.dot(nrm, t))));
    const bin = v3.cross(t, nrm);
    if (i > 0) v += v3.len(v3.sub(b.pts[i], b.pts[i - 1]));
    for (let j = 0; j <= seg; j++) {
      const ang = (j / seg) * Math.PI * 2;
      const dir = v3.add(v3.mul(nrm, Math.cos(ang)), v3.mul(bin, Math.sin(ang)));
      const k = flare ? flare(b.pts[i][1], ang) : 1;
      const p = v3.add(b.pts[i], v3.mul(dir, b.rad[i] * k));
      // where the foot spreads the surface leans outwards
      const nn = flare && k > 1.05 ? v3.norm(v3.add(dir, [0, 0.55 * Math.min(1, k - 1), 0])) : dir;
      V.push(p[0], p[1], p[2], nn[0], nn[1], nn[2], (j / seg) * uRep, v / (b.level === 0 ? 1.6 : 1.1));
    }
  }
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < seg; j++) {
      const a = base + i * (seg + 1) + j, c = a + seg + 1;
      I.push(a, a + 1, c, a + 1, c + 1, c);
    }
  }
}

/** One leaf card: a strip of three rows bent a little along its length. */
function card(L: number[], LI: number[], c: V3, a: V3, w: V3, len: number, wid: number, bend: number, cell: number, ao: number, sway: number, phase: number, colour: number, canopy: (p: V3) => V3) {
  const base = L.length / L_STRIDE;
  let n = v3.norm(v3.cross(a, w));
  const out = canopy(c);
  if (v3.dot(n, out) < 0) n = v3.mul(n, -1);
  // atlas: 2 x 2 clusters, the twig entering at the bottom (v = 1)
  const [u0, v0] = [(cell % 2) * 0.5 + 0.004, Math.floor(cell / 2) * 0.5 + 0.004];
  for (let row = 0; row < 3; row++) {
    const s = row / 2; // 0 at the twig, 1 at the tip
    const sag = bend * len * (s - 0.5) * (s - 0.5) * 4;
    for (let side = 0; side < 2; side++) {
      const p = v3.add(v3.add(v3.add(c, v3.mul(a, (s - 0.5) * len)), v3.mul(w, (side - 0.5) * wid)), v3.mul(n, -sag));
      const cn = canopy(p);
      L.push(p[0], p[1], p[2], n[0], n[1], n[2], cn[0], cn[1], cn[2], u0 + side * 0.492, v0 + (1 - s) * 0.492, ao, sway * (0.5 + 0.5 * s), phase, colour, c[0], c[1], c[2]);
    }
  }
  LI.push(base, base + 1, base + 2, base + 1, base + 3, base + 2, base + 2, base + 3, base + 4, base + 3, base + 5, base + 4);
}

function finish(V: number[], I: number[], lodCount: number, L: number[], LI: number[], branches: Branch[], cards: TreeMesh['cards'], height: number, r: Rng): TreeMesh {
  // shuffle the cards so that any prefix of them is an even thinning of the crown
  const nCards = LI.length / 12;
  const order = Array.from({ length: nCards }, (_, i) => i);
  for (let i = nCards - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const idx = new Uint32Array(LI.length);
  order.forEach((o, i) => idx.set(LI.slice(o * 12, o * 12 + 12), i * 12));
  let radius = 0;
  for (let i = 0; i < L.length; i += L_STRIDE) radius = Math.max(radius, Math.hypot(L[i], L[i + 2]));
  return { trunk: new Float32Array(V), trunkIdx: new Uint32Array(I), trunkLod: lodCount, leaf: new Float32Array(L), leafIdx: idx, branches, cards: order.map((o) => cards[o]), radius, height };
}

/** Broadleaf: trunk with a root flare -> limbs -> boughs -> twigs, leaf clusters on the twigs. */
function broadleaf(seed: number): TreeMesh {
  const r = rng(seed * 7919 + 13);
  const V: number[] = [], I: number[] = [], L: number[] = [], LI: number[] = [];
  const branches: Branch[] = [];
  const cards: TreeMesh['cards'] = [];
  const lobes = [r() * 6.28, r() * 6.28, 3 + Math.floor(r() * 3)];
  const flare = (y: number, ang: number) => 1 + 1.5 * Math.exp(-Math.max(0, y + 0.35) / 0.42) * (1 + 0.38 * Math.sin(ang * lobes[2] + lobes[0])) + 0.07 * Math.sin(ang * 2 + lobes[1] + y * 2.2);
  const up = (k: number) => (): V3 => [0, k, 0];
  const lean: V3 = [(r() - 0.5) * 0.14, 1, (r() - 0.5) * 0.14];
  const trunk = grow(r, [0, -0.35, 0], lean, 4.6, 0.34, 0.21, 8, 0.07, up(0.1), 0);
  branches.push(trunk);
  tube(V, I, trunk, 10, flare);
  const centre: V3 = [0, H_BROAD * 0.66, 0];
  const canopyR = H_BROAD * 0.45;
  const canopy = (p: V3): V3 => v3.norm(v3.sub(p, centre));
  const limbs: Branch[] = [];
  const nLimb = 4 + Math.floor(r() * 2);
  const a0 = r() * 6.28;
  for (let k = 0; k <= nLimb; k++) {
    const leader = k === nLimb;
    const at = along(trunk, leader ? 1 : 0.72 + 0.26 * (k / nLimb));
    const ang = a0 + k * 2.39996;
    const spread = leader ? 0.12 : 0.55 + r() * 0.35;
    const dir = v3.norm([Math.cos(ang) * spread, 1 - spread * 0.55, Math.sin(ang) * spread]);
    const b = grow(r, at.p, dir, leader ? 3.6 : 3.3 + r() * 1.3, at.r * (leader ? 0.9 : 0.72), 0.045, 6, 0.13, up(0.1), 1);
    limbs.push(b);
    branches.push(b);
    tube(V, I, b, 7, null);
  }
  const boughs: Branch[] = [];
  for (const lb of limbs) {
    const n = 5 + Math.floor(r() * 2);
    for (let k = 0; k < n; k++) {
      const at = along(lb, 0.3 + 0.7 * (k / (n - 1)));
      const side = perp(at.d, r);
      const dir = v3.norm(v3.add(v3.add(v3.mul(at.d, 0.55), v3.mul(side, 0.8)), [0, 0.12, 0]));
      const b = grow(r, at.p, dir, 1.5 + r() * 1.1, Math.max(0.03, at.r * 0.6), 0.016, 4, 0.2, up(0.06), 2);
      boughs.push(b);
      branches.push(b);
    }
  }
  const lodCount = I.length;
  for (const b of boughs) tube(V, I, b, 5, null);
  for (const bg of boughs) {
    const n = 4;
    for (let k = 0; k < n; k++) {
      const at = along(bg, 0.35 + 0.65 * (k / (n - 1)));
      const dir = v3.norm(v3.add(v3.add(v3.mul(at.d, 0.5), v3.mul(perp(at.d, r), 0.85)), [0, 0.1, 0]));
      const tw = grow(r, at.p, dir, 0.65 + r() * 0.5, Math.max(0.012, at.r * 0.55), 0.005, 3, 0.25, up(0.04), 3);
      branches.push(tw);
      tube(V, I, tw, 4, null);
      // leaf clusters along the twig and at its tip: two or three cards crossed
      for (const t of [0.3, 0.65, 1.0]) {
        const q = along(tw, t);
        const len = 1.1 + r() * 0.6, wid = len * (0.8 + r() * 0.25);
        const c = v3.add(q.p, v3.mul(q.d, len * 0.38));
        const d = v3.len(v3.sub(c, centre)) / canopyR;
        const ao = Math.max(0.16, Math.min(1, Math.pow(d, 1.6)));
        const w0 = perp(q.d, r);
        const cell = Math.floor(r() * 4);
        const phase = r(), colour = r();
        const nc = 2;
        for (let m = 0; m < nc; m++) {
          const ang = (m / nc) * Math.PI + (r() - 0.5) * 0.4;
          const w = v3.norm(v3.add(v3.mul(w0, Math.cos(ang)), v3.mul(v3.cross(q.d, w0), Math.sin(ang))));
          card(L, LI, c, q.d, w, len, wid, 0.12 + r() * 0.1, (cell + m) % 4, ao, 0.6 + 0.4 * t, phase, colour, canopy);
          cards.push({ c, a: q.d, w, len, wid, cell: (cell + m) % 4, ao });
        }
      }
    }
  }
  return finish(V, I, lodCount, L, LI, branches, cards, H_BROAD, r);
}

/** Conifer: a slender leader with whorls of boughs that droop and lift at the tip, each with side twigs; needle fans along them. */
function conifer(seed: number): TreeMesh {
  const r = rng(seed * 104729 + 71);
  const V: number[] = [], I: number[] = [], L: number[] = [], LI: number[] = [];
  const branches: Branch[] = [];
  const cards: TreeMesh['cards'] = [];
  const lobes = [r() * 6.28, r() * 6.28];
  const flare = (y: number, ang: number) => 1 + 0.9 * Math.exp(-Math.max(0, y + 0.35) / 0.36) * (1 + 0.3 * Math.sin(ang * 4 + lobes[0])) + 0.04 * Math.sin(ang * 3 + lobes[1] + y);
  const trunk = grow(r, [0, -0.35, 0], [(r() - 0.5) * 0.05, 1, (r() - 0.5) * 0.05], H_CONIFER + 0.3, 0.3, 0.018, 16, 0.02, () => [0, 0.2, 0], 0);
  branches.push(trunk);
  tube(V, I, trunk, 10, flare);
  const reach = (y: number) => 0.35 + 3.1 * Math.pow(Math.max(0, 1 - (y - 1.8) / (H_CONIFER - 1.8)), 0.85);
  const canopy = (p: V3): V3 => v3.norm([p[0], 0.45 * Math.hypot(p[0], p[2]) + 0.15, p[2]]);
  const boughs: Branch[] = [];
  let y = 1.9 + r() * 0.5;
  let turn = r() * 6.28;
  while (y < H_CONIFER - 0.25) {
    const k = (y - 1.8) / (H_CONIFER - 1.8);
    const n = 5 + Math.floor(r() * 2);
    turn += 0.7 + r();
    for (let j = 0; j < n; j++) {
      const ang = turn + (j / n) * Math.PI * 2 + (r() - 0.5) * 0.3;
      const len = reach(y) * (0.8 + r() * 0.3);
      const droop = 0.42 - 0.3 * k + r() * 0.12;
      const dir = v3.norm([Math.cos(ang), -droop, Math.sin(ang)]);
      const start: V3 = [trunk.pts[0][0], y + (r() - 0.5) * 0.12, trunk.pts[0][2]];
      // sags along its length, the tip turning up to the light
      const b = grow(r, start, dir, len, 0.022 + 0.016 * len, 0.006, 5, 0.06, (t) => [0, t > 0.55 ? 0.1 : -0.03, 0], 1);
      boughs.push(b);
      branches.push(b);
      tube(V, I, b, 4, null);
    }
    y += 0.66 - 0.22 * k;
  }
  const lodCount = I.length;
  for (const b of boughs) {
    const len = v3.len(v3.sub(b.pts[b.pts.length - 1], b.pts[0]));
    const top = b.pts[0][1];
    const ao = (c: V3) => Math.max(0.2, Math.min(1, Math.pow(Math.hypot(c[0], c[2]) / Math.max(0.4, reach(top)), 1.3)));
    const phase = r(), colour = r();
    // needle fans along the bough itself
    const nFan = Math.max(1, Math.round(len / 0.75));
    for (let m = 0; m < nFan; m++) {
      const q = along(b, (m + 0.6) / nFan);
      const side = v3.norm(v3.cross(q.d, [0, 1, 0]));
      const fl = len / nFan * 1.45, fw = 0.55 + 0.35 * Math.min(1, len / 2);
      const cell = Math.floor(r() * 4);
      card(L, LI, q.p, q.d, side, fl, fw, 0.1, cell, ao(q.p), 0.35 + 0.65 * (m + 1) / nFan, phase, colour, canopy);
      cards.push({ c: q.p, a: q.d, w: side, len: fl, wid: fw, cell, ao: ao(q.p) });
      if (m % 2 === 0) {
        const upw = v3.norm(v3.cross(side, q.d));
        card(L, LI, q.p, q.d, upw, fl, fw * 0.7, 0.05, (cell + 1) % 4, ao(q.p), 0.35 + 0.65 * (m + 1) / nFan, phase, colour, canopy);
        cards.push({ c: q.p, a: q.d, w: upw, len: fl, wid: fw * 0.7, cell: (cell + 1) % 4, ao: ao(q.p) });
      }
    }
    // side twigs in pairs, each a fan of its own
    if (len < 0.9) continue;
    for (const t of [0.32, 0.55, 0.78]) {
      const q = along(b, t);
      const side = v3.norm(v3.cross(q.d, [0, 1, 0]));
      for (const s of [1, -1]) {
        const dir = v3.norm(v3.add(v3.add(v3.mul(q.d, 0.62), v3.mul(side, 0.78 * s)), [0, -0.12, 0]));
        const tl = len * (0.5 - 0.3 * t) * (0.8 + r() * 0.4);
        const tw = grow(r, q.p, dir, tl, 0.012, 0.004, 2, 0.08, () => [0, 0.03, 0], 2);
        branches.push(tw);
        tube(V, I, tw, 3, null);
        const mid = along(tw, 0.55);
        const w = v3.norm(v3.cross(mid.d, [0, 1, 0]));
        const cell = Math.floor(r() * 4);
        card(L, LI, mid.p, mid.d, w, tl * 1.25, 0.34 + 0.3 * tl, 0.12, cell, ao(mid.p), 0.5 + 0.5 * t, r(), colour, canopy);
        cards.push({ c: mid.p, a: mid.d, w, len: tl * 1.25, wid: 0.34 + 0.3 * tl, cell, ao: ao(mid.p) });
      }
    }
  }
  return finish(V, I, lodCount, L, LI, branches, cards, H_CONIFER, r);
}

// ------------------------------------------------------------------ textures
type Ctx = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
function canvas(w: number, h: number): { c: OffscreenCanvas | HTMLCanvasElement; x: Ctx } {
  const c = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : Object.assign(document.createElement('canvas'), { width: w, height: h });
  return { c, x: c.getContext('2d', { willReadFrequently: true }) as Ctx };
}
type Px = Uint8ClampedArray;

/** Mip chain by box filter. `kind`: colour is weighted by alpha so cut-out edges do not darken; normals are renormalised. */
function mipChain(base: Px, size: number, kind: 'colour' | 'normal' | 'data'): Px[] {
  const out = [base];
  let src = base;
  for (let s = size >> 1; s >= 1; s >>= 1) {
    const dst = new Uint8ClampedArray(s * s * 4);
    const w = s * 2;
    for (let y = 0; y < s; y++) {
      for (let x = 0; x < s; x++) {
        const o = (y * s + x) * 4;
        const i = [(y * 2 * w + x * 2) * 4, (y * 2 * w + x * 2 + 1) * 4, ((y * 2 + 1) * w + x * 2) * 4, ((y * 2 + 1) * w + x * 2 + 1) * 4];
        if (kind === 'colour') {
          let a = 0, r = 0, g = 0, b = 0;
          for (const k of i) {
            const wa = src[k + 3] + 1;
            a += src[k + 3];
            r += src[k] * wa;
            g += src[k + 1] * wa;
            b += src[k + 2] * wa;
          }
          const ws = a + 4;
          dst[o] = r / ws;
          dst[o + 1] = g / ws;
          dst[o + 2] = b / ws;
          dst[o + 3] = a / 4;
        } else {
          for (let c = 0; c < 4; c++) dst[o + c] = (src[i[0] + c] + src[i[1] + c] + src[i[2] + c] + src[i[3] + c]) / 4;
          if (kind === 'normal') {
            const nx = dst[o] / 127.5 - 1, ny = dst[o + 1] / 127.5 - 1, nz = dst[o + 2] / 127.5 - 1;
            const l = Math.hypot(nx, ny, nz) || 1;
            dst[o] = (nx / l + 1) * 127.5;
            dst[o + 1] = (ny / l + 1) * 127.5;
            dst[o + 2] = (nz / l + 1) * 127.5;
          }
        }
      }
    }
    out.push(dst);
    src = dst;
  }
  return out;
}

/**
 * Box-filtered alpha thins out with every mip, so distant leaves would melt
 * away. Rescale the alpha of each level until the share of texels that pass
 * the test (alpha >= 0.5) is the same as in level 0.
 */
function preserveCoverage(mips: Px[]) {
  const coverage = (p: Px, scale: number) => {
    let n = 0;
    for (let i = 3; i < p.length; i += 4) if (p[i] * scale >= 127.5) n++;
    return n / (p.length / 4);
  };
  const want = coverage(mips[0], 1);
  for (let m = 1; m < mips.length; m++) {
    const p = mips[m];
    let lo = 0.25, hi = 8;
    for (let it = 0; it < 14; it++) {
      const mid = (lo + hi) / 2;
      if (coverage(p, mid) < want) lo = mid;
      else hi = mid;
    }
    const s = (lo + hi) / 2;
    for (let i = 3; i < p.length; i += 4) p[i] = Math.min(255, p[i] * s);
  }
}

/** Tangent-space normal map from a height field (wrapping or clamped at the edges). */
function normalsFrom(h: Float32Array, size: number, strength: number, wrap: boolean): Px {
  const out = new Uint8ClampedArray(size * size * 4);
  const at = (x: number, y: number) => {
    if (wrap) return h[((y + size) % size) * size + ((x + size) % size)];
    return h[Math.max(0, Math.min(size - 1, y)) * size + Math.max(0, Math.min(size - 1, x))];
  };
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (at(x + 1, y) - at(x - 1, y)) * strength, dy = (at(x, y + 1) - at(x, y - 1)) * strength;
      const l = Math.hypot(dx, dy, 1);
      const o = (y * size + x) * 4;
      out[o] = (-dx / l + 1) * 127.5;
      out[o + 1] = (-dy / l + 1) * 127.5;
      out[o + 2] = (1 / l + 1) * 127.5;
      out[o + 3] = 255;
    }
  }
  return out;
}

/** Tiling value noise (period in lattice cells). */
function tnoise(x: number, y: number, px: number, py: number, seed: number): number {
  const xi = Math.floor(x), yi = Math.floor(y);
  const fx = x - xi, fy = y - yi;
  const h = (i: number, j: number) => {
    let n = (((i % px) + px) % px) * 374761393 + (((j % py) + py) % py) * 668265263 + seed * 2147483647;
    n = (n ^ (n >>> 13)) * 1274126177;
    return ((n ^ (n >>> 16)) >>> 0) / 4294967296;
  };
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  return (h(xi, yi) * (1 - sx) + h(xi + 1, yi) * sx) * (1 - sy) + (h(xi, yi + 1) * (1 - sx) + h(xi + 1, yi + 1) * sx) * sy;
}

/** Bark: deep vertical fissures between plates, tiling both ways. */
function barkTextures(): { albedo: Px; normal: Px; rta: Px } {
  const S = TEX;
  const h = new Float32Array(S * S);
  const albedo = new Uint8ClampedArray(S * S * 4), rta = new Uint8ClampedArray(S * S * 4);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const u = x / S, v = y / S;
      const warp = tnoise(u * 6, v * 3, 6, 3, 1) - 0.5 + 0.5 * (tnoise(u * 14, v * 6, 14, 6, 2) - 0.5);
      // ridged noise stretched along the trunk: the fissures
      let f = 0, amp = 0.6, fu = 12, fv = 2;
      for (let o = 0; o < 4; o++) {
        const n = tnoise((u + warp * 0.035) * fu, v * fv, fu, fv, 10 + o);
        f += (1 - Math.abs(n * 2 - 1)) * amp;
        amp *= 0.5;
        fu *= 2;
        fv *= 2;
      }
      const plate = Math.pow(Math.min(1, f / 1.05), 2.2);
      const grain = tnoise(u * 90, v * 22, 90, 22, 31) * 0.5 + tnoise(u * 220, v * 60, 220, 60, 37) * 0.5;
      const hh = plate * 0.85 + grain * 0.15;
      h[y * S + x] = hh;
      const tone = 0.32 + 0.75 * hh + (tnoise(u * 5, v * 5, 5, 5, 44) - 0.5) * 0.25;
      const o = (y * S + x) * 4;
      // grey-brown plates, dark damp cracks, a little lichen
      const lichen = Math.max(0, tnoise(u * 9, v * 7, 9, 7, 51) - 0.68) * 2.4 * plate;
      albedo[o] = Math.min(255, (158 * tone) * (1 - lichen) + 168 * lichen);
      albedo[o + 1] = Math.min(255, (132 * tone) * (1 - lichen) + 178 * lichen);
      albedo[o + 2] = Math.min(255, (108 * tone) * (1 - lichen) + 132 * lichen);
      albedo[o + 3] = 255;
      rta[o] = 255 * (0.95 - 0.1 * plate * grain); // roughness 0.85 .. 0.95
      rta[o + 1] = 0;
      rta[o + 2] = 255 * (0.35 + 0.65 * Math.pow(hh, 0.7)); // ao: dark in the cracks
      rta[o + 3] = 255;
    }
  }
  return { albedo, normal: normalsFrom(h, S, 9, true), rta };
}

interface Atlas {
  albedo: Px;
  normal: Px;
  rta: Px;
  /** The colour canvas, for painting the far impostors. */
  source: OffscreenCanvas | HTMLCanvasElement;
}

/**
 * A leaf-cluster atlas: four clusters (2 x 2), each a twig with dozens of
 * overlapping leaves or hundreds of needles. Everything is drawn three times
 * from one recipe: colour (+ alpha), height (for the normal map) and
 * roughness / translucency / ao.
 */
function leafAtlas(conifer: boolean): Atlas {
  const S = TEX, C = S / 2;
  const col = canvas(S, S), hgt = canvas(S, S), mat = canvas(S, S);
  hgt.x.fillStyle = '#000';
  hgt.x.fillRect(0, 0, S, S);
  const each = (f: (x: Ctx, mode: number) => void) => [col.x, hgt.x, mat.x].forEach((x, m) => f(x, m));
  const grey = (v: number) => `rgb(${v * 255},${v * 255},${v * 255})`;
  for (let cell = 0; cell < 4; cell++) {
    const r = rng(cell * 97 + (conifer ? 5 : 1));
    const ox = (cell % 2) * C, oy = Math.floor(cell / 2) * C;
    each((x) => {
      x.save();
      x.beginPath();
      x.rect(ox + 2, oy + 2, C - 4, C - 4);
      x.clip();
      x.translate(ox, oy);
      x.lineCap = 'round';
      x.lineJoin = 'round';
    });
    // the twig: a main stem from the bottom edge and side shoots
    const stems: { x0: number; y0: number; x1: number; y1: number; w: number }[] = [];
    const top = C * (0.1 + r() * 0.06);
    const sway = (r() - 0.5) * C * 0.16;
    const stemX = (t: number) => C / 2 + sway * Math.sin(t * 2.6);
    const stemY = (t: number) => C - 2 - t * (C - 2 - top);
    const N = 14;
    for (let i = 0; i < N; i++) stems.push({ x0: stemX(i / N), y0: stemY(i / N), x1: stemX((i + 1) / N), y1: stemY((i + 1) / N), w: (conifer ? 5 : 7) * (1 - 0.75 * i / N) });
    const shoots: { x: number; y: number; ang: number; len: number }[] = [];
    const nShoot = conifer ? 9 : 7;
    for (let i = 0; i < nShoot; i++) {
      const t = 0.14 + 0.8 * (i / nShoot) + r() * 0.04;
      const s = i % 2 ? 1 : -1;
      const ang = -Math.PI / 2 + s * (conifer ? 0.75 + r() * 0.3 : 0.7 + r() * 0.5);
      const len = C * (conifer ? 0.34 : 0.3) * (1 - 0.55 * t) * (0.8 + r() * 0.4);
      shoots.push({ x: stemX(t), y: stemY(t), ang, len });
      stems.push({ x0: stemX(t), y0: stemY(t), x1: stemX(t) + Math.cos(ang) * len, y1: stemY(t) + Math.sin(ang) * len, w: conifer ? 2.5 : 3 });
    }
    each((x, m) => {
      x.strokeStyle = m === 0 ? (conifer ? '#4a3524' : '#4f3d28') : m === 1 ? grey(0.7) : 'rgb(235,0,190)';
      for (const s of stems) {
        x.lineWidth = s.w;
        x.beginPath();
        x.moveTo(s.x0, s.y0);
        x.lineTo(s.x1, s.y1);
        x.stroke();
      }
    });
    if (conifer) {
      // needles: hundreds of fine strokes swept forward along every shoot
      const needles = (px: number, py: number, ang: number, len: number, taper: number) => {
        const steps = Math.floor(len / 2.6);
        for (let i = 0; i < steps; i++) {
          const t = i / steps;
          const bx = px + Math.cos(ang) * len * t, by = py + Math.sin(ang) * len * t;
          for (const s of [1, -1]) {
            const na = ang + s * (0.75 + (r() - 0.5) * 0.5);
            const nl = (20 + r() * 12) * (1 - taper * t) * (C / 512);
            const g = 62 + r() * 46, shade = 0.7 + r() * 0.5;
            each((x, m) => {
              x.lineWidth = m === 1 ? 2.6 : 2.1;
              x.strokeStyle = m === 0 ? `rgb(${(20 + r() * 0 + g * 0.34) * shade},${g * shade},${(30 + g * 0.3) * shade})` : m === 1 ? grey(0.45 + 0.5 * (1 - t * 0.3)) : 'rgb(165,150,255)';
              x.beginPath();
              x.moveTo(bx, by);
              x.lineTo(bx + Math.cos(na) * nl, by + Math.sin(na) * nl);
              x.stroke();
            });
          }
        }
      };
      for (const s of shoots) needles(s.x, s.y, s.ang, s.len * 1.08, 0.55);
      needles(stemX(0.05), stemY(0.05), -Math.PI / 2 + sway * 0.002, (C - top) * 0.97, 0.35);
    } else {
      // leaves: ovate, serrated, veined; drawn from the back of the cluster forward
      const leaves: { x: number; y: number; ang: number; len: number; hue: number; lit: number }[] = [];
      for (const s of shoots) {
        const n = 4 + Math.floor(r() * 3);
        for (let i = 0; i < n; i++) {
          const t = 0.25 + 0.75 * (i / (n - 1));
          leaves.push({ x: s.x + Math.cos(s.ang) * s.len * t, y: s.y + Math.sin(s.ang) * s.len * t, ang: s.ang + (i % 2 ? 0.75 : -0.75) + (r() - 0.5) * 0.5, len: C * (0.15 + r() * 0.07), hue: r(), lit: 0.72 + r() * 0.4 });
        }
        leaves.push({ x: s.x + Math.cos(s.ang) * s.len, y: s.y + Math.sin(s.ang) * s.len, ang: s.ang + (r() - 0.5) * 0.3, len: C * (0.17 + r() * 0.06), hue: r(), lit: 0.9 + r() * 0.3 });
      }
      for (let i = 0; i < 5; i++) leaves.push({ x: stemX(0.86 + i * 0.03), y: stemY(0.86 + i * 0.03), ang: -Math.PI / 2 + (i - 2) * 0.55, len: C * (0.16 + r() * 0.06), hue: r(), lit: 0.95 + r() * 0.25 });
      leaves.forEach((lf, li) => {
        const depth = li / leaves.length;
        // outline: half width follows a rounded profile with a saw-toothed margin
        const pts: [number, number][] = [];
        const T = 22;
        const prof = (t: number) => Math.pow(Math.sin(Math.PI * Math.pow(t, 0.72)), 0.9) * 0.34 * (1 + 0.07 * ((t * 11) % 1 < 0.5 ? 1 : -0.4));
        for (let i = 0; i <= T; i++) pts.push([lf.len * (i / T), lf.len * prof(i / T)]);
        for (let i = T; i >= 0; i--) pts.push([lf.len * (i / T), -lf.len * prof(i / T)]);
        each((x, m) => {
          x.save();
          x.translate(lf.x, lf.y);
          x.rotate(lf.ang);
          x.beginPath();
          pts.forEach(([px, py], i) => (i ? x.lineTo(px, py) : x.moveTo(px, py)));
          x.closePath();
          if (m === 0) {
            const g = x.createLinearGradient(0, 0, lf.len, 0);
            const k = lf.lit * (0.62 + 0.38 * depth);
            const c0 = [36 + 34 * lf.hue, 92 + 26 * lf.hue, 22 + 8 * lf.hue], c1 = [70 + 60 * lf.hue, 128 + 22 * lf.hue, 30];
            g.addColorStop(0, `rgb(${c0[0] * k},${c0[1] * k},${c0[2] * k})`);
            g.addColorStop(1, `rgb(${c1[0] * k},${c1[1] * k},${c1[2] * k})`);
            x.fillStyle = g;
            x.fill();
            x.strokeStyle = `rgba(20,45,12,0.55)`;
            x.lineWidth = 1;
            x.stroke();
            x.strokeStyle = `rgba(${150 * k},${178 * k},${82 * k},0.85)`;
          } else if (m === 1) {
            // a shallow dish: high along the edges, a groove at every vein
            const g = x.createLinearGradient(0, -lf.len * 0.34, 0, lf.len * 0.34);
            g.addColorStop(0, grey(0.72));
            g.addColorStop(0.5, grey(0.5 + 0.2 * depth));
            g.addColorStop(1, grey(0.72));
            x.fillStyle = g;
            x.fill();
            x.strokeStyle = grey(0.3);
          } else {
            // roughness 0.55 .. 0.8, translucent, shaded where it lies under others
            x.fillStyle = `rgb(${255 * (0.58 + 0.2 * lf.hue)},235,${255 * (0.55 + 0.45 * depth)})`;
            x.fill();
            x.strokeStyle = `rgb(${255 * 0.8},150,${255 * (0.55 + 0.45 * depth)})`;
          }
          // midrib and side veins
          x.lineWidth = 1.6;
          x.beginPath();
          x.moveTo(0, 0);
          x.lineTo(lf.len * 0.97, 0);
          x.stroke();
          x.lineWidth = 0.8;
          for (let v = 1; v <= 6; v++) {
            const t = v / 7.5;
            for (const s of [1, -1]) {
              x.beginPath();
              x.moveTo(lf.len * t, 0);
              x.quadraticCurveTo(lf.len * (t + 0.08), s * lf.len * prof(t) * 0.6, lf.len * (t + 0.17), s * lf.len * prof(Math.min(1, t + 0.17)) * 0.95);
              x.stroke();
            }
          }
          x.restore();
        });
      });
    }
    each((x) => x.restore());
  }
  const albedo = col.x.getImageData(0, 0, S, S).data;
  const hd = hgt.x.getImageData(0, 0, S, S).data;
  const md = mat.x.getImageData(0, 0, S, S).data;
  const h = new Float32Array(S * S);
  const rta = new Uint8ClampedArray(S * S * 4);
  // texels outside the leaves take a leaf colour, so filtering never bleeds black into the edges
  const fill = conifer ? [34, 70, 40] : [52, 104, 28];
  for (let i = 0; i < S * S; i++) {
    const o = i * 4;
    h[i] = (hd[o] / 255) * (albedo[o + 3] / 255);
    if (albedo[o + 3] < 8) {
      albedo[o] = fill[0];
      albedo[o + 1] = fill[1];
      albedo[o + 2] = fill[2];
    }
    const cov = md[o + 3] / 255;
    rta[o] = md[o] * cov + 178 * (1 - cov);
    rta[o + 1] = md[o + 1] * cov + 200 * (1 - cov);
    rta[o + 2] = md[o + 2] * cov + 255 * (1 - cov);
    rta[o + 3] = 255;
  }
  return { albedo, normal: normalsFrom(h, S, 5, false), rta, source: col.c };
}

/**
 * The far level of detail: the tree painted once from the side (its real
 * branches and leaf cards, projected and layered back to front) onto a card.
 */
function impostor(mesh: TreeMesh, atlas: Atlas): Px {
  const W = 256, H = 512;
  const { x } = canvas(W, H);
  const R = mesh.radius * 1.05, top = mesh.height * 1.08;
  const px = (p: V3): [number, number] => [(p[0] / R * 0.5 + 0.5) * W, (1 - (p[1] + 0.4) / (top + 0.4)) * H];
  const sc = W / (2 * R);
  x.lineCap = 'round';
  for (const b of mesh.branches) {
    if (b.level > 1) continue;
    for (let i = 0; i + 1 < b.pts.length; i++) {
      x.strokeStyle = '#4a3a2c';
      x.lineWidth = Math.max(1, b.rad[i] * 2 * sc);
      x.beginPath();
      x.moveTo(...px(b.pts[i]));
      x.lineTo(...px(b.pts[i + 1]));
      x.stroke();
    }
  }
  const C = TEX / 2;
  const cards = [...mesh.cards].sort((a, b) => a.c[2] - b.c[2]);
  for (const k of cards) {
    const tipL = px(v3.add(v3.add(k.c, v3.mul(k.a, k.len / 2)), v3.mul(k.w, -k.wid / 2)));
    const tipR = px(v3.add(v3.add(k.c, v3.mul(k.a, k.len / 2)), v3.mul(k.w, k.wid / 2)));
    const baseL = px(v3.add(v3.add(k.c, v3.mul(k.a, -k.len / 2)), v3.mul(k.w, -k.wid / 2)));
    x.setTransform((tipR[0] - tipL[0]) / C, (tipR[1] - tipL[1]) / C, (baseL[0] - tipL[0]) / C, (baseL[1] - tipL[1]) / C, tipL[0], tipL[1]);
    // deep in the crown the cards are dark
    x.globalAlpha = 1;
    x.drawImage(atlas.source, (k.cell % 2) * C, Math.floor(k.cell / 2) * C, C, C, 0, 0, C, C);
  }
  x.setTransform(1, 0, 0, 1, 0, 0);
  const d = x.getImageData(0, 0, W, H).data;
  // pad to a square power of two for the mip chain (the card uses the left half)
  const out = new Uint8ClampedArray(H * H * 4);
  const fill = [44, 86, 34];
  for (let y = 0; y < H; y++) {
    for (let xx = 0; xx < H; xx++) {
      const o = (y * H + xx) * 4;
      if (xx < W && d[(y * W + xx) * 4 + 3] >= 8) {
        const s = (y * W + xx) * 4;
        // shade towards the middle and the underside, as the canopy ao does up close
        const shade = 0.62 + 0.38 * Math.min(1, Math.abs(xx / W - 0.5) * 2.2 + (1 - y / H) * 0.5);
        out[o] = d[s] * shade;
        out[o + 1] = d[s + 1] * shade;
        out[o + 2] = d[s + 2] * shade;
        out[o + 3] = d[s + 3];
      } else {
        out[o] = fill[0];
        out[o + 1] = fill[1];
        out[o + 2] = fill[2];
      }
    }
  }
  return out;
}

/** Two crossed cards showing the impostor, in the leaf vertex layout. */
function billboard(mesh: TreeMesh): { v: Float32Array; i: Uint32Array } {
  const R = mesh.radius * 1.05, top = mesh.height * 1.08;
  const V: number[] = [], I: number[] = [];
  const centre: V3 = [0, mesh.height * 0.62, 0];
  for (const [ax, az] of [[1, 0], [0, 1], [0.707, 0.707], [0.707, -0.707]]) {
    const base = V.length / L_STRIDE;
    for (const [s, t] of [[-1, 0], [1, 0], [-1, 1], [1, 1]]) {
      const p: V3 = [ax * s * R, -0.4 + t * (top + 0.4), az * s * R];
      const cn = v3.norm(v3.sub(p, centre));
      V.push(p[0], p[1], p[2], -az, 0, ax, cn[0], cn[1] * 0.6 + 0.4, cn[2], (s * 0.5 + 0.5) * 0.5, 1 - t, 0.5, 0, 0, 0.5, p[0], p[1], p[2]);
    }
    I.push(base, base + 1, base + 2, base + 1, base + 3, base + 2);
  }
  return { v: new Float32Array(V), i: new Uint32Array(I) };
}

// ------------------------------------------------------------------ renderer
interface Variant {
  conifer: boolean;
  trunkVB: GPUBuffer;
  trunkIB: GPUBuffer;
  trunkCount: number;
  trunkLod: number;
  leafVB: GPUBuffer;
  leafIB: GPUBuffer;
  leafCount: number;
  farVB: GPUBuffer;
  farIB: GPUBuffer;
  farCount: number;
  farBG: GPUBindGroup;
  radius: number;
  height: number;
}
interface Group {
  first: number;
  count: number;
}

export interface TreeDeps {
  bglMain: GPUBindGroupLayout;
  bglShadow: GPUBindGroupLayout;
  format: GPUTextureFormat;
}

export class TreeRenderer {
  private d: GPUDevice;
  private bgl: GPUBindGroupLayout;
  private pipes: Record<'trunk' | 'leaf' | 'trunkShadow' | 'leafShadow', GPURenderPipeline>;
  private sampler: GPUSampler;
  private variants: Variant[] = [];
  private barkBG: GPUBindGroup | null = null;
  private leafBG: GPUBindGroup[] = [];
  private inst: GPUBuffer | null = null;
  private data = new Float32Array(0);
  private trees: Float32Array = new Float32Array(0);
  /** Per variant and level of detail: a run of instances in the storage buffer. */
  private groups: Group[][] = [];
  private textures: { bark: [GPUTexture, GPUTexture, GPUTexture]; leaf: [GPUTexture, GPUTexture, GPUTexture][]; flatN: GPUTexture; farRTA: GPUTexture; far: GPUTexture[] } | null = null;
  ready = false;
  tris = 0;

  constructor(device: GPUDevice, deps: TreeDeps) {
    this.d = device;
    const d = device;
    this.bgl = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    });
    this.sampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', maxAnisotropy: 8 });
    const main = d.createShaderModule({ code: TREE_WGSL });
    const shadow = d.createShaderModule({ code: TREE_SHADOW_WGSL });
    const lMain = d.createPipelineLayout({ bindGroupLayouts: [deps.bglMain, this.bgl] });
    const lShadow = d.createPipelineLayout({ bindGroupLayouts: [deps.bglShadow, this.bgl] });
    const trunkVB: GPUVertexBufferLayout[] = [{
      arrayStride: T_STRIDE * 4,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
        { shaderLocation: 2, offset: 24, format: 'float32x2' },
      ],
    }];
    const leafVB: GPUVertexBufferLayout[] = [{
      arrayStride: L_STRIDE * 4,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
        { shaderLocation: 2, offset: 24, format: 'float32x3' },
        { shaderLocation: 3, offset: 36, format: 'float32x2' },
        { shaderLocation: 4, offset: 44, format: 'float32x4' },
        { shaderLocation: 5, offset: 60, format: 'float32x3' },
      ],
    }];
    const depth: GPUDepthStencilState = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' };
    const shadowDepth: GPUDepthStencilState = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2.0 };
    const msaa = MSAA_SAMPLES > 1;
    this.pipes = {
      trunk: d.createRenderPipeline({
        layout: lMain,
        vertex: { module: main, entryPoint: 'vs_trunk', buffers: trunkVB },
        fragment: { module: main, entryPoint: 'fs_trunk', targets: [{ format: deps.format }] },
        primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
        depthStencil: depth,
        multisample: { count: MSAA_SAMPLES },
      }),
      leaf: d.createRenderPipeline({
        layout: lMain,
        vertex: { module: main, entryPoint: 'vs_leaf', buffers: leafVB },
        fragment: { module: main, entryPoint: msaa ? 'fs_leaf_a2c' : 'fs_leaf', targets: [{ format: deps.format }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: depth,
        multisample: { count: MSAA_SAMPLES, alphaToCoverageEnabled: msaa },
      }),
      trunkShadow: d.createRenderPipeline({
        layout: lShadow,
        vertex: { module: shadow, entryPoint: 'vs_trunk', buffers: trunkVB },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: shadowDepth,
      }),
      leafShadow: d.createRenderPipeline({
        layout: lShadow,
        vertex: { module: shadow, entryPoint: 'vs_leaf', buffers: leafVB },
        fragment: { module: shadow, entryPoint: 'fs_leaf', targets: [] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: shadowDepth,
      }),
    };
  }

  private texture(levels: Px[], size: number, format: GPUTextureFormat): GPUTexture {
    const t = this.d.createTexture({ size: [size, size], format, mipLevelCount: levels.length, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    levels.forEach((p, m) => {
      const s = Math.max(1, size >> m);
      this.d.queue.writeTexture({ texture: t, mipLevel: m }, p.buffer as ArrayBuffer, { offset: p.byteOffset, bytesPerRow: s * 4, rowsPerImage: s }, [s, s]);
    });
    return t;
  }

  private bind(a: GPUTexture, n: GPUTexture, r: GPUTexture): GPUBindGroup {
    return this.d.createBindGroup({
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: this.inst! } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: a.createView() },
        { binding: 3, resource: n.createView() },
        { binding: 4, resource: r.createView() },
      ],
    });
  }

  /** Grow the meshes and paint the textures (once), a step at a time so the page keeps breathing. */
  private async build() {
    const pause = () => new Promise<void>((res) => setTimeout(res, 0));
    const bark = barkTextures();
    await pause();
    const atlases = [leafAtlas(true), leafAtlas(false)];
    await pause();
    const leafTex = atlases.map((a) => {
      const col = mipChain(a.albedo, TEX, 'colour');
      preserveCoverage(col);
      return [this.texture(col, TEX, 'rgba8unorm-srgb'), this.texture(mipChain(a.normal, TEX, 'normal'), TEX, 'rgba8unorm'), this.texture(mipChain(a.rta, TEX, 'data'), TEX, 'rgba8unorm')] as [GPUTexture, GPUTexture, GPUTexture];
    });
    await pause();
    const barkTex: [GPUTexture, GPUTexture, GPUTexture] = [
      this.texture(mipChain(bark.albedo, TEX, 'colour'), TEX, 'rgba8unorm-srgb'),
      this.texture(mipChain(bark.normal, TEX, 'normal'), TEX, 'rgba8unorm'),
      this.texture(mipChain(bark.rta, TEX, 'data'), TEX, 'rgba8unorm'),
    ];
    const flatN = this.texture([new Uint8ClampedArray([128, 128, 255, 255])], 1, 'rgba8unorm');
    const farRTA = this.texture([new Uint8ClampedArray([178, 90, 235, 255])], 1, 'rgba8unorm');
    const far: GPUTexture[] = [];
    const buf = (data: Float32Array | Uint32Array, usage: number) => {
      const b = this.d.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
      this.d.queue.writeBuffer(b, 0, data);
      return b;
    };
    const variants: Omit<Variant, 'farBG'>[] = [];
    for (let k = 0; k < VARIANTS * 2; k++) {
      const isConifer = k < VARIANTS;
      const m = isConifer ? conifer(k + 1) : broadleaf(k + 1);
      const imp = mipChain(impostor(m, atlases[isConifer ? 0 : 1]), 512, 'colour');
      preserveCoverage(imp);
      far.push(this.texture(imp, 512, 'rgba8unorm-srgb'));
      const bb = billboard(m);
      variants.push({
        conifer: isConifer,
        trunkVB: buf(m.trunk, GPUBufferUsage.VERTEX), trunkIB: buf(m.trunkIdx, GPUBufferUsage.INDEX), trunkCount: m.trunkIdx.length, trunkLod: m.trunkLod,
        leafVB: buf(m.leaf, GPUBufferUsage.VERTEX), leafIB: buf(m.leafIdx, GPUBufferUsage.INDEX), leafCount: m.leafIdx.length,
        farVB: buf(bb.v, GPUBufferUsage.VERTEX), farIB: buf(bb.i, GPUBufferUsage.INDEX), farCount: bb.i.length,
        radius: m.radius, height: m.height,
      });
      await pause();
    }
    this.textures = { bark: barkTex, leaf: leafTex, flatN, farRTA, far };
    this.pending = variants;
    console.info('trees:', variants.map((v) => `${v.conifer ? 'conifer' : 'broadleaf'} ${v.trunkCount / 3 | 0}+${v.leafCount / 3 | 0} tris, ${v.leafCount / 12 | 0} cards`).join('; '));
    this.rebind();
    this.ready = true;
  }
  private pending: Omit<Variant, 'farBG'>[] = [];
  private building = false;

  /** The bind groups hold the instance buffer, so they are made again whenever it is replaced. */
  private rebind() {
    const t = this.textures;
    if (!t || !this.inst) return;
    this.barkBG = this.bind(...t.bark);
    this.leafBG = t.leaf.map((l) => this.bind(...l));
    this.variants = this.pending.map((v, k) => ({ ...v, farBG: this.bind(t.far[k], t.flatN, t.farRTA) }));
  }

  /** The trees of the world: x, y, z, height, radius, kind (0 conifer), seed, - per tree. */
  setTrees(list: Float32Array) {
    this.trees = list;
    const n = Math.max(1, list.length >> 3);
    this.inst?.destroy();
    this.inst = this.d.createBuffer({ size: n * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.data = new Float32Array(n * 8);
    this.rebind();
    if (!this.building) {
      this.building = true;
      this.build().catch((e) => console.warn('trees:', e));
    }
  }

  /**
   * Sort the trees into runs by variant and level of detail and write them to
   * the storage buffer. `lod`: distances (m) up to which the full tree and the
   * thinned tree are drawn; beyond, the crossed impostor cards up to `far`.
   */
  prepare(cam: ArrayLike<number>, planes: Float32Array, lod: [number, number], far: number) {
    if (!this.ready || !this.inst) return;
    const t = this.trees;
    const nV = this.variants.length;
    const lists: number[][] = Array.from({ length: nV * 3 }, () => []);
    for (let i = 0; i + 8 <= t.length; i += 8) {
      const dx = t[i] - cam[0], dz = t[i + 2] - cam[2];
      const d = Math.hypot(dx, dz);
      if (d > far) continue;
      const seed = t[i + 6];
      const v = (t[i + 5] < 0.5 ? 0 : VARIANTS) + (seed % VARIANTS);
      const level = d < lod[0] ? 0 : d < lod[1] ? 1 : 2;
      if (level === 2) {
        const h = t[i + 3], r = t[i + 4] * 2;
        if (!aabbVisible(planes, [t[i] - r, t[i + 1] - 1, t[i + 2] - r], [t[i] + r, t[i + 1] + h * 1.15, t[i + 2] + r])) continue;
      }
      lists[v * 3 + level].push(i);
    }
    let n = 0;
    const out = this.data;
    this.groups = [];
    this.tris = 0;
    for (let v = 0; v < nV; v++) {
      const vr = this.variants[v];
      const g: Group[] = [];
      for (let level = 0; level < 3; level++) {
        const first = n;
        for (const i of lists[v * 3 + level]) {
          const o = n * 8;
          const seed = t[i + 6];
          out[o] = t[i];
          out[o + 1] = t[i + 1];
          out[o + 2] = t[i + 2];
          out[o + 3] = (seed * 0.61803) % (Math.PI * 2);
          out[o + 4] = t[i + 3] / vr.height;
          out[o + 5] = ((seed * 0.137) % 1 + 1) % 1;
          out[o + 6] = level === 1 ? 1.7 : 1;
          out[o + 7] = vr.conifer ? 0 : 1;
          n++;
        }
        g.push({ first, count: n - first });
      }
      this.groups.push(g);
      this.tris += (g[0].count * (vr.trunkCount + vr.leafCount) + g[1].count * (vr.trunkLod + vr.leafCount * LOD1_CARDS) + g[2].count * vr.farCount) / 3;
    }
    if (n) this.d.queue.writeBuffer(this.inst, 0, out.buffer, 0, n * 32);
  }

  /** Scene pass: trunks (opaque), then the leaves (alpha cut-out, both sides). */
  draw(pass: GPURenderPassEncoder) {
    if (!this.ready || !this.barkBG) return;
    pass.setPipeline(this.pipes.trunk);
    pass.setBindGroup(1, this.barkBG);
    this.variants.forEach((v, k) => {
      const g = this.groups[k];
      if (!g || (!g[0].count && !g[1].count)) return;
      pass.setVertexBuffer(0, v.trunkVB);
      pass.setIndexBuffer(v.trunkIB, 'uint32');
      if (g[0].count) pass.drawIndexed(v.trunkCount, g[0].count, 0, 0, g[0].first);
      if (g[1].count) pass.drawIndexed(v.trunkLod, g[1].count, 0, 0, g[1].first);
    });
    pass.setPipeline(this.pipes.leaf);
    this.variants.forEach((v, k) => {
      const g = this.groups[k];
      if (!g) return;
      if (g[0].count || g[1].count) {
        pass.setBindGroup(1, this.leafBG[v.conifer ? 0 : 1]);
        pass.setVertexBuffer(0, v.leafVB);
        pass.setIndexBuffer(v.leafIB, 'uint32');
        if (g[0].count) pass.drawIndexed(v.leafCount, g[0].count, 0, 0, g[0].first);
        if (g[1].count) pass.drawIndexed(lod1(v.leafCount), g[1].count, 0, 0, g[1].first);
      }
      if (g[2].count) {
        pass.setBindGroup(1, v.farBG);
        pass.setVertexBuffer(0, v.farVB);
        pass.setIndexBuffer(v.farIB, 'uint32');
        pass.drawIndexed(v.farCount, g[2].count, 0, 0, g[2].first);
      }
    });
  }

  /** A shadow cascade (0 near, 1 far): depth only, the leaves through their alpha. */
  drawShadow(pass: GPURenderPassEncoder, cascade: number) {
    if (!this.ready || !this.barkBG) return;
    const levels = cascade === 0 ? [0] : [0, 1];
    pass.setPipeline(this.pipes.trunkShadow);
    pass.setBindGroup(1, this.barkBG);
    this.variants.forEach((v, k) => {
      const g = this.groups[k];
      if (!g) return;
      pass.setVertexBuffer(0, v.trunkVB);
      pass.setIndexBuffer(v.trunkIB, 'uint32');
      for (const l of levels) if (g[l].count) pass.drawIndexed(v.trunkLod, g[l].count, 0, 0, g[l].first);
    });
    pass.setPipeline(this.pipes.leafShadow);
    this.variants.forEach((v, k) => {
      const g = this.groups[k];
      if (!g) return;
      pass.setBindGroup(1, this.leafBG[v.conifer ? 0 : 1]);
      pass.setVertexBuffer(0, v.leafVB);
      pass.setIndexBuffer(v.leafIB, 'uint32');
      for (const l of levels) {
        if (!g[l].count) continue;
        // the far cascade and the thinned trees make do with fewer cards
        pass.drawIndexed(cascade === 0 && l === 0 ? v.leafCount : lod1(v.leafCount), g[l].count, 0, 0, g[l].first);
      }
    });
  }
}

/** Share of the leaf cards kept at the middle level of detail (they are drawn larger). */
const LOD1_CARDS = 0.3;
function lod1(indexCount: number): number {
  return Math.floor((indexCount / 12) * LOD1_CARDS) * 12;
}
