// WGSL shaders. The scene is lit in linear HDR (Cook-Torrance sun light, sky
// ambient, two shadow cascades, height fog) into a float target; a post pass
// adds bloom, tonemaps (ACES), anti-aliases (FXAA) and grades.

export const FRAME = /* wgsl */ `
struct Frame {
  viewProj: mat4x4f,
  lightVP0: mat4x4f,   // near shadow cascade (the cascade being drawn, in shadow passes)
  lightVP1: mat4x4f,   // far shadow cascade
  invViewProj: mat4x4f,
  camPos: vec4f,       // xyz camera, w time
  sunDir: vec4f,       // xyz toward sun, w shadows on
  sunColor: vec4f,     // rgb, w ambient
  fogColor: vec4f,     // rgb, w density
  skyTop: vec4f,
  skyHorizon: vec4f,
  params: vec4f,       // x shadow texel, y map size, z hurt flash, w water level
  zone: vec4f,         // cx, cz, r, visible
  focus: vec4f,        // xyz camera target (player), w enabled
  post: vec4f,         // 1/width, 1/height, bloom strength, exposure
  ray0: vec4f,         // view ray through the screen centre ...
  rayX: vec4f,         // ... and its change across the screen (ndc x, y)
  rayY: vec4f,
};
`;

const NOISE = /* wgsl */ `
fn hash2(p: vec2f) -> f32 {
  var q = fract(p * vec2f(0.1031, 0.1030));
  q += dot(q, q.yx + 33.33);
  return fract((q.x + q.y) * q.x);
}
fn hash3(p: vec3f) -> f32 {
  var q = fract(p * 0.1031);
  q += dot(q, q.zyx + 31.32);
  return fract((q.x + q.y) * q.z);
}
fn vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash2(i), hash2(i + vec2f(1.0, 0.0)), u.x), mix(hash2(i + vec2f(0.0, 1.0)), hash2(i + vec2f(1.0, 1.0)), u.x), u.y);
}
fn vnoise3(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = mix(mix(hash3(i), hash3(i + vec3f(1.0, 0.0, 0.0)), u.x), mix(hash3(i + vec3f(0.0, 1.0, 0.0)), hash3(i + vec3f(1.0, 1.0, 0.0)), u.x), u.y);
  let b = mix(mix(hash3(i + vec3f(0.0, 0.0, 1.0)), hash3(i + vec3f(1.0, 0.0, 1.0)), u.x), mix(hash3(i + vec3f(0.0, 1.0, 1.0)), hash3(i + vec3f(1.0, 1.0, 1.0)), u.x), u.y);
  return mix(a, b, u.z);
}
fn fbm2(p: vec2f) -> f32 {
  return vnoise(p) * 0.5 + vnoise(p * 2.03 + 17.1) * 0.25 + vnoise(p * 4.01 + 5.3) * 0.125 + vnoise(p * 8.07 + 9.7) * 0.0625;
}
fn fbm3(p: vec3f) -> f32 {
  return vnoise3(p) * 0.5 + vnoise3(p * 2.03 + 7.1) * 0.25 + vnoise3(p * 4.01 + 3.3) * 0.125;
}
fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}
`;

export const COMMON = /* wgsl */ `
${FRAME}
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var shadowMap: texture_depth_2d_array;
@group(0) @binding(2) var shadowSamp: sampler_comparison;
${NOISE}

fn shadowCascade(lp: vec4f, layer: i32, bias: f32) -> f32 {
  let uv = lp.xy * vec2f(0.5, -0.5) + vec2f(0.5);
  let ts = F.params.x;
  var s = 0.0;
  for (var i = -1; i <= 1; i++) {
    for (var j = -1; j <= 1; j++) {
      s += textureSampleCompareLevel(shadowMap, shadowSamp, uv + vec2f(f32(i), f32(j)) * ts, layer, lp.z - bias);
    }
  }
  return s / 9.0;
}

fn shadowFactor(wp: vec3f, n: vec3f) -> f32 {
  if (F.sunDir.w < 0.5) { return 1.0; }
  let slope = 1.0 - max(dot(n, F.sunDir.xyz), 0.0);
  let lp0 = F.lightVP0 * vec4f(wp + n * (0.04 + 0.08 * slope), 1.0);
  let e0 = max(abs(lp0.x), abs(lp0.y));
  if (e0 < 0.96 && lp0.z < 1.0) {
    let s0 = shadowCascade(lp0, 0, 0.0008);
    if (e0 < 0.86) { return s0; }
    let lp1 = F.lightVP1 * vec4f(wp + n * (0.25 + 0.5 * slope), 1.0);
    return mix(s0, shadowCascade(lp1, 1, 0.0016), smoothstep(0.86, 0.96, e0));
  }
  let lp1 = F.lightVP1 * vec4f(wp + n * (0.25 + 0.5 * slope), 1.0);
  let e1 = max(abs(lp1.x), abs(lp1.y));
  if (e1 > 0.99 || lp1.z > 1.0) { return 1.0; }
  return mix(shadowCascade(lp1, 1, 0.0016), 1.0, smoothstep(0.88, 0.99, e1));
}

fn skyColor(dir: vec3f) -> vec3f {
  let h = dir.y;
  var c = mix(F.skyHorizon.rgb, F.skyTop.rgb, pow(clamp(h, 0.0, 1.0), 0.5));
  c = mix(c, F.fogColor.rgb, smoothstep(0.03, -0.2, h));
  let sd = max(dot(dir, F.sunDir.xyz), 0.0);
  // forward scattering around the sun
  c += F.sunColor.rgb * (pow(sd, 6.0) * 0.06 + pow(sd, 48.0) * 0.18);
  return c;
}

const PI = 3.14159265;

struct Surface {
  albedo: vec3f,
  n: vec3f,
  rough: f32,
  metal: f32,
  ao: f32,
  trans: f32,   // foliage translucency
};

/// Cook-Torrance sun light + sky ambient (diffuse and specular), in linear HDR.
fn shade(s: Surface, wp: vec3f) -> vec3f {
  let L = F.sunDir.xyz;
  let V = normalize(F.camPos.xyz - wp);
  let n = s.n;
  let ndl = max(dot(n, L), 0.0);
  let ndv = max(dot(n, V), 1e-3);
  let H = normalize(L + V);
  let ndh = max(dot(n, H), 0.0);
  let vdh = max(dot(V, H), 0.0);
  let a = max(s.rough * s.rough, 0.02);
  let a2 = a * a;
  let dd = ndh * ndh * (a2 - 1.0) + 1.0;
  let D = a2 / (PI * dd * dd);
  let k = (s.rough + 1.0) * (s.rough + 1.0) / 8.0;
  let G = (ndl / (ndl * (1.0 - k) + k)) * (ndv / (ndv * (1.0 - k) + k));
  let f0 = mix(vec3f(0.04), s.albedo, s.metal);
  let Fr = f0 + (1.0 - f0) * pow(1.0 - vdh, 5.0);
  let diffCol = s.albedo * (1.0 - s.metal);
  let sh = shadowFactor(wp, n);
  let spec = Fr * (D * G / (4.0 * ndl * ndv + 1e-3));
  var direct = (diffCol / PI * (1.0 - Fr) + spec) * ndl;
  // light through thin leaves
  direct += diffCol / PI * s.trans * max(dot(-n, L), 0.0) * 0.6;
  var c = direct * F.sunColor.rgb * sh * PI;
  // ambient: sky above, ground bounce below
  let up = n.y * 0.5 + 0.5;
  let ground = F.sunColor.rgb * vec3f(0.11, 0.1, 0.08);
  let irr = mix(ground, mix(F.skyHorizon.rgb, F.skyTop.rgb, 0.5) * 0.75 + F.sunColor.rgb * 0.03, up);
  c += diffCol * irr * F.sunColor.w * s.ao;
  // ambient specular from the sky in the mirror direction
  let R = reflect(-V, n);
  let env = mix(skyColor(R), irr, clamp(s.rough * 1.15, 0.0, 1.0));
  let fa = f0 + (max(vec3f(1.0 - s.rough), f0) - f0) * pow(1.0 - ndv, 5.0);
  c += env * fa * s.ao * F.sunColor.w * mix(1.0, 0.5 + 0.5 * sh, 0.6);
  return c;
}

fn applyFog(c: vec3f, wp: vec3f) -> vec3f {
  let v = wp - F.camPos.xyz;
  let d = length(v);
  // denser near the ground / sea
  let hgt = max(0.5 * (wp.y + F.camPos.y), 0.0);
  let dens = F.fogColor.w * (0.35 + 0.65 * exp(-hgt * 0.012));
  let f = 1.0 - exp(-pow(d * dens, 1.35));
  let sd = max(dot(v / max(d, 1e-3), F.sunDir.xyz), 0.0);
  let col = F.fogColor.rgb + F.sunColor.rgb * pow(sd, 8.0) * 0.12;
  return mix(c, col, clamp(f, 0.0, 1.0));
}

/// Bump a normal with the world-space gradient of a height function.
fn bump(n: vec3f, grad: vec3f, k: f32) -> vec3f {
  return normalize(n - k * (grad - n * dot(grad, n)));
}
`;

// ---------------------------------------------------------------- static world
export const LIT_WGSL = /* wgsl */ `
${COMMON}
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f };
struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) wp: vec3f,
  @location(1) n: vec3f,
  @location(2) col: vec4f,
};

fn sway(p: vec3f, m: f32) -> vec3f {
  if (m > 0.75) { return p; }
  let t = F.camPos.w;
  let s = sin(t * 1.6 + p.x * 0.31 + p.z * 0.23) * 0.07 + sin(t * 2.7 + p.z * 0.47) * 0.03;
  return p + vec3f(s, 0.0, s * 0.6);
}

@vertex fn vs(v: VIn) -> VOut {
  var o: VOut;
  let p = sway(v.pos, v.col.a);
  o.wp = p;
  o.clip = F.viewProj * vec4f(p, 1.0);
  o.n = v.nrm.xyz;
  o.col = v.col;
  return o;
}

@fragment fn fs_terrain(i: VOut) -> @location(0) vec4f {
  var n = normalize(i.n);
  let p = i.wp.xz;
  let dist = distance(i.wp, F.camPos.xyz);
  let detail = 1.0 - smoothstep(40.0, 160.0, dist);
  let base = pow(i.col.rgb, vec3f(2.2));
  let pave = i.col.a;
  let slope = 1.0 - n.y;
  let wl = F.params.w;
  // mvar variation breaks up the vertex colours
  let mvar = fbm2(p * 0.021) * 0.6 + vnoise(p * 0.11) * 0.4;
  // vertex colours are picked for a flat look: pull them to natural reflectance
  let lum0 = dot(base, vec3f(0.3, 0.55, 0.15));
  var albedo = mix(vec3f(lum0), base, 0.62) * (0.5 + 0.4 * mvar);
  // grass blades / soil
  let g1 = vnoise(p * 3.1);
  let g2 = vnoise(p * 9.7 + 31.0);
  let fine = g1 * 0.6 + g2 * 0.4;
  let grass = (1.0 - smoothstep(0.2, 0.42, slope)) * smoothstep(1.4, 3.2, i.wp.y) * (1.0 - pave);
  albedo = mix(albedo, albedo * mix(vec3f(0.55, 0.62, 0.42), vec3f(1.25, 1.3, 0.95), fine), grass * (0.35 + 0.65 * detail));
  // dry patches
  albedo = mix(albedo, albedo * vec3f(1.25, 1.12, 0.8), grass * smoothstep(0.55, 0.8, mvar) * 0.5);
  // rock on slopes
  let rk = fbm2(p * 0.35 + i.wp.y * 0.2);
  let rock = smoothstep(0.24, 0.5, slope + (rk - 0.5) * 0.2) * (1.0 - pave);
  albedo = mix(albedo, vec3f(0.26, 0.25, 0.23) * (0.6 + 0.8 * rk), rock * 0.8);
  // paved ground: asphalt / packed dirt with fine grain and cracks
  let grain = vnoise(p * 14.0);
  let crack = mix(1.0, smoothstep(0.485, 0.5, abs(fbm2(p * 1.9) - 0.5) * 2.0 + 0.485 - 0.012 * grain), 0.55 * detail);
  let asphalt = vec3f(0.085, 0.083, 0.08) * (0.8 + 0.4 * grain) * mix(0.55, 1.0, crack) * (0.85 + 0.3 * mvar);
  albedo = mix(albedo, asphalt, clamp(pave * 1.25, 0.0, 1.0));
  // wet sand line + underwater darkening
  let wet = 1.0 - smoothstep(wl - 2.5, wl + 0.6, i.wp.y);
  albedo *= mix(1.0, 0.5, wet);
  // surface relief from the same noise
  let e = 0.12;
  let h0 = fine + rk * rock * 2.0;
  let hx = (vnoise((p + vec2f(e, 0.0)) * 3.1) * 0.6 + vnoise((p + vec2f(e, 0.0)) * 9.7 + 31.0) * 0.4) + fbm2((p + vec2f(e, 0.0)) * 0.35 + i.wp.y * 0.2) * rock * 2.0;
  let hz = (vnoise((p + vec2f(0.0, e)) * 3.1) * 0.6 + vnoise((p + vec2f(0.0, e)) * 9.7 + 31.0) * 0.4) + fbm2((p + vec2f(0.0, e)) * 0.35 + i.wp.y * 0.2) * rock * 2.0;
  n = bump(n, vec3f((hx - h0) / e, 0.0, (hz - h0) / e), 0.09 * detail * (1.0 - 0.6 * pave));
  var s: Surface;
  s.albedo = albedo;
  s.n = n;
  s.rough = mix(mix(0.92, 0.8, rock), 0.72, pave) - wet * 0.45;
  s.metal = 0.0;
  s.ao = 0.75 + 0.25 * fine;
  s.trans = 0.0;
  var c = shade(s, i.wp);
  c = applyFog(c, i.wp);
  return vec4f(c, 1.0);
}

@fragment fn fs_props(i: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  var n = normalize(i.n);
  var albedo = pow(i.col.rgb, vec3f(2.2));
  let mat = i32(round(i.col.a * 255.0));
  let dist = distance(i.wp, F.camPos.xyz);
  let detail = 1.0 - smoothstep(30.0, 120.0, dist);
  var s: Surface;
  s.rough = 0.85;
  s.metal = 0.0;
  s.ao = 1.0;
  s.trans = 0.0;
  let wp = i.wp;
  if (mat < 192) {
    // foliage: see-through between the camera and the player
    if (F.focus.w > 0.5) {
      let c = F.camPos.xyz;
      let seg = F.focus.xyz - c;
      let L = length(seg);
      if (L > 0.5) {
        let dir = seg / L;
        let t = dot(wp - c, dir);
        if (t > 0.0 && t < L + 0.6) {
          let perp = length((wp - c) - dir * t);
          let radius = 0.9 + 1.4 * clamp(t / L, 0.0, 1.0);
          let px = vec2i(i.clip.xy);
          if (perp < radius * 0.65 || (perp < radius && ((px.x + px.y) & 1) == 0)) { discard; }
        }
      }
    }
    // leaf clusters: light and dark clumps, a little yellowing
    let cl = vnoise3(wp * 2.6);
    let fine = vnoise3(wp * 9.0);
    albedo *= 0.55 + 0.75 * cl * (0.6 + 0.4 * fine);
    albedo = mix(albedo, albedo * vec3f(1.3, 1.15, 0.6), smoothstep(0.62, 0.9, vnoise3(wp * 0.7)) * 0.35);
    let e = 0.09;
    let g = vec3f(vnoise3((wp + vec3f(e, 0.0, 0.0)) * 2.6) - cl, vnoise3((wp + vec3f(0.0, e, 0.0)) * 2.6) - cl, vnoise3((wp + vec3f(0.0, 0.0, e)) * 2.6) - cl) / e;
    n = bump(n, g, 0.35 * detail);
    s.rough = 0.62;
    s.ao = 0.45 + 0.55 * cl;
    s.trans = 1.0;
  } else if (mat == 250) {
    // roof: rows of tiles / sheet ribs
    let u = wp.x + wp.z;
    let rib = abs(fract(u * 1.6) - 0.5) * 2.0;
    let row = fract(wp.x * 0.9 - wp.z * 0.9);
    albedo *= 0.75 + 0.3 * smoothstep(0.1, 0.5, rib) + (vnoise(wp.xz * 0.8) - 0.5) * 0.25;
    albedo *= 0.9 + 0.1 * step(0.08, row);
    n = bump(n, vec3f(1.0, 0.0, 1.0) * (step(0.5, fract(u * 1.6)) - 0.5) * (1.0 - smoothstep(0.3, 0.5, rib)) * 2.0, 0.5 * detail);
    s.rough = 0.6;
  } else if (mat == 240) {
    // wood: planks and grain
    let ax = abs(n);
    var uv = wp.xy;
    if (ax.x > ax.z && ax.x > ax.y) { uv = wp.zy; } else if (ax.y > ax.z) { uv = wp.xz; }
    let plank = fract(uv.y * 4.0);
    let gr = vnoise(vec2f(uv.x * 2.0, uv.y * 40.0 + floor(uv.y * 4.0) * 7.0));
    albedo *= (0.7 + 0.45 * gr) * mix(0.55, 1.0, smoothstep(0.0, 0.06, plank) * smoothstep(1.0, 0.94, plank));
    s.rough = 0.78;
    s.ao = mix(0.6, 1.0, smoothstep(0.0, 0.1, plank));
  } else if (mat == 235) {
    // painted steel with rust and chips
    let r = fbm3(wp * 1.3);
    let rust = smoothstep(0.58, 0.75, r + (1.0 - clamp(wp.y * 0.0 + 1.0, 0.0, 1.0)));
    albedo = mix(albedo * (0.85 + 0.25 * vnoise3(wp * 6.0)), vec3f(0.23, 0.1, 0.045) * (0.6 + 0.8 * vnoise3(wp * 11.0)), rust * 0.75);
    s.rough = mix(0.42, 0.85, rust);
    s.metal = mix(0.55, 0.1, rust);
  } else if (mat == 220 || mat == 225) {
    // rock / bark
    let f = select(vec3f(1.4, 1.4, 1.4), vec3f(7.0, 1.2, 7.0), mat == 225);
    let h0 = fbm3(wp * f);
    let e = 0.05;
    let g = vec3f(fbm3((wp + vec3f(e, 0.0, 0.0)) * f) - h0, fbm3((wp + vec3f(0.0, e, 0.0)) * f) - h0, fbm3((wp + vec3f(0.0, 0.0, e)) * f) - h0) / e;
    n = bump(n, g, 0.45 * detail + 0.1);
    albedo *= 0.55 + 0.9 * h0;
    if (mat == 220) {
      // moss on top
      albedo = mix(albedo, vec3f(0.09, 0.13, 0.04), smoothstep(0.55, 0.9, n.y) * smoothstep(0.45, 0.6, vnoise3(wp * 0.9)) * 0.7);
    }
    s.rough = 0.9;
    s.ao = 0.6 + 0.4 * h0;
  } else if (mat == 215 && abs(n.y) < 0.5) {
    // brick: running bond, every brick its own shade, recessed mortar
    let ax = abs(n);
    let u = select(wp.x, wp.z, ax.x > ax.z);
    let row = floor(wp.y / 0.085);
    let bu = u / 0.22 + 0.5 * (row - 2.0 * floor(row * 0.5));
    let cell = vec2f(fract(bu), fract(wp.y / 0.085));
    let id = vec2f(floor(bu), row);
    let edge = min(min(cell.x, 1.0 - cell.x) * 0.22, min(cell.y, 1.0 - cell.y) * 0.085);
    let mortar = 1.0 - smoothstep(0.004, 0.011, edge);
    let tone = 0.72 + 0.5 * hash2(id) + (vnoise3(wp * 14.0) - 0.5) * 0.25;
    let soot = fbm3(wp * vec3f(0.5, 0.3, 0.5));
    albedo = mix(albedo * tone * (0.8 + 0.3 * soot), vec3f(0.62, 0.6, 0.56) * (0.8 + 0.3 * vnoise3(wp * 30.0)), mortar);
    var g = vec3f(0.0);
    let de = smoothstep(0.0, 0.011, edge);
    // the brick faces stand a little proud of the joints
    g.y = (smoothstep(0.0, 0.14, cell.y) - smoothstep(0.86, 1.0, cell.y)) * -(1.0 - de) * 2.5;
    let gx = (smoothstep(0.0, 0.05, cell.x) - smoothstep(0.95, 1.0, cell.x)) * -(1.0 - de) * 2.5;
    if (ax.x > ax.z) { g.z = gx; } else { g.x = gx; }
    let gn = vnoise3(wp * 40.0);
    g += (vec3f(vnoise3((wp + vec3f(0.01, 0.0, 0.0)) * 40.0), vnoise3((wp + vec3f(0.0, 0.01, 0.0)) * 40.0), vnoise3((wp + vec3f(0.0, 0.0, 0.01)) * 40.0)) - gn) * 6.0 * (1.0 - mortar);
    n = bump(n, g, detail);
    s.rough = mix(0.82, 0.95, mortar);
    s.ao = mix(1.0, 0.7, mortar);
  } else if (mat == 205) {
    // paint and upholstery: even colour, a soft sheen
    albedo *= 0.92 + 0.12 * vnoise3(wp * 9.0);
    s.rough = 0.62;
  } else {
    // plaster, concrete, floors: stains, streaks under the eaves, fine grain
    let stain = fbm3(wp * vec3f(0.45, 0.25, 0.45));
    let streak = vnoise(vec2f((wp.x + wp.z) * 2.2, wp.y * 0.35));
    let grain = vnoise3(wp * 18.0);
    var k = 0.78 + 0.3 * stain + (grain - 0.5) * 0.12 * detail;
    if (abs(n.y) < 0.5) { k -= 0.18 * smoothstep(0.55, 0.85, streak) * stain; }
    if (mat == 230) { k *= 0.9 + 0.2 * vnoise3(wp * 3.0); }
    albedo *= k;
    let e = 0.03;
    var g = vec3f(vnoise3((wp + vec3f(e, 0.0, 0.0)) * 18.0) - grain, vnoise3((wp + vec3f(0.0, e, 0.0)) * 18.0) - grain, vnoise3((wp + vec3f(0.0, 0.0, e)) * 18.0) - grain) / e * 0.012;
    if (mat == 255 && abs(n.y) < 0.5) {
      // lap siding: boards 22 cm tall, each casting a thin shadow on the next
      let b = fract(wp.y / 0.22);
      albedo *= mix(0.62, 1.0, smoothstep(0.0, 0.12, b)) * (0.94 + 0.12 * hash2(vec2f(floor(wp.y / 0.22), 3.0)));
      g.y += (smoothstep(0.0, 0.12, b) - smoothstep(0.85, 1.0, b)) * -0.9;
    }
    if (mat == 210) {
      // stucco: a rough trowelled coat, darker towards the ground where rain splashes
      let h0 = fbm3(wp * 7.0);
      let e2 = 0.02;
      g += (vec3f(fbm3((wp + vec3f(e2, 0.0, 0.0)) * 7.0), fbm3((wp + vec3f(0.0, e2, 0.0)) * 7.0), fbm3((wp + vec3f(0.0, 0.0, e2)) * 7.0)) - h0) / e2 * 0.05;
      albedo *= 0.9 + 0.25 * h0;
    }
    if (mat == 230 && abs(n.y) < 0.5) {
      // cast concrete: formwork panel joints and tie holes, rain streaks
      let ax = abs(n);
      let u = select(wp.x, wp.z, ax.x > ax.z);
      let pj = min(abs(fract(u / 2.4) - 0.5) * 2.4, abs(fract(wp.y / 1.2) - 0.5) * 1.2);
      let joint = 1.0 - smoothstep(0.585, 0.6, pj);
      albedo *= mix(0.72, 1.0, joint);
      let hole = length(vec2f(fract(u / 1.2) - 0.5, fract(wp.y / 0.6) - 0.5) * vec2f(1.2, 0.6));
      albedo *= mix(0.6, 1.0, smoothstep(0.018, 0.03, hole));
      albedo *= 1.0 - 0.22 * smoothstep(0.5, 0.9, vnoise(vec2f(u * 3.0, wp.y * 0.18)));
    }
    n = bump(n, g, detail);
    s.rough = select(0.88, 0.8, mat == 245);
  }
  if (!front) { n = -n; }
  s.albedo = albedo;
  s.n = n;
  var c = shade(s, wp);
  c = applyFog(c, wp);
  return vec4f(c, 1.0);
}
`;

export const LIT_SHADOW_WGSL = /* wgsl */ `
${FRAME}
@group(0) @binding(0) var<uniform> F: Frame;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f };
@vertex fn vs(v: VIn) -> @builtin(position) vec4f {
  return F.lightVP0 * vec4f(v.pos, 1.0);
}
`;

// ---------------------------------------------------------------- instanced shapes
const BOX_IO = /* wgsl */ `
struct BIn {
  @location(0) pos: vec3f,
  @location(1) nrm: vec3f,
  @location(2) ipos: vec3f,
  @location(3) iquat: vec4f,
  @location(4) iscale: vec3f,
  @location(5) icol: vec4f,
  @location(6) imat: vec4f,
};
`;

export const BOX_WGSL = /* wgsl */ `
${COMMON}
${BOX_IO}
struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) wp: vec3f,
  @location(1) n: vec3f,
  @location(2) col: vec4f,
  @location(3) mat: vec2f,
  @location(4) lp: vec3f,
};
@vertex fn vs(v: BIn) -> VOut {
  var o: VOut;
  let wp = v.ipos + qrot(v.iquat, v.pos * v.iscale);
  o.wp = wp;
  o.clip = F.viewProj * vec4f(wp, 1.0);
  o.n = normalize(qrot(v.iquat, v.nrm / max(v.iscale, vec3f(1e-4))));
  o.col = v.icol;
  o.mat = v.imat.xy;
  o.lp = v.pos * v.iscale;
  return o;
}
@fragment fn fs(i: VOut) -> @location(0) vec4f {
  var albedo = pow(i.col.rgb, vec3f(2.2));
  let lit = i.col.a;
  // a little surface grain so flat colours do not look like plastic
  let g = vnoise3(i.lp * 60.0);
  albedo *= 0.9 + 0.2 * g * (1.0 - i.mat.y);
  var s: Surface;
  s.albedo = albedo;
  s.n = normalize(i.n);
  s.rough = clamp(i.mat.x + (g - 0.5) * 0.12, 0.05, 1.0);
  s.metal = i.mat.y;
  s.ao = 1.0;
  s.trans = 0.0;
  var c = shade(s, i.wp) * lit + albedo * (1.0 - lit) * 6.0;
  c = applyFog(c, i.wp);
  return vec4f(c, 1.0);
}
@fragment fn fs_glow(i: VOut) -> @location(0) vec4f {
  let albedo = pow(i.col.rgb, vec3f(2.2)) * 5.0;
  let d = distance(i.wp, F.camPos.xyz);
  let fade = exp(-d * 0.002);
  return vec4f(albedo, i.col.a * fade);
}
// Window glass: mostly clear, mirrors the sky at grazing angles, glints in the sun.
@fragment fn fs_glass(i: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  var n = normalize(i.n);
  let V = normalize(F.camPos.xyz - i.wp);
  if (dot(n, V) < 0.0) { n = -n; }
  let fres = 0.05 + 0.95 * pow(1.0 - max(dot(n, V), 0.0), 5.0);
  let R = reflect(-V, n);
  let sky = mix(F.skyHorizon.rgb, F.skyTop.rgb, clamp(R.y * 1.3 + 0.15, 0.0, 1.0)) * (0.7 + 0.5 * fbm2(R.xz * 2.0 + R.y));
  let ground = vec3f(0.12, 0.13, 0.11);
  let env = mix(ground, sky, smoothstep(-0.15, 0.1, R.y));
  let glint = pow(max(dot(R, F.sunDir.xyz), 0.0), 600.0) * 30.0 + pow(max(dot(R, F.sunDir.xyz), 0.0), 40.0) * 0.5;
  // faint grime so the pane reads as a surface even head-on
  let dirt = 0.04 + 0.05 * fbm2(i.wp.xy * 1.7 + i.wp.z * 1.7);
  let a = clamp(0.08 + dirt + fres * 0.85 + glint, 0.0, 1.0);
  var c = (pow(i.col.rgb, vec3f(2.2)) * (0.08 + dirt) + env * fres + F.sunColor.rgb * glint) / max(a, 0.001);
  c = applyFog(c, i.wp);
  return vec4f(c, a);
}
@fragment fn fs_smoke(i: VOut) -> @location(0) vec4f {
  let albedo = pow(i.col.rgb, vec3f(2.2));
  var c = albedo * (0.55 + 0.45 * max(dot(normalize(i.n), F.sunDir.xyz), 0.0)) * F.sunColor.rgb;
  c = applyFog(c, i.wp);
  return vec4f(c, i.col.a);
}
`;

export const BOX_SHADOW_WGSL = /* wgsl */ `
${FRAME}
@group(0) @binding(0) var<uniform> F: Frame;
${BOX_IO}
fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}
@vertex fn vs(v: BIn) -> @builtin(position) vec4f {
  if (v.icol.a < 0.5) { return vec4f(0.0, 0.0, -2.0, 1.0); } // emissive parts cast no shadow
  let wp = v.ipos + qrot(v.iquat, v.pos * v.iscale);
  return F.lightVP0 * vec4f(wp, 1.0);
}
`;

// ---------------------------------------------------------------- glTF models
const MODEL_IO = /* wgsl */ `
struct MIn {
  @location(0) pos: vec3f,
  @location(1) nrm: vec3f,
  @location(2) tan: vec4f,
  @location(3) uv: vec2f,
  @location(4) ipos: vec3f,
  @location(5) iquat: vec4f,
  /** 1 = as painted, towards 0 = burnt out. */
  @location(6) itint: f32,
};
`;

export const MODEL_WGSL = /* wgsl */ `
${COMMON}
${MODEL_IO}
struct Mat {
  base: vec4f,
  emissive: vec4f,
  mr: vec4f,     // metallic, roughness, has textures, unused
};
@group(1) @binding(0) var<uniform> M: Mat;
@group(1) @binding(1) var samp: sampler;
@group(1) @binding(2) var baseTex: texture_2d<f32>;
@group(1) @binding(3) var normalTex: texture_2d<f32>;
@group(1) @binding(4) var ormTex: texture_2d<f32>;
@group(1) @binding(5) var emisTex: texture_2d<f32>;
struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) wp: vec3f,
  @location(1) n: vec3f,
  @location(2) t: vec4f,
  @location(3) uv: vec2f,
  @location(4) tint: f32,
};
@vertex fn vs(v: MIn) -> VOut {
  var o: VOut;
  o.tint = v.itint;
  let wp = v.ipos + qrot(v.iquat, v.pos);
  o.wp = wp;
  o.clip = F.viewProj * vec4f(wp, 1.0);
  o.n = qrot(v.iquat, v.nrm);
  o.t = vec4f(qrot(v.iquat, v.tan.xyz), v.tan.w);
  o.uv = v.uv;
  return o;
}
fn surface(i: VOut, front: bool) -> Surface {
  var n = normalize(i.n);
  var s: Surface;
  var base = M.base;
  s.metal = M.mr.x;
  s.rough = M.mr.y;
  s.ao = 1.0;
  if (M.mr.z > 0.5) {
    base *= textureSample(baseTex, samp, i.uv);
    let orm = textureSample(ormTex, samp, i.uv).rgb;
    s.ao = orm.r;
    s.rough *= orm.g;
    s.metal *= orm.b;
    let tn = textureSample(normalTex, samp, i.uv).xyz * 2.0 - 1.0;
    let t = normalize(i.t.xyz - n * dot(i.t.xyz, n));
    let b = cross(n, t) * i.t.w;
    n = normalize(t * tn.x + b * tn.y + n * tn.z);
  }
  if (!front) { n = -n; }
  // a burnt-out wreck: sooty, dull paint
  let burnt = 1.0 - smoothstep(0.3, 0.9, i.tint);
  s.albedo = base.rgb * i.tint;
  s.metal *= 1.0 - 0.8 * burnt;
  s.n = n;
  s.rough = clamp(mix(s.rough, 0.9, burnt), 0.04, 1.0);
  s.trans = 0.0;
  return s;
}
@fragment fn fs(i: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let s = surface(i, front);
  var c = shade(s, i.wp);
  if (M.mr.z > 0.5) { c += textureSample(emisTex, samp, i.uv).rgb * M.emissive.rgb * 2.0 * step(0.5, i.tint); }
  c = applyFog(c, i.wp);
  return vec4f(c, 1.0);
}
@fragment fn fs_blend(i: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let s = surface(i, front);
  var c = shade(s, i.wp);
  // glass: reflections stay bright where the surface is see-through
  let V = normalize(F.camPos.xyz - i.wp);
  let fres = 0.04 + 0.96 * pow(1.0 - max(dot(s.n, V), 0.0), 5.0);
  // the glass of a wreck is blackened
  let a = mix(0.96, clamp(M.base.a + fres * 0.6, 0.0, 1.0), step(0.5, i.tint));
  c = applyFog(c, i.wp);
  return vec4f(c, a);
}
`;

export const MODEL_SHADOW_WGSL = /* wgsl */ `
${FRAME}
@group(0) @binding(0) var<uniform> F: Frame;
${MODEL_IO}
fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}
@vertex fn vs(v: MIn) -> @builtin(position) vec4f {
  return F.lightVP0 * vec4f(v.ipos + qrot(v.iquat, v.pos), 1.0);
}
`;

// ---------------------------------------------------------------- grass
// Blades of grass around the camera, placed and shaped in the vertex shader:
// no buffers. Instances index the cells of a grid snapped to the world, so a
// blade is always the same blade wherever the camera is. Ranges of the
// instance index select the ring:
//   [0, 65536)        far ring (bigger, sparser blades), hollow in the middle
//   [65536, 131072)   near ring (fine blades)
//   [200000, 265536)  far ring without the hole (when the near ring is off)
export const GRASS_FAR = 256;
export const GRASS_NEAR = 256;
// Everything about a blade that the scene pass and the shadow pass share:
// where it grows, its shape and how the wind moves it, so a blade and its
// shadow are always the same blade in the same pose.
const GRASS_CORE = /* wgsl */ `
@group(1) @binding(0) var heightTex: texture_2d<f32>;
@group(1) @binding(1) var maskTex: texture_2d<f32>;
// the terrain exactly as the simulation has it: 4 m cells split into two triangles
fn groundH(p: vec2f) -> f32 {
  let f = clamp(p / 4.0, vec2f(0.0), vec2f(255.999));
  let i = vec2i(floor(f));
  let t = f - floor(f);
  let h00 = textureLoad(heightTex, i, 0).r;
  let h10 = textureLoad(heightTex, i + vec2i(1, 0), 0).r;
  let h01 = textureLoad(heightTex, i + vec2i(0, 1), 0).r;
  let h11 = textureLoad(heightTex, i + vec2i(1, 1), 0).r;
  if (t.x + t.y <= 1.0) { return h00 + (h10 - h00) * t.x + (h01 - h00) * t.y; }
  return h11 + (h01 - h11) * (1.0 - t.x) + (h10 - h11) * (1.0 - t.y);
}
fn grassAt(p: vec2f) -> f32 {
  let f = clamp(p - 0.5, vec2f(0.0), vec2f(1022.999));
  let i = vec2i(floor(f));
  let t = f - floor(f);
  let a = mix(textureLoad(maskTex, i, 0).r, textureLoad(maskTex, i + vec2i(1, 0), 0).r, t.x);
  let b = mix(textureLoad(maskTex, i + vec2i(0, 1), 0).r, textureLoad(maskTex, i + vec2i(1, 1), 0).r, t.x);
  return mix(a, b, t.y);
}
// Wind in two layers, as for the trees:
//  1. gusts roll across the field and lean whole tufts over, slowly;
//  2. each blade shivers quickly on its own, mostly at the tip.
// Returns the sideways shift of a point at height t (0 root .. 1 tip).
fn applyGrassWind(p: vec2f, t: f32, hgt: f32, phase: f32, time: f32) -> vec2f {
  let gust = vnoise(p * 0.06 + vec2f(time * 0.33, time * 0.2));
  let roll = sin(time * 1.5 + p.x * 0.33 + p.y * 0.27) * 0.5 + 0.5;
  let lean = (0.25 + 1.15 * gust) * (0.35 + 0.65 * roll);
  let shiver = sin(time * 5.3 + phase * 6.2831 + p.x * 2.1 - p.y * 1.7) * 0.5 + sin(time * 8.9 + phase * 17.0) * 0.25;
  return (vec2f(0.86, 0.5) * lean * 0.5 * t * t + vec2f(-0.5, 0.86) * shiver * (0.03 + 0.09 * gust) * t * t * t) * hgt;
}
struct Blade {
  wp: vec3f,
  n: vec3f,
  tv: vec3f,     // height along the blade, per-blade tint, dryness
  grows: bool,
};
fn blade(vi: u32, ii: u32) -> Blade {
  var o: Blade;
  o.grows = false;
  // which ring
  var k = ii;
  var cell = 0.5;
  var n = ${GRASS_FAR}u;
  var hole = 17.5;
  var hmul = 1.75;
  var wmul = 3.0;
  if (ii >= 200000u) { k = ii - 200000u; hole = 0.0; }
  else if (ii >= 65536u) { k = ii - 65536u; cell = 0.14; n = ${GRASS_NEAR}u; hole = 0.0; hmul = 1.4; wmul = 2.0; }
  let reach = f32(n) * cell * 0.5;
  let c = floor(F.camPos.xz / cell) - f32(n / 2u) + vec2f(f32(k % n), f32(k / n));
  let r1 = hash2(c + cell * 13.7);
  let r2 = hash2(c.yx * 1.31 + 71.9);
  let r3 = hash2(c + 191.3);
  let r4 = hash2(c * 0.77 + 37.1);
  let p = (c + vec2f(r1, r2)) * cell;
  let dist = length(p - F.camPos.xz);
  var fade = 1.0 - smoothstep(reach * 0.78, reach * 0.98, dist);
  if (hole > 0.0) { fade *= smoothstep(hole * 0.82, hole, dist); }
  let mask = grassAt(p);
  // tufts and bare patches
  let clump = vnoise(p * 0.45);
  let patchy = vnoise(p * 0.11 + 40.0);
  if (fade <= 0.01 || r3 > mask * (0.82 + 0.4 * patchy)) { return o; }
  let hgt = (0.2 + 0.34 * r4) * (0.55 + 0.95 * clump) * hmul * fade * (0.5 + 0.5 * mask);

  // the blade: three quads narrowing to a point
  var idx = array<u32, 15>(0u, 1u, 2u, 1u, 3u, 2u, 2u, 3u, 4u, 3u, 5u, 4u, 4u, 5u, 6u);
  let v = idx[vi];
  let level = v / 2u;
  let side = f32(v % 2u) * 2.0 - 1.0;
  var ts = array<f32, 4>(0.0, 0.36, 0.7, 1.0);
  var ws = array<f32, 4>(1.0, 0.82, 0.5, 0.0);
  let t = ts[level];
  let ang = r2 * 6.2831853 + r1 * 3.0;
  let dir = vec2f(cos(ang), sin(ang));
  let across = vec2f(-dir.y, dir.x);
  let halfW = (0.011 + 0.008 * r3) * wmul * ws[level] * select(side, 0.0, level == 3u);
  // it arcs over under its own weight, the taller the more
  let droop = (0.18 + 0.5 * r1) * hgt;
  var off = dir * droop * t * t + across * halfW;
  var y = hgt * t * (1.0 - 0.22 * t * t);
  off += applyGrassWind(p, t, hgt, r1, F.camPos.w);
  // pressed down around the player's feet
  if (F.focus.w > 0.5) {
    let d = p - F.focus.xz;
    let l = length(d);
    let near = 1.0 - smoothstep(0.25, 1.05, l);
    if (near > 0.0 && abs(F.focus.y - groundH(p)) < 2.5) {
      off += d / max(l, 0.05) * near * 0.4 * hgt * t;
      y *= 1.0 - 0.55 * near;
    }
  }
  o.wp = vec3f(p.x + off.x, groundH(p) + y - 0.02, p.y + off.y);
  // rounded shading: mostly up, a little along the lean and across the width
  o.n = normalize(vec3f(dir.x * 0.5 * t + across.x * side * 0.35, 1.0, dir.y * 0.5 * t + across.y * side * 0.35));
  o.tv = vec3f(t, r4, smoothstep(0.55, 0.85, vnoise(p * 0.05 + 9.0)) * (0.4 + 0.6 * r1));
  o.grows = true;
  return o;
}
`;

export const GRASS_WGSL = /* wgsl */ `
${COMMON}
${GRASS_CORE}
struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) wp: vec3f,
  @location(1) n: vec3f,
  @location(2) tv: vec3f,
};
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  o.clip = vec4f(2.0, 2.0, 2.0, 1.0);   // culled unless a blade grows here
  let b = blade(vi, ii);
  if (!b.grows) { return o; }
  o.wp = b.wp;
  o.clip = F.viewProj * vec4f(b.wp, 1.0);
  o.n = b.n;
  o.tv = b.tv;
  return o;
}
@fragment fn fs(i: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let t = i.tv.x;
  // dark and damp at the root, brighter and yellower towards the tip
  var albedo = mix(vec3f(0.022, 0.045, 0.012), vec3f(0.095, 0.165, 0.04), pow(t, 0.8));
  albedo = mix(albedo, albedo * vec3f(1.25, 1.08, 0.7), i.tv.y * 0.5);
  // dry straw-coloured stretches
  albedo = mix(albedo, vec3f(0.36, 0.31, 0.12) * (0.35 + 0.75 * t), i.tv.z * 0.75);
  var s: Surface;
  s.albedo = albedo;
  var n = normalize(i.n);
  if (!front) { n = vec3f(-n.x, n.y, -n.z); }
  s.n = n;
  s.rough = 0.72;
  s.metal = 0.0;
  s.ao = mix(0.22, 1.0, smoothstep(0.0, 0.7, t));
  s.trans = 1.0;
  var c = shade(s, i.wp);
  c = applyFog(c, i.wp);
  return vec4f(c, 1.0);
}
`;

// Depth only, into a shadow cascade: the same blades in the same wind.
export const GRASS_SHADOW_WGSL = /* wgsl */ `
${FRAME}
@group(0) @binding(0) var<uniform> F: Frame;
${NOISE}
${GRASS_CORE}
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> @builtin(position) vec4f {
  let b = blade(vi, ii);
  if (!b.grows) { return vec4f(2.0, 2.0, 2.0, 1.0); }
  return F.lightVP0 * vec4f(b.wp, 1.0);
}
`;

// ---------------------------------------------------------------- soft particles
// Camera-facing puffs of smoke and fire. The shape comes from animated noise
// (no flat faces, no hard edges); smoke is lit by the sun, fire is emissive.
export const PUFF_WGSL = /* wgsl */ `
${COMMON}
struct PIn {
  @builtin(vertex_index) vi: u32,
  @location(0) ipos: vec3f,
  @location(1) isize: f32,
  @location(2) icol: vec4f,
  @location(3) iparm: vec3f,   // rotation, seed, kind (0 smoke, 1 flame, 2 fireball)
};
struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) col: vec4f,
  @location(2) parm: vec3f,
  @location(3) wp: vec3f,
  @location(4) soft: vec2f,   // near plane, fade distance
};
@group(1) @binding(0) var sceneDepth: texture_depth_2d;
// 0 where the particle meets what is behind it, 1 well in front of it
fn depthFade(i: VOut) -> f32 {
  let d = textureLoad(sceneDepth, vec2i(i.clip.xy), 0);
  let sceneZ = i.soft.x / max(d, 1e-7);
  let partZ = 1.0 / i.clip.w;
  return clamp((sceneZ - partZ) / i.soft.y, 0.0, 1.0);
}
@vertex fn vs(v: PIn) -> VOut {
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c = corners[v.vi];
  let right = normalize(F.rayX.xyz);
  let up = normalize(F.rayY.xyz);
  // flames stand upright: they rise whatever the camera does
  let upw = select(up, normalize(vec3f(0.0, 1.0, 0.0) + up * 0.25), v.iparm.z > 0.5 && v.iparm.z < 1.5);
  let cr = cos(v.iparm.x);
  let sr = sin(v.iparm.x);
  let q = vec2f(c.x * cr - c.y * sr, c.x * sr + c.y * cr);
  let stretch = select(1.0, 1.7, v.iparm.z > 0.5 && v.iparm.z < 1.5);
  let wp = v.ipos + (right * q.x + upw * q.y * stretch) * v.isize;
  var o: VOut;
  o.clip = F.viewProj * vec4f(wp, 1.0);
  o.uv = c;
  o.col = v.icol;
  o.parm = v.iparm;
  o.wp = wp;
  // (reverse-Z with the far plane at infinity: clip z is the near distance)
  o.soft = vec2f(o.clip.z, max(v.isize * 0.45, 0.15));
  return o;
}
fn heatRamp(h: f32) -> vec3f {
  // deep red -> orange -> yellow-white
  let a = mix(vec3f(0.55, 0.04, 0.0), vec3f(1.0, 0.32, 0.03), smoothstep(0.0, 0.45, h));
  return mix(a, vec3f(1.0, 0.86, 0.5), smoothstep(0.45, 1.0, h));
}
@fragment fn fs(i: VOut) -> @location(0) vec4f {
  let t = F.camPos.w;
  let seed = i.parm.y;
  let r = length(i.uv);
  if (r > 1.0) { discard; }
  let kind = i.parm.z;
  let df = depthFade(i);
  if (kind < 0.5) {
    // ---- smoke: billowing density with a ragged, wispy edge
    let p = vec3f(i.uv * 1.25 + vec2f(seed * 17.3, seed * 7.1), t * 0.22 + seed * 5.0);
    let warp = vec2f(fbm3(p * 1.1 + 3.7), fbm3(p * 1.1 - 9.2)) - 0.5;
    let n = fbm3(vec3f(p.xy + warp * 0.9, p.z) * 1.6);
    let fine = fbm3(p * 4.2 + 11.0);
    var dens = smoothstep(1.0, 0.15, r + (n - 0.5) * 1.1);
    dens *= 0.5 + 0.9 * n;
    dens *= 0.8 + 0.4 * fine;
    // rounded: lit like a lumpy ball, sunward side brighter
    let z = sqrt(max(1.0 - r * r, 0.0));
    let right = normalize(F.rayX.xyz);
    let up = normalize(F.rayY.xyz);
    let fw = normalize(F.camPos.xyz - i.wp);
    let nrm = normalize(right * (i.uv.x + warp.x) + up * (i.uv.y + warp.y) + fw * (z + 0.25));
    let ndl = dot(nrm, F.sunDir.xyz) * 0.5 + 0.5;
    let sky = mix(F.skyHorizon.rgb, F.skyTop.rgb, 0.5) * F.sunColor.w;
    var c = i.col.rgb * (sky * 0.9 + F.sunColor.rgb * (0.25 + 0.75 * ndl * ndl) * 0.55);
    // thick cores swallow light
    c *= 1.0 - 0.35 * dens;
    let a = clamp(i.col.a * dens * 1.35 * df, 0.0, 1.0);
    c = applyFog(c, i.wp);
    return vec4f(c * a, a);
  }
  if (kind < 1.5) {
    // ---- flame: noise scrolling upward through a teardrop
    let y = i.uv.y * 0.5 + 0.5;                       // 0 at the base, 1 at the tip
    let sc = vec3f(i.uv.x * 1.9 + seed * 13.0, i.uv.y * 1.5 - t * 2.6 - seed * 9.0, t * 0.9 + seed * 3.0);
    let n = fbm3(sc);
    let lick = fbm3(sc * 2.3 + 5.0);
    let width = mix(0.85, 0.12, pow(y, 1.3));
    let d = abs(i.uv.x + (n - 0.5) * 0.7 * y) / width;
    var heat = (1.0 - smoothstep(0.35, 1.0, d)) * (1.0 - smoothstep(0.55, 1.0, y + (lick - 0.5) * 0.8));
    heat *= smoothstep(0.0, 0.12, y) * (0.55 + 0.9 * lick);
    heat = clamp(heat * i.col.a * 1.5 * df, 0.0, 1.0);
    let c = heatRamp(heat) * i.col.rgb * heat * 9.0;
    return vec4f(c, heat * 0.2);
  }
  // ---- fireball: a roiling ball that is brightest inside
  let p = vec3f(i.uv * 1.7 + seed * 23.0, t * 1.4 + seed * 4.0);
  let n = fbm3(p);
  let n2 = fbm3(p * 2.6 + 4.0);
  var heat = smoothstep(1.0, 0.05, r + (n - 0.5) * 0.95);
  heat *= 0.6 + 0.8 * n2;
  heat = clamp(heat * i.col.a * 1.4 * df, 0.0, 1.0);
  // as it cools the rim goes to dark soot
  let soot = smoothstep(0.55, 0.0, heat) * smoothstep(0.03, 0.3, heat);
  // (the faint rim adds almost nothing: no coloured halo against the sky)
  let c = heatRamp(heat) * i.col.rgb * heat * heat * 18.0;
  return vec4f(c, clamp(heat * 0.4 + soot * 0.75, 0.0, 1.0));
}
`;

// ---------------------------------------------------------------- sky
export const SKY_WGSL = /* wgsl */ `
${COMMON}
struct VOut { @builtin(position) clip: vec4f, @location(0) ndc: vec2f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut;
  o.clip = vec4f(p[vi], 0.0, 1.0);
  o.ndc = p[vi];
  return o;
}
@fragment fn fs(i: VOut) -> @location(0) vec4f {
  let dir = normalize(F.ray0.xyz + F.rayX.xyz * i.ndc.x + F.rayY.xyz * i.ndc.y);
  let h = dir.y;
  var c = skyColor(dir);
  let sd = max(dot(dir, F.sunDir.xyz), 0.0);
  c += F.sunColor.rgb * smoothstep(0.99985, 0.99995, sd) * 40.0;
  // layered clouds lit from the sun side
  if (h > 0.0) {
    let t = F.camPos.w;
    let uv = dir.xz / (h + 0.1) * 1.3 + vec2f(t * 0.006, t * 0.0025);
    let d0 = fbm2(uv) + vnoise(uv * 5.0) * 0.05 * smoothstep(0.05, 0.3, h);
    let cover = smoothstep(0.42, 0.72, d0) * smoothstep(0.0, 0.14, h);
    let toSun = normalize(F.sunDir.xz + vec2f(1e-4)) * 0.12;
    let d1 = fbm2(uv + toSun);
    let lit = clamp(0.55 + (d0 - d1) * 3.5, 0.0, 1.0);
    let cloud = mix(F.skyTop.rgb * 0.55 + vec3f(0.25), F.sunColor.rgb * 0.95 + vec3f(0.2), lit) * (0.9 + 0.5 * pow(sd, 8.0));
    c = mix(c, cloud, cover * 0.9);
  }
  return vec4f(c, 1.0);
}
`;

// ---------------------------------------------------------------- water
export const WATER_WGSL = /* wgsl */ `
${COMMON}
@group(1) @binding(0) var heightTex: texture_2d<f32>;
struct VOut { @builtin(position) clip: vec4f, @location(0) wp: vec3f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var q = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c = F.params.y * 0.5;
  let p = vec3f(c + q[vi].x * 4000.0, F.params.w, c + q[vi].y * 4000.0);
  var o: VOut;
  o.wp = p;
  o.clip = F.viewProj * vec4f(p, 1.0);
  return o;
}
fn terrainH(x: f32, z: f32) -> f32 {
  let size = F.params.y;
  let n = 256.0;
  let fx = clamp(x / size * n, 0.0, n - 0.001);
  let fz = clamp(z / size * n, 0.0, n - 0.001);
  let i = vec2i(i32(fx), i32(fz));
  let t = vec2f(fx - f32(i.x), fz - f32(i.y));
  let h00 = textureLoad(heightTex, i, 0).r;
  let h10 = textureLoad(heightTex, i + vec2i(1, 0), 0).r;
  let h01 = textureLoad(heightTex, i + vec2i(0, 1), 0).r;
  let h11 = textureLoad(heightTex, i + vec2i(1, 1), 0).r;
  return mix(mix(h00, h10, t.x), mix(h01, h11, t.x), t.y);
}
fn waves(p: vec2f, t: f32) -> f32 {
  return vnoise(p * 0.07 + vec2f(t * 0.22, t * 0.1)) * 1.0
    + vnoise(p * 0.19 - vec2f(t * 0.16, t * 0.27)) * 0.5
    + vnoise(p * 0.53 + vec2f(t * 0.35, -t * 0.3)) * 0.22
    + vnoise(p * 1.7 - vec2f(t * 0.5, t * 0.4)) * 0.08;
}
@fragment fn fs(i: VOut) -> @location(0) vec4f {
  let t = F.camPos.w;
  let p = i.wp.xz;
  let dist = distance(i.wp, F.camPos.xyz);
  let e = 0.35 + dist * 0.004;
  let w0 = waves(p, t);
  let gx = waves(p + vec2f(e, 0.0), t) - w0;
  let gz = waves(p + vec2f(0.0, e), t) - w0;
  let amp = 1.1 / (1.0 + dist * 0.01);
  let n = normalize(vec3f(-gx / e * amp, 1.0, -gz / e * amp));
  let v = normalize(F.camPos.xyz - i.wp);
  let fres = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
  let depth = F.params.w - terrainH(i.wp.x, i.wp.z);
  let deep = vec3f(0.006, 0.045, 0.085);
  let shallow = vec3f(0.035, 0.22, 0.24);
  let sun = F.sunColor.rgb;
  var body = mix(shallow, deep, smoothstep(0.0, 8.0, depth)) * sun * (0.45 + 0.55 * max(dot(n, F.sunDir.xyz), 0.0));
  // light scattered back through wave crests
  body += shallow * sun * pow(max(dot(v, -F.sunDir.xyz), 0.0), 4.0) * 0.25 * smoothstep(0.9, 1.5, w0);
  let r = reflect(-v, n);
  let sky = skyColor(vec3f(r.x, abs(r.y), r.z));
  var c = mix(body, sky, fres);
  c += sun * pow(max(dot(r, F.sunDir.xyz), 0.0), 380.0) * 14.0 * shadowFactor(i.wp, vec3f(0.0, 1.0, 0.0));
  let foam = smoothstep(0.85, 0.0, depth) * (0.5 + 0.5 * sin(t * 1.7 + depth * 7.0 + w0 * 4.0)) + smoothstep(1.62, 1.75, w0) * 0.25;
  c = mix(c, sun * 0.85, clamp(foam, 0.0, 1.0) * 0.6);
  c = applyFog(c, i.wp);
  let alpha = clamp(0.5 + depth * 0.16 + fres * 0.5, 0.0, 0.97);
  return vec4f(c, alpha);
}
`;

// ---------------------------------------------------------------- zone wall
export const ZONE_WGSL = /* wgsl */ `
${COMMON}
struct VOut { @builtin(position) clip: vec4f, @location(0) wp: vec3f, @location(1) uv: vec2f };
@vertex fn vs(@location(0) p: vec3f) -> VOut {
  let wp = vec3f(F.zone.x + p.x * F.zone.z, p.y, F.zone.y + p.z * F.zone.z);
  var o: VOut;
  o.wp = wp;
  o.clip = F.viewProj * vec4f(wp, 1.0);
  o.uv = vec2f(atan2(p.z, p.x), p.y);
  return o;
}
@fragment fn fs(i: VOut) -> @location(0) vec4f {
  let t = F.camPos.w;
  let d = distance(i.wp.xz, F.camPos.xz);
  let near = 1.0 - smoothstep(30.0, 500.0, d);
  // the moving bands fade into a flat tint far away (they would only shimmer there)
  let stripes = mix(0.5, 0.5 + 0.5 * sin(i.uv.x * F.zone.z * 0.9 + i.uv.y * 0.35 - t * 2.0), near * near);
  let base = vec3f(0.12, 0.38, 1.0) * 1.6;
  let a = (0.2 + 0.3 * stripes) * (0.3 + 0.7 * near) * F.zone.w * smoothstep(-30.0, 10.0, i.wp.y);
  return vec4f(base * (0.9 + 0.8 * stripes), a * (1.0 - smoothstep(150.0, 300.0, i.wp.y)));
}
`;

// ---------------------------------------------------------------- post
const FULLSCREEN = /* wgsl */ `
struct VOut { @builtin(position) clip: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut;
  o.clip = vec4f(p[vi], 0.0, 1.0);
  o.uv = p[vi] * vec2f(0.5, -0.5) + vec2f(0.5);
  return o;
}
`;

/** Bloom downsample (13-tap would be nicer; four bilinear taps are enough here). */
export const BLOOM_WGSL = /* wgsl */ `
${FULLSCREEN}
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
fn taps(uv: vec2f) -> vec3f {
  let ts = 1.0 / vec2f(textureDimensions(src));
  return (textureSampleLevel(src, samp, uv + ts * vec2f(-1.0, -1.0), 0.0).rgb
    + textureSampleLevel(src, samp, uv + ts * vec2f(1.0, -1.0), 0.0).rgb
    + textureSampleLevel(src, samp, uv + ts * vec2f(-1.0, 1.0), 0.0).rgb
    + textureSampleLevel(src, samp, uv + ts * vec2f(1.0, 1.0), 0.0).rgb) * 0.25;
}
@fragment fn fs_first(i: VOut) -> @location(0) vec4f {
  let c = min(taps(i.uv), vec3f(40.0));
  let l = max(c.r, max(c.g, c.b));
  // soft threshold: only the bright end of the range blooms
  let k = max(l - 1.1, 0.0) / max(l, 1e-4);
  return vec4f(c * k, 1.0);
}
@fragment fn fs_down(i: VOut) -> @location(0) vec4f {
  return vec4f(taps(i.uv), 1.0);
}
`;

/** Screen-space ambient occlusion at half resolution, from the depth buffer. */
export const AO_WGSL = /* wgsl */ `
${FRAME}
${FULLSCREEN}
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var depthTex: texture_depth_2d;
fn wpos(px: vec2i, dim: vec2f) -> vec4f {
  let d = textureLoad(depthTex, px, 0);
  let uv = (vec2f(px) + vec2f(0.5)) / dim;
  let w = F.invViewProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, max(d, 1e-6), 1.0);
  return vec4f(w.xyz / w.w, d);
}
@fragment fn fs(i: VOut) -> @location(0) vec4f {
  let dim = vec2f(textureDimensions(depthTex));
  let px = vec2i(i.uv * dim);
  let c = wpos(px, dim);
  if (c.w <= 0.0) { return vec4f(1.0); }
  let P = c.xyz;
  let toCam = F.camPos.xyz - P;
  let dist = length(toCam);
  if (dist > 90.0) { return vec4f(1.0); }
  // normal from the nearer neighbours on each axis (keeps edges clean)
  let xa = wpos(px + vec2i(1, 0), dim);
  let xb = wpos(px - vec2i(1, 0), dim);
  let ya = wpos(px + vec2i(0, 1), dim);
  let yb = wpos(px - vec2i(0, 1), dim);
  var dx = xa.xyz - P;
  if (abs(xb.w - c.w) < abs(xa.w - c.w)) { dx = P - xb.xyz; }
  var dy = ya.xyz - P;
  if (abs(yb.w - c.w) < abs(ya.w - c.w)) { dy = P - yb.xyz; }
  var N = normalize(cross(dy, dx));
  if (dot(N, toCam) < 0.0) { N = -N; }
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(N.y) > 0.9);
  let T = normalize(cross(up, N));
  let B = cross(N, T);
  // interleaved gradient noise turns the kernel per pixel
  let ang = fract(52.9829189 * fract(dot(vec2f(px), vec2f(0.06711056, 0.00583715)))) * 6.2831853;
  let ca = cos(ang);
  let sa = sin(ang);
  let radius = clamp(dist * 0.06, 0.25, 1.1);
  var occ = 0.0;
  for (var k = 0; k < 10; k++) {
    let f = (f32(k) + 0.5) / 10.0;
    let a = f32(k) * 2.399963;
    let r = sqrt(f);
    let kx = cos(a) * r;
    let ky = sin(a) * r;
    let dir = T * (kx * ca - ky * sa) + B * (kx * sa + ky * ca) + N * sqrt(max(1.0 - r * r, 0.0)) * 0.9 + N * 0.1;
    let sp = P + N * 0.03 + dir * radius * (0.35 + 0.65 * f);
    let cl = F.viewProj * vec4f(sp, 1.0);
    if (cl.w <= 0.0) { continue; }
    let ndc = cl.xyz / cl.w;
    let suv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (suv.x < 0.0 || suv.y < 0.0 || suv.x > 1.0 || suv.y > 1.0) { continue; }
    let td = textureLoad(depthTex, vec2i(suv * dim), 0);
    // reversed depth: the scene is in front of the sample when its depth is larger
    let zs = 1.0 / max(ndc.z, 1e-7);
    let zt = 1.0 / max(td, 1e-7);
    let gap = (zs - zt) / zs * dist;
    occ += step(0.02, gap) * smoothstep(radius * 2.5, radius * 0.5, gap);
  }
  let fade = 1.0 - smoothstep(60.0, 90.0, dist);
  return vec4f(1.0 - occ / 10.0 * 0.85 * fade);
}
`;

export const POST_WGSL = /* wgsl */ `
${FRAME}
${FULLSCREEN}
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var hdr: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var b0: texture_2d<f32>;
@group(0) @binding(4) var b1: texture_2d<f32>;
@group(0) @binding(5) var b2: texture_2d<f32>;
@group(0) @binding(6) var b3: texture_2d<f32>;
@group(0) @binding(7) var aoTex: texture_2d<f32>;

fn aces(x: vec3f) -> vec3f {
  // Narkowicz fit of the ACES filmic curve
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}
fn grade(c0: vec3f, bloom: vec4f) -> vec3f {
  var c = (c0 * bloom.a + bloom.rgb) * F.post.w;
  c = aces(c);
  // a touch of contrast and saturation
  let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c = mix(vec3f(l), c, 1.08);
  c = c * c * (3.0 - 2.0 * c) * 0.35 + c * 0.65;
  return pow(clamp(c, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));
}
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }

@fragment fn fs(i: VOut) -> @location(0) vec4f {
  let ts = F.post.xy;
  let uv = i.uv;
  let glow = (textureSampleLevel(b0, samp, uv, 0.0).rgb * 0.3 + textureSampleLevel(b1, samp, uv, 0.0).rgb * 0.3
    + textureSampleLevel(b2, samp, uv, 0.0).rgb * 0.25 + textureSampleLevel(b3, samp, uv, 0.0).rgb * 0.15) * F.post.z;
  // ambient occlusion (half resolution; four taps blur its sampling pattern)
  let at = ts * 1.5;
  let ao = (textureSampleLevel(aoTex, samp, uv + at * vec2f(-1.0, -1.0), 0.0).r + textureSampleLevel(aoTex, samp, uv + at * vec2f(1.0, -1.0), 0.0).r
    + textureSampleLevel(aoTex, samp, uv + at * vec2f(-1.0, 1.0), 0.0).r + textureSampleLevel(aoTex, samp, uv + at * vec2f(1.0, 1.0), 0.0).r) * 0.25;
  let bloom = vec4f(glow, ao);
  // FXAA on the graded image
  let cM = grade(textureSampleLevel(hdr, samp, uv, 0.0).rgb, bloom);
  let cNW = grade(textureSampleLevel(hdr, samp, uv + ts * vec2f(-0.5, -0.5), 0.0).rgb, bloom);
  let cNE = grade(textureSampleLevel(hdr, samp, uv + ts * vec2f(0.5, -0.5), 0.0).rgb, bloom);
  let cSW = grade(textureSampleLevel(hdr, samp, uv + ts * vec2f(-0.5, 0.5), 0.0).rgb, bloom);
  let cSE = grade(textureSampleLevel(hdr, samp, uv + ts * vec2f(0.5, 0.5), 0.0).rgb, bloom);
  let lM = luma(cM);
  let lNW = luma(cNW);
  let lNE = luma(cNE);
  let lSW = luma(cSW);
  let lSE = luma(cSE);
  let lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
  let lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
  var c = cM;
  if (lMax - lMin > max(0.04, lMax * 0.125)) {
    var dir = vec2f(-((lNW + lNE) - (lSW + lSE)), (lNW + lSW) - (lNE + lSE));
    let reduce = max((lNW + lNE + lSW + lSE) * 0.03125, 0.0078125);
    let rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcp, vec2f(-8.0), vec2f(8.0)) * ts;
    let a = 0.5 * (grade(textureSampleLevel(hdr, samp, uv + dir * (1.0 / 3.0 - 0.5), 0.0).rgb, bloom) + grade(textureSampleLevel(hdr, samp, uv + dir * (2.0 / 3.0 - 0.5), 0.0).rgb, bloom));
    let b = a * 0.5 + 0.25 * (grade(textureSampleLevel(hdr, samp, uv - dir * 0.5, 0.0).rgb, bloom) + grade(textureSampleLevel(hdr, samp, uv + dir * 0.5, 0.0).rgb, bloom));
    let lB = luma(b);
    c = select(b, a, lB < lMin || lB > lMax);
  }
  // vignette, hurt flash, a little dither against banding
  let q = uv - vec2f(0.5);
  c *= 1.0 - dot(q, q) * 0.35;
  c = mix(c, vec3f(0.6, 0.02, 0.02), clamp(F.params.z, 0.0, 1.0) * dot(q, q) * 1.6);
  let dn = fract(sin(dot(i.clip.xy, vec2f(12.9898, 78.233))) * 43758.5453);
  c += (dn - 0.5) / 255.0;
  return vec4f(c, 1.0);
}
`;

/** Mip generation for model textures. */
export const MIP_WGSL = /* wgsl */ `
${FULLSCREEN}
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment fn fs(i: VOut) -> @location(0) vec4f {
  return textureSampleLevel(src, samp, i.uv, 0.0);
}
`;
