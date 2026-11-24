// WGSL shaders. All lighting is done in linear space with a filmic tonemap.

const FRAME = /* wgsl */ `
struct Frame {
  viewProj: mat4x4f,
  lightVP: mat4x4f,
  invViewProj: mat4x4f,
  camPos: vec4f,     // xyz camera, w time
  sunDir: vec4f,     // xyz toward sun, w shadows on
  sunColor: vec4f,   // rgb, w ambient
  fogColor: vec4f,   // rgb, w density
  skyTop: vec4f,
  skyHorizon: vec4f,
  params: vec4f,     // x shadow texel, y map size, z hurt flash, w water level
  zone: vec4f,       // cx, cz, r, visible
  focus: vec4f,      // xyz camera target (player), w enabled
};
`;

const COMMON = /* wgsl */ `
${FRAME}
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var shadowMap: texture_depth_2d;
@group(0) @binding(2) var shadowSamp: sampler_comparison;

fn shadowFactor(wp: vec3f, n: vec3f) -> f32 {
  if (F.sunDir.w < 0.5) { return 1.0; }
  let lp = F.lightVP * vec4f(wp + n * 0.06, 1.0);
  let uv = lp.xy * vec2f(0.5, -0.5) + vec2f(0.5);
  let d = lp.z;
  if (uv.x < 0.002 || uv.y < 0.002 || uv.x > 0.998 || uv.y > 0.998 || d > 1.0) { return 1.0; }
  let ts = F.params.x;
  var s = 0.0;
  for (var i = -1; i <= 1; i++) {
    for (var j = -1; j <= 1; j++) {
      s += textureSampleCompareLevel(shadowMap, shadowSamp, uv + vec2f(f32(i), f32(j)) * ts, d - 0.0012);
    }
  }
  // fade out toward the edge of the shadow region
  let edge = smoothstep(0.38, 0.5, max(abs(uv.x - 0.5), abs(uv.y - 0.5)));
  return mix(s / 9.0, 1.0, edge);
}

fn lighting(albedo: vec3f, n: vec3f, wp: vec3f) -> vec3f {
  let L = F.sunDir.xyz;
  let ndl = max(dot(n, L), 0.0);
  let sh = shadowFactor(wp, n);
  let hemi = mix(vec3f(0.36, 0.33, 0.30), vec3f(0.55, 0.66, 0.85), n.y * 0.5 + 0.5);
  return albedo * (hemi * F.sunColor.w + F.sunColor.rgb * ndl * sh);
}

fn applyFog(c: vec3f, wp: vec3f) -> vec3f {
  let d = distance(wp, F.camPos.xyz);
  let f = 1.0 - exp(-pow(d * F.fogColor.w, 1.6));
  return mix(c, F.fogColor.rgb, clamp(f, 0.0, 1.0));
}

fn tonemap(c0: vec3f) -> vec3f {
  let c = c0 * 0.8;
  let a = c * (2.51 * c + 0.03);
  let b = c * (2.43 * c + 0.59) + 0.14;
  let t = clamp(a / b, vec3f(0.0), vec3f(1.0));
  return pow(t, vec3f(1.0 / 2.2));
}

fn hash2(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}
fn vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash2(i), hash2(i + vec2f(1.0, 0.0)), u.x), mix(hash2(i + vec2f(0.0, 1.0)), hash2(i + vec2f(1.0, 1.0)), u.x), u.y);
}
fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
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
  let n = normalize(i.n);
  var albedo = pow(i.col.rgb, vec3f(2.2)) * 0.9;
  let nz = vnoise(i.wp.xz * 0.35) * 0.6 + vnoise(i.wp.xz * 1.7) * 0.4;
  albedo *= 0.84 + 0.3 * nz;
  // wet sand line + underwater darkening
  let wl = F.params.w;
  albedo *= mix(0.55, 1.0, smoothstep(wl - 2.5, wl + 0.6, i.wp.y));
  var c = lighting(albedo, n, i.wp);
  c = applyFog(c, i.wp);
  return vec4f(tonemap(c), 1.0);
}

@fragment fn fs_props(i: VOut) -> @location(0) vec4f {
  let n = normalize(i.n);
  var albedo = pow(i.col.rgb, vec3f(2.2));
  if (i.col.a < 0.75) {
    // see-through foliage between the camera and the player
    if (F.focus.w > 0.5) {
      let c = F.camPos.xyz;
      let seg = F.focus.xyz - c;
      let L = length(seg);
      if (L > 0.5) {
        let dir = seg / L;
        let t = dot(i.wp - c, dir);
        if (t > 0.0 && t < L + 0.6) {
          let perp = length((i.wp - c) - dir * t);
          let radius = 0.9 + 1.4 * clamp(t / L, 0.0, 1.0);
          let px = vec2i(i.clip.xy);
          if (perp < radius * 0.65 || (perp < radius && ((px.x + px.y) & 1) == 0)) { discard; }
        }
      }
    }
    albedo *= 0.85 + 0.3 * vnoise(i.wp.xz * 2.0 + i.wp.y);
  }
  var c = lighting(albedo, n, i.wp);
  c = applyFog(c, i.wp);
  return vec4f(tonemap(c), 1.0);
}
`;

export const LIT_SHADOW_WGSL = /* wgsl */ `
${FRAME}
@group(0) @binding(0) var<uniform> F: Frame;
struct VIn { @location(0) pos: vec3f, @location(1) nrm: vec4f, @location(2) col: vec4f };
@vertex fn vs(v: VIn) -> @builtin(position) vec4f {
  return F.lightVP * vec4f(v.pos, 1.0);
}
`;

// ---------------------------------------------------------------- instanced boxes
const BOX_IO = /* wgsl */ `
struct BIn {
  @location(0) pos: vec3f,
  @location(1) nrm: vec3f,
  @location(2) ipos: vec3f,
  @location(3) iquat: vec4f,
  @location(4) iscale: vec3f,
  @location(5) icol: vec4f,
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
};
@vertex fn vs(v: BIn) -> VOut {
  var o: VOut;
  let wp = v.ipos + qrot(v.iquat, v.pos * v.iscale);
  o.wp = wp;
  o.clip = F.viewProj * vec4f(wp, 1.0);
  o.n = normalize(qrot(v.iquat, v.nrm / max(v.iscale, vec3f(1e-4))));
  o.col = v.icol;
  return o;
}
@fragment fn fs(i: VOut) -> @location(0) vec4f {
  let albedo = pow(i.col.rgb, vec3f(2.2));
  let lit = i.col.a;
  var c = lighting(albedo, normalize(i.n), i.wp) * lit + albedo * (1.0 - lit) * 3.0;
  c = applyFog(c, i.wp);
  return vec4f(tonemap(c), 1.0);
}
@fragment fn fs_glow(i: VOut) -> @location(0) vec4f {
  let albedo = pow(i.col.rgb, vec3f(2.2)) * 2.2;
  let d = distance(i.wp, F.camPos.xyz);
  let fade = exp(-d * 0.002);
  return vec4f(tonemap(albedo), i.col.a * fade);
}
@fragment fn fs_smoke(i: VOut) -> @location(0) vec4f {
  let albedo = pow(i.col.rgb, vec3f(2.2));
  var c = albedo * (0.55 + 0.45 * max(dot(normalize(i.n), F.sunDir.xyz), 0.0)) * F.sunColor.rgb;
  c = applyFog(c, i.wp);
  return vec4f(tonemap(c), i.col.a);
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
  return F.lightVP * vec4f(wp, 1.0);
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
  let w = F.invViewProj * vec4f(i.ndc, 0.5, 1.0);
  let dir = normalize(w.xyz / w.w - F.camPos.xyz);
  let h = dir.y;
  var c = mix(F.skyHorizon.rgb, F.skyTop.rgb, pow(clamp(h, 0.0, 1.0), 0.55));
  c = mix(c, F.fogColor.rgb * 0.92, smoothstep(0.02, -0.25, h));
  let sd = max(dot(dir, F.sunDir.xyz), 0.0);
  c += F.sunColor.rgb * (pow(sd, 900.0) * 6.0 + pow(sd, 24.0) * 0.25);
  // drifting clouds
  if (h > 0.0) {
    let uv = dir.xz / (h + 0.12) * 1.6 + vec2f(F.camPos.w * 0.008, F.camPos.w * 0.003);
    var n = vnoise(uv) * 0.55 + vnoise(uv * 2.3) * 0.3 + vnoise(uv * 5.1) * 0.15;
    n = smoothstep(0.52, 0.85, n) * smoothstep(0.0, 0.18, h);
    c = mix(c, vec3f(1.0, 0.99, 0.97) * (0.85 + 0.3 * sd), n * 0.85);
  }
  return vec4f(tonemap(c), 1.0);
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
@fragment fn fs(i: VOut) -> @location(0) vec4f {
  let t = F.camPos.w;
  let p = i.wp.xz;
  let e = 0.6;
  let w1 = vnoise(p * 0.08 + vec2f(t * 0.25, t * 0.11));
  let w2 = vnoise(p * 0.21 - vec2f(t * 0.17, t * 0.29));
  let gx = (vnoise((p + vec2f(e, 0.0)) * 0.08 + vec2f(t * 0.25, t * 0.11)) - w1) + (vnoise((p + vec2f(e, 0.0)) * 0.21 - vec2f(t * 0.17, t * 0.29)) - w2) * 0.5;
  let gz = (vnoise((p + vec2f(0.0, e)) * 0.08 + vec2f(t * 0.25, t * 0.11)) - w1) + (vnoise((p + vec2f(0.0, e)) * 0.21 - vec2f(t * 0.17, t * 0.29)) - w2) * 0.5;
  let n = normalize(vec3f(-gx * 1.6, 1.0, -gz * 1.6));
  let v = normalize(F.camPos.xyz - i.wp);
  let fres = pow(1.0 - max(dot(n, v), 0.0), 4.0) * 0.85 + 0.08;
  let depth = F.params.w - terrainH(i.wp.x, i.wp.z);
  let deep = vec3f(0.02, 0.12, 0.2);
  let shallow = vec3f(0.12, 0.45, 0.5);
  var c = mix(shallow, deep, smoothstep(0.0, 9.0, depth));
  let r = reflect(-v, n);
  let sky = mix(F.skyHorizon.rgb, F.skyTop.rgb, clamp(r.y, 0.0, 1.0));
  c = mix(c * (0.5 + 0.5 * max(dot(n, F.sunDir.xyz), 0.0)) * F.sunColor.rgb, sky, fres);
  c += F.sunColor.rgb * pow(max(dot(r, F.sunDir.xyz), 0.0), 220.0) * 3.0;
  let foam = smoothstep(0.9, 0.0, depth) * (0.55 + 0.45 * sin(t * 2.0 + depth * 6.0 + w1 * 5.0));
  c = mix(c, vec3f(0.95), clamp(foam, 0.0, 1.0) * 0.6);
  c = applyFog(c, i.wp);
  let alpha = clamp(0.55 + depth * 0.12 + fres * 0.3, 0.0, 0.94);
  return vec4f(tonemap(c), alpha);
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
  let stripes = 0.5 + 0.5 * sin(i.uv.x * F.zone.z * 0.9 + i.uv.y * 0.35 - t * 2.0);
  let near = 1.0 - smoothstep(30.0, 600.0, d);
  let base = vec3f(0.25, 0.55, 1.0);
  let a = (0.10 + 0.22 * stripes) * (0.35 + 0.65 * near) * F.zone.w * smoothstep(-30.0, 10.0, i.wp.y);
  return vec4f(base * (1.2 + stripes), a * (1.0 - smoothstep(150.0, 300.0, i.wp.y)));
}
`;
