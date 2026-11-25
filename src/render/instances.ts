// Growable CPU-side instance lists for the instanced shape pipelines.
// Layout per instance (48 bytes): pos f32x3, quat f32x4, scale f32x3, rgba u8x4,
// material u8x4 (roughness, metallic, unused, unused).
// Every shape has its own list so each can be drawn with one call.

import type { Q, V3 } from './math';

export const INSTANCE_STRIDE = 48;
const WORDS = INSTANCE_STRIDE / 4;

/** Unit meshes an instance can use. */
export const SHAPE_BOX = 0; // bevelled box
export const SHAPE_SPHERE = 1; // ellipsoid
export const SHAPE_PILL = 2; // capsule along Y
export const SHAPE_ROD = 3; // cylinder along Z
export const SHAPE_COUNT = 4;
/** Shapes registered at run time (model parts) follow the built-in ones. */
export const MAX_SHAPES = 32;

/** Pack 0xRRGGBB + alpha byte (lit factor / opacity) into little-endian RGBA. */
export function rgba(hex: number, a = 255): number {
  return (((hex >> 16) & 255) | (((hex >> 8) & 255) << 8) | ((hex & 255) << 16) | ((a & 255) << 24)) >>> 0;
}

/** Surface: roughness and metallic in 0..1. */
export function surf(rough: number, metal = 0): number {
  return ((Math.round(rough * 255) & 255) | ((Math.round(metal * 255) & 255) << 8)) >>> 0;
}
export const SURF_DEFAULT = surf(0.85, 0);

class List {
  buf: ArrayBuffer;
  f: Float32Array;
  u: Uint32Array;
  count = 0;
  private cap: number;

  constructor(cap: number) {
    this.cap = cap;
    this.buf = new ArrayBuffer(cap * INSTANCE_STRIDE);
    this.f = new Float32Array(this.buf);
    this.u = new Uint32Array(this.buf);
  }

  private grow() {
    const cap = this.cap * 2;
    const nb = new ArrayBuffer(cap * INSTANCE_STRIDE);
    new Uint8Array(nb).set(new Uint8Array(this.buf));
    this.buf = nb;
    this.f = new Float32Array(nb);
    this.u = new Uint32Array(nb);
    this.cap = cap;
  }

  push(p: V3, q: Q, s: V3, color: number, material: number) {
    if (this.count >= this.cap) this.grow();
    const o = this.count * WORDS;
    const f = this.f;
    f[o] = p[0];
    f[o + 1] = p[1];
    f[o + 2] = p[2];
    f[o + 3] = q[0];
    f[o + 4] = q[1];
    f[o + 5] = q[2];
    f[o + 6] = q[3];
    f[o + 7] = s[0];
    f[o + 8] = s[1];
    f[o + 9] = s[2];
    this.u[o + 10] = color;
    this.u[o + 11] = material;
    this.count++;
  }
}

export class Instances {
  readonly lists: List[] = [];

  constructor(cap = 1024) {
    for (let k = 0; k < MAX_SHAPES; k++) this.lists.push(new List(k === 0 ? cap : k < SHAPE_COUNT ? 256 : 32));
  }

  reset() {
    for (const l of this.lists) l.count = 0;
  }

  get count() {
    let n = 0;
    for (const l of this.lists) n += l.count;
    return n;
  }

  push(p: V3, q: Q, s: V3, color: number, shape = SHAPE_BOX, material = SURF_DEFAULT) {
    this.lists[shape].push(p, q, s, color, material);
  }
}
