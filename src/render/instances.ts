// Growable CPU-side instance list for the box pipelines.
// Layout per instance (44 bytes): pos f32x3, quat f32x4, scale f32x3, rgba u8x4.

import type { Q, V3 } from './math';

export const INSTANCE_STRIDE = 44;
const FLOATS = INSTANCE_STRIDE / 4;

/** Pack 0xRRGGBB + alpha byte (lit factor / opacity) into little-endian RGBA. */
export function rgba(hex: number, a = 255): number {
  return (((hex >> 16) & 255) | (((hex >> 8) & 255) << 8) | ((hex & 255) << 16) | ((a & 255) << 24)) >>> 0;
}

export class Instances {
  buf: ArrayBuffer;
  f: Float32Array;
  u: Uint32Array;
  count = 0;
  private cap: number;

  constructor(cap = 2048) {
    this.cap = cap;
    this.buf = new ArrayBuffer(cap * INSTANCE_STRIDE);
    this.f = new Float32Array(this.buf);
    this.u = new Uint32Array(this.buf);
  }

  reset() {
    this.count = 0;
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

  push(p: V3, q: Q, s: V3, color: number) {
    if (this.count >= this.cap) this.grow();
    const o = this.count * FLOATS;
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
    this.count++;
  }
}
