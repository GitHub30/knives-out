// Unit meshes for the instanced shape pipeline (pos f32x3, normal f32x3 per
// vertex, unindexed). All fit the unit cube [-0.5, 0.5]^3 so one instance
// scale works for every shape.

type V = [number, number, number];
const nrm = (v: V): V => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
};

/** Box with rounded edges: a subdivided cube pushed onto a rounded-box surface. */
export function roundedBox(bevel = 0.07, seg = 2): Float32Array {
  const out: number[] = [];
  const h = 0.5 - bevel;
  const place = (p: V): [V, V] => {
    const c: V = [Math.max(-h, Math.min(h, p[0])), Math.max(-h, Math.min(h, p[1])), Math.max(-h, Math.min(h, p[2]))];
    const d: V = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
    const l = Math.hypot(d[0], d[1], d[2]);
    if (l < 1e-6) return [p, [0, 0, 0]];
    const n: V = [d[0] / l, d[1] / l, d[2] / l];
    return [[c[0] + n[0] * bevel, c[1] + n[1] * bevel, c[2] + n[2] * bevel], n];
  };
  // grid lines: the flat middle plus `seg` steps across each bevel
  const ticks: number[] = [-0.5];
  for (let k = 1; k <= seg; k++) ticks.push(-0.5 + (bevel * k) / seg);
  for (let k = seg - 1; k >= 0; k--) ticks.push(0.5 - (bevel * k) / seg);
  const faces: [V, V, V][] = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
  ];
  for (const [n, u, v] of faces) {
    const at = (a: number, b: number): [V, V] => {
      const p: V = [n[0] * 0.5 + u[0] * a + v[0] * b, n[1] * 0.5 + u[1] * a + v[1] * b, n[2] * 0.5 + u[2] * a + v[2] * b];
      // corners of the cube are pulled in onto the rounded surface
      const [q, nn] = place(p);
      return [q, nn[0] === 0 && nn[1] === 0 && nn[2] === 0 ? n : nn];
    };
    for (let i = 0; i + 1 < ticks.length; i++) {
      for (let j = 0; j + 1 < ticks.length; j++) {
        const q = [at(ticks[i], ticks[j]), at(ticks[i + 1], ticks[j]), at(ticks[i + 1], ticks[j + 1]), at(ticks[i], ticks[j + 1])];
        for (const k of [0, 1, 2, 0, 2, 3]) out.push(...q[k][0], ...q[k][1]);
      }
    }
  }
  return new Float32Array(out);
}

/** Surface of revolution around `axis` (0 x, 1 y, 2 z) from a profile of (radius, height, normal r, normal h). */
function revolve(profile: [number, number, number, number][], sides: number, axis: 1 | 2): Float32Array {
  const out: number[] = [];
  const pt = (k: number, a: number): [V, V] => {
    const [r, h, nr, nh] = profile[k];
    const c = Math.cos(a), s = Math.sin(a);
    return axis === 1 ? [[c * r, h, s * r], nrm([c * nr, nh, s * nr])] : [[c * r, s * r, h], nrm([c * nr, s * nr, nh])];
  };
  for (let k = 0; k + 1 < profile.length; k++) {
    for (let i = 0; i < sides; i++) {
      const a0 = (i / sides) * Math.PI * 2, a1 = ((i + 1) / sides) * Math.PI * 2;
      const q = [pt(k, a0), pt(k + 1, a0), pt(k + 1, a1), pt(k, a1)];
      // winding: counter-clockwise seen from outside
      const order = axis === 1 ? [0, 1, 2, 0, 2, 3] : [0, 2, 1, 0, 3, 2];
      for (const j of order) out.push(...q[j][0], ...q[j][1]);
    }
  }
  return new Float32Array(out);
}

export function sphere(rings = 10, sides = 16): Float32Array {
  const prof: [number, number, number, number][] = [];
  for (let k = 0; k <= rings; k++) {
    const t = -Math.PI / 2 + (k / rings) * Math.PI;
    prof.push([Math.cos(t) * 0.5, Math.sin(t) * 0.5, Math.cos(t), Math.sin(t)]);
  }
  return revolve(prof, sides, 1);
}

/** Capsule along Y: rounded ends take `cap` of the height each. */
export function pill(cap = 0.22, rings = 5, sides = 12): Float32Array {
  const prof: [number, number, number, number][] = [];
  for (let k = 0; k <= rings; k++) {
    const t = -Math.PI / 2 + (k / rings) * (Math.PI / 2);
    prof.push([Math.cos(t) * 0.5, -0.5 + cap + Math.sin(t) * cap, Math.cos(t), Math.sin(t)]);
  }
  for (let k = 0; k <= rings; k++) {
    const t = (k / rings) * (Math.PI / 2);
    prof.push([Math.cos(t) * 0.5, 0.5 - cap + Math.sin(t) * cap, Math.cos(t), Math.sin(t)]);
  }
  return revolve(prof, sides, 1);
}

/** Cylinder along Z with flat, slightly chamfered ends. */
export function rod(sides = 12): Float32Array {
  const c = 0.06;
  const prof: [number, number, number, number][] = [
    [0, -0.5, 0, -1], [0.5 - c, -0.5, 0, -1], [0.5 - c, -0.5, 0.7, -0.7], [0.5, -0.5 + c, 0.7, -0.7], [0.5, -0.5 + c, 1, 0],
    [0.5, 0.5 - c, 1, 0], [0.5, 0.5 - c, 0.7, 0.7], [0.5 - c, 0.5, 0.7, 0.7], [0.5 - c, 0.5, 0, 1], [0, 0.5, 0, 1],
  ];
  return revolve(prof, sides, 2);
}
