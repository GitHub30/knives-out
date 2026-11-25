//! Render geometry for the static world + the minimap image.
//! Vertex layout (20 bytes): pos f32x3, normal snorm8x4, color unorm8x4 (alpha = material).

use crate::math::*;
use crate::rng::Rng;
use crate::world::*;

pub const CH: usize = 8; // chunks per side
pub const CHW: f32 = MAP / CH as f32;
// Material ids (vertex colour alpha): the shader picks its procedural surface
// detail from them. Anything below 192 is foliage (sways, fades near the camera).
pub const MAT_SOLID: u8 = 255; // plaster wall
pub const MAT_ROOF: u8 = 250;
pub const MAT_FLOOR: u8 = 245;
pub const MAT_WOOD: u8 = 240;
pub const MAT_METAL: u8 = 235;
pub const MAT_CONCRETE: u8 = 230;
pub const MAT_BARK: u8 = 225;
pub const MAT_ROCK: u8 = 220;
pub const MAT_LEAF: u8 = 128;
pub const MAP_PX: usize = 512;

pub struct Meshes {
    pub tv: Vec<u32>,
    pub ti: Vec<u32>,
    pub tchunks: Vec<f32>,
    pub pv: Vec<u32>,
    pub pchunks: Vec<f32>,
    pub map: Vec<u8>,
}

#[inline]
fn pack_n(n: V3) -> u32 {
    let q = |v: f32| ((clampf(v, -1.0, 1.0) * 127.0).round() as i32 as i8) as u8 as u32;
    q(n.x) | (q(n.y) << 8) | (q(n.z) << 16)
}
#[inline]
fn pack_c(c: (f32, f32, f32), a: u8) -> u32 {
    let q = |v: f32| (clampf(v, 0.0, 1.0) * 255.0).round() as u32;
    q(c.0) | (q(c.1) << 8) | (q(c.2) << 16) | ((a as u32) << 24)
}
#[inline]
fn rgb(c: u32) -> (f32, f32, f32) {
    (((c >> 16) & 255) as f32 / 255.0, ((c >> 8) & 255) as f32 / 255.0, (c & 255) as f32 / 255.0)
}
#[inline]
fn mix(a: (f32, f32, f32), b: (f32, f32, f32), t: f32) -> (f32, f32, f32) {
    (lerpf(a.0, b.0, t), lerpf(a.1, b.1, t), lerpf(a.2, b.2, t))
}
#[inline]
fn shade(a: (f32, f32, f32), k: f32) -> (f32, f32, f32) {
    (a.0 * k, a.1 * k, a.2 * k)
}

fn terrain_color(w: &World, x: f32, z: f32, y: f32, n: V3, var: f32) -> (f32, f32, f32) {
    let sand = rgb(0xD9C89A);
    let wet = rgb(0xA8996F);
    let deep = rgb(0x5E6F66);
    let g1 = rgb(0x5C8A38);
    let g2 = rgb(0x7FA145);
    let alpine = rgb(0x6F7A4C);
    let rock = rgb(0x8A857A);
    let road = rgb(0x9C8B6C);
    let paved = rgb(0xA49C8C);
    let slope = 1.0 - n.y;
    let mut col = if y < -0.4 {
        mix(wet, deep, smoothstep(-0.4, -9.0, y))
    } else {
        let mut g = mix(g1, g2, var);
        g = mix(g, alpine, smoothstep(36.0, 56.0, y));
        g = mix(g, rock, smoothstep(0.22, 0.42, slope));
        mix(sand, g, smoothstep(1.4, 3.2, y))
    };
    if y > 0.8 {
        let rd = w.road_dist(x, z);
        if rd < 4.8 {
            col = mix(col, road, smoothstep(4.8, 2.4, rd) * 0.85);
        }
        for t in &w.towns {
            let d = dist_xz(x, z, t.x, t.z);
            if d < t.r + 8.0 {
                col = mix(col, paved, smoothstep(t.r + 8.0, t.r - 6.0, d) * 0.75);
            }
        }
    }
    col
}

struct Soup {
    chunks: Vec<Vec<u32>>,
    bounds: Vec<[f32; 6]>,
    cur: usize,
}

impl Soup {
    fn new() -> Soup {
        Soup { chunks: vec![Vec::new(); CH * CH], bounds: vec![[f32::MAX, f32::MAX, f32::MAX, f32::MIN, f32::MIN, f32::MIN]; CH * CH], cur: 0 }
    }
    fn select(&mut self, x: f32, z: f32) {
        let cx = clampf(x / CHW, 0.0, CH as f32 - 1.0) as usize;
        let cz = clampf(z / CHW, 0.0, CH as f32 - 1.0) as usize;
        self.cur = cz * CH + cx;
    }
    fn vert(&mut self, p: V3, n: u32, c: u32) {
        let v = &mut self.chunks[self.cur];
        v.push(p.x.to_bits());
        v.push(p.y.to_bits());
        v.push(p.z.to_bits());
        v.push(n);
        v.push(c);
        let b = &mut self.bounds[self.cur];
        b[0] = b[0].min(p.x);
        b[1] = b[1].min(p.y);
        b[2] = b[2].min(p.z);
        b[3] = b[3].max(p.x);
        b[4] = b[4].max(p.y);
        b[5] = b[5].max(p.z);
    }
    /// Triangle oriented to face away from `center`.
    fn tri(&mut self, a: V3, b: V3, c: V3, center: V3, col: u32) {
        let mut n = (b - a).cross(c - a);
        let centroid = (a + b + c) * (1.0 / 3.0);
        let (b, c) = if n.dot(centroid - center) < 0.0 {
            n = -n;
            (c, b)
        } else {
            (b, c)
        };
        let nn = pack_n(n.norm());
        self.vert(a, nn, col);
        self.vert(b, nn, col);
        self.vert(c, nn, col);
    }
    fn quad(&mut self, a: V3, b: V3, c: V3, d: V3, center: V3, col: u32) {
        self.tri(a, b, c, center, col);
        self.tri(a, c, d, center, col);
    }
    fn cuboid(&mut self, mn: V3, mx: V3, col: u32, bottom: bool, top_col: Option<u32>) {
        let c = (mn + mx) * 0.5;
        let (x0, y0, z0, x1, y1, z1) = (mn.x, mn.y, mn.z, mx.x, mx.y, mx.z);
        let tc = top_col.unwrap_or(col);
        self.quad(V3::new(x0, y1, z0), V3::new(x0, y1, z1), V3::new(x1, y1, z1), V3::new(x1, y1, z0), c, tc);
        if bottom {
            self.quad(V3::new(x0, y0, z0), V3::new(x1, y0, z0), V3::new(x1, y0, z1), V3::new(x0, y0, z1), c, col);
        }
        self.quad(V3::new(x1, y0, z0), V3::new(x1, y1, z0), V3::new(x1, y1, z1), V3::new(x1, y0, z1), c, col);
        self.quad(V3::new(x0, y0, z0), V3::new(x0, y0, z1), V3::new(x0, y1, z1), V3::new(x0, y1, z0), c, col);
        self.quad(V3::new(x0, y0, z1), V3::new(x1, y0, z1), V3::new(x1, y1, z1), V3::new(x0, y1, z1), c, col);
        self.quad(V3::new(x0, y0, z0), V3::new(x0, y1, z0), V3::new(x1, y1, z0), V3::new(x1, y0, z0), c, col);
    }
    /// Triangle with per-vertex normals (smooth shading), wound to face `out`.
    fn tri_n(&mut self, a: (V3, V3), b: (V3, V3), c: (V3, V3), col: u32) {
        let n = (b.0 - a.0).cross(c.0 - a.0);
        let (b, c) = if n.dot(a.1 + b.1 + c.1) < 0.0 { (c, b) } else { (b, c) };
        self.vert(a.0, pack_n(a.1), col);
        self.vert(b.0, pack_n(b.1), col);
        self.vert(c.0, pack_n(c.1), col);
    }
    /// Smooth tapered trunk / pole.
    fn trunk(&mut self, x: f32, z: f32, y0: f32, y1: f32, r0: f32, r1: f32, sides: usize, lean: (f32, f32), col: u32) {
        for k in 0..sides {
            let (a0, a1) = (k as f32 * TAU / sides as f32, (k + 1) as f32 * TAU / sides as f32);
            let n0 = V3::new(a0.cos(), 0.15, a0.sin()).norm();
            let n1 = V3::new(a1.cos(), 0.15, a1.sin()).norm();
            let b0 = V3::new(x + a0.cos() * r0, y0, z + a0.sin() * r0);
            let b1 = V3::new(x + a1.cos() * r0, y0, z + a1.sin() * r0);
            let t0 = V3::new(x + lean.0 + a0.cos() * r1, y1, z + lean.1 + a0.sin() * r1);
            let t1 = V3::new(x + lean.0 + a1.cos() * r1, y1, z + lean.1 + a1.sin() * r1);
            self.tri_n((b0, n0), (t0, n0), (t1, n1), col);
            self.tri_n((b0, n0), (t1, n1), (b1, n1), col);
        }
    }
    /// Smooth cone skirt (a pine bough layer), drooping at the rim.
    fn skirt(&mut self, x: f32, z: f32, y0: f32, y1: f32, r: f32, sides: usize, rot: f32, seed: u32, col: u32, col_in: u32) {
        let mut rng = Rng::new(seed as u64 | 1);
        let apex = V3::new(x, y1, z);
        let slope = r / (y1 - y0).max(0.1);
        let mut rim = Vec::with_capacity(sides + 1);
        for k in 0..sides {
            let a = rot + k as f32 * TAU / sides as f32;
            let rr = r * rng.range(0.82, 1.1);
            let n = V3::new(a.cos(), slope * 0.8, a.sin()).norm();
            rim.push((V3::new(x + a.cos() * rr, y0 - rng.range(0.0, 0.25) * r * 0.3, z + a.sin() * rr), n));
        }
        rim.push(rim[0]);
        let up = V3::new(0.0, 1.0, 0.0);
        for k in 0..sides {
            let (p0, p1) = (rim[k], rim[k + 1]);
            let na = (p0.1 + p1.1 + up).norm();
            self.tri_n(p0, (apex, na), p1, col);
            // underside, darker
            let c = V3::new(x, y0 + (y1 - y0) * 0.12, z);
            self.tri_n((p0.0, -up), (p1.0, -up), (c, -up), col_in);
        }
    }
    /// Smooth lumpy ellipsoid (foliage clump, boulder): a subdivided
    /// icosahedron pushed in and out by noise, with smooth normals.
    fn lump(&mut self, c: V3, rx: f32, ry: f32, rz: f32, seed: u32, rough: f32, subdiv: u32, cols: &[u32], flat_bottom: bool) {
        let t = (1.0 + 5.0f32.sqrt()) * 0.5;
        let l = (1.0 + t * t).sqrt();
        let raw = [
            (-1.0, t, 0.0), (1.0, t, 0.0), (-1.0, -t, 0.0), (1.0, -t, 0.0),
            (0.0, -1.0, t), (0.0, 1.0, t), (0.0, -1.0, -t), (0.0, 1.0, -t),
            (t, 0.0, -1.0), (t, 0.0, 1.0), (-t, 0.0, -1.0), (-t, 0.0, 1.0),
        ];
        let faces: [(usize, usize, usize); 20] = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ];
        let mut tris: Vec<[V3; 3]> = faces.iter().map(|&(a, b, cc)| {
            let v = |k: usize| V3::new(raw[k].0 / l, raw[k].1 / l, raw[k].2 / l);
            [v(a), v(b), v(cc)]
        }).collect();
        for _ in 0..subdiv {
            let mut next = Vec::with_capacity(tris.len() * 4);
            for [a, b, cc] in tris {
                let (ab, bc, ca) = ((a + b).norm(), (b + cc).norm(), (cc + a).norm());
                next.extend_from_slice(&[[a, ab, ca], [ab, b, bc], [ca, bc, cc], [ab, bc, ca]]);
            }
            tris = next;
        }
        let sf = seed as f32 * 0.137;
        // radial displacement: a few sine lobes keyed by the seed (cheap, seamless)
        let disp = |d: V3| -> f32 {
            let a = (d.x * 3.1 + sf).sin() * (d.y * 2.7 + sf * 1.7).cos();
            let b = (d.z * 4.3 - sf * 0.6).sin() * (d.x * 5.1 + d.y * 3.3).cos();
            1.0 + rough * (0.6 * a + 0.4 * b)
        };
        let place = |d: V3| -> (V3, V3) {
            let k = disp(d);
            let mut y = d.y * ry * k;
            if flat_bottom && y < -ry * 0.35 {
                y = -ry * 0.35 + (y + ry * 0.35) * 0.25;
            }
            let p = c + V3::new(d.x * rx * k, y, d.z * rz * k);
            // ellipsoid normal, bent a little by the bumps
            let n = V3::new(d.x / rx, d.y / ry, d.z / rz).norm();
            (p, n)
        };
        let mut rng = Rng::new(seed as u64 ^ 0x9E37);
        for (k, [a, b, cc]) in tris.iter().enumerate() {
            let col = cols[(k / 4 + rng.below(cols.len() as u32) as usize) % cols.len()];
            self.tri_n(place(*a), place(*b), place(*cc), col);
        }
    }
    fn finish(self) -> (Vec<u32>, Vec<f32>) {
        let mut out = Vec::new();
        let mut table = Vec::new();
        for (k, ch) in self.chunks.into_iter().enumerate() {
            let first = (out.len() / 5) as f32;
            let count = (ch.len() / 5) as f32;
            out.extend_from_slice(&ch);
            let b = self.bounds[k];
            table.push(first);
            table.push(count);
            if count > 0.0 {
                table.extend_from_slice(&b);
            } else {
                table.extend_from_slice(&[0.0; 6]);
            }
        }
        (out, table)
    }
}

pub fn build(w: &World) -> Meshes {
    // ---------------- terrain
    let mut tv = Vec::with_capacity(NV * NV * 5);
    let mut rng = Rng::new(w.seed as u64 ^ 0x5151);
    let mut vcol = vec![(0.0f32, 0.0f32, 0.0f32); NV * NV];
    for j in 0..NV {
        for i in 0..NV {
            let x = i as f32 * CELL;
            let z = j as f32 * CELL;
            let y = w.h[j * NV + i];
            let hl = w.h[j * NV + i.saturating_sub(1)];
            let hr = w.h[j * NV + (i + 1).min(N)];
            let hd = w.h[j.saturating_sub(1) * NV + i];
            let hu = w.h[(j + 1).min(N) * NV + i];
            let n = V3::new(hl - hr, 2.0 * CELL, hd - hu).norm();
            let var = clampf(0.5 + (rng.f() - 0.5) * 0.5 + ((x * 0.013).sin() * (z * 0.011).cos()) * 0.3, 0.0, 1.0);
            let col = terrain_color(w, x, z, y, n, var);
            vcol[j * NV + i] = col;
            // alpha: how much of this vertex is road or town paving (0..255)
            let mut pave = 0.0f32;
            if y > 0.8 {
                let rd = w.road_dist(x, z);
                if rd < 4.8 {
                    pave = smoothstep(4.8, 2.4, rd) * 0.85;
                }
                for t in &w.towns {
                    let d = dist_xz(x, z, t.x, t.z);
                    if d < t.r + 8.0 {
                        pave = pave.max(smoothstep(t.r + 8.0, t.r - 6.0, d) * 0.75);
                    }
                }
            }
            tv.push(x.to_bits());
            tv.push(y.to_bits());
            tv.push(z.to_bits());
            tv.push(pack_n(n));
            tv.push(pack_c(col, (pave * 255.0) as u8));
        }
    }
    let per = N / CH;
    let mut ti = Vec::with_capacity(N * N * 6);
    let mut tchunks = Vec::new();
    for cz in 0..CH {
        for cx in 0..CH {
            let start = ti.len() as f32;
            let mut ymin = f32::MAX;
            let mut ymax = f32::MIN;
            for j in cz * per..(cz + 1) * per {
                for i in cx * per..(cx + 1) * per {
                    let v00 = (j * NV + i) as u32;
                    let v10 = v00 + 1;
                    let v01 = v00 + NV as u32;
                    let v11 = v01 + 1;
                    ti.extend_from_slice(&[v00, v01, v10, v10, v01, v11]);
                    for v in [v00, v10, v01, v11] {
                        let y = w.h[v as usize];
                        ymin = ymin.min(y);
                        ymax = ymax.max(y);
                    }
                }
            }
            let count = ti.len() as f32 - start;
            tchunks.extend_from_slice(&[
                start,
                count,
                cx as f32 * CHW,
                ymin,
                cz as f32 * CHW,
                (cx + 1) as f32 * CHW,
                ymax,
                (cz + 1) as f32 * CHW,
            ]);
        }
    }

    // ---------------- props
    let mut s = Soup::new();
    for b in &w.boxes {
        s.select((b.min.x + b.max.x) * 0.5, (b.min.z + b.max.z) * 0.5);
        let c = rgb(b.col);
        if b.kind == BK_CAR {
            continue; // drawn from the car model by the client
        }
        let mat = match b.kind {
            BK_ROOF => MAT_ROOF,
            BK_FLOOR => MAT_FLOOR,
            BK_CRATE => MAT_WOOD,
            BK_CONTAINER => MAT_METAL,
            BK_CONCRETE => MAT_CONCRETE,
            _ => MAT_SOLID,
        };
        let col = pack_c(c, mat);
        match b.kind {
            BK_ROOF => s.cuboid(b.min, b.max, pack_c(shade(c, 0.8), mat), true, Some(col)),
            BK_CONTAINER => {
                s.cuboid(b.min, b.max, col, false, Some(pack_c(shade(c, 0.85), mat)));
                // corrugation ribs
                let ribs = 7;
                let along_x = b.max.x - b.min.x > b.max.z - b.min.z;
                let dark = pack_c(shade(c, 0.72), mat);
                for k in 1..ribs {
                    let t = k as f32 / ribs as f32;
                    if along_x {
                        let x = lerpf(b.min.x, b.max.x, t);
                        s.cuboid(V3::new(x - 0.06, b.min.y + 0.1, b.min.z - 0.04), V3::new(x + 0.06, b.max.y - 0.1, b.max.z + 0.04), dark, false, None);
                    } else {
                        let z = lerpf(b.min.z, b.max.z, t);
                        s.cuboid(V3::new(b.min.x - 0.04, b.min.y + 0.1, z - 0.06), V3::new(b.max.x + 0.04, b.max.y - 0.1, z + 0.06), dark, false, None);
                    }
                }
            }
            BK_CRATE => {
                s.cuboid(b.min, b.max, col, false, Some(pack_c(shade(c, 1.1), mat)));
                let dark = pack_c(shade(c, 0.7), mat);
                let e = 0.07;
                // edge planks
                s.cuboid(V3::new(b.min.x - 0.02, b.max.y - e, b.min.z - 0.02), V3::new(b.max.x + 0.02, b.max.y + 0.01, b.max.z + 0.02), dark, false, None);
                s.cuboid(V3::new(b.min.x - 0.02, b.min.y, b.min.z - 0.02), V3::new(b.max.x + 0.02, b.min.y + e, b.max.z + 0.02), dark, false, None);
            }
            _ => s.cuboid(b.min, b.max, col, false, None),
        }
    }
    for t in &w.trees {
        s.select(t.x, t.z);
        let mut r = Rng::new(t.seed as u64);
        let trunk = pack_c(rgb(0x4A3A2A), MAT_BARK);
        let rot = r.range(0.0, TAU);
        if t.kind == 0 {
            // pine: a straight trunk under layered, drooping boughs
            let top = t.y + t.h;
            s.trunk(t.x, t.z, t.y - 0.4, t.y + t.h * 0.8, 0.3, 0.07, 8, (0.0, 0.0), trunk);
            let base = [0x24482A, 0x2B5530, 0x213F28, 0x315A33];
            let ci = r.below(4) as usize;
            let c1 = pack_c(rgb(base[ci]), MAT_LEAF);
            let c2 = pack_c(shade(rgb(base[ci]), 0.55), MAT_LEAF);
            let layers = 5;
            for k in 0..layers {
                let f = k as f32 / layers as f32;
                let y0 = t.y + t.h * (0.2 + f * 0.62);
                let y1 = if k == layers - 1 { top } else { y0 + t.h * 0.3 };
                let rr = t.r * (1.0 - f * 0.78);
                s.skirt(t.x, t.z, y0, y1, rr, 9, rot + k as f32 * 0.7, t.seed ^ (k as u32 * 977), c1, c2);
            }
        } else {
            let lean = (r.range(-0.4, 0.4), r.range(-0.4, 0.4));
            s.trunk(t.x, t.z, t.y - 0.4, t.y + t.h * 0.6, 0.42, 0.2, 8, lean, trunk);
            let palettes: [[u32; 3]; 4] = [[0x3F7A2E, 0x4C8A36, 0x356627], [0x58893A, 0x669940, 0x4A7730], [0xB07E2C, 0xC28C33, 0x966A26], [0x7A9238, 0x889F40, 0x687E2E]];
            let pal = palettes[if r.chance(0.07) { 2 } else { [0usize, 1, 3][r.below(3) as usize] }];
            let cols = [pack_c(rgb(pal[0]), MAT_LEAF), pack_c(rgb(pal[1]), MAT_LEAF), pack_c(rgb(pal[2]), MAT_LEAF)];
            let c = V3::new(t.x + lean.0, t.y + t.h * 0.7, t.z + lean.1);
            s.lump(c, t.r * 1.2, t.r * 0.95, t.r * 1.2, t.seed, 0.16, 1, &cols, false);
            for k in 0..2u32 {
                let a = rot + k as f32 * 2.4 + r.range(-0.4, 0.4);
                let off = V3::new(a.cos() * t.r * 0.8, r.range(-0.4, 0.35) * t.r, a.sin() * t.r * 0.8);
                let q = r.range(0.7, 0.95);
                s.lump(c + off, t.r * q, t.r * q * 0.8, t.r * q, t.seed ^ (k + 3) * 7919, 0.18, 1, &cols, false);
            }
        }
    }
    for rk in &w.rocks {
        s.select(rk.x, rk.z);
        let g = [0x7F7D77, 0x74726D, 0x8A867D, 0x6A6965];
        let cols = [pack_c(rgb(g[0]), MAT_ROCK), pack_c(rgb(g[1]), MAT_ROCK), pack_c(rgb(g[2]), MAT_ROCK), pack_c(rgb(g[3]), MAT_ROCK)];
        s.lump(V3::new(rk.x, rk.y + rk.h * 0.22, rk.z), rk.r, rk.h * 0.78, rk.r * 0.92, rk.seed, 0.22, 2, &cols, true);
    }
    let (pv, pchunks) = s.finish();

    // ---------------- minimap
    let mut map = vec![0u8; MAP_PX * MAP_PX * 4];
    let scale = MAP / MAP_PX as f32;
    let light = V3::new(-0.6, 0.65, -0.45).norm();
    for v in 0..MAP_PX {
        for u in 0..MAP_PX {
            let x = (u as f32 + 0.5) * scale;
            let z = (v as f32 + 0.5) * scale;
            let y = w.height(x, z);
            let col = if y < WATER {
                let d = smoothstep(0.0, -12.0, y);
                mix(rgb(0x5AA6C8), rgb(0x1E4F78), d)
            } else {
                let n = w.normal(x, z);
                let i = ((u as f32 / scale) as usize).min(N);
                let j = ((v as f32 / scale) as usize).min(N);
                let _ = (i, j);
                let c = terrain_color(w, x, z, y, n, 0.5);
                let lit = clampf(0.55 + 0.6 * n.dot(light), 0.35, 1.2);
                shade(c, lit)
            };
            let o = (v * MAP_PX + u) * 4;
            map[o] = (clampf(col.0, 0.0, 1.0) * 255.0) as u8;
            map[o + 1] = (clampf(col.1, 0.0, 1.0) * 255.0) as u8;
            map[o + 2] = (clampf(col.2, 0.0, 1.0) * 255.0) as u8;
            map[o + 3] = 255;
        }
    }
    let mut plot = |x: f32, z: f32, c: (f32, f32, f32), a: f32| {
        let u = (x / scale) as i32;
        let v = (z / scale) as i32;
        if u < 0 || v < 0 || u >= MAP_PX as i32 || v >= MAP_PX as i32 {
            return;
        }
        let o = (v as usize * MAP_PX + u as usize) * 4;
        for k in 0..3 {
            let src = [c.0, c.1, c.2][k] * 255.0;
            map[o + k] = lerpf(map[o + k] as f32, src, a) as u8;
        }
    };
    for t in &w.trees {
        let c = if t.kind == 0 { rgb(0x2E5530) } else { rgb(0x3F6E33) };
        plot(t.x, t.z, c, 0.7);
        plot(t.x + scale, t.z, c, 0.4);
        plot(t.x, t.z + scale, c, 0.4);
    }
    for b in &w.boxes {
        if b.kind != BK_ROOF && b.kind != BK_CONTAINER && b.kind != BK_CONCRETE {
            continue;
        }
        let c = if b.kind == BK_CONTAINER { shade(rgb(b.col), 0.85) } else if b.kind == BK_CONCRETE { rgb(0x8E8D86) } else { shade(rgb(b.col), 1.25) };
        let mut x = b.min.x;
        while x < b.max.x {
            let mut z = b.min.z;
            while z < b.max.z {
                plot(x, z, c, 0.95);
                z += scale * 0.9;
            }
            x += scale * 0.9;
        }
    }
    Meshes { tv, ti, tchunks, pv, pchunks, map }
}
