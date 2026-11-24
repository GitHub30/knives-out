//! Render geometry for the static world + the minimap image.
//! Vertex layout (20 bytes): pos f32x3, normal snorm8x4, color unorm8x4 (alpha = material).

use crate::math::*;
use crate::rng::Rng;
use crate::world::*;

pub const CH: usize = 8; // chunks per side
pub const CHW: f32 = MAP / CH as f32;
pub const MAT_SOLID: u8 = 255;
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
    fn prism(&mut self, x: f32, z: f32, y0: f32, y1: f32, r0: f32, r1: f32, sides: usize, rot: f32, col: u32) {
        let center = V3::new(x, (y0 + y1) * 0.5, z);
        for k in 0..sides {
            let a0 = rot + k as f32 * TAU / sides as f32;
            let a1 = rot + (k + 1) as f32 * TAU / sides as f32;
            let b0 = V3::new(x + a0.cos() * r0, y0, z + a0.sin() * r0);
            let b1 = V3::new(x + a1.cos() * r0, y0, z + a1.sin() * r0);
            let t0 = V3::new(x + a0.cos() * r1, y1, z + a0.sin() * r1);
            let t1 = V3::new(x + a1.cos() * r1, y1, z + a1.sin() * r1);
            self.quad(b0, t0, t1, b1, center, col);
        }
    }
    fn cone(&mut self, x: f32, z: f32, y0: f32, y1: f32, r: f32, sides: usize, rot: f32, col: u32, col_dark: u32) {
        let apex = V3::new(x, y1, z);
        let base_c = V3::new(x, y0, z);
        let center = V3::new(x, y0 + (y1 - y0) * 0.3, z);
        for k in 0..sides {
            let a0 = rot + k as f32 * TAU / sides as f32;
            let a1 = rot + (k + 1) as f32 * TAU / sides as f32;
            let b0 = V3::new(x + a0.cos() * r, y0, z + a0.sin() * r);
            let b1 = V3::new(x + a1.cos() * r, y0, z + a1.sin() * r);
            self.tri(b0, apex, b1, center, if k % 2 == 0 { col } else { col_dark });
            self.tri(base_c, b0, b1, V3::new(x, y0 + 1.0, z), col_dark);
        }
    }
    fn blob(&mut self, c: V3, rx: f32, ry: f32, rz: f32, seed: u32, jitter: f32, cols: &[u32]) {
        let t = (1.0 + 5.0f32.sqrt()) * 0.5;
        let raw = [
            (-1.0, t, 0.0), (1.0, t, 0.0), (-1.0, -t, 0.0), (1.0, -t, 0.0),
            (0.0, -1.0, t), (0.0, 1.0, t), (0.0, -1.0, -t), (0.0, 1.0, -t),
            (t, 0.0, -1.0), (t, 0.0, 1.0), (-t, 0.0, -1.0), (-t, 0.0, 1.0),
        ];
        let faces: [(usize, usize, usize); 20] = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ];
        let mut rng = Rng::new(seed as u64);
        let l = (1.0 + t * t).sqrt();
        let mut v = [V3::ZERO; 12];
        for (k, &(x, y, z)) in raw.iter().enumerate() {
            let j = 1.0 + rng.range(-jitter, jitter);
            v[k] = c + V3::new(x / l * rx * j, y / l * ry * j, z / l * rz * j);
        }
        for (k, &(a, b, cc)) in faces.iter().enumerate() {
            let col = cols[(k + rng.below(cols.len() as u32) as usize) % cols.len()];
            self.tri(v[a], v[b], v[cc], c, col);
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
            tv.push(x.to_bits());
            tv.push(y.to_bits());
            tv.push(z.to_bits());
            tv.push(pack_n(n));
            tv.push(pack_c(col, MAT_SOLID));
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
        let col = pack_c(c, MAT_SOLID);
        match b.kind {
            BK_ROOF => s.cuboid(b.min, b.max, pack_c(shade(c, 0.8), MAT_SOLID), true, Some(col)),
            BK_CONTAINER => {
                s.cuboid(b.min, b.max, col, false, Some(pack_c(shade(c, 0.85), MAT_SOLID)));
                // corrugation ribs
                let ribs = 7;
                let along_x = b.max.x - b.min.x > b.max.z - b.min.z;
                let dark = pack_c(shade(c, 0.72), MAT_SOLID);
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
                s.cuboid(b.min, b.max, col, false, Some(pack_c(shade(c, 1.1), MAT_SOLID)));
                let dark = pack_c(shade(c, 0.7), MAT_SOLID);
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
        let trunk = pack_c(rgb(0x5E4630), MAT_SOLID);
        let rot = r.range(0.0, TAU);
        if t.kind == 0 {
            // pine
            let top = t.y + t.h;
            s.prism(t.x, t.z, t.y - 0.4, t.y + t.h * 0.35, 0.32, 0.22, 5, rot, trunk);
            let base = [0x2F5A2E, 0x356B34, 0x2A4F2C, 0x3D6B3A];
            let ci = r.below(4) as usize;
            let c1 = pack_c(rgb(base[ci]), MAT_LEAF);
            let c2 = pack_c(shade(rgb(base[ci]), 0.82), MAT_LEAF);
            let layers = 3;
            for k in 0..layers {
                let f = k as f32 / layers as f32;
                let y0 = t.y + t.h * (0.22 + f * 0.25);
                let y1 = if k == layers - 1 { top } else { t.y + t.h * (0.62 + f * 0.2) };
                let rr = t.r * (1.0 - f * 0.32);
                s.cone(t.x, t.z, y0, y1, rr, 7, rot + k as f32, c1, c2);
            }
        } else {
            s.prism(t.x, t.z, t.y - 0.4, t.y + t.h * 0.6, 0.3, 0.2, 5, rot, trunk);
            let palettes: [[u32; 3]; 4] = [[0x4F8A35, 0x5E9A3E, 0x46782F], [0x6A9A3A, 0x7BAA45, 0x5B8A33], [0xC08A2E, 0xD39A35, 0xA8742A], [0x8FA33A, 0x9DB045, 0x7E9132]];
            let pal = palettes[if r.chance(0.12) { 2 } else { r.below(4) as usize % 4 }];
            let cols = [pack_c(rgb(pal[0]), MAT_LEAF), pack_c(rgb(pal[1]), MAT_LEAF), pack_c(rgb(pal[2]), MAT_LEAF)];
            let c = V3::new(t.x, t.y + t.h * 0.7, t.z);
            s.blob(c, t.r, t.r * 0.85, t.r, t.seed, 0.18, &cols);
            if r.chance(0.6) {
                let off = V3::new(r.range(-1.0, 1.0), r.range(0.3, 1.2), r.range(-1.0, 1.0));
                s.blob(c + off, t.r * 0.7, t.r * 0.6, t.r * 0.7, t.seed ^ 77, 0.2, &cols);
            }
        }
    }
    for rk in &w.rocks {
        s.select(rk.x, rk.z);
        let g = [0x8C8A84, 0x7D7B76, 0x9A968C, 0x6F6E6A];
        let cols = [pack_c(rgb(g[0]), MAT_SOLID), pack_c(rgb(g[1]), MAT_SOLID), pack_c(rgb(g[2]), MAT_SOLID), pack_c(rgb(g[3]), MAT_SOLID)];
        s.blob(V3::new(rk.x, rk.y + rk.h * 0.25, rk.z), rk.r, rk.h * 0.75, rk.r * 0.9, rk.seed, 0.25, &cols);
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
