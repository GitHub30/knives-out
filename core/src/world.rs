//! Procedural island: heightmap, towns, buildings, props, loot spots, and all
//! collision / ray queries. Generation is fully deterministic from the seed.

use crate::math::*;
use crate::rng::{Noise, Rng};

pub const MAP: f32 = 1024.0;
pub const N: usize = 256; // cells per side
pub const NV: usize = N + 1; // vertices per side
pub const CELL: f32 = MAP / N as f32; // 4 m
pub const WATER: f32 = 0.0;
pub const GC: f32 = 16.0; // collision grid cell
pub const GN: usize = 64;
pub const STEP: f32 = 0.55;
pub const WALL_H: f32 = 3.2;

pub const BK_WALL: u8 = 0;
pub const BK_ROOF: u8 = 1;
pub const BK_FLOOR: u8 = 2;
pub const BK_CRATE: u8 = 3;
pub const BK_CONTAINER: u8 = 4;
pub const BK_CONCRETE: u8 = 5;

pub const CK_TREE: u8 = 0;
pub const CK_ROCK: u8 = 1;

pub const HIT_TERRAIN: u8 = 100;

#[derive(Clone, Copy, Debug)]
pub struct Bx {
    pub min: V3,
    pub max: V3,
    pub kind: u8,
    pub col: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Cyl {
    pub x: f32,
    pub z: f32,
    pub r: f32,
    pub y0: f32,
    pub y1: f32,
    pub kind: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct Building {
    pub x0: f32,
    pub z0: f32,
    pub x1: f32,
    pub z1: f32,
    pub y: f32,
    pub door_out: (f32, f32),
    pub door_in: (f32, f32),
    pub town: u8,
}

impl Building {
    pub fn contains(&self, x: f32, z: f32) -> bool {
        x > self.x0 && x < self.x1 && z > self.z0 && z < self.z1
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Town {
    pub x: f32,
    pub z: f32,
    pub r: f32,
    pub y: f32,
    /// 0 village, 1 industrial, 2 military
    pub kind: u8,
    pub name: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct Tree {
    pub x: f32,
    pub z: f32,
    pub y: f32,
    pub h: f32,
    pub r: f32,
    pub kind: u8,
    pub seed: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Rock {
    pub x: f32,
    pub z: f32,
    pub y: f32,
    pub r: f32,
    pub h: f32,
    pub seed: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Spot {
    pub pos: V3,
    pub tier: u8,
}

pub struct World {
    pub seed: u32,
    pub h: Vec<f32>,
    pub maxh: f32,
    pub boxes: Vec<Bx>,
    pub cyls: Vec<Cyl>,
    pub buildings: Vec<Building>,
    pub towns: Vec<Town>,
    pub roads: Vec<(f32, f32, f32, f32)>,
    pub trees: Vec<Tree>,
    pub rocks: Vec<Rock>,
    pub spots: Vec<Spot>,
    gbox: Vec<Vec<u16>>,
    gcyl: Vec<Vec<u16>>,
    pub checksum: u32,
}

#[inline]
fn cell_range(minx: f32, maxx: f32, minz: f32, maxz: f32) -> (i32, i32, i32, i32) {
    let c = |v: f32| ((v / GC).floor() as i32).clamp(0, GN as i32 - 1);
    (c(minx), c(maxx), c(minz), c(maxz))
}

const GRID_MARGIN: f32 = 0.6;

const WALL_COLS: [u32; 7] = [0xE6DDCB, 0xCDB89A, 0xA9604A, 0x93A6AF, 0xD9C47E, 0x86946E, 0xBFC6C9];
const ROOF_COLS: [u32; 5] = [0x5B4A41, 0x3E4B54, 0x70392C, 0x4B513A, 0x2F3438];
const CONTAINER_COLS: [u32; 5] = [0xA8382C, 0x2D5DA8, 0x3B7C45, 0xD5802A, 0x6E6E6E];

impl World {
    // ------------------------------------------------------------------
    // Generation
    // ------------------------------------------------------------------
    pub fn generate(seed: u32) -> World {
        let mut rng = Rng::new(seed as u64 ^ 0xA5A5_1234_0000);
        let n1 = Noise::new(seed as u64 * 7 + 1);
        let n2 = Noise::new(seed as u64 * 13 + 5);
        let n3 = Noise::new(seed as u64 * 31 + 9);

        let mut h = vec![0.0f32; NV * NV];
        for j in 0..NV {
            for i in 0..NV {
                let x = i as f32 * CELL;
                let z = j as f32 * CELL;
                let nx = x / MAP - 0.5;
                let nz = z / MAP - 0.5;
                let d = (nx * nx + nz * nz).sqrt() * 2.0;
                let en = n2.fbm(nx * 4.0 + 10.0, nz * 4.0 + 10.0, 3) * 0.25;
                let mask = 1.0 - smoothstep(0.68 + en, 0.96 + en, d);
                let hills = n1.fbm(x / 240.0, z / 240.0, 5);
                let ridge = 1.0 - n2.perlin(x / 170.0 + 37.0, z / 170.0 + 11.0).abs();
                let mtn = smoothstep(0.0, 0.5, n3.perlin(x / 430.0 + 91.0, z / 430.0 + 7.0));
                let mut y = 12.0 + hills * 26.0 + ridge * ridge * ridge * 38.0 * mtn;
                y = y.max(3.0 + hills * 4.0);
                y = y * mask + (-14.0) * (1.0 - mask);
                h[j * NV + i] = y.max(-16.0);
            }
        }

        let mut w = World {
            seed,
            h,
            maxh: 0.0,
            boxes: Vec::new(),
            cyls: Vec::new(),
            buildings: Vec::new(),
            towns: Vec::new(),
            roads: Vec::new(),
            trees: Vec::new(),
            rocks: Vec::new(),
            spots: Vec::new(),
            gbox: Vec::new(),
            gcyl: Vec::new(),
            checksum: 0,
        };

        // ---- towns
        let mut names: Vec<u8> = (0..16).collect();
        for i in (1..names.len()).rev() {
            let j = rng.below(i as u32 + 1) as usize;
            names.swap(i, j);
        }
        let mut attempts = 0;
        while w.towns.len() < 8 && attempts < 4000 {
            attempts += 1;
            let x = rng.range(140.0, MAP - 140.0);
            let z = rng.range(140.0, MAP - 140.0);
            let cy = w.height(x, z);
            if !(3.0..=34.0).contains(&cy) {
                continue;
            }
            let mut mn = cy;
            let mut mx = cy;
            for k in 0..12 {
                let a = k as f32 * TAU / 12.0;
                for rr in [25.0, 50.0, 75.0] {
                    let s = w.height(x + a.cos() * rr, z + a.sin() * rr);
                    mn = mn.min(s);
                    mx = mx.max(s);
                }
            }
            if mx - mn > 16.0 || mn < 1.5 {
                continue;
            }
            if w.towns.iter().any(|t| dist_xz(t.x, t.z, x, z) < 200.0) {
                continue;
            }
            let idx = w.towns.len();
            let kind = match idx {
                0 => 2,
                1 | 2 => 1,
                _ => 0,
            };
            let r = if kind == 0 { rng.range(46.0, 60.0) } else { rng.range(56.0, 68.0) };
            let y = ((mn + mx) * 0.5).max(2.5);
            w.towns.push(Town { x, z, r, y, kind, name: names[idx] });
        }
        for ti in 0..w.towns.len() {
            let t = w.towns[ti];
            w.flatten(t.x, t.z, t.r, t.r + 38.0, t.y);
        }

        // ---- roads: connect each town to its nearest already-connected town
        if !w.towns.is_empty() {
            let mut connected = vec![0usize];
            while connected.len() < w.towns.len() {
                let mut best = (f32::MAX, 0usize, 0usize);
                for ti in 0..w.towns.len() {
                    if connected.contains(&ti) {
                        continue;
                    }
                    for &ci in &connected {
                        let d = dist_xz(w.towns[ti].x, w.towns[ti].z, w.towns[ci].x, w.towns[ci].z);
                        if d < best.0 {
                            best = (d, ti, ci);
                        }
                    }
                }
                let a = w.towns[best.1];
                let b = w.towns[best.2];
                w.roads.push((a.x, a.z, b.x, b.z));
                connected.push(best.1);
            }
        }

        // ---- huts: lone cabins in the wilderness
        let mut huts: Vec<(f32, f32, f32)> = Vec::new();
        let mut tries = 0;
        while huts.len() < 14 && tries < 3000 {
            tries += 1;
            let x = rng.range(90.0, MAP - 90.0);
            let z = rng.range(90.0, MAP - 90.0);
            let y = w.height(x, z);
            if !(3.0..=45.0).contains(&y) {
                continue;
            }
            let slope = (w.height(x + 8.0, z) - w.height(x - 8.0, z)).abs() + (w.height(x, z + 8.0) - w.height(x, z - 8.0)).abs();
            if slope > 6.0 {
                continue;
            }
            if w.towns.iter().any(|t| dist_xz(t.x, t.z, x, z) < t.r + 70.0) {
                continue;
            }
            if huts.iter().any(|&(hx, hz, _)| dist_xz(hx, hz, x, z) < 90.0) {
                continue;
            }
            w.flatten(x, z, 9.0, 18.0, y);
            huts.push((x, z, y));
        }

        // ---- buildings & props in towns
        for ti in 0..w.towns.len() {
            let t = w.towns[ti];
            let spacing = if t.kind == 0 { 17.0 } else { 23.0 };
            let n = (t.r / spacing) as i32 + 1;
            let mut lots: Vec<(f32, f32)> = Vec::new();
            for gx in -n..=n {
                for gz in -n..=n {
                    if gx == 0 && gz == 0 {
                        continue; // plaza
                    }
                    let lx = t.x + gx as f32 * spacing + rng.range(-2.0, 2.0);
                    let lz = t.z + gz as f32 * spacing + rng.range(-2.0, 2.0);
                    if dist_xz(lx, lz, t.x, t.z) > t.r - 5.0 {
                        continue;
                    }
                    lots.push((lx, lz));
                }
            }
            for (lx, lz) in lots {
                let roll = rng.f();
                if roll < 0.18 {
                    continue;
                }
                if t.kind > 0 && roll < 0.42 {
                    w.container_stack(lx, lz, &mut rng);
                    continue;
                }
                let big = t.kind > 0 && rng.chance(0.55);
                let (mut bw, mut bd) = if big {
                    (rng.range(15.0, 20.0), rng.range(10.0, 14.0))
                } else {
                    (rng.range(7.0, 11.0), rng.range(7.0, 10.0))
                };
                if rng.chance(0.5) {
                    std::mem::swap(&mut bw, &mut bd);
                }
                let x0 = lx - bw * 0.5;
                let z0 = lz - bd * 0.5;
                let (dx, dz) = (t.x - lx, t.z - lz);
                let side = if dx.abs() > dz.abs() {
                    if dx > 0.0 { 1 } else { 3 }
                } else if dz > 0.0 {
                    2
                } else {
                    0
                };
                let tier = t.kind;
                w.try_building(x0, z0, x0 + bw, z0 + bd, side, if big { 1 } else { 0 }, ti as u8, tier, &mut rng);
            }
            // crates around town
            let ncr = 6 + rng.below(6);
            for _ in 0..ncr {
                let a = rng.range(0.0, TAU);
                let rr = rng.range(4.0, t.r - 3.0);
                let x = t.x + a.cos() * rr;
                let z = t.z + a.sin() * rr;
                if w.buildings.iter().any(|b| x > b.x0 - 2.5 && x < b.x1 + 2.5 && z > b.z0 - 2.5 && z < b.z1 + 2.5) {
                    continue;
                }
                let y = w.height(x, z) - 0.05;
                let s = 1.3;
                w.boxes.push(Bx { min: V3::new(x - s * 0.5, y, z - s * 0.5), max: V3::new(x + s * 0.5, y + s, z + s * 0.5), kind: BK_CRATE, col: 0x9C7A4B });
                if rng.chance(0.3) {
                    w.boxes.push(Bx { min: V3::new(x - s * 0.5, y + s, z - s * 0.5), max: V3::new(x + s * 0.5, y + 2.0 * s, z + s * 0.5), kind: BK_CRATE, col: 0x8E6E42 });
                }
                if rng.chance(0.5) {
                    w.spots.push(Spot { pos: V3::new(x + 1.2, y + 0.05, z), tier: t.kind });
                }
            }
            // military perimeter wall
            if t.kind == 2 {
                let r = t.r + 4.0;
                let segs = 6;
                for side in 0..4 {
                    for k in 0..segs {
                        if k == segs / 2 {
                            continue; // gate
                        }
                        let a0 = -r + 2.0 * r * k as f32 / segs as f32 + 0.6;
                        let a1 = -r + 2.0 * r * (k + 1) as f32 / segs as f32 - 0.6;
                        let (cx, cz) = match side {
                            0 => (t.x + (a0 + a1) * 0.5, t.z - r),
                            1 => (t.x + r, t.z + (a0 + a1) * 0.5),
                            2 => (t.x + (a0 + a1) * 0.5, t.z + r),
                            _ => (t.x - r, t.z + (a0 + a1) * 0.5),
                        };
                        let half = (a1 - a0) * 0.5;
                        let (hx, hz) = if side % 2 == 0 { (half, 0.3) } else { (0.3, half) };
                        let y = w.min_height_rect(cx - hx, cz - hz, cx + hx, cz + hz) - 0.3;
                        let top = w.height(cx, cz) + 2.1;
                        w.boxes.push(Bx { min: V3::new(cx - hx, y, cz - hz), max: V3::new(cx + hx, top, cz + hz), kind: BK_CONCRETE, col: 0xA7A69E });
                    }
                }
            }
        }
        for (x, z, _) in huts.clone() {
            let bw = rng.range(6.0, 7.5);
            let bd = rng.range(5.5, 7.0);
            let side = rng.below(4) as u8;
            w.try_building(x - bw * 0.5, z - bd * 0.5, x + bw * 0.5, z + bd * 0.5, side, 2, 255, 0, &mut rng);
        }

        // ---- trees
        let mut tries = 0;
        while w.trees.len() < 1700 && tries < 60000 {
            tries += 1;
            let x = rng.range(6.0, MAP - 6.0);
            let z = rng.range(6.0, MAP - 6.0);
            let y = w.height(x, z);
            if !(1.8..=70.0).contains(&y) {
                continue;
            }
            let dens = n3.fbm(x / 140.0 + 300.0, z / 140.0 + 300.0, 3);
            if rng.f() > smoothstep(-0.15, 0.25, dens) * 0.9 + 0.04 {
                continue;
            }
            let sl = (w.height(x + 2.0, z) - w.height(x - 2.0, z)).abs() + (w.height(x, z + 2.0) - w.height(x, z - 2.0)).abs();
            if sl > 3.2 {
                continue;
            }
            if w.near_structure(x, z, 6.0) || w.road_dist(x, z) < 5.0 {
                continue;
            }
            let kind = if y > 28.0 || rng.chance(0.55) { 0 } else { 1 };
            let th = rng.range(7.0, 13.0);
            let tr = rng.range(2.0, 3.4);
            w.trees.push(Tree { x, z, y, h: th, r: tr, kind, seed: rng.u32() });
            w.cyls.push(Cyl { x, z, r: 0.38, y0: y - 1.0, y1: y + th, kind: CK_TREE });
        }

        // ---- rocks
        let mut tries = 0;
        while w.rocks.len() < 240 && tries < 20000 {
            tries += 1;
            let x = rng.range(10.0, MAP - 10.0);
            let z = rng.range(10.0, MAP - 10.0);
            let y = w.height(x, z);
            if !(0.5..=80.0).contains(&y) {
                continue;
            }
            if w.near_structure(x, z, 8.0) || w.road_dist(x, z) < 6.0 {
                continue;
            }
            let r = rng.range(0.9, 3.2);
            let hh = r * rng.range(0.7, 1.3);
            w.rocks.push(Rock { x, z, y, r, h: hh, seed: rng.u32() });
            w.cyls.push(Cyl { x, z, r: r * 0.85, y0: y - 1.0, y1: y + hh * 0.85, kind: CK_ROCK });
        }

        // ---- finalize
        w.maxh = w.h.iter().cloned().fold(f32::MIN, f32::max);
        w.build_grid();
        w.checksum = w.compute_checksum();
        w
    }

    fn flatten(&mut self, cx: f32, cz: f32, r_in: f32, r_out: f32, y: f32) {
        let i0 = (((cx - r_out) / CELL).floor() as i32).max(0) as usize;
        let i1 = (((cx + r_out) / CELL).ceil() as i32).min(N as i32) as usize;
        let j0 = (((cz - r_out) / CELL).floor() as i32).max(0) as usize;
        let j1 = (((cz + r_out) / CELL).ceil() as i32).min(N as i32) as usize;
        for j in j0..=j1 {
            for i in i0..=i1 {
                let x = i as f32 * CELL;
                let z = j as f32 * CELL;
                let d = dist_xz(x, z, cx, cz);
                if d < r_out {
                    let k = smoothstep(r_out, r_in, d);
                    let v = &mut self.h[j * NV + i];
                    *v = lerpf(*v, y, k);
                }
            }
        }
    }

    pub fn road_dist(&self, x: f32, z: f32) -> f32 {
        let mut best = f32::MAX;
        for &(ax, az, bx, bz) in &self.roads {
            let (dx, dz) = (bx - ax, bz - az);
            let l2 = dx * dx + dz * dz;
            let t = if l2 > 0.0 { clampf(((x - ax) * dx + (z - az) * dz) / l2, 0.0, 1.0) } else { 0.0 };
            let d = dist_xz(x, z, ax + dx * t, az + dz * t);
            if d < best {
                best = d;
            }
        }
        best
    }

    fn near_structure(&self, x: f32, z: f32, margin: f32) -> bool {
        if self.towns.iter().any(|t| dist_xz(t.x, t.z, x, z) < t.r + margin + 4.0) {
            return true;
        }
        self.buildings.iter().any(|b| x > b.x0 - margin && x < b.x1 + margin && z > b.z0 - margin && z < b.z1 + margin)
    }

    fn min_height_rect(&self, x0: f32, z0: f32, x1: f32, z1: f32) -> f32 {
        let mut m = f32::MAX;
        for k in 0..9 {
            let fx = (k % 3) as f32 * 0.5;
            let fz = (k / 3) as f32 * 0.5;
            m = m.min(self.height(lerpf(x0, x1, fx), lerpf(z0, z1, fz)));
        }
        m
    }
    fn max_height_rect(&self, x0: f32, z0: f32, x1: f32, z1: f32) -> f32 {
        let mut m = f32::MIN;
        for k in 0..9 {
            let fx = (k % 3) as f32 * 0.5;
            let fz = (k / 3) as f32 * 0.5;
            m = m.max(self.height(lerpf(x0, x1, fx), lerpf(z0, z1, fz)));
        }
        m
    }

    fn container_stack(&mut self, x: f32, z: f32, rng: &mut Rng) {
        if self.buildings.iter().any(|b| x > b.x0 - 8.0 && x < b.x1 + 8.0 && z > b.z0 - 8.0 && z < b.z1 + 8.0) {
            return;
        }
        let along_x = rng.chance(0.5);
        let (hx, hz) = if along_x { (3.03, 1.22) } else { (1.22, 3.03) };
        let count = 1 + rng.below(3);
        for k in 0..count {
            let off = (k as f32 - (count - 1) as f32 * 0.5) * 2.7;
            let (cx, cz) = if along_x { (x, z + off) } else { (x + off, z) };
            let y = self.min_height_rect(cx - hx, cz - hz, cx + hx, cz + hz) - 0.1;
            let col = CONTAINER_COLS[rng.below(CONTAINER_COLS.len() as u32) as usize];
            self.boxes.push(Bx { min: V3::new(cx - hx, y, cz - hz), max: V3::new(cx + hx, y + 2.6, cz + hz), kind: BK_CONTAINER, col });
            if rng.chance(0.35) {
                let col2 = CONTAINER_COLS[rng.below(CONTAINER_COLS.len() as u32) as usize];
                self.boxes.push(Bx { min: V3::new(cx - hx, y + 2.6, cz - hz), max: V3::new(cx + hx, y + 5.2, cz + hz), kind: BK_CONTAINER, col: col2 });
            }
        }
        let y = self.height(x, z);
        let (sx, sz) = if along_x { (x + 4.2, z) } else { (x, z + 4.2) };
        self.spots.push(Spot { pos: V3::new(sx, y + 0.05, sz), tier: 1 });
    }

    /// kind: 0 house, 1 warehouse, 2 hut. side: door side 0=-z 1=+x 2=+z 3=-x
    #[allow(clippy::too_many_arguments)]
    fn try_building(&mut self, x0: f32, z0: f32, x1: f32, z1: f32, side: u8, kind: u8, town: u8, tier: u8, rng: &mut Rng) -> bool {
        if x0 < 20.0 || z0 < 20.0 || x1 > MAP - 20.0 || z1 > MAP - 20.0 {
            return false;
        }
        if self.buildings.iter().any(|b| x0 < b.x1 + 3.0 && x1 > b.x0 - 3.0 && z0 < b.z1 + 3.0 && z1 > b.z0 - 3.0) {
            return false;
        }
        if self.boxes.iter().any(|b| b.kind == BK_CONTAINER && x0 < b.max.x + 2.0 && x1 > b.min.x - 2.0 && z0 < b.max.z + 2.0 && z1 > b.min.z - 2.0) {
            return false;
        }
        let lo = self.min_height_rect(x0, z0, x1, z1);
        let hi = self.max_height_rect(x0, z0, x1, z1);
        if hi - lo > 0.7 || lo < 1.0 {
            return false;
        }
        let y = hi + 0.12;
        let t = 0.3;
        let wall_col = if kind == 1 { 0x8D949C } else if kind == 2 { 0x8B6A4A } else { WALL_COLS[rng.below(WALL_COLS.len() as u32) as usize] };
        let roof_col = if kind == 1 { 0x5E646C } else if kind == 2 { 0x4E3B2C } else { ROOF_COLS[rng.below(ROOF_COLS.len() as u32) as usize] };
        let floor_col = if kind == 1 { 0x7D7B76 } else { 0x9B7B58 };
        let hgt = if kind == 1 { 4.4 } else { WALL_H };

        // floor slab
        self.boxes.push(Bx { min: V3::new(x0, lo - 0.6, z0), max: V3::new(x1, y, z1), kind: BK_FLOOR, col: floor_col });

        let mut door = (0.0f32, 0.0f32, 0.0f32, 0.0f32); // door_out x,z, door_in x,z
        for s in 0..4u8 {
            let along_x = s == 0 || s == 2;
            let (a0, a1, p0, p1) = match s {
                0 => (x0, x1, z0, z0 + t),
                2 => (x0, x1, z1 - t, z1),
                1 => (z0 + t, z1 - t, x1 - t, x1),
                _ => (z0 + t, z1 - t, x0, x0 + t),
            };
            let len = a1 - a0;
            let mut opens: Vec<(f32, f32, f32, f32)> = Vec::new();
            let mut door_c = -100.0;
            if s == side {
                let dw = if kind == 1 { 1.4 } else { 0.8 };
                let c = rng.range(a0 + dw + 0.8, a1 - dw - 0.8);
                opens.push((c - dw, c + dw, 0.0, if kind == 1 { 3.0 } else { 2.3 }));
                door_c = c;
                let out = 1.6;
                door = match s {
                    0 => (c, z0 - out, c, z0 + out),
                    2 => (c, z1 + out, c, z1 - out),
                    1 => (x1 + out, c, x1 - out, c),
                    _ => (x0 - out, c, x0 + out, c),
                };
            }
            let nwin = if len > 8.0 { 2 } else if len > 4.5 { 1 } else { 0 };
            for k in 0..nwin {
                let c = a0 + len * (k as f32 + 1.0) / (nwin as f32 + 1.0) + rng.range(-0.6, 0.6);
                if (c - door_c).abs() < 2.2 || c < a0 + 1.0 || c > a1 - 1.0 {
                    continue;
                }
                if rng.chance(0.2) {
                    continue;
                }
                opens.push((c - 0.65, c + 0.65, 1.0, 2.1));
            }
            self.wall(along_x, a0, a1, p0, p1, y, hgt, &mut opens, wall_col);
        }
        // door step when the floor sits noticeably above the terrain
        let dy = y - self.height(door.0, door.1);
        if dy > 0.42 {
            let st = y - dy * 0.5;
            let (sx0, sz0, sx1, sz1) = match side {
                0 => (door.0 - 1.0, z0 - 1.4, door.0 + 1.0, z0),
                2 => (door.0 - 1.0, z1, door.0 + 1.0, z1 + 1.4),
                1 => (x1, door.1 - 1.0, x1 + 1.4, door.1 + 1.0),
                _ => (x0 - 1.4, door.1 - 1.0, x0, door.1 + 1.0),
            };
            self.boxes.push(Bx { min: V3::new(sx0, lo - 0.6, sz0), max: V3::new(sx1, st, sz1), kind: BK_FLOOR, col: 0x8A8178 });
        }
        // roof
        self.boxes.push(Bx { min: V3::new(x0 - 0.3, y + hgt, z0 - 0.3), max: V3::new(x1 + 0.3, y + hgt + 0.3, z1 + 0.3), kind: BK_ROOF, col: roof_col });

        // interior: warehouse crates / divider
        let w_ = x1 - x0;
        let d_ = z1 - z0;
        if kind == 1 {
            let nc = 2 + rng.below(3);
            for _ in 0..nc {
                let cx = rng.range(x0 + 2.0, x1 - 2.0);
                let cz = rng.range(z0 + 2.0, z1 - 2.0);
                if dist_xz(cx, cz, door.2, door.3) < 3.0 {
                    continue;
                }
                let s = 1.3;
                self.boxes.push(Bx { min: V3::new(cx - s * 0.5, y, cz - s * 0.5), max: V3::new(cx + s * 0.5, y + s, cz + s * 0.5), kind: BK_CRATE, col: 0x9C7A4B });
            }
        } else if kind == 0 && w_.max(d_) > 9.0 {
            // partial divider wall across the longer axis
            if w_ > d_ {
                let xm = (x0 + x1) * 0.5;
                if (door.2 - xm).abs() > 1.5 {
                    let mut opens = vec![(z0 + d_ * 0.5 - 0.6, z0 + d_ * 0.5 + 0.6, 0.0, 2.2)];
                    self.wall(false, z0 + t, z1 - t, xm - 0.1, xm + 0.1, y, hgt, &mut opens, wall_col);
                }
            } else {
                let zm = (z0 + z1) * 0.5;
                if (door.3 - zm).abs() > 1.5 {
                    let mut opens = vec![(x0 + w_ * 0.5 - 0.6, x0 + w_ * 0.5 + 0.6, 0.0, 2.2)];
                    self.wall(true, x0 + t, x1 - t, zm - 0.1, zm + 0.1, y, hgt, &mut opens, wall_col);
                }
            }
        }

        // loot spots
        let area = w_ * d_;
        let n = clampf(area / 22.0, 2.0, 8.0) as u32;
        for _ in 0..n {
            let px = rng.range(x0 + 0.9, x1 - 0.9);
            let pz = rng.range(z0 + 0.9, z1 - 0.9);
            self.spots.push(Spot { pos: V3::new(px, y + 0.05, pz), tier });
        }

        self.buildings.push(Building { x0, z0, x1, z1, y, door_out: (door.0, door.1), door_in: (door.2, door.3), town });
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn wall(&mut self, along_x: bool, a0: f32, a1: f32, p0: f32, p1: f32, y: f32, hgt: f32, opens: &mut [(f32, f32, f32, f32)], col: u32) {
        opens.sort_by(|a, b| a.0.total_cmp(&b.0));
        let push = |boxes: &mut Vec<Bx>, s0: f32, s1: f32, ylo: f32, yhi: f32| {
            if s1 - s0 < 0.02 || yhi - ylo < 0.02 {
                return;
            }
            let (mn, mx) = if along_x {
                (V3::new(s0, ylo, p0), V3::new(s1, yhi, p1))
            } else {
                (V3::new(p0, ylo, s0), V3::new(p1, yhi, s1))
            };
            boxes.push(Bx { min: mn, max: mx, kind: BK_WALL, col });
        };
        let mut cur = a0;
        for &(s, e, bot, top) in opens.iter() {
            let s = s.max(cur);
            if s > cur {
                push(&mut self.boxes, cur, s, y, y + hgt);
            }
            if e <= s {
                continue;
            }
            if bot > 0.0 {
                push(&mut self.boxes, s, e, y, y + bot);
            }
            if top < hgt {
                push(&mut self.boxes, s, e, y + top, y + hgt);
            }
            cur = e;
        }
        if cur < a1 {
            push(&mut self.boxes, cur, a1, y, y + hgt);
        }
    }

    fn build_grid(&mut self) {
        self.gbox = vec![Vec::new(); GN * GN];
        self.gcyl = vec![Vec::new(); GN * GN];
        for (i, b) in self.boxes.iter().enumerate() {
            let (x0, x1, z0, z1) = cell_range(b.min.x - GRID_MARGIN, b.max.x + GRID_MARGIN, b.min.z - GRID_MARGIN, b.max.z + GRID_MARGIN);
            for cz in z0..=z1 {
                for cx in x0..=x1 {
                    self.gbox[cz as usize * GN + cx as usize].push(i as u16);
                }
            }
        }
        for (i, c) in self.cyls.iter().enumerate() {
            let m = c.r + GRID_MARGIN;
            let (x0, x1, z0, z1) = cell_range(c.x - m, c.x + m, c.z - m, c.z + m);
            for cz in z0..=z1 {
                for cx in x0..=x1 {
                    self.gcyl[cz as usize * GN + cx as usize].push(i as u16);
                }
            }
        }
    }

    fn compute_checksum(&self) -> u32 {
        let mut hsh: u32 = 0x811C_9DC5;
        let mut feed = |v: u32| {
            for k in 0..4 {
                hsh ^= (v >> (k * 8)) & 0xFF;
                hsh = hsh.wrapping_mul(0x0100_0193);
            }
        };
        for v in &self.h {
            feed(v.to_bits());
        }
        for b in &self.boxes {
            feed(b.min.x.to_bits());
            feed(b.min.y.to_bits());
            feed(b.min.z.to_bits());
            feed(b.max.x.to_bits());
            feed(b.max.y.to_bits());
            feed(b.max.z.to_bits());
        }
        for c in &self.cyls {
            feed(c.x.to_bits());
            feed(c.z.to_bits());
            feed(c.r.to_bits());
        }
        for s in &self.spots {
            feed(s.pos.x.to_bits());
            feed(s.pos.z.to_bits());
        }
        hsh
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /// Terrain height, interpolated on the same triangles the renderer uses.
    pub fn height(&self, x: f32, z: f32) -> f32 {
        let fx = clampf(x / CELL, 0.0, N as f32 - 0.0001);
        let fz = clampf(z / CELL, 0.0, N as f32 - 0.0001);
        let i = fx as usize;
        let j = fz as usize;
        let tx = fx - i as f32;
        let tz = fz - j as f32;
        let h00 = self.h[j * NV + i];
        let h10 = self.h[j * NV + i + 1];
        let h01 = self.h[(j + 1) * NV + i];
        let h11 = self.h[(j + 1) * NV + i + 1];
        if tx + tz <= 1.0 {
            h00 + (h10 - h00) * tx + (h01 - h00) * tz
        } else {
            h11 + (h01 - h11) * (1.0 - tx) + (h10 - h11) * (1.0 - tz)
        }
    }

    pub fn normal(&self, x: f32, z: f32) -> V3 {
        let e = 1.0;
        let hx = self.height(x + e, z) - self.height(x - e, z);
        let hz = self.height(x, z + e) - self.height(x, z - e);
        V3::new(-hx, 2.0 * e, -hz).norm()
    }

    pub fn building_at(&self, x: f32, z: f32) -> Option<usize> {
        self.buildings.iter().position(|b| b.contains(x, z))
    }

    #[inline]
    fn first_cell(minx: f32, minz: f32, qx0: i32, qz0: i32) -> (i32, i32) {
        let (bx0, _, bz0, _) = cell_range(minx - GRID_MARGIN, minx - GRID_MARGIN, minz - GRID_MARGIN, minz - GRID_MARGIN);
        (bx0.max(qx0), bz0.max(qz0))
    }

    /// Calls `f` once per box overlapping the XZ rect (deduplicated across cells).
    #[inline]
    fn for_boxes(&self, x0: f32, z0: f32, x1: f32, z1: f32, mut f: impl FnMut(&Bx)) {
        let (cx0, cx1, cz0, cz1) = cell_range(x0, x1, z0, z1);
        for cz in cz0..=cz1 {
            for cx in cx0..=cx1 {
                for &bi in &self.gbox[cz as usize * GN + cx as usize] {
                    let b = &self.boxes[bi as usize];
                    if b.max.x < x0 || b.min.x > x1 || b.max.z < z0 || b.min.z > z1 {
                        continue;
                    }
                    let (fx, fz) = Self::first_cell(b.min.x, b.min.z, cx0, cz0);
                    if fx != cx || fz != cz {
                        continue;
                    }
                    f(b);
                }
            }
        }
    }
    #[inline]
    fn for_cyls(&self, x0: f32, z0: f32, x1: f32, z1: f32, mut f: impl FnMut(&Cyl)) {
        let (cx0, cx1, cz0, cz1) = cell_range(x0, x1, z0, z1);
        for cz in cz0..=cz1 {
            for cx in cx0..=cx1 {
                for &ci in &self.gcyl[cz as usize * GN + cx as usize] {
                    let c = &self.cyls[ci as usize];
                    if c.x + c.r < x0 || c.x - c.r > x1 || c.z + c.r < z0 || c.z - c.r > z1 {
                        continue;
                    }
                    let (fx, fz) = Self::first_cell(c.x - c.r, c.z - c.r, cx0, cz0);
                    if fx != cx || fz != cz {
                        continue;
                    }
                    f(c);
                }
            }
        }
    }

    /// Highest walkable surface under (x,z) not higher than feet+STEP.
    pub fn ground_below(&self, x: f32, z: f32, feet: f32, r: f32) -> f32 {
        let mut g = self.height(x, z);
        let rr = r * 0.6;
        let lim = feet + STEP;
        self.for_boxes(x - rr, z - rr, x + rr, z + rr, |b| {
            if b.max.y <= lim && b.max.y > g {
                g = b.max.y;
            }
        });
        self.for_cyls(x - 0.1, z - 0.1, x + 0.1, z + 0.1, |c| {
            if c.kind == CK_ROCK && c.y1 <= lim && c.y1 > g && dist_xz(x, z, c.x, c.z) < c.r * 0.9 {
                g = c.y1;
            }
        });
        g
    }

    /// Lowest ceiling (box bottom) above the head, if any.
    pub fn ceiling(&self, x: f32, z: f32, feet: f32, hgt: f32, r: f32) -> Option<f32> {
        let mut c: Option<f32> = None;
        let lim = feet + hgt * 0.75;
        self.for_boxes(x - r, z - r, x + r, z + r, |b| {
            if b.min.y >= lim && c.map_or(true, |v| b.min.y < v) {
                c = Some(b.min.y);
            }
        });
        c
    }

    /// Push a vertical cylinder body (feet at p.y) out of walls/trees/rocks.
    pub fn collide_body(&self, p: &mut V3, r: f32, hgt: f32) -> bool {
        let mut hit = false;
        for _ in 0..3 {
            let mut moved = false;
            let feet = p.y;
            let q = r + 0.05;
            let (px, pz) = (p.x, p.z);
            let mut dx_acc = 0.0f32;
            let mut dz_acc = 0.0f32;
            self.for_boxes(px - q, pz - q, px + q, pz + q, |b| {
                if b.max.y <= feet + STEP || b.min.y >= feet + hgt * 0.75 {
                    return;
                }
                let cx = px + dx_acc;
                let cz = pz + dz_acc;
                let qx = clampf(cx, b.min.x, b.max.x);
                let qz = clampf(cz, b.min.z, b.max.z);
                let dx = cx - qx;
                let dz = cz - qz;
                let d2 = dx * dx + dz * dz;
                if d2 >= r * r {
                    return;
                }
                if d2 > 1e-10 {
                    let d = d2.sqrt();
                    let push = r - d;
                    dx_acc += dx / d * push;
                    dz_acc += dz / d * push;
                } else {
                    // center inside the box: exit along the shallowest axis
                    let l = cx - b.min.x;
                    let rr_ = b.max.x - cx;
                    let bk = cz - b.min.z;
                    let fr = b.max.z - cz;
                    let m = l.min(rr_).min(bk).min(fr);
                    if m == l {
                        dx_acc -= l + r;
                    } else if m == rr_ {
                        dx_acc += rr_ + r;
                    } else if m == bk {
                        dz_acc -= bk + r;
                    } else {
                        dz_acc += fr + r;
                    }
                }
                moved = true;
            });
            self.for_cyls(px - q - 3.5, pz - q - 3.5, px + q + 3.5, pz + q + 3.5, |c| {
                if c.y1 <= feet + STEP || c.y0 >= feet + hgt {
                    return;
                }
                let cx = px + dx_acc;
                let cz = pz + dz_acc;
                let dx = cx - c.x;
                let dz = cz - c.z;
                let rr_ = r + c.r;
                let d2 = dx * dx + dz * dz;
                if d2 >= rr_ * rr_ {
                    return;
                }
                let d = d2.sqrt().max(1e-4);
                let (nx, nz) = if d2 > 1e-8 { (dx / d, dz / d) } else { (1.0, 0.0) };
                let push = rr_ - d;
                dx_acc += nx * push;
                dz_acc += nz * push;
                moved = true;
            });
            p.x += dx_acc;
            p.z += dz_acc;
            if !moved {
                break;
            }
            hit = true;
        }
        hit
    }

    fn ray_terrain(&self, o: V3, d: V3, maxt: f32) -> Option<f32> {
        let mut t = 0.0f32;
        let start = o.y - self.height(o.x, o.z);
        if start < -0.3 {
            return Some(0.0);
        }
        let mut guard = 0;
        while t < maxt && guard < 4000 {
            guard += 1;
            let p = o + d * t;
            let above = p.y - self.height(p.x, p.z);
            if p.y > self.maxh + 1.0 && d.y >= 0.0 {
                return None;
            }
            if p.x < -60.0 || p.z < -60.0 || p.x > MAP + 60.0 || p.z > MAP + 60.0 {
                return None;
            }
            let step = (above * 0.45).max(0.6);
            let nt = (t + step).min(maxt);
            let q = o + d * nt;
            if q.y - self.height(q.x, q.z) < 0.0 {
                let (mut a, mut b) = (t, nt);
                for _ in 0..10 {
                    let m = (a + b) * 0.5;
                    let r = o + d * m;
                    if r.y - self.height(r.x, r.z) < 0.0 {
                        b = m
                    } else {
                        a = m
                    }
                }
                return Some(a);
            }
            if nt >= maxt {
                break;
            }
            t = nt;
        }
        None
    }

    /// Ray vs boxes/cylinders via 2D DDA over the collision grid.
    fn ray_objects(&self, o: V3, d: V3, maxt: f32) -> Option<(f32, V3, u8)> {
        let mut best = maxt;
        let mut res: Option<(V3, u8)> = None;
        let mut cx = (o.x / GC).floor() as i32;
        let mut cz = (o.z / GC).floor() as i32;
        let sx: i32 = if d.x > 0.0 { 1 } else { -1 };
        let sz: i32 = if d.z > 0.0 { 1 } else { -1 };
        let tdx = if d.x.abs() > 1e-9 { GC / d.x.abs() } else { f32::INFINITY };
        let tdz = if d.z.abs() > 1e-9 { GC / d.z.abs() } else { f32::INFINITY };
        let mut tmx = if d.x > 1e-9 {
            ((cx + 1) as f32 * GC - o.x) / d.x
        } else if d.x < -1e-9 {
            (cx as f32 * GC - o.x) / d.x
        } else {
            f32::INFINITY
        };
        let mut tmz = if d.z > 1e-9 {
            ((cz + 1) as f32 * GC - o.z) / d.z
        } else if d.z < -1e-9 {
            (cz as f32 * GC - o.z) / d.z
        } else {
            f32::INFINITY
        };
        for _ in 0..(GN * 2 + 8) {
            if cx >= 0 && cz >= 0 && (cx as usize) < GN && (cz as usize) < GN {
                let ci = cz as usize * GN + cx as usize;
                for &bi in &self.gbox[ci] {
                    let b = &self.boxes[bi as usize];
                    if let Some((t, ax)) = ray_aabb(o, d, b.min, b.max, best) {
                        if t < best {
                            best = t;
                            let n = match ax {
                                0 => V3::new(-d.x.signum(), 0.0, 0.0),
                                1 => V3::new(0.0, -d.y.signum(), 0.0),
                                _ => V3::new(0.0, 0.0, -d.z.signum()),
                            };
                            res = Some((n, b.kind));
                        }
                    }
                }
                for &ii in &self.gcyl[ci] {
                    let c = &self.cyls[ii as usize];
                    if let Some(t) = ray_cyl(o, d, c.x, c.z, c.r, c.y0, c.y1, best) {
                        if t < best {
                            best = t;
                            let p = o + d * t;
                            let n = V3::new(p.x - c.x, 0.0, p.z - c.z).norm();
                            res = Some((n, 10 + c.kind));
                        }
                    }
                }
            }
            let exit = tmx.min(tmz);
            if best <= exit || exit > maxt {
                break;
            }
            if tmx < tmz {
                cx += sx;
                tmx += tdx;
            } else {
                cz += sz;
                tmz += tdz;
            }
            if (cx < 0 && sx < 0) || (cx >= GN as i32 && sx > 0) || (cz < 0 && sz < 0) || (cz >= GN as i32 && sz > 0) {
                break;
            }
        }
        res.map(|(n, k)| (best, n, k))
    }

    /// First hit along a normalized ray: (distance, normal, kind).
    pub fn raycast(&self, o: V3, d: V3, maxt: f32) -> Option<(f32, V3, u8)> {
        let obj = self.ray_objects(o, d, maxt);
        let lim = obj.map_or(maxt, |h| h.0);
        if let Some(t) = self.ray_terrain(o, d, lim) {
            let p = o + d * t;
            return Some((t, self.normal(p.x, p.z), HIT_TERRAIN));
        }
        obj
    }

    /// True if the segment a->b is obstructed.
    pub fn blocked(&self, a: V3, b: V3) -> bool {
        let dv = b - a;
        let l = dv.len();
        if l < 1e-4 {
            return false;
        }
        self.raycast(a, dv * (1.0 / l), l - 0.05).is_some()
    }
}
