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
/// A parked car (drawn from a model by the client; two collision boxes).
pub const BK_CAR: u8 = 6;
/// Decoration only (`World::deco`): painted trim and metalwork.
pub const BK_TRIM: u8 = 7;
pub const BK_METAL: u8 = 8;
/// Solid but not drawn as a box: the steps under a pitched roof, furniture
/// (both are drawn from their own shapes).
pub const BK_HIDDEN: u8 = 9;
/// Wall finishes besides lap siding (`BK_WALL`).
pub const BK_BRICK: u8 = 11;
pub const BK_STUCCO: u8 = 12;
/// The hull of a helicopter (ray hits only: it is not a box of the world).
pub const BK_HELI: u8 = 13;
/// A door leaf: solid while shut, drawn by the client (it swings).
pub const BK_DOOR: u8 = 14;

/// A pitched roof as it is drawn (it is solid as a stack of `BK_HIDDEN` steps).
#[derive(Clone, Copy, Debug)]
pub struct Roof {
    pub x0: f32,
    pub z0: f32,
    pub x1: f32,
    pub z1: f32,
    /// Height of the eaves and of the ridge above them.
    pub y: f32,
    pub rise: f32,
    /// The ridge runs along x.
    pub along_x: bool,
    pub col: u32,
    pub wall_col: u32,
}
/// Storey height: walls plus the slab above them.
pub const STOREY: f32 = WALL_H + 0.3;
/// Car half width, half length and height (m).
pub const CAR_HW: f32 = 0.9;
pub const CAR_HL: f32 = 2.45;
pub const CAR_H: f32 = 1.45;
/// Vehicle kinds: 0 = police car, 1 = fire truck.
pub const VK_CAR: u8 = 0;
pub const VK_TRUCK: u8 = 1;
/// Half width, half length and height per vehicle kind.
pub const CAR_DIMS: [(f32, f32, f32); 2] = [(CAR_HW, CAR_HL, CAR_H), (1.25, 3.05, 3.0)];

/// Helicopter hull: half width over the side pods, half length, and the top
/// of the cabin / of the rear pylon above the model origin (on the ground).
pub const HELI_HW: f32 = 1.9;
pub const HELI_HL: f32 = 7.75;
pub const HELI_H: f32 = 3.5;
pub const HELI_TOP: f32 = 5.5;
/// The hull as boxes in its own frame (x = right, y up, z = nose): the
/// cabin with its pods, the rear pylon and the forward pylon.
pub const HELI_BOXES: [(V3, V3); 3] = [
    (V3::new(-HELI_HW, 0.3, -HELI_HL), V3::new(HELI_HW, HELI_H, HELI_HL)),
    (V3::new(-0.7, HELI_H, -7.6), V3::new(0.7, HELI_TOP, -3.6)),
    (V3::new(-0.7, HELI_H, 3.2), V3::new(0.7, 4.7, 6.2)),
];

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

/// A hinged door in a doorway. Shut, its box is part of the world; open, the
/// box is taken out of the grid and the leaf stands along `open` (not solid).
#[derive(Clone, Copy, Debug)]
pub struct Door {
    pub bx: u32,
    /// hinge (x, floor y, z), leaf width and height
    pub hinge: V3,
    pub w: f32,
    pub h: f32,
    /// heading of the leaf from the hinge when shut and when open
    pub shut: f32,
    pub open: f32,
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
    /// Storeys (1 = a bungalow or warehouse).
    pub floors: u8,
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
    /// Parked cars: x, y (ground), z, yaw (a multiple of a quarter turn).
    /// Drawn but not solid: window and door frames, sills, pipes, bands.
    /// Loose props (crates and containers): indices into `boxes`. They can be
    /// knocked about; `move_box` keeps the collision grid in step.
    pub props: Vec<u32>,
    /// Doors, and which of them stand open (all shut as generated).
    pub doors: Vec<Door>,
    pub door_open: Vec<bool>,
    pub deco: Vec<Bx>,
    /// Window panes (min, max): drawn as glass, not solid.
    pub glass: Vec<[f32; 6]>,
    pub roofs: Vec<Roof>,
    pub cars: Vec<[f32; 4]>,
    /// Vehicle kind of each car (never changes during a match).
    pub car_kind: Vec<u8>,
    /// Velocity (x, z) of each car as the server knows it, for collisions
    /// between vehicles. All zero in a client's copy of the world, which
    /// only hears of the poses.
    pub car_vel: Vec<[f32; 2]>,
    /// Helicopters: x, y (the model origin: on the ground when parked), z,
    /// heading. Where they are now (they fly); generated parked.
    pub helis: Vec<[f32; 4]>,
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
            props: Vec::new(),
            doors: Vec::new(),
            door_open: Vec::new(),
            deco: Vec::new(),
            glass: Vec::new(),
            roofs: Vec::new(),
            cars: Vec::new(),
            car_kind: Vec::new(),
            car_vel: Vec::new(),
            helis: Vec::new(),
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
                // residential towns: some two-storey houses and a few apartment blocks
                let mut frng = Rng::new((seed as u64) << 20 ^ ((lx as u64) << 10) ^ lz as u64 ^ 0xF100);
                if !big && t.kind == 0 && frng.chance(0.2) {
                    let (aw, ad) = if bw > bd { (frng.range(17.0, 21.0), frng.range(9.0, 10.5)) } else { (frng.range(9.0, 10.5), frng.range(17.0, 21.0)) };
                    let floors = 3 + frng.below(2) as u8;
                    if w.try_building(lx - aw * 0.5, lz - ad * 0.5, lx + aw * 0.5, lz + ad * 0.5, side, 3, ti as u8, tier, floors, &mut rng) {
                        continue;
                    }
                }
                let floors = if !big && frng.chance(if t.kind == 0 { 0.5 } else { 0.25 }) { 2 } else { 1 };
                w.try_building(x0, z0, x0 + bw, z0 + bd, side, if big { 1 } else { 0 }, ti as u8, tier, floors, &mut rng);
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
                if rng.chance(0.5) && !w.spot_blocked(x + 1.2, y + 0.3, z, 0) {
                    w.spots.push(Spot { pos: V3::new(x + 1.2, y + 0.05, z), tier: t.kind });
                }
            }
            // parked cars (their own random stream, so adding them moved
            // nothing else in the world)
            let mut crng = Rng::new((seed as u64) << 8 ^ 0xCA75 ^ ti as u64);
            let want = 2 + crng.below(2) as usize;
            let mut placed = 0;
            // the police cars first, then one fire truck per town
            for n in 0..70 {
                let truck = n >= 40;
                if (!truck && placed >= want) || (truck && placed > want) {
                    continue;
                }
                if truck {
                    placed = placed.max(want);
                }
                let a = crng.range(0.0, TAU);
                let rr = crng.range(6.0, (t.r - 4.0).max(7.0));
                let (x, z) = (t.x + a.cos() * rr, t.z + a.sin() * rr);
                let quarter = crng.below(4);
                // half extents on x / z (the car is 4.9 m long, 1.8 m wide)
                let kind = if truck { VK_TRUCK } else { VK_CAR };
                let (dw, dl, _) = CAR_DIMS[kind as usize];
                let (hx, hz) = if quarter % 2 == 0 { (dw, dl) } else { (dl, dw) };
                // not on top of a car already parked here
                if w.cars.iter().any(|c| (c[0] - x).abs() < 7.0 && (c[2] - z).abs() < 7.0) {
                    continue;
                }
                let m = 1.6;
                if w.buildings.iter().any(|b| x + hx > b.x0 - m && x - hx < b.x1 + m && z + hz > b.z0 - m && z - hz < b.z1 + m) {
                    continue;
                }
                if w.boxes.iter().any(|b| x + hx > b.min.x - m && x - hx < b.max.x + m && z + hz > b.min.z - m && z - hz < b.max.z + m) {
                    continue;
                }
                let (lo, hi) = (w.min_height_rect(x - hx, z - hz, x + hx, z + hz), w.max_height_rect(x - hx, z - hz, x + hx, z + hz));
                if hi - lo > 0.25 || lo < 1.0 {
                    continue;
                }
                let y = hi;
                let yaw = quarter as f32 * (TAU / 4.0);
                // cars are dynamic (drivable): they collide through `cars`, not as boxes
                let _ = lo;
                w.cars.push([x, y, z, yaw]);
                w.car_kind.push(kind);
                w.car_vel.push([0.0; 2]);
                placed += 1;
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
            w.try_building(x - bw * 0.5, z - bd * 0.5, x + bw * 0.5, z + bd * 0.5, side, 2, 255, 0, 1, &mut rng);
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

        // ---- helicopters: two, parked on open ground out of town. Placed
        // last and from their own random stream, so nothing else moved.
        let mut hrng = Rng::new((seed as u64) << 8 ^ 0x4E11_C0);
        for n in 0..6000 {
            if w.helis.len() >= 2 {
                break;
            }
            // the demands relax as the tries run out: two are always placed
            let relax = (n / 1500) as f32;
            let clear = 25.0 - relax * 5.0;
            let x = hrng.range(110.0, MAP - 110.0);
            let z = hrng.range(110.0, MAP - 110.0);
            let yaw = hrng.range(0.0, TAU);
            let y = w.height(x, z);
            if !(3.0..=45.0).contains(&y) {
                continue;
            }
            let (lo, hi) = (w.min_height_rect(x - 12.0, z - 12.0, x + 12.0, z + 12.0), w.max_height_rect(x - 12.0, z - 12.0, x + 12.0, z + 12.0));
            if hi - lo > 2.2 + relax * 1.5 || lo < 2.0 {
                continue;
            }
            if w.towns.iter().any(|t| dist_xz(t.x, t.z, x, z) < t.r + 45.0 - relax * 10.0) || w.road_dist(x, z) < 14.0 - relax * 3.0 {
                continue;
            }
            if w.helis.iter().any(|h| dist_xz(h[0], h[2], x, z) < 260.0 - relax * 60.0) {
                continue;
            }
            if w.buildings.iter().any(|b| x > b.x0 - clear && x < b.x1 + clear && z > b.z0 - clear && z < b.z1 + clear) {
                continue;
            }
            if w.boxes.iter().any(|b| x > b.min.x - clear && x < b.max.x + clear && z > b.min.z - clear && z < b.max.z + clear) {
                continue;
            }
            if w.cyls.iter().any(|c| dist_xz(c.x, c.z, x, z) < clear + c.r) || w.cars.iter().any(|c| dist_xz(c[0], c[2], x, z) < clear) {
                continue;
            }
            // a level pad (nothing stands within its rim, so only bare
            // terrain changes)
            let y = (lo + hi) * 0.5;
            w.flatten(x, z, 10.0, (clear - 3.0).min(20.0), y);
            w.helis.push([x, w.height(x, z), z, yaw]);
        }

        // ---- finalize
        // A box placed after a loot spot (a crate of a later town) can cover
        // it: every item spawned there would sit inside the box.
        let blocked: Vec<bool> = w.spots.iter().map(|sp| w.spot_blocked(sp.pos.x, sp.pos.y + 0.3, sp.pos.z, 0)).collect();
        let mut k = 0;
        w.spots.retain(|_| {
            k += 1;
            !blocked[k - 1]
        });
        w.maxh = w.h.iter().cloned().fold(f32::MIN, f32::max);
        w.build_grid();
        w.door_open = vec![false; w.doors.len()];
        w.props = (0..w.boxes.len()).filter(|&i| w.boxes[i].kind == BK_CRATE || w.boxes[i].kind == BK_CONTAINER).map(|i| i as u32).collect();
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

    /// True if a loot spot at (x,y,z) would sit inside or right against a box
    /// (checks boxes from index `from`, the grid is not built yet).
    /// True if a small item at (x, y, z) keeps `m` metres from every solid
    /// box (walls, crates, containers, props).
    pub fn item_spot_clear(&self, x: f32, y: f32, z: f32, m: f32) -> bool {
        let mut clear = true;
        self.for_boxes(x - m, z - m, x + m, z + m, |b| {
            if b.kind != BK_FLOOR && x > b.min.x - m && x < b.max.x + m && z > b.min.z - m && z < b.max.z + m && y > b.min.y - 0.2 && y < b.max.y + 0.2 {
                clear = false;
            }
        });
        clear
    }

    fn spot_blocked(&self, x: f32, y: f32, z: f32, from: usize) -> bool {
        let m = 0.35;
        self.boxes[from..].iter().any(|b| {
            b.kind != BK_FLOOR && x > b.min.x - m && x < b.max.x + m && z > b.min.z - m && z < b.max.z + m && y > b.min.y - 0.2 && y < b.max.y + 0.2
        })
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
        let (sx, sz) = if along_x { (x + 4.2, z) } else { (x, z + 4.2) };
        let y = self.height(sx, sz);
        if !self.spot_blocked(sx, y + 0.3, sz, 0) {
            self.spots.push(Spot { pos: V3::new(sx, y + 0.05, sz), tier: 1 });
        }
    }

    /// kind: 0 house, 1 warehouse, 2 hut. side: door side 0=-z 1=+x 2=+z 3=-x
    #[allow(clippy::too_many_arguments)]
    /// kind: 0 house, 1 warehouse, 2 cabin, 3 apartment block.
    #[allow(clippy::too_many_arguments)]
    fn try_building(&mut self, x0: f32, z0: f32, x1: f32, z1: f32, side: u8, kind: u8, town: u8, tier: u8, floors: u8, rng: &mut Rng) -> bool {
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
        let apt = kind == 3;
        let wall_col = if kind == 1 { 0x8D949C } else if kind == 2 { 0x8B6A4A } else if apt { [0xD9D4C8, 0xC9CED2, 0xD8C9B4, 0xBFC6BE][rng.below(4) as usize] } else { WALL_COLS[rng.below(WALL_COLS.len() as u32) as usize] };
        let roof_col = if kind == 1 { 0x5E646C } else if kind == 2 { 0x4E3B2C } else if apt { 0x6A6D70 } else { ROOF_COLS[rng.below(ROOF_COLS.len() as u32) as usize] };
        let floor_col = if kind == 1 || apt { 0x7D7B76 } else { 0x9B7B58 };
        let hgt = if kind == 1 { 4.4 } else { WALL_H };
        // the stairs need room: a long side of 7.6 m or more
        let floors = if (x1 - x0).max(z1 - z0) < 7.6 || kind == 1 || kind == 2 { 1 } else { floors.clamp(1, 4) };
        // houses: lap siding, brick or stucco
        let wall_kind = if apt { BK_CONCRETE } else if kind == 0 { [BK_WALL, BK_BRICK, BK_STUCCO, BK_WALL][rng.below(4) as usize] } else { BK_WALL };
        let wall_col = if wall_kind == BK_BRICK { [0x9A4A38, 0x8A4634, 0xA65C40, 0x7C4A3A][rng.below(4) as usize] } else { wall_col };
        let trim_col = if apt { 0x3E4348 } else { 0xECE9E2 };

        let first_box = self.boxes.len();
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
            self.wall(along_x, a0, a1, p0, p1, y, hgt, &mut opens, wall_col, wall_kind);
            if s == side && kind != 1 {
                // the door leaf: hinged at the low end of the opening, swinging outwards
                let (d0, d1, top) = (door_c - 0.8, door_c + 0.8, 2.3);
                let pc = (p0 + p1) * 0.5;
                let (mn, mx, hinge) = if along_x {
                    (V3::new(d0, y, pc - 0.04), V3::new(d1, y + top, pc + 0.04), V3::new(d0, y, pc))
                } else {
                    (V3::new(pc - 0.04, y, d0), V3::new(pc + 0.04, y + top, d1), V3::new(pc, y, d0))
                };
                let shut = if along_x { core::f32::consts::FRAC_PI_2 } else { 0.0 };
                let open = match s {
                    0 => core::f32::consts::PI,
                    2 => 0.0,
                    1 => core::f32::consts::FRAC_PI_2,
                    _ => -core::f32::consts::FRAC_PI_2,
                };
                let col = [0x6B4A2E, 0x7A3B2A, 0x2F4F5E, 0xD8D2C4][(door_c * 3.7) as usize % 4];
                self.doors.push(Door { bx: self.boxes.len() as u32, hinge, w: d1 - d0, h: top, shut, open, col });
                self.boxes.push(Bx { min: mn, max: mx, kind: BK_DOOR, col });
            }
            for &(s0, s1, bot, top) in opens.iter() {
                self.frame(along_x, s0, s1, y + bot, y + top, p0, p1, bot <= 0.0, trim_col);
            }
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
        // ---- upper storeys: a dog-leg staircase, a slab with the stairwell cut out, walls
        let along = x1 - x0 >= z1 - z0; // the stairs run along x
        let (a_lo, _a_hi, p_lo, p_hi) = if along { (x0 + t, x1 - t, z0 + t, z1 - t) } else { (z0 + t, z1 - t, x0 + t, x1 - t) };
        const STEPS: usize = 14;
        let (run, sw) = (0.3f32, 1.15f32);
        let rl = run * STEPS as f32;
        // against the side wall away from the door
        let far_side = if along { side == 0 } else { side == 3 };
        let a_s = a_lo + 1.35;
        let lane = |k: usize| if far_side { (p_hi - sw * (k + 1) as f32, p_hi - sw * k as f32) } else { (p_lo + sw * k as f32, p_lo + sw * (k + 1) as f32) };
        let rect = |a0: f32, a1: f32, q0: f32, q1: f32, ylo: f32, yhi: f32| {
            if along { (V3::new(a0, ylo, q0), V3::new(a1, yhi, q1)) } else { (V3::new(q0, ylo, a0), V3::new(q1, yhi, a1)) }
        };
        for f in 1..floors as usize {
            let yf = y + STOREY * f as f32;
            let below = yf - STOREY;
            // the flight up to this floor: lanes alternate and so does the direction
            let (l0, l1) = lane((f - 1) % 2);
            let up = (f - 1) % 2 == 0;
            for k in 0..STEPS {
                let (s0, s1) = if up { (a_s + run * k as f32, a_s + run * (k + 1) as f32) } else { (a_s + rl - run * (k + 1) as f32, a_s + rl - run * k as f32) };
                let top = below + STOREY * (k + 1) as f32 / STEPS as f32;
                let (mn, mx) = rect(s0, s1, l0, l1, below, top);
                self.boxes.push(Bx { min: mn, max: mx, kind: BK_FLOOR, col: 0x8A8178 });
            }
            // slab around the stairwell
            let (h0, h1) = (a_s, a_s + rl);
            let (xa, xb, za, zb) = if along { (x0, x1, z0, z1) } else { (z0, z1, x0, x1) };
            for (a0, a1, q0, q1) in [(xa, h0, za, zb), (h1, xb, za, zb), (h0, h1, za, l0), (h0, h1, l1, zb)] {
                if a1 - a0 > 0.01 && q1 - q0 > 0.01 {
                    let (mn, mx) = rect(a0, a1, q0, q1, yf - 0.3, yf);
                    self.boxes.push(Bx { min: mn, max: mx, kind: BK_FLOOR, col: floor_col });
                }
            }
            // walls of this storey; an apartment block opens onto balconies at the front
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
                let n = ((len - 1.5) / if apt { 3.2 } else { 4.0 }).floor().max(1.0) as usize;
                for k in 0..n {
                    let c = a0 + len * (k as f32 + 0.5) / n as f32;
                    if apt && s == side && k % 2 == 0 {
                        opens.push((c - 0.5, c + 0.5, 0.0, 2.15));
                    } else {
                        opens.push((c - 0.7, c + 0.7, 0.95, 2.15));
                    }
                }
                self.wall(along_x, a0, a1, p0, p1, yf, hgt, &mut opens, wall_col, wall_kind);
                for &(s0, s1, bot, top) in opens.iter() {
                    self.frame(along_x, s0, s1, yf + bot, yf + top, p0, p1, bot <= 0.0, trim_col);
                }
            }
            if apt {
                // balcony along the front: slab, solid guard and end walls
                let (bw, g) = (1.35f32, 0.14f32);
                let (mn, mx) = match side {
                    0 => (V3::new(x0, yf - 0.25, z0 - bw), V3::new(x1, yf, z0)),
                    2 => (V3::new(x0, yf - 0.25, z1), V3::new(x1, yf, z1 + bw)),
                    1 => (V3::new(x1, yf - 0.25, z0), V3::new(x1 + bw, yf, z1)),
                    _ => (V3::new(x0 - bw, yf - 0.25, z0), V3::new(x0, yf, z1)),
                };
                self.boxes.push(Bx { min: mn, max: mx, kind: BK_CONCRETE, col: 0xB9B7B0 });
                let (gl, gh) = (yf, yf + 1.05);
                let guards = match side {
                    0 => [(mn.x, mn.z, mx.x, mn.z + g), (mn.x, mn.z, mn.x + g, mx.z), (mx.x - g, mn.z, mx.x, mx.z)],
                    2 => [(mn.x, mx.z - g, mx.x, mx.z), (mn.x, mn.z, mn.x + g, mx.z), (mx.x - g, mn.z, mx.x, mx.z)],
                    1 => [(mx.x - g, mn.z, mx.x, mx.z), (mn.x, mn.z, mx.x, mn.z + g), (mn.x, mx.z - g, mx.x, mx.z)],
                    _ => [(mn.x, mn.z, mn.x + g, mx.z), (mn.x, mn.z, mx.x, mn.z + g), (mn.x, mx.z - g, mx.x, mx.z)],
                };
                for (gx0, gz0, gx1, gz1) in guards {
                    self.boxes.push(Bx { min: V3::new(gx0, gl, gz0), max: V3::new(gx1, gh, gz1), kind: BK_CONCRETE, col: wall_col });
                }
                // an air-conditioner on every other balcony bay
                let (cx, cz) = ((mn.x + mx.x) * 0.5, (mn.z + mx.z) * 0.5);
                let half = if side == 0 || side == 2 { (x1 - x0) * 0.5 } else { (z1 - z0) * 0.5 };
                for k in 0..3 {
                    let o = (k as f32 - 1.0) * (half - 1.1);
                    let (ax, az) = if side == 0 || side == 2 { (cx + o, cz) } else { (cx, cz + o) };
                    self.deco.push(Bx { min: V3::new(ax - 0.4, yf, az - 0.4), max: V3::new(ax + 0.4, yf + 0.6, az + 0.4), kind: BK_METAL, col: 0xD4D6D2 });
                }
            }
        }
        let top = y + STOREY * (floors as f32 - 1.0) + hgt;
        // ---- outside trim: plinth, a band at every floor line, downpipes, a canopy over the door
        {
            let e = 0.06;
            let band = if apt { 0x9FA29F } else { 0xD8D4CA };
            self.deco.push(Bx { min: V3::new(x0 - e, y - 0.3, z0 - e), max: V3::new(x1 + e, y + 0.35, z1 + e), kind: BK_CONCRETE, col: 0x8C8A84 });
            for f in 1..floors as usize {
                let yf = y + STOREY * f as f32;
                self.deco.push(Bx { min: V3::new(x0 - e, yf - 0.32, z0 - e), max: V3::new(x1 + e, yf - 0.1, z1 + e), kind: BK_CONCRETE, col: band });
            }
            if floors > 1 || apt {
                for (px, pz) in [(x0 - 0.1, z0 - 0.1), (x1 + 0.1, z1 + 0.1)] {
                    self.deco.push(Bx { min: V3::new(px - 0.05, y, pz - 0.05), max: V3::new(px + 0.05, top, pz + 0.05), kind: BK_METAL, col: 0x5A5E63 });
                }
            }
            let (cx, cz) = (door.0, door.1);
            let (mn, mx) = match side {
                0 => (V3::new(cx - 1.2, y + 2.45, z0 - 1.0), V3::new(cx + 1.2, y + 2.57, z0)),
                2 => (V3::new(cx - 1.2, y + 2.45, z1), V3::new(cx + 1.2, y + 2.57, z1 + 1.0)),
                1 => (V3::new(x1, y + 2.45, cz - 1.2), V3::new(x1 + 1.0, y + 2.57, cz + 1.2)),
                _ => (V3::new(x0 - 1.0, y + 2.45, cz - 1.2), V3::new(x0, y + 2.57, cz + 1.2)),
            };
            if kind != 1 && kind != 2 {
                self.deco.push(Bx { min: mn, max: mx, kind: BK_CONCRETE, col: band });
            }
        }
        // roof (an apartment block has a parapet and plant on it)
        self.boxes.push(Bx { min: V3::new(x0 - 0.3, top, z0 - 0.3), max: V3::new(x1 + 0.3, top + 0.3, z1 + 0.3), kind: BK_ROOF, col: roof_col });
        if kind == 0 || kind == 2 {
            // a gable roof: solid as a stack of low steps (so it can be walked
            // and landed on), drawn as two smooth slopes
            let (lo, hi) = if along { (z0 - 0.3, z1 + 0.3) } else { (x0 - 0.3, x1 + 0.3) };
            let half = (hi - lo) * 0.5;
            let rise = half * 0.42;
            let n = (rise / 0.25).ceil().max(1.0) as usize;
            let (r, run) = (rise / n as f32, half / n as f32);
            for k in 0..n {
                let (q0, q1) = (lo + run * k as f32, hi - run * k as f32);
                let (ylo, yhi) = (top + 0.3 + r * k as f32, top + 0.3 + r * (k + 1) as f32);
                let (mn, mx) = if along { (V3::new(x0 - 0.3, ylo, q0), V3::new(x1 + 0.3, yhi, q1)) } else { (V3::new(q0, ylo, z0 - 0.3), V3::new(q1, yhi, z1 + 0.3)) };
                self.boxes.push(Bx { min: mn, max: mx, kind: BK_HIDDEN, col: roof_col });
            }
            self.roofs.push(Roof { x0: x0 - 0.3, z0: z0 - 0.3, x1: x1 + 0.3, z1: z1 + 0.3, y: top + 0.3, rise, along_x: along, col: roof_col, wall_col });
        }
        if apt {
            let (rt, ph, pt) = (top + 0.3, 0.8f32, 0.2f32);
            for (px0, pz0, px1, pz1) in [(x0 - 0.3, z0 - 0.3, x1 + 0.3, z0 - 0.3 + pt), (x0 - 0.3, z1 + 0.3 - pt, x1 + 0.3, z1 + 0.3), (x0 - 0.3, z0 - 0.3, x0 - 0.3 + pt, z1 + 0.3), (x1 + 0.3 - pt, z0 - 0.3, x1 + 0.3, z1 + 0.3)] {
                self.boxes.push(Bx { min: V3::new(px0, rt, pz0), max: V3::new(px1, rt + ph, pz1), kind: BK_CONCRETE, col: wall_col });
            }
            let (cx, cz) = ((x0 + x1) * 0.5, (z0 + z1) * 0.5);
            self.boxes.push(Bx { min: V3::new(cx - 1.4, rt, cz - 1.1), max: V3::new(cx + 1.4, rt + 1.9, cz + 1.1), kind: BK_CONCRETE, col: 0xA9A8A2 });
            self.boxes.push(Bx { min: V3::new(cx + 3.0, rt, cz - 0.6), max: V3::new(cx + 4.2, rt + 1.1, cz + 0.6), kind: BK_CONTAINER, col: 0x8F979C });
        }

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
        } else if kind == 0 && floors == 1 && w_.max(d_) > 9.0 {
            // partial divider wall across the longer axis
            if w_ > d_ {
                let xm = (x0 + x1) * 0.5;
                if (door.2 - xm).abs() > 1.5 {
                    let mut opens = vec![(z0 + d_ * 0.5 - 0.6, z0 + d_ * 0.5 + 0.6, 0.0, 2.2)];
                    self.wall(false, z0 + t, z1 - t, xm - 0.1, xm + 0.1, y, hgt, &mut opens, wall_col, wall_kind);
                }
            } else {
                let zm = (z0 + z1) * 0.5;
                if (door.3 - zm).abs() > 1.5 {
                    let mut opens = vec![(x0 + w_ * 0.5 - 0.6, x0 + w_ * 0.5 + 0.6, 0.0, 2.2)];
                    self.wall(true, x0 + t, x1 - t, zm - 0.1, zm + 0.1, y, hgt, &mut opens, wall_col, wall_kind);
                }
            }
        }

        // furniture (houses and flats), clear of the door and the stairs
        if kind == 0 || apt {
            let stairs = if floors > 1 {
                let (q0, q1) = if far_side { (p_hi - sw * 2.0 - 0.9, p_hi) } else { (p_lo, p_lo + sw * 2.0 + 0.9) };
                let (mn, mx) = rect(a_s - 1.6, a_s + rl + 1.9, q0, q1, 0.0, 0.0);
                (mn.x, mn.z, mx.x, mx.z)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };
            for f in 0..floors as usize {
                let fy = y + STOREY * f as f32;
                let mut keep = vec![stairs];
                if f == 0 {
                    keep.push((door.2 - 2.0, door.3 - 2.0, door.2 + 2.0, door.3 + 2.0));
                } else if apt {
                    // the balcony doors
                    keep.push(match side {
                        0 => (x0, z0, x1, z0 + 1.6),
                        2 => (x0, z1 - 1.6, x1, z1),
                        1 => (x1 - 1.6, z0, x1, z1),
                        _ => (x0, z0, x0 + 1.6, z1),
                    });
                }
                self.furnish(x0 + t, z0 + t, x1 - t, z1 - t, fy, &keep, rng);
            }
        }

        // loot spots (never inside crates / interior walls of this building)
        let area = w_ * d_;
        let n = clampf(area / 22.0, 2.0, 8.0) as u32;
        for f in 0..floors as usize {
            // upstairs the loot is a little better and there is less of it
            let fy = y + STOREY * f as f32;
            let want = if f == 0 { n } else { (n * 2 / 3).max(2) };
            let ftier = if f > 0 { (tier + 1).min(2) } else { tier };
            let mut placed = 0;
            let mut tries = 0;
            while placed < want && tries < want * 8 {
                tries += 1;
                let px = rng.range(x0 + 0.9, x1 - 0.9);
                let pz = rng.range(z0 + 0.9, z1 - 0.9);
                if self.spot_blocked(px, fy + 0.3, pz, first_box) {
                    continue;
                }
                // not in the stairwell: there must be floor right here
                if f > 0 && !self.boxes[first_box..].iter().any(|b| b.kind == BK_FLOOR && (b.max.y - fy).abs() < 0.01 && px > b.min.x + 0.3 && px < b.max.x - 0.3 && pz > b.min.z + 0.3 && pz < b.max.z - 0.3 && b.max.y - b.min.y < 0.35) {
                    continue;
                }
                self.spots.push(Spot { pos: V3::new(px, fy + 0.05, pz), tier: ftier });
                placed += 1;
            }
        }

        self.buildings.push(Building { x0, z0, x1, z1, y, door_out: (door.0, door.1), door_in: (door.2, door.3), town, floors });
        true
    }

    #[allow(clippy::too_many_arguments)]
    /// Frame of a window or door opening: jambs, head, and for a window a
    /// sill and glazing bars. Decoration only.
    #[allow(clippy::too_many_arguments)]
    fn frame(&mut self, along_x: bool, s0: f32, s1: f32, ylo: f32, yhi: f32, p0: f32, p1: f32, door: bool, col: u32) {
        if !door {
            let pc = (p0 + p1) * 0.5;
            self.glass.push(if along_x { [s0, ylo, pc - 0.012, s1, yhi, pc + 0.012] } else { [pc - 0.012, ylo, s0, pc + 0.012, yhi, s1] });
        }
        let mut bar = |a0: f32, a1: f32, y0: f32, y1: f32, q0: f32, q1: f32, kind: u8, col: u32| {
            let (mn, mx) = if along_x { (V3::new(a0, y0, q0), V3::new(a1, y1, q1)) } else { (V3::new(q0, y0, a0), V3::new(q1, y1, a1)) };
            self.deco.push(Bx { min: mn, max: mx, kind, col });
        };
        let (w, o) = (0.07, 0.035);
        bar(s0 - w, s0, ylo, yhi + w, p0 - o, p1 + o, BK_TRIM, col);
        bar(s1, s1 + w, ylo, yhi + w, p0 - o, p1 + o, BK_TRIM, col);
        bar(s0 - w, s1 + w, yhi, yhi + w, p0 - o, p1 + o, BK_TRIM, col);
        if !door {
            bar(s0 - w - 0.04, s1 + w + 0.04, ylo - 0.06, ylo, p0 - 0.12, p1 + 0.12, BK_CONCRETE, 0xCFCBC2);
            let pm = (p0 + p1) * 0.5;
            let (sm, ym) = ((s0 + s1) * 0.5, (ylo + yhi) * 0.5);
            bar(sm - 0.02, sm + 0.02, ylo, yhi, pm - 0.025, pm + 0.025, BK_TRIM, col);
            bar(s0, s1, ym - 0.02, ym + 0.02, pm - 0.025, pm + 0.025, BK_TRIM, col);
        }
    }

    /// Furniture in the corners of a room (interior x0..x1, z0..z1, floor at
    /// `y`), skipping what would stand in a `keep` rectangle. Each piece is a
    /// simple hidden solid with its real shape as decoration.
    #[allow(clippy::too_many_arguments)]
    fn furnish(&mut self, x0: f32, z0: f32, x1: f32, z1: f32, y: f32, keep: &[(f32, f32, f32, f32)], rng: &mut Rng) {
        let wood = [0x6B4A2E, 0x7E5B3A, 0x4E3828][rng.below(3) as usize];
        let cloth = [0x55606B, 0x7A4B3A, 0x49604E, 0x6A6257][rng.below(4) as usize];
        let first = rng.below(4);
        for c in 0..4u32 {
            if rng.chance(0.3) {
                continue;
            }
            // corner c: which x wall and which z wall it sits against
            let (wx, wz) = (c % 2 == 0, c < 2);
            // long side along x or z, footprint (lx, lz) and height of the solid part
            let item = (first + c) % 4;
            let along_x = rng.chance(0.5);
            let (l, d, h) = match item {
                0 => (2.0, 1.05, 0.5),   // bed
                1 => (1.3, 0.42, 1.8),   // shelf
                2 => (1.3, 0.8, 0.76),   // table
                _ => (1.9, 0.85, 0.42),  // sofa
            };
            let (lx, lz) = if along_x { (l, d) } else { (d, l) };
            let g = 0.06;
            let bx0 = if wx { x0 + g } else { x1 - g - lx };
            let bz0 = if wz { z0 + g } else { z1 - g - lz };
            let (bx1, bz1) = (bx0 + lx, bz0 + lz);
            if bx1 > x1 - 0.8 && bx0 < x0 + 0.8 || bz1 > z1 - 0.8 && bz0 < z0 + 0.8 {
                continue; // the room is too small for it
            }
            // (a table has a chair standing out from it)
            let m = if item == 2 { 1.3 } else { 0.5 };
            if keep.iter().any(|k| bx0 < k.2 + m && bx1 > k.0 - m && bz0 < k.3 + m && bz1 > k.1 - m) {
                continue;
            }
            if self.boxes.iter().rev().take(400).any(|b| b.max.y > y + 0.05 && b.min.y < y + h && bx0 < b.max.x && bx1 > b.min.x && bz0 < b.max.z && bz1 > b.min.z) {
                continue;
            }
            let solid = |boxes: &mut Vec<Bx>, a: (f32, f32, f32), b: (f32, f32, f32)| boxes.push(Bx { min: V3::new(a.0, a.1, a.2), max: V3::new(b.0, b.1, b.2), kind: BK_HIDDEN, col: 0 });
            let mut part = |a: (f32, f32, f32), b: (f32, f32, f32), kind: u8, col: u32| self.deco.push(Bx { min: V3::new(a.0, a.1, a.2), max: V3::new(b.0, b.1, b.2), kind, col });
            // the wall end of the long side (for headboards and backrests)
            let head_lo = if along_x { wx } else { wz };
            match item {
                0 => {
                    part((bx0, y, bz0), (bx1, y + 0.28, bz1), BK_CRATE, wood);
                    part((bx0 + 0.03, y + 0.28, bz0 + 0.03), (bx1 - 0.03, y + 0.5, bz1 - 0.03), BK_TRIM, 0xE4E0D6);
                    // headboard, pillow and a blanket over the foot end
                    let (hb0, hb1, pl0, pl1, bl0, bl1) = if head_lo { (0.0, 0.08, 0.14, 0.6, 0.75, l) } else { (l - 0.08, l, l - 0.6, l - 0.14, 0.0, l - 0.75) };
                    let seg = |a: f32, b: f32, y0: f32, y1: f32, inset: f32| if along_x { ((bx0 + a, y0, bz0 + inset), (bx0 + b, y1, bz1 - inset)) } else { ((bx0 + inset, y0, bz0 + a), (bx1 - inset, y1, bz0 + b)) };
                    let (a, b) = seg(hb0, hb1, y, y + 0.95, 0.0);
                    part(a, b, BK_CRATE, wood);
                    let (a, b) = seg(pl0, pl1, y + 0.5, y + 0.62, 0.22);
                    part(a, b, BK_TRIM, 0xF2F0EA);
                    let (a, b) = seg(bl0, bl1, y + 0.5, y + 0.54, 0.0);
                    part(a, b, BK_TRIM, cloth);
                }
                1 => {
                    // carcass: sides, back, five shelves with a few books
                    part((bx0, y, bz0), (bx1, y + 0.06, bz1), BK_CRATE, wood);
                    for k in 1..=5 {
                        let sy = y + 0.36 * k as f32;
                        part((bx0, sy - 0.03, bz0), (bx1, sy, bz1), BK_CRATE, wood);
                    }
                    if along_x {
                        part((bx0, y, bz0), (bx0 + 0.04, y + 1.8, bz1), BK_CRATE, wood);
                        part((bx1 - 0.04, y, bz0), (bx1, y + 1.8, bz1), BK_CRATE, wood);
                        let zb = if wz { bz0 } else { bz1 - 0.03 };
                        part((bx0, y, zb), (bx1, y + 1.8, zb + 0.03), BK_CRATE, wood);
                        for k in 0..3 {
                            let (bx, by) = (bx0 + 0.12 + 0.36 * k as f32, y + 0.36 * (k + 1) as f32);
                            part((bx, by, bz0 + 0.08), (bx + 0.26, by + 0.24, bz1 - 0.08), BK_TRIM, [0x8A3B32, 0x33506A, 0x4E6B45][k]);
                        }
                    } else {
                        part((bx0, y, bz0), (bx1, y + 1.8, bz0 + 0.04), BK_CRATE, wood);
                        part((bx0, y, bz1 - 0.04), (bx1, y + 1.8, bz1), BK_CRATE, wood);
                        let xb = if wx { bx0 } else { bx1 - 0.03 };
                        part((xb, y, bz0), (xb + 0.03, y + 1.8, bz1), BK_CRATE, wood);
                        for k in 0..3 {
                            let (bz, by) = (bz0 + 0.12 + 0.36 * k as f32, y + 0.36 * (k + 1) as f32);
                            part((bx0 + 0.08, by, bz), (bx1 - 0.08, by + 0.24, bz + 0.26), BK_TRIM, [0x8A3B32, 0x33506A, 0x4E6B45][k]);
                        }
                    }
                }
                2 => {
                    part((bx0, y + 0.71, bz0), (bx1, y + 0.76, bz1), BK_CRATE, wood);
                    for (lx_, lz_) in [(bx0 + 0.05, bz0 + 0.05), (bx1 - 0.11, bz0 + 0.05), (bx0 + 0.05, bz1 - 0.11), (bx1 - 0.11, bz1 - 0.11)] {
                        part((lx_, y, lz_), (lx_ + 0.06, y + 0.71, lz_ + 0.06), BK_CRATE, wood);
                    }
                    // a chair on the open long side
                    let (cx, cz) = if along_x { ((bx0 + bx1) * 0.5, if wz { bz1 + 0.45 } else { bz0 - 0.45 }) } else { (if wx { bx1 + 0.45 } else { bx0 - 0.45 }, (bz0 + bz1) * 0.5) };
                    part((cx - 0.22, y + 0.42, cz - 0.22), (cx + 0.22, y + 0.46, cz + 0.22), BK_CRATE, wood);
                    for (sx, sz) in [(-0.2, -0.2), (0.16, -0.2), (-0.2, 0.16), (0.16, 0.16)] {
                        part((cx + sx, y, cz + sz), (cx + sx + 0.04, y + 0.42, cz + sz + 0.04), BK_CRATE, wood);
                    }
                    let (b0, b1) = if along_x { let zb = if wz { cz + 0.18 } else { cz - 0.22 }; ((cx - 0.22, y + 0.46, zb), (cx + 0.22, y + 0.9, zb + 0.04)) } else { let xb = if wx { cx + 0.18 } else { cx - 0.22 }; ((xb, y + 0.46, cz - 0.22), (xb + 0.04, y + 0.9, cz + 0.22)) };
                    part(b0, b1, BK_CRATE, wood);
                    solid(&mut self.boxes, (cx - 0.22, y, cz - 0.22), (cx + 0.22, y + 0.46, cz + 0.22));
                }
                _ => {
                    part((bx0, y + 0.08, bz0), (bx1, y + 0.42, bz1), BK_TRIM, cloth);
                    // backrest against the wall, arms at the ends
                    let (b0, b1) = if along_x { let zb = if wz { bz0 } else { bz1 - 0.22 }; ((bx0, y + 0.42, zb), (bx1, y + 0.86, zb + 0.22)) } else { let xb = if wx { bx0 } else { bx1 - 0.22 }; ((xb, y + 0.42, bz0), (xb + 0.22, y + 0.86, bz1)) };
                    part(b0, b1, BK_TRIM, cloth);
                    solid(&mut self.boxes, b0, b1);
                    if along_x {
                        part((bx0, y + 0.42, bz0), (bx0 + 0.16, y + 0.62, bz1), BK_TRIM, cloth);
                        part((bx1 - 0.16, y + 0.42, bz0), (bx1, y + 0.62, bz1), BK_TRIM, cloth);
                    } else {
                        part((bx0, y + 0.42, bz0), (bx1, y + 0.62, bz0 + 0.16), BK_TRIM, cloth);
                        part((bx0, y + 0.42, bz1 - 0.16), (bx1, y + 0.62, bz1), BK_TRIM, cloth);
                    }
                }
            }
            solid(&mut self.boxes, (bx0, y, bz0), (bx1, y + h, bz1));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn wall(&mut self, along_x: bool, a0: f32, a1: f32, p0: f32, p1: f32, y: f32, hgt: f32, opens: &mut [(f32, f32, f32, f32)], col: u32, kind: u8) {
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
            boxes.push(Bx { min: mn, max: mx, kind, col });
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
        for h in &self.helis {
            for v in h {
                feed(v.to_bits());
            }
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
        self.collide_body_ex(p, r, hgt, false)
    }

    /// Take box `i` out of the collision grid / put it back (around moving it).
    fn grid_cells(b: &Bx) -> (i32, i32, i32, i32) {
        cell_range(b.min.x - GRID_MARGIN, b.max.x + GRID_MARGIN, b.min.z - GRID_MARGIN, b.max.z + GRID_MARGIN)
    }
    pub fn unlink_box(&mut self, i: usize) {
        let (x0, x1, z0, z1) = Self::grid_cells(&self.boxes[i]);
        for cz in z0..=z1 {
            for cx in x0..=x1 {
                self.gbox[cz as usize * GN + cx as usize].retain(|&k| k as usize != i);
            }
        }
    }
    pub fn link_box(&mut self, i: usize) {
        let (x0, x1, z0, z1) = Self::grid_cells(&self.boxes[i]);
        for cz in z0..=z1 {
            for cx in x0..=x1 {
                self.gbox[cz as usize * GN + cx as usize].push(i as u16);
            }
        }
    }
    /// Open or shut door `k` (a shut door is solid, an open one is not).
    pub fn set_door(&mut self, k: usize, open: bool) {
        if k >= self.doors.len() || self.door_open[k] == open {
            return;
        }
        self.door_open[k] = open;
        let i = self.doors[k].bx as usize;
        if open {
            self.unlink_box(i);
        } else {
            self.link_box(i);
        }
    }
    /// Move box `i` so that the middle of its underside is at `pos`.
    pub fn move_box(&mut self, i: usize, pos: V3) {
        self.unlink_box(i);
        let b = &mut self.boxes[i];
        let (hx, hy, hz) = ((b.max.x - b.min.x) * 0.5, b.max.y - b.min.y, (b.max.z - b.min.z) * 0.5);
        b.min = V3::new(pos.x - hx, pos.y, pos.z - hz);
        b.max = V3::new(pos.x + hx, pos.y + hy, pos.z + hz);
        self.link_box(i);
    }

    /// `through_crates`: crates do not stop the body (a car ploughs through them).
    pub fn collide_body_ex(&self, p: &mut V3, r: f32, hgt: f32, through_crates: bool) -> bool {
        let mut hit = false;
        for _ in 0..3 {
            let mut moved = false;
            let feet = p.y;
            let q = r + 0.05;
            let (px, pz) = (p.x, p.z);
            let mut dx_acc = 0.0f32;
            let mut dz_acc = 0.0f32;
            self.for_boxes(px - q, pz - q, px + q, pz + q, |b| {
                // Whole body height: a crouched (1.25 m) body cannot slip
                // through a 1.1 m window gap. Ceilings are resolved first by
                // the caller, so door lintels above the head are not walls.
                if b.max.y <= feet + STEP || b.min.y >= feet + hgt - 0.02 || (through_crates && b.kind == BK_CRATE) {
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
        self.raycast_opt(o, d, maxt, true)
    }

    /// First hit on the world itself (terrain, buildings, trees, props):
    /// neither cars nor helicopters.
    pub fn raycast_static(&self, o: V3, d: V3, maxt: f32) -> Option<(f32, V3, u8)> {
        self.raycast_opt(o, d, maxt, false)
    }

    /// True if something other than a vehicle obstructs the segment a->b.
    pub fn blocked_static(&self, a: V3, b: V3) -> bool {
        let dv = b - a;
        let l = dv.len();
        l > 1e-4 && self.raycast_opt(a, dv * (1.0 / l), l - 0.05, false).is_some()
    }

    fn raycast_opt(&self, o: V3, d: V3, maxt: f32, cars: bool) -> Option<(f32, V3, u8)> {
        let mut obj = self.ray_objects(o, d, maxt);
        if cars {
            if let Some((t, n, _)) = self.ray_cars(o, d, obj.map_or(maxt, |h| h.0)) {
                obj = Some((t, n, BK_CAR));
            }
            if let Some((t, n, _)) = self.ray_helis(o, d, obj.map_or(maxt, |h| h.0), usize::MAX) {
                obj = Some((t, n, BK_HELI));
            }
        }
        let lim = obj.map_or(maxt, |h| h.0);
        if let Some(t) = self.ray_terrain(o, d, lim) {
            let p = o + d * t;
            return Some((t, self.normal(p.x, p.z), HIT_TERRAIN));
        }
        obj
    }

    /// Where grass grows, one byte per square metre of the map (255 = thick):
    /// not on sand, roads or town paving, steep rock, nor under anything built.
    pub fn grass_mask(&self) -> Vec<u8> {
        let n = MAP as usize;
        let mut m = vec![0u8; n * n];
        for j in 0..n {
            for i in 0..n {
                let (x, z) = (i as f32 + 0.5, j as f32 + 0.5);
                let y = self.height(x, z);
                if y < 1.7 {
                    continue;
                }
                let mut pave = 0.0f32;
                for t in &self.towns {
                    let d = dist_xz(x, z, t.x, t.z);
                    if d < t.r + 8.0 {
                        pave = pave.max(smoothstep(t.r + 8.0, t.r - 6.0, d) * 0.75);
                    }
                }
                if pave < 0.6 {
                    let rd = self.road_dist(x, z);
                    if rd < 4.8 {
                        pave = pave.max(smoothstep(4.8, 2.4, rd) * 0.85);
                    }
                }
                let g = smoothstep(1.7, 3.0, y) * (1.0 - clampf(pave * 1.7, 0.0, 1.0)) * smoothstep(0.7, 0.86, self.normal(x, z).y);
                m[j * n + i] = (g * 255.0) as u8;
            }
        }
        // weeds on the paving: scattered patches where it is breaking up, and
        // along the foot of walls
        let wh = |i: usize, j: usize, k: u32| -> f32 {
            let mut h = (i as u32).wrapping_mul(0x9E37_79B1) ^ (j as u32).wrapping_mul(0x85EB_CA6B) ^ k.wrapping_mul(0xC2B2_AE35);
            h ^= h >> 15;
            h = h.wrapping_mul(0x2C1B_3C6D);
            h ^= h >> 12;
            (h & 0xFFFF) as f32 / 65535.0
        };
        for t in &self.towns {
            let (i0, i1) = (((t.x - t.r).max(0.0)) as usize, ((t.x + t.r).min(MAP - 1.0)) as usize);
            let (j0, j1) = (((t.z - t.r).max(0.0)) as usize, ((t.z + t.r).min(MAP - 1.0)) as usize);
            for j in j0..=j1 {
                for i in i0..=i1 {
                    if dist_xz(i as f32 + 0.5, j as f32 + 0.5, t.x, t.z) > t.r - 2.0 || self.road_dist(i as f32 + 0.5, j as f32 + 0.5) < 3.2 {
                        continue;
                    }
                    // patches a few metres across (a coarse cell picks, a fine one thins)
                    if wh(i / 5, j / 5, 1) > 0.88 && wh(i, j, 2) > 0.45 {
                        m[j * n + i] = m[j * n + i].max(26 + (wh(i, j, 3) * 34.0) as u8);
                    }
                }
            }
        }
        for b in &self.boxes {
            let wall = matches!(b.kind, BK_WALL | BK_BRICK | BK_STUCCO | BK_CONCRETE);
            if !wall || b.min.y > self.height((b.min.x + b.max.x) * 0.5, (b.min.z + b.max.z) * 0.5) + 1.0 {
                continue;
            }
            let (i0, i1) = (((b.min.x - 1.5).max(0.0)) as usize, ((b.max.x + 1.5).min(MAP - 1.0)) as usize);
            let (j0, j1) = (((b.min.z - 1.5).max(0.0)) as usize, ((b.max.z + 1.5).min(MAP - 1.0)) as usize);
            for j in j0..=j1 {
                for i in i0..=i1 {
                    if wh(i, j, 4) > 0.6 && self.road_dist(i as f32 + 0.5, j as f32 + 0.5) > 3.2 {
                        m[j * n + i] = m[j * n + i].max(36 + (wh(i, j, 5) * 50.0) as u8);
                    }
                }
            }
        }
        for b in &self.boxes {
            if b.min.y > self.height((b.min.x + b.max.x) * 0.5, (b.min.z + b.max.z) * 0.5) + 2.5 {
                continue; // well above the ground (an upper floor, a roof)
            }
            let (i0, i1) = (((b.min.x - 0.6).max(0.0)) as usize, ((b.max.x + 0.6).min(MAP - 1.0)) as usize);
            let (j0, j1) = (((b.min.z - 0.6).max(0.0)) as usize, ((b.max.z + 0.6).min(MAP - 1.0)) as usize);
            for j in j0..=j1 {
                for i in i0..=i1 {
                    m[j * n + i] = 0;
                }
            }
        }
        for c in &self.cyls {
            let (i, j) = ((c.x as usize).min(n - 1), (c.z as usize).min(n - 1));
            m[j * n + i] = m[j * n + i].min(90);
        }
        // helicopter pads: bare ground, the grass coming back towards the rim
        for h in &self.helis {
            let r = 12.0f32;
            let (i0, i1) = ((h[0] - r).max(0.0) as usize, ((h[0] + r) as usize).min(n - 1));
            let (j0, j1) = ((h[2] - r).max(0.0) as usize, ((h[2] + r) as usize).min(n - 1));
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let k = smoothstep(8.5, r, dist_xz(i as f32 + 0.5, j as f32 + 0.5, h[0], h[2]));
                    m[j * n + i] = (m[j * n + i] as f32 * k) as u8;
                }
            }
        }
        m
    }

    /// First helicopter hull hit along a normalized ray: (distance, normal,
    /// index). `skip` is one to ignore (usize::MAX for none). The hull is
    /// taken level: the few degrees it tilts in flight are not modelled.
    pub fn ray_helis(&self, o: V3, d: V3, maxt: f32, skip: usize) -> Option<(f32, V3, usize)> {
        let mut best: Option<(f32, V3, usize)> = None;
        let mut lim = maxt;
        for (k, h) in self.helis.iter().enumerate() {
            if k == skip {
                continue;
            }
            // quick reject: closest approach of the ray to the hull's middle
            let rel = V3::new(h[0] - o.x, h[1] + 2.6 - o.y, h[2] - o.z);
            let along = rel.dot(d);
            if along < -9.0 || along > lim + 9.0 {
                continue;
            }
            if (rel - d * along).len2() > 81.0 {
                continue;
            }
            // into the hull's frame: x = right, z = forward
            let (f, r) = (fwd(h[3]), right(h[3]));
            let lo = V3::new(-(rel.x * r.x + rel.z * r.z), -rel.y + 2.6, -(rel.x * f.x + rel.z * f.z));
            let ld = V3::new(d.x * r.x + d.z * r.z, d.y, d.x * f.x + d.z * f.z);
            for (mn, mx) in HELI_BOXES {
                if let Some((t, ax)) = ray_aabb(lo, ld, mn, mx, lim) {
                    if t < lim {
                        lim = t;
                        let n = match ax {
                            0 => r * -ld.x.signum(),
                            1 => V3::new(0.0, -ld.y.signum(), 0.0),
                            _ => f * -ld.z.signum(),
                        };
                        best = Some((t, n, k));
                    }
                }
            }
        }
        best
    }

    /// Push a body (circle of radius `r` at `p`, `hgt` tall) out of the
    /// cabin of helicopter `k`. True if it was touching.
    pub fn push_heli(&self, k: usize, p: &mut V3, r: f32, hgt: f32) -> bool {
        let Some(h) = self.helis.get(k) else { return false };
        let (dx, dz) = (p.x - h[0], p.z - h[2]);
        // (not under one that hovers, nor on top of it)
        if dx * dx + dz * dz > 144.0 || p.y > h[1] + HELI_H - 0.3 || p.y + hgt < h[1] + 0.3 {
            return false;
        }
        let (f, rt) = (fwd(h[3]), right(h[3]));
        let (lx, lz) = (dx * rt.x + dz * rt.z, dx * f.x + dz * f.z);
        let (qx, qz) = (clampf(lx, -HELI_HW, HELI_HW), clampf(lz, -HELI_HL, HELI_HL));
        let (ex, ez) = (lx - qx, lz - qz);
        let d2 = ex * ex + ez * ez;
        if d2 >= r * r {
            return false;
        }
        let (nx, nz, push);
        if d2 > 1e-8 {
            let dl = d2.sqrt();
            nx = ex / dl;
            nz = ez / dl;
            push = r - dl;
        } else {
            // centre inside the hull: leave through the nearest side
            let (px, pz) = (HELI_HW - lx.abs(), HELI_HL - lz.abs());
            if px < pz {
                nx = if lx >= 0.0 { 1.0 } else { -1.0 };
                nz = 0.0;
                push = px + r;
            } else {
                nx = 0.0;
                nz = if lz >= 0.0 { 1.0 } else { -1.0 };
                push = pz + r;
            }
        }
        p.x += (rt.x * nx + f.x * nz) * push;
        p.z += (rt.z * nx + f.z * nz) * push;
        true
    }

    /// Push a standing body out of every helicopter (`skip`: one to ignore).
    pub fn push_helis(&self, p: &mut V3, r: f32, hgt: f32, skip: usize) -> bool {
        let mut hit = false;
        for k in 0..self.helis.len() {
            if k != skip {
                hit |= self.push_heli(k, p, r, hgt);
            }
        }
        hit
    }

    /// First car hit along a normalized ray: (distance, normal, car index).
    pub fn ray_cars(&self, o: V3, d: V3, maxt: f32) -> Option<(f32, V3, usize)> {
        let mut best: Option<(f32, V3, usize)> = None;
        let mut lim = maxt;
        for (k, c) in self.cars.iter().enumerate() {
            // quick reject: closest approach of the ray to the car centre
            let rel = V3::new(c[0] - o.x, c[1] + 0.7 - o.y, c[2] - o.z);
            let along = rel.dot(d);
            if along < -3.0 || along > lim + 3.0 {
                continue;
            }
            if (rel - d * along).len2() > 16.0 {
                continue;
            }
            // into the car's frame: x = right, z = forward
            let (f, r) = (fwd(c[3]), right(c[3]));
            let lo = V3::new(-(rel.x * r.x + rel.z * r.z), -rel.y + 0.7, -(rel.x * f.x + rel.z * f.z));
            let ld = V3::new(d.x * r.x + d.z * r.z, d.y, d.x * f.x + d.z * f.z);
            let (hw, hl, hh) = self.car_dims(k);
            // a car is a body with a cabin on it, a truck one tall box
            let boxes = if self.car_kind.get(k) == Some(&VK_TRUCK) {
                [(V3::new(-hw, -0.2, -hl), V3::new(hw, hh, hl)), (V3::ZERO, V3::ZERO)]
            } else {
                [(V3::new(-hw, -0.2, -hl), V3::new(hw, 0.98, hl)), (V3::new(-0.8, 0.98, -1.5), V3::new(0.8, hh, 1.0))]
            };
            for (mn, mx) in boxes {
                if mx.y <= mn.y {
                    continue;
                }
                if let Some((t, ax)) = ray_aabb(lo, ld, mn, mx, lim) {
                    if t < lim {
                        lim = t;
                        let n = match ax {
                            0 => r * -ld.x.signum(),
                            1 => V3::new(0.0, -ld.y.signum(), 0.0),
                            _ => f * -ld.z.signum(),
                        };
                        best = Some((t, n, k));
                    }
                }
            }
        }
        best
    }

    /// Push a standing body (circle of radius `r` at `p`) out of the cars.
    /// `skip` is a car index to ignore (usize::MAX for none).
    pub fn push_cars(&self, p: &mut V3, r: f32, skip: usize) -> bool {
        let mut hit = false;
        for k in 0..self.cars.len() {
            if k != skip {
                hit |= self.push_car(k, p, r);
            }
        }
        hit
    }

    /// Half width, half length and height of car `k`.
    pub fn car_dims(&self, k: usize) -> (f32, f32, f32) {
        CAR_DIMS[(*self.car_kind.get(k).unwrap_or(&0) as usize).min(CAR_DIMS.len() - 1)]
    }

    /// Push a body out of car `k`. True if it was touching.
    pub fn push_car(&self, k: usize, p: &mut V3, r: f32) -> bool {
        let mut hit = false;
        if let Some(c) = self.cars.get(k) {
            let (hw, hl, hh) = self.car_dims(k);
            let (dx, dz) = (p.x - c[0], p.z - c[2]);
            if dx * dx + dz * dz > 36.0 || p.y > c[1] + hh || p.y + 1.2 < c[1] {
                return false;
            }
            let (f, rt) = (fwd(c[3]), right(c[3]));
            let (lx, lz) = (dx * rt.x + dz * rt.z, dx * f.x + dz * f.z);
            let (qx, qz) = (clampf(lx, -hw, hw), clampf(lz, -hl, hl));
            let (ex, ez) = (lx - qx, lz - qz);
            let d2 = ex * ex + ez * ez;
            if d2 >= r * r {
                return false;
            }
            let (nx, nz, push);
            if d2 > 1e-8 {
                let dl = d2.sqrt();
                nx = ex / dl;
                nz = ez / dl;
                push = r - dl;
            } else {
                // centre inside the car: leave through the nearest side
                let (px, pz) = (hw - lx.abs(), hl - lz.abs());
                if px < pz {
                    nx = if lx >= 0.0 { 1.0 } else { -1.0 };
                    nz = 0.0;
                    push = px + r;
                } else {
                    nx = 0.0;
                    nz = if lz >= 0.0 { 1.0 } else { -1.0 };
                    push = pz + r;
                }
            }
            p.x += (rt.x * nx + f.x * nz) * push;
            p.z += (rt.z * nx + f.z * nz) * push;
            hit = true;
        }
        hit
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
