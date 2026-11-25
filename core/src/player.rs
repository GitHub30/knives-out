//! Player movement. This is the code path shared by the authoritative server
//! and client-side prediction, so it must depend only on (state, cmd, mods).

use crate::defs::*;
use crate::math::*;
use crate::net::{dq_pitch, dq_yaw};
use crate::world::*;

pub const P_RADIUS: f32 = 0.38;
pub const PITCH_LIMIT: f32 = 1.45;

#[inline]
pub fn body_h(crouch: bool) -> f32 {
    if crouch {
        1.25
    } else {
        1.8
    }
}
#[inline]
pub fn eye_h(crouch: bool) -> f32 {
    if crouch {
        1.12
    } else {
        1.62
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct MoveState {
    pub pos: V3,
    pub vel: V3,
    pub mode: u8,
    pub on_ground: bool,
    pub crouch: bool,
    pub in_water: bool,
    pub airtime: f32,
    /// Index of the car being driven (MODE_DRIVE) or of the helicopter
    /// being flown (MODE_FLY).
    pub car: u8,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Cmd {
    pub seq: u32,
    /// Server tick (x256) the client was rendering when this command was made.
    pub view: u32,
    pub mx: f32,
    pub mz: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub buttons: u16,
}

impl Cmd {
    pub fn from_quant(seq: u32, view: u32, mxq: i8, mzq: i8, yawq: u16, pitchq: i16, buttons: u16) -> Cmd {
        let mut mx = (mxq as f32 / 127.0).clamp(-1.0, 1.0);
        let mut mz = (mzq as f32 / 127.0).clamp(-1.0, 1.0);
        let l = (mx * mx + mz * mz).sqrt();
        if l > 1.0 {
            mx /= l;
            mz /= l;
        }
        Cmd {
            seq,
            view,
            mx,
            mz,
            yaw: dq_yaw(yawq),
            pitch: clampf(dq_pitch(pitchq), -PITCH_LIMIT, PITCH_LIMIT),
            buttons: buttons & B_MASK,
        }
    }
    #[inline]
    pub fn has(&self, b: u16) -> bool {
        self.buttons & b != 0
    }
    #[inline]
    pub fn slot(&self) -> u8 {
        ((self.buttons >> 8) & 3) as u8
    }
    #[inline]
    pub fn heal(&self) -> u8 {
        ((self.buttons >> 10) & 3) as u8
    }
}

pub const EV_LANDED: u8 = 1;
pub const EV_JUMPED: u8 = 2;
pub const EV_CHUTE: u8 = 4;

/// Advance one command. `speed_mult` folds in weapon weight, healing and boost.
pub fn step_move(w: &World, s: &mut MoveState, c: &Cmd, speed_mult: f32, dt: f32) -> u8 {
    let mut hits = DriveHits::default();
    step_move_hits(w, s, c, speed_mult, dt, &mut hits)
}

/// The other cars a driven one ran into during its step, each with the
/// closing speed it paid its share of (`CarHit::closing`).
#[derive(Clone, Copy, Default, Debug)]
pub struct DriveHits {
    pub n: usize,
    pub hits: [CarHit; CAR_HITS],
}

/// `step_move` that also tells which cars a driven car ran into, so that
/// whoever moves them (the server, or the predicting client) gives them
/// exactly what the driver paid for.
pub fn step_move_hits(w: &World, s: &mut MoveState, c: &Cmd, speed_mult: f32, dt: f32, out: &mut DriveHits) -> u8 {
    out.n = 0;
    let mut ev = 0u8;
    let f = fwd(c.yaw);
    let r = right(c.yaw);
    let wish = r * c.mx + f * c.mz;
    let prev = s.pos;
    match s.mode {
        MODE_FALL => {
            s.airtime += dt;
            let tv = wish * 24.0;
            let tvy = if c.pitch < -0.6 { -52.0 } else { -38.0 };
            s.vel.x = approach(s.vel.x, tv.x, 18.0 * dt);
            s.vel.z = approach(s.vel.z, tv.z, 18.0 * dt);
            s.vel.y = approach(s.vel.y, tvy, 25.0 * dt);
            s.pos += s.vel * dt;
            w.collide_body(&mut s.pos, P_RADIUS, 1.8);
            let g = w.ground_below(s.pos.x, s.pos.z, prev.y.max(s.pos.y), P_RADIUS);
            let land = g.max(WATER - 1.25);
            if s.pos.y <= land {
                s.pos.y = land;
                s.vel = V3::ZERO;
                s.mode = MODE_GROUND;
                s.on_ground = true;
                s.in_water = g < WATER - 1.2;
                ev |= EV_LANDED;
            } else if s.pos.y - g < 80.0 || (c.has(B_JUMP) && s.airtime > 0.8 && s.pos.y - g > 12.0) {
                s.mode = MODE_PARA;
                s.vel.y = s.vel.y.max(-16.0);
                ev |= EV_CHUTE;
            }
        }
        MODE_PARA => {
            s.airtime += dt;
            let tv = wish * 10.0;
            let tvy = if c.mz > 0.5 { -8.5 } else { -6.0 };
            s.vel.x = approach(s.vel.x, tv.x, 8.0 * dt);
            s.vel.z = approach(s.vel.z, tv.z, 8.0 * dt);
            s.vel.y = approach(s.vel.y, tvy, 12.0 * dt);
            s.pos += s.vel * dt;
            w.collide_body(&mut s.pos, P_RADIUS, 1.8);
            let g = w.ground_below(s.pos.x, s.pos.z, prev.y.max(s.pos.y), P_RADIUS);
            let land = g.max(WATER - 1.25);
            if s.pos.y <= land {
                s.pos.y = land;
                s.vel = V3::ZERO;
                s.mode = MODE_GROUND;
                s.on_ground = true;
                s.in_water = g < WATER - 1.2;
                ev |= EV_LANDED;
            }
        }
        MODE_GROUND => {
            let want_crouch = c.has(B_CROUCH) && !s.in_water;
            if s.crouch && !want_crouch && !s.in_water {
                // stand up only with headroom (never into a roof or lintel)
                let stand = body_h(false);
                let blocked = w.ceiling(s.pos.x, s.pos.z, s.pos.y, body_h(true), P_RADIUS).map_or(false, |cy| cy < s.pos.y + stand);
                s.crouch = blocked;
            } else {
                s.crouch = want_crouch;
            }
            let ads = c.has(B_ADS);
            let sprint = c.has(B_SPRINT) && c.mz > 0.3 && !ads && !s.crouch;
            let mut speed: f32 = if s.crouch {
                2.5
            } else if sprint {
                6.9
            } else {
                5.2
            };
            if ads {
                speed = speed.min(3.0);
            }
            speed *= speed_mult;
            if s.in_water {
                speed = 2.7;
            }
            let tv = wish * speed;
            let acc = if s.on_ground { 45.0 } else { 9.0 };
            s.vel.x = approach(s.vel.x, tv.x, acc * dt);
            s.vel.z = approach(s.vel.z, tv.z, acc * dt);
            if s.on_ground && c.has(B_JUMP) && !s.in_water {
                s.vel.y = 6.8;
                s.on_ground = false;
                ev |= EV_JUMPED;
            }
            if !s.on_ground {
                s.vel.y -= 20.0 * dt;
                s.airtime += dt;
            } else {
                s.vel.y = 0.0;
                s.airtime = 0.0;
            }
            let hgt = body_h(s.crouch);
            let mut np = s.pos + s.vel * dt;
            if s.vel.y > 0.0 {
                if let Some(cy) = w.ceiling(np.x, np.z, prev.y, hgt, P_RADIUS) {
                    if np.y + hgt > cy {
                        np.y = (cy - hgt).max(prev.y);
                        s.vel.y = 0.0;
                    }
                }
            }
            w.collide_body(&mut np, P_RADIUS, hgt);
            if w.push_cars(&mut np, P_RADIUS, usize::MAX) | w.push_helis(&mut np, P_RADIUS, hgt, usize::MAX) {
                // squeezed between a car and a wall: the wall wins
                w.collide_body(&mut np, P_RADIUS, hgt);
            }
            let g = w.ground_below(np.x, np.z, prev.y.max(np.y), P_RADIUS);
            let was_ground = s.on_ground;
            let (free_y, free_vy, ev0) = (np.y, s.vel.y, ev);
            if np.y <= g {
                np.y = g;
                if s.vel.y < 0.0 {
                    s.vel.y = 0.0;
                }
                if !was_ground {
                    ev |= EV_LANDED;
                }
                s.on_ground = true;
            } else if was_ground && s.vel.y <= 0.0 && np.y - g < 0.6 {
                np.y = g;
                s.on_ground = true;
            } else {
                s.on_ground = false;
            }
            if np.y > prev.y + 0.01 {
                // stepped or jumped up: the taller body may now touch something
                let (bx, bz) = (np.x, np.z);
                w.collide_body(&mut np, P_RADIUS, hgt);
                if s.on_ground && (np.x != bx || np.z != bz) {
                    // Pushed back off the step it just snapped onto (a window
                    // sill): redo the ground test where the body really is,
                    // instead of standing in mid-air.
                    let g2 = w.ground_below(np.x, np.z, prev.y.max(free_y), P_RADIUS);
                    if free_y <= g2 || (was_ground && free_vy <= 0.0 && free_y - g2 < 0.6) {
                        np.y = g2;
                    } else {
                        np.y = free_y;
                        s.vel.y = free_vy;
                        s.on_ground = false;
                        ev = ev0;
                    }
                }
            }
            // deep water: float at the surface
            if g < WATER - 1.2 {
                s.in_water = true;
                let sy = WATER - 1.25;
                if np.y <= sy + 0.05 {
                    np.y = sy;
                    s.vel.y = 0.0;
                    s.on_ground = true;
                }
            } else {
                s.in_water = false;
            }
            s.pos = np;
        }
        MODE_DRIVE => out.n = step_drive(w, s, c, dt, &mut out.hits),
        MODE_FLY => step_fly(w, s, c, dt),
        _ => {}
    }
    s.pos.x = clampf(s.pos.x, 2.0, MAP - 2.0);
    s.pos.z = clampf(s.pos.z, 2.0, MAP - 2.0);
    if !s.pos.finite() || !s.vel.finite() {
        // Never let NaNs escape into the simulation.
        s.pos = V3::new(MAP * 0.5, 50.0, MAP * 0.5);
        s.vel = V3::ZERO;
    }
    ev
}

pub const CAR_VMAX: f32 = 24.0;
pub const CAR_VREV: f32 = 7.0;
/// Offsets along the car of the three circles it collides with.
pub const CAR_CIRCLES: [f32; 3] = [-1.5, 0.0, 1.5];
pub const CAR_CR: f32 = 0.95;

/// Collision circles (offsets along the body, radius) and top speed per vehicle kind.
pub fn car_shape(kind: u8) -> ([f32; 3], f32, f32) {
    if kind == VK_TRUCK {
        ([-1.85, 0.0, 1.85], 1.3, 18.0)
    } else {
        (CAR_CIRCLES, CAR_CR, CAR_VMAX)
    }
}

/// Signed speed of a driven car along its heading.
pub fn car_speed(s: &MoveState) -> f32 {
    let f = fwd(s.airtime);
    s.vel.x * f.x + s.vel.z * f.z
}

/// How heavy the vehicles are next to a police car: it decides who moves
/// how much when two of them collide.
pub const CAR_MASS: f32 = 1.0;
pub const TRUCK_MASS: f32 = 3.0;
pub const HELI_MASS: f32 = 20.0;
/// A helicopter on the ground slower than this stands: cars do not push
/// it about (a knock moves it a hand's breadth), and it does not shove them.
pub const HELI_STAND_V: f32 = 3.0;
/// How much of the closing speed two vehicles bounce apart with.
pub const VEH_BOUNCE: f32 = 0.15;
/// A shoved car that slides: what slows it along its wheels and across
/// them (m/s2), the fastest it goes and turns, how much a push off its
/// middle turns it, and the crawl below which it stands.
pub const CAR_ROLL_FRICTION: f32 = 4.0;
pub const CAR_SIDE_FRICTION: f32 = 11.0;
pub const CAR_SLIDE_MAX: f32 = 45.0;
pub const CAR_SPIN_MAX: f32 = 2.0;
pub const CAR_SPIN_GAIN: f32 = 0.5;
pub const CAR_REST_V: f32 = 0.3;
pub const CAR_REST_SPIN: f32 = 0.12;

pub fn car_mass(kind: u8) -> f32 {
    if kind == VK_TRUCK {
        TRUCK_MASS
    } else {
        CAR_MASS
    }
}

/// The share of the closing speed a body of mass `m` loses when it runs
/// into one of mass `other` (which gains `hit_share(other, m)` of it).
#[inline]
pub fn hit_share(m: f32, other: f32) -> f32 {
    other * (1.0 + VEH_BOUNCE) / (m + other)
}

/// Another car touched by a car: the summed push out of it (pointing from
/// it to the car that asked) and where they touch, weighted by `w`.
#[derive(Clone, Copy, Default, Debug)]
pub struct CarHit {
    pub k: usize,
    pub px: f32,
    pub pz: f32,
    pub cx: f32,
    pub cz: f32,
    pub w: f32,
    /// How fast the two closed in when the driver that touched it paid for
    /// the hit (filled by `step_drive` only; 0: they were not closing).
    pub closing: f32,
}
pub const CAR_HITS: usize = 4;

impl CarHit {
    /// Unit direction from the other car to this one, and the contact point.
    pub fn normal(&self) -> Option<(f32, f32, f32, f32)> {
        let pl = (self.px * self.px + self.pz * self.pz).sqrt();
        if pl < 1e-5 || self.w < 1e-6 {
            return None;
        }
        Some((self.px / pl, self.pz / pl, self.cx / self.w, self.cz / self.w))
    }
}

fn add_hit(hits: &mut [CarHit; CAR_HITS], n: &mut usize, k: usize, nx: f32, nz: f32, depth: f32, cx: f32, cz: f32) {
    let slot = match hits[..*n].iter().position(|h| h.k == k) {
        Some(s) => s,
        None if *n < CAR_HITS => {
            hits[*n] = CarHit { k, ..Default::default() };
            *n += 1;
            *n - 1
        }
        None => return,
    };
    let h = &mut hits[slot];
    h.px += nx * depth;
    h.pz += nz * depth;
    h.cx += cx * depth;
    h.cz += cz * depth;
    h.w += depth;
}

/// Push the body of car `me` (of `kind`, at `np` heading `yaw`) out of
/// walls, trees, the other cars and the helicopters: three circles along
/// the body. Returns how far the static world and the helicopters pushed
/// it (the sum over the circles, and as a vector) and how many other cars
/// it touched, which are listed in `hits`.
pub fn car_collide(w: &World, me: usize, kind: u8, np: &mut V3, yaw: f32, hits: &mut [CarHit; CAR_HITS]) -> (f32, V3, usize) {
    let f = fwd(yaw);
    let (circles, cr, _) = car_shape(kind);
    let mut pushed = 0.0f32;
    let mut solid = V3::ZERO;
    let mut n = 0usize;
    for _ in 0..2 {
        for off in circles {
            let base = V3::new(np.x + f.x * off, w.height(np.x + f.x * off, np.z + f.z * off) + 0.45, np.z + f.z * off);
            let mut q = base;
            // (crates do not stop a car: it knocks them flying)
            w.collide_body_ex(&mut q, cr, 1.0, true);
            let (mut sx, mut sz) = (q.x - base.x, q.z - base.z);
            for (k, o) in w.cars.iter().enumerate() {
                if k == me || (o[0] - q.x).abs() > 8.0 || (o[2] - q.z).abs() > 8.0 || (o[1] - np.y).abs() > 3.0 {
                    continue;
                }
                let of = fwd(o[3]);
                let (ocircles, ocr, _) = car_shape(*w.car_kind.get(k).unwrap_or(&0));
                for ooff in ocircles {
                    let (ox, oz) = (o[0] + of.x * ooff, o[2] + of.z * ooff);
                    let (dx, dz) = (q.x - ox, q.z - oz);
                    let d2 = dx * dx + dz * dz;
                    let rr = cr + ocr;
                    if d2 < rr * rr && d2 > 1e-6 {
                        let dl = d2.sqrt();
                        let (nx, nz) = (dx / dl, dz / dl);
                        q.x += nx * (rr - dl);
                        q.z += nz * (rr - dl);
                        add_hit(hits, &mut n, k, nx, nz, rr - dl, ox + nx * ocr, oz + nz * ocr);
                    }
                }
            }
            let (bx, bz) = (q.x, q.z);
            w.push_helis(&mut q, cr, 1.0, usize::MAX);
            sx += q.x - bx;
            sz += q.z - bz;
            solid.x += sx;
            solid.z += sz;
            pushed += (sx * sx + sz * sz).sqrt();
            np.x += q.x - base.x;
            np.z += q.z - base.z;
        }
    }
    (pushed, solid, n)
}

/// Push a car of `kind` (at `np` heading `yaw`) out of walls and trees
/// only, a few rounds over its circles: what it comes to rest in is never
/// a wall.
pub fn car_out_of_walls(w: &World, kind: u8, np: &mut V3, yaw: f32) {
    let f = fwd(yaw);
    let (circles, cr, _) = car_shape(kind);
    for _ in 0..3 {
        let mut any = false;
        for off in circles {
            let base = V3::new(np.x + f.x * off, w.height(np.x + f.x * off, np.z + f.z * off) + 0.45, np.z + f.z * off);
            let mut q = base;
            if w.collide_body_ex(&mut q, cr, 1.0, true) {
                np.x += q.x - base.x;
                np.z += q.z - base.z;
                any = true;
            }
        }
        if !any {
            break;
        }
    }
}

/// How two cars (poses `a` and `b`) overlap where they overlap deepest:
/// the direction from `b` to `a` there, and how deep.
pub fn car_overlap(a: [f32; 4], akind: u8, b: [f32; 4], bkind: u8) -> Option<(f32, f32, f32)> {
    let (ac, ar, _) = car_shape(akind);
    let (bc, br, _) = car_shape(bkind);
    let (af, bf) = (fwd(a[3]), fwd(b[3]));
    let mut best: Option<(f32, f32, f32)> = None;
    for ao in ac {
        for bo in bc {
            let (dx, dz) = (a[0] + af.x * ao - (b[0] + bf.x * bo), a[2] + af.z * ao - (b[2] + bf.z * bo));
            let d = (dx * dx + dz * dz).sqrt();
            let depth = ar + br - d;
            if depth > 1e-4 && d > 1e-4 && best.map_or(true, |x| depth > x.2) {
                best = Some((dx / d, dz / d, depth));
            }
        }
    }
    best
}

/// The other cars that car `me` (at `pos` heading `yaw`) touches or comes
/// within `margin` of. Nothing is moved.
pub fn car_touches(w: &World, me: usize, kind: u8, pos: V3, yaw: f32, margin: f32, hits: &mut [CarHit; CAR_HITS]) -> usize {
    let f = fwd(yaw);
    let (circles, cr, _) = car_shape(kind);
    let mut n = 0usize;
    for off in circles {
        let (qx, qz) = (pos.x + f.x * off, pos.z + f.z * off);
        for (k, o) in w.cars.iter().enumerate() {
            if k == me || (o[0] - qx).abs() > 8.0 || (o[2] - qz).abs() > 8.0 || (o[1] - pos.y).abs() > 3.0 {
                continue;
            }
            let of = fwd(o[3]);
            let (ocircles, ocr, _) = car_shape(*w.car_kind.get(k).unwrap_or(&0));
            for ooff in ocircles {
                let (ox, oz) = (o[0] + of.x * ooff, o[2] + of.z * ooff);
                let (dx, dz) = (qx - ox, qz - oz);
                let d2 = dx * dx + dz * dz;
                let rr = cr + ocr + margin;
                if d2 < rr * rr && d2 > 1e-6 {
                    let dl = d2.sqrt();
                    let (nx, nz) = (dx / dl, dz / dl);
                    add_hit(hits, &mut n, k, nx, nz, rr - dl, ox + nx * ocr, oz + nz * ocr);
                }
            }
        }
    }
    n
}

/// One step of a driven car: `mz` is the throttle, `mx` steers, jump brakes.
fn step_drive(w: &World, s: &mut MoveState, c: &Cmd, dt: f32, hits: &mut [CarHit; CAR_HITS]) -> usize {
    let mut yaw = if s.airtime.is_finite() { s.airtime } else { 0.0 };
    let mut v = car_speed(s);
    let (thr, steer) = (clampf(c.mz, -1.0, 1.0), clampf(c.mx, -1.0, 1.0));
    let wet = w.height(s.pos.x, s.pos.z) < WATER - 0.45;
    let kind = *w.car_kind.get(s.car as usize).unwrap_or(&0);
    let (circles, _, top) = car_shape(kind);
    let power = if kind == VK_TRUCK { 7.0 } else { 11.0 };
    let vmax = if wet { 3.0 } else { top };
    if c.has(B_JUMP) {
        v = approach(v, 0.0, 22.0 * dt);
    } else if thr > 0.05 {
        if v < 0.0 {
            v = approach(v, 0.0, 16.0 * dt);
        } else {
            v += power * thr * (1.0 - v / vmax).max(0.0) * dt;
        }
    } else if thr < -0.05 {
        if v > 0.0 {
            v = approach(v, 0.0, 18.0 * dt);
        } else {
            v = approach(v, -CAR_VREV.min(vmax), 5.0 * -thr * dt);
        }
    } else {
        v = approach(v, 0.0, (1.5 + v.abs() * 0.12) * dt);
    }
    if v > vmax {
        v = approach(v, vmax, 14.0 * dt);
    }
    // bicycle model: the steering lock closes up with speed
    let lock = 0.62 / (1.0 + v.abs() * 0.085);
    let wheelbase = if kind == VK_TRUCK { 3.6 } else { 2.8 };
    yaw = wrap_angle(yaw - steer * lock.tan() * v / wheelbase * dt);
    let f = fwd(yaw);
    let mut np = V3::new(s.pos.x + f.x * v * dt, s.pos.y, s.pos.z + f.z * v * dt);
    // too steep to climb
    let ahead = (circles[2] + 0.7) * if v >= 0.0 { 1.0 } else { -1.0 };
    let rise = w.height(np.x + f.x * ahead, np.z + f.z * ahead) - w.height(np.x, np.z);
    if rise > 1.9 {
        np.x = s.pos.x;
        np.z = s.pos.z;
        v = 0.0;
    } else if rise > 0.5 {
        v = approach(v, 0.0, (rise - 0.5) * 9.0 * dt);
    }
    // walls, trees and other cars against three circles along the body
    *hits = [CarHit::default(); CAR_HITS];
    let (pushed, _, nhit) = car_collide(w, s.car as usize, kind, &mut np, yaw, hits);
    if pushed > 1e-4 {
        // scraping costs speed; a square hit stops the car
        v *= (1.0 - pushed * 1.5).clamp(0.0, 1.0);
    }
    // Another vehicle is shoved along instead (the server moves it): this
    // one keeps the share of the closing speed that its weight earns it.
    for h in hits[..nhit].iter_mut() {
        let pl = (h.px * h.px + h.pz * h.pz).sqrt();
        if pl < 1e-5 {
            continue;
        }
        let (nx, nz) = (h.px / pl, h.pz / pl);
        let ov = w.car_vel.get(h.k).copied().unwrap_or([0.0; 2]);
        let closing = -((f.x * v - ov[0]) * nx + (f.z * v - ov[1]) * nz);
        if closing > 0.0 {
            let share = hit_share(car_mass(kind), car_mass(*w.car_kind.get(h.k).unwrap_or(&0)));
            v += (nx * f.x + nz * f.z) * closing * share;
            h.closing = closing;
        }
    }
    np.x = clampf(np.x, 4.0, MAP - 4.0);
    np.z = clampf(np.z, 4.0, MAP - 4.0);
    np.y = w.height(np.x, np.z);
    s.pos = np;
    s.vel = V3::new(f.x * v, 0.0, f.z * v);
    s.airtime = yaw;
    s.on_ground = true;
    s.crouch = false;
    s.in_water = false;
    nhit
}

/// How much a push of (`dvx`, `dvz`) at (`rx`, `rz`) from the middle of a
/// vehicle of `kind` turns it (a point at r turns with the car at
/// spin * (r.z, -r.x)).
pub fn shove_spin(kind: u8, dvx: f32, dvz: f32, rx: f32, rz: f32) -> f32 {
    let gyr2 = if kind == VK_TRUCK { 5.0 } else { 3.0 };
    (dvx * rz - dvz * rx) / gyr2 * CAR_SPIN_GAIN
}

/// One step of the friction on a car that slides at (`sx`, `sz`) and turns
/// at `spin`, heading `yaw`: rolling along its wheels is easy, sideways the
/// tyres bite. Below a crawl it stands.
pub fn slide_friction(sx: f32, sz: f32, spin: f32, yaw: f32, wet: bool, dt: f32) -> (f32, f32, f32) {
    let f = fwd(yaw);
    let wet = if wet { 2.5 } else { 1.0 };
    let along = sx * f.x + sz * f.z;
    let (lx, lz) = (sx - f.x * along, sz - f.z * along);
    let ll = (lx * lx + lz * lz).sqrt();
    let along2 = approach(along, 0.0, (CAR_ROLL_FRICTION + along.abs() * 0.25) * wet * dt);
    let ll2 = approach(ll, 0.0, CAR_SIDE_FRICTION * wet * dt);
    let lk = if ll > 1e-5 { ll2 / ll } else { 0.0 };
    let (sx, sz) = (f.x * along2 + lx * lk, f.z * along2 + lz * lk);
    let spin = approach(spin, 0.0, (2.5 + spin.abs() * 2.0) * dt);
    (sx, sz, spin)
}

/// What the predicting client keeps about the cars besides `World.cars`
/// and `World.car_vel`: how fast each turns, and which ones nobody drives
/// (those slide on by themselves; a driven one is its driver's business).
#[derive(Default, Clone)]
pub struct CarsPred {
    pub spin: Vec<f32>,
    pub free: Vec<bool>,
}

/// Client prediction of what the server's `car_moved_by` does to the cars
/// a driven car (`me`, of `kind`) ran into: they are shoved along.
pub fn predict_shoves(w: &mut World, cp: &mut CarsPred, kind: u8, hits: &DriveHits) {
    for h in &hits.hits[..hits.n] {
        let Some((nx, nz, cx, cz)) = h.normal() else { continue };
        if !(h.closing > 0.05) || h.k >= w.cars.len() || h.k >= w.car_vel.len() || h.k >= cp.spin.len() {
            continue;
        }
        let jk = *w.car_kind.get(h.k).unwrap_or(&0);
        let dv = h.closing * hit_share(car_mass(jk), car_mass(kind));
        let c = w.cars[h.k];
        let v = &mut w.car_vel[h.k];
        v[0] -= nx * dv;
        v[1] -= nz * dv;
        let l = (v[0] * v[0] + v[1] * v[1]).sqrt();
        if l > CAR_SLIDE_MAX {
            v[0] *= CAR_SLIDE_MAX / l;
            v[1] *= CAR_SLIDE_MAX / l;
        }
        if cp.free[h.k] {
            cp.spin[h.k] = clampf(cp.spin[h.k] + shove_spin(jk, -nx * dv, -nz * dv, cx - c[0], cz - c[2]), -CAR_SPIN_MAX, CAR_SPIN_MAX);
        }
    }
}

/// Client prediction of the server's `update_cars`: the driverless cars
/// that move slide on for one step, slow down and stop at walls. (Cruder
/// than the server: no damage, no people, no props; the next snapshot
/// puts right what this gets wrong.) `me` is the car the client drives.
pub fn predict_slides(w: &mut World, cp: &mut CarsPred, me: usize, dt: f32) {
    for k in 0..w.cars.len().min(w.car_vel.len()).min(cp.spin.len()) {
        let [mut sx, mut sz] = w.car_vel[k];
        let mut spin = cp.spin[k];
        if k == me || !cp.free[k] || (sx == 0.0 && sz == 0.0 && spin == 0.0) {
            continue;
        }
        let c = w.cars[k];
        if c[1] < -500.0 {
            continue;
        }
        let kind = *w.car_kind.get(k).unwrap_or(&0);
        let sp = (sx * sx + sz * sz).sqrt();
        let dir = if sp > 1e-4 { V3::new(sx / sp, 0.0, sz / sp) } else { V3::ZERO };
        let yaw = wrap_angle(c[3] + spin * dt);
        let mut np = V3::new(c[0], c[1], c[2]);
        let (mut ex, mut ez) = (sx * dt, sz * dt);
        let reach = car_shape(kind).0[2] + 0.7;
        let rise = w.height(c[0] + ex + dir.x * reach, c[2] + ez + dir.z * reach) - w.height(c[0] + ex, c[2] + ez);
        if rise > 1.9 {
            ex = 0.0;
            ez = 0.0;
            sx = 0.0;
            sz = 0.0;
        } else if rise > 0.5 {
            let k2 = (1.0 - (rise - 0.5) * 9.0 * dt / sp.max(0.5)).max(0.0);
            sx *= k2;
            sz *= k2;
        }
        let mut hits = [CarHit::default(); CAR_HITS];
        let (mut solid, mut n) = (V3::ZERO, 0);
        let steps = ((sp * dt / 0.5).ceil() as usize).clamp(1, 4);
        for _ in 0..steps {
            np.x += ex / steps as f32;
            np.z += ez / steps as f32;
            let (_, s2, n2) = car_collide(w, k, kind, &mut np, yaw, &mut hits);
            solid = s2;
            n = n2;
            if n2 > 0 || s2.x != 0.0 || s2.z != 0.0 {
                break;
            }
        }
        let sl = (solid.x * solid.x + solid.z * solid.z).sqrt();
        if sl > 1e-4 {
            let (nx, nz) = (solid.x / sl, solid.z / sl);
            let into = sx * nx + sz * nz;
            if into < 0.0 {
                sx -= nx * into * 1.1;
                sz -= nz * into * 1.1;
            }
            spin *= 0.5;
        }
        let mk = car_mass(kind);
        for h in &hits[..n] {
            let Some((nx, nz, cx, cz)) = h.normal() else { continue };
            let ov = w.car_vel[h.k.min(w.car_vel.len() - 1)];
            let closing = -((sx - ov[0]) * nx + (sz - ov[1]) * nz);
            if !(closing > 0.05) {
                continue;
            }
            let jk = *w.car_kind.get(h.k).unwrap_or(&0);
            let mj = car_mass(jk);
            let dv = closing * hit_share(mk, mj);
            sx += nx * dv;
            sz += nz * dv;
            spin = clampf(spin + shove_spin(kind, nx * dv, nz * dv, cx - np.x, cz - np.z), -CAR_SPIN_MAX, CAR_SPIN_MAX);
            if h.k < cp.free.len() && cp.free[h.k] && h.k < w.car_vel.len() {
                let dj = closing * hit_share(mj, mk);
                w.car_vel[h.k][0] -= nx * dj;
                w.car_vel[h.k][1] -= nz * dj;
            }
        }
        let wet = w.height(np.x, np.z) < WATER - 0.45;
        (sx, sz, spin) = slide_friction(sx, sz, spin, yaw, wet, dt);
        let (bx, bz) = (clampf(np.x, 4.0, MAP - 4.0), clampf(np.z, 4.0, MAP - 4.0));
        if bx != np.x {
            sx = 0.0;
        }
        if bz != np.z {
            sz = 0.0;
        }
        np.x = bx;
        np.z = bz;
        np.y = w.height(np.x, np.z);
        if sx * sx + sz * sz < CAR_REST_V * CAR_REST_V && spin.abs() < CAR_REST_SPIN {
            sx = 0.0;
            sz = 0.0;
            spin = 0.0;
        }
        if !(np.finite() && sx.is_finite() && sz.is_finite() && spin.is_finite() && yaw.is_finite()) {
            w.car_vel[k] = [0.0; 2];
            cp.spin[k] = 0.0;
            continue;
        }
        w.cars[k] = [np.x, np.y, np.z, yaw];
        w.car_vel[k] = [sx, sz];
        cp.spin[k] = spin;
    }
}

pub const HELI_VMAX: f32 = 45.0;
/// Top speed backwards and sideways.
pub const HELI_VSIDE: f32 = 14.0;
pub const HELI_CLIMB: f32 = 9.0;
/// Highest the model origin goes.
pub const HELI_CEIL: f32 = 260.0;
/// How fast the heading follows the view (rad/s).
pub const HELI_TURN: f32 = 1.3;
/// Offsets along the hull of the three circles it collides with, their
/// radius, and the height of the hull they stand for.
pub const HELI_CIRCLES: [f32; 3] = [-5.4, 0.0, 5.4];
pub const HELI_CR: f32 = 2.3;
pub const HELI_BODY_H: f32 = 3.4;
/// How deep the hull sits in water it has landed on.
pub const HELI_DRAUGHT: f32 = 0.7;

/// What a helicopter at `pos` heading `yaw` would stand on: the highest
/// ground or roof under its hull that is not above `feet` (or the water).
pub fn heli_floor(w: &World, pos: V3, yaw: f32, feet: f32) -> f32 {
    let f = fwd(yaw);
    let mut g = WATER - HELI_DRAUGHT;
    for off in [-5.6f32, 0.0, 5.6] {
        g = g.max(w.ground_below(pos.x + f.x * off, pos.z + f.z * off, feet, 1.6));
    }
    g
}

/// Push the hull of helicopter `k` (at `np`, heading `yaw`) out of walls,
/// trees, cars and the other helicopters. Returns the push applied.
pub fn heli_collide(w: &World, k: usize, np: &mut V3, yaw: f32) -> V3 {
    let (solid, cars, _) = heli_collide_parts(w, k, np, yaw, true);
    solid + cars
}

/// Like `heli_collide`, but tells the push by the cars (which give way:
/// the server shoves them along) from the rest, and how heavy the
/// heaviest car touched is. Without `with_cars` the cars are ignored (one
/// standing on the ground is far too heavy for a car to move aside).
pub fn heli_collide_parts(w: &World, k: usize, np: &mut V3, yaw: f32, with_cars: bool) -> (V3, V3, f32) {
    let f = fwd(yaw);
    let mut solid = V3::ZERO;
    let mut cars = V3::ZERO;
    let mut mass = 0.0f32;
    for _ in 0..2 {
        for off in HELI_CIRCLES {
            let base = V3::new(np.x + f.x * off, np.y + 0.2, np.z + f.z * off);
            let mut q = base;
            w.collide_body(&mut q, HELI_CR, HELI_BODY_H);
            solid.x += q.x - base.x;
            solid.z += q.z - base.z;
            let (bx, bz) = (q.x, q.z);
            for c in 0..if with_cars { w.cars.len() } else { 0 } {
                if w.push_car(c, &mut q, HELI_CR) {
                    mass = mass.max(car_mass(*w.car_kind.get(c).unwrap_or(&0)));
                }
            }
            cars.x += q.x - bx;
            cars.z += q.z - bz;
            let (bx, bz) = (q.x, q.z);
            for (j, o) in w.helis.iter().enumerate() {
                if j == k || (o[1] - np.y).abs() > HELI_TOP || (o[0] - q.x).abs() > 22.0 || (o[2] - q.z).abs() > 22.0 {
                    continue;
                }
                let of = fwd(o[3]);
                for ooff in HELI_CIRCLES {
                    let (dx, dz) = (q.x - (o[0] + of.x * ooff), q.z - (o[2] + of.z * ooff));
                    let d2 = dx * dx + dz * dz;
                    let rr = HELI_CR * 2.0;
                    if d2 < rr * rr && d2 > 1e-6 {
                        let dl = d2.sqrt();
                        q.x += dx / dl * (rr - dl);
                        q.z += dz / dl * (rr - dl);
                    }
                }
            }
            solid.x += q.x - bx;
            solid.z += q.z - bz;
            np.x += q.x - base.x;
            np.z += q.z - base.z;
        }
    }
    (solid, cars, mass)
}

/// One step of a flown helicopter: the stick moves it in the frame of its
/// heading, the heading follows the view, jump climbs and crouch descends.
/// On the ground without the climb held it stands still.
pub fn step_fly(w: &World, s: &mut MoveState, c: &Cmd, dt: f32) {
    let mut yaw = if s.airtime.is_finite() { s.airtime } else { 0.0 };
    let up = c.has(B_JUMP);
    let down = c.has(B_CROUCH) && !up;
    let alt = s.pos.y - heli_floor(w, s.pos, yaw, s.pos.y);
    // standing on its wheels (at most nudged along a little by a vehicle)
    let standing = s.on_ground && !up && s.vel.x * s.vel.x + s.vel.z * s.vel.z < HELI_STAND_V * HELI_STAND_V;
    let mut tv = V3::ZERO;
    let mut acc = 14.0;
    if !(s.on_ground && !up) {
        let dy = wrap_angle(c.yaw - yaw);
        yaw = wrap_angle(yaw + clampf(dy, -HELI_TURN * dt, HELI_TURN * dt));
        let (mz, mx) = (clampf(c.mz, -1.0, 1.0), clampf(c.mx, -1.0, 1.0));
        tv = fwd(yaw) * (mz * if mz >= 0.0 { HELI_VMAX } else { HELI_VSIDE }) + right(yaw) * (mx * HELI_VSIDE);
        acc = 10.0;
    }
    let (dvx, dvz) = (tv.x - s.vel.x, tv.z - s.vel.z);
    let dl = (dvx * dvx + dvz * dvz).sqrt();
    let k = if dl > acc * dt { acc * dt / dl } else { 1.0 };
    s.vel.x += dvx * k;
    s.vel.z += dvz * k;
    // the descent flares out near the ground: it cannot be flown into it
    let mut tvy = if up {
        HELI_CLIMB
    } else if down {
        -HELI_CLIMB.min(1.2 + alt.max(0.0) * 0.7)
    } else {
        0.0
    };
    if s.pos.y >= HELI_CEIL {
        tvy = tvy.min(0.0);
    }
    s.vel.y = approach(s.vel.y, tvy, 8.0 * dt);
    let mut np = s.pos + s.vel * dt;
    np.x = clampf(np.x, 10.0, MAP - 10.0);
    np.z = clampf(np.z, 10.0, MAP - 10.0);
    np.y = np.y.min(HELI_CEIL);
    let (push, cpush, cmass) = heli_collide_parts(w, s.car as usize, &mut np, yaw, !standing);
    let pl = push.len();
    if pl > 1e-4 {
        // what it flew into takes the speed towards it
        let n = push * (1.0 / pl);
        let vn = s.vel.dot(n);
        if vn < 0.0 {
            s.vel -= n * vn;
        }
    }
    let cl = cpush.len();
    if cl > 1e-4 {
        // a car gives way (the server shoves it along): that costs only
        // the little speed its weight is worth
        let n = cpush * (1.0 / cl);
        let vn = s.vel.dot(n);
        if vn < 0.0 {
            s.vel -= n * (vn * hit_share(HELI_MASS, cmass.max(CAR_MASS)));
        }
    }
    let g = heli_floor(w, np, yaw, s.pos.y.max(np.y));
    if np.y <= g {
        np.y = g;
        if s.vel.y < 0.0 {
            s.vel.y = 0.0;
        }
        // skidding along the ground
        let fr = (1.0 - 3.0 * dt).max(0.0);
        s.vel.x *= fr;
        s.vel.z *= fr;
        s.on_ground = true;
    } else {
        s.on_ground = false;
    }
    s.pos = np;
    s.airtime = yaw;
    s.crouch = false;
    s.in_water = false;
}

/// The ray through the crosshair of the over-the-shoulder camera, with its
/// origin advanced to the plane of the head so nothing behind the player is hit.
/// `scoped` (ADS with a >= 3.5x scope) uses the first-person eye instead.
pub fn aim_ray(pos: V3, yaw: f32, pitch: f32, crouch: bool, ads: bool, scoped: bool) -> (V3, V3) {
    let head = pos + V3::new(0.0, eye_h(crouch), 0.0);
    let d = dir_from(yaw, pitch);
    if scoped {
        // scoped weapons: first-person view from the eye
        return (head, d);
    }
    let r = right(yaw);
    let side = if ads { 0.38 } else { 0.55 };
    let back = if ads { 1.3 } else { 2.7 };
    let cam = head + r * side + V3::UP * 0.28 - d * back;
    let t0 = (head - cam).dot(d).max(0.0);
    (cam + d * t0, d)
}

/// Server-side aim ray: like `aim_ray`, but the shoulder offset is clamped by
/// raycasts from the head so the origin can never start inside a wall the
/// player is hugging (shots through walls).
pub fn aim_ray_safe(w: &World, pos: V3, yaw: f32, pitch: f32, crouch: bool, ads: bool, scoped: bool) -> (V3, V3) {
    let head = pos + V3::new(0.0, eye_h(crouch), 0.0);
    let d = dir_from(yaw, pitch);
    if scoped {
        return (head, d);
    }
    let r = right(yaw);
    let side_max = if ads { 0.38 } else { 0.55 };
    let side = match w.raycast(head, r, side_max + 0.2) {
        Some((t, _, _)) => clampf(t - 0.2, 0.0, side_max),
        None => side_max,
    };
    let sh = head + r * side;
    let up = match w.raycast(sh, V3::UP, 0.28 + 0.15) {
        Some((t, _, _)) => clampf(t - 0.15, 0.0, 0.28),
        None => 0.28,
    };
    let back = if ads { 1.3 } else { 2.7 };
    let cam = sh + V3::UP * up - d * back;
    let t0 = (head - cam).dot(d).max(0.0);
    (cam + d * t0, d)
}

/// Approximate muzzle position (visual start of tracers).
/// Where a player's head and body are drawn (mirrors `emitCharacter` in
/// src/render/models.ts), so hits match the model: the torso leans forward
/// around the hips (more when crouched or sprinting), the head pitches with
/// the view, and a free-falling body is tilted face down around the hips.
#[derive(Clone, Copy, Debug)]
pub struct BodyPose {
    /// Centre of the head.
    pub head: V3,
    /// Body capsule axis, from just above the feet to below the head.
    pub a: V3,
    pub b: V3,
    /// The hip pivot (on the axis): legs below it, the torso above.
    pub hip: V3,
    /// The drawn torso's frame (models.ts `torso`): its base, leaning.
    pub torso: Frame,
    lean: f32,
    pitch: f32,
    mode: u8,
}

/// A rigid frame of the client's model (models.ts `T`): origin and the world
/// images of the local x, y and z axes (its quaternion as a matrix).
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub p: V3,
    pub m: [V3; 3],
}

/// Rotations as the client's qX / qY / qZ (right-handed; qX turns +y towards
/// +z, the model's forward).
fn rot_x(a: f32) -> [V3; 3] {
    let (s, c) = a.sin_cos();
    [V3::new(1.0, 0.0, 0.0), V3::new(0.0, c, s), V3::new(0.0, -s, c)]
}
fn rot_y(a: f32) -> [V3; 3] {
    let (s, c) = a.sin_cos();
    [V3::new(c, 0.0, -s), V3::new(0.0, 1.0, 0.0), V3::new(s, 0.0, c)]
}
fn rot_z(a: f32) -> [V3; 3] {
    let (s, c) = a.sin_cos();
    [V3::new(c, s, 0.0), V3::new(-s, c, 0.0), V3::new(0.0, 0.0, 1.0)]
}
/// qMul(a, b): b first, then a.
fn rot_mul(a: [V3; 3], b: [V3; 3]) -> [V3; 3] {
    let f = |v: V3| a[0] * v.x + a[1] * v.y + a[2] * v.z;
    [f(b[0]), f(b[1]), f(b[2])]
}

impl Frame {
    pub fn at(&self, v: [f32; 3]) -> V3 {
        self.p + self.m[0] * v[0] + self.m[1] * v[1] + self.m[2] * v[2]
    }
    /// models.ts `child`: a pivot in this frame, then a local rotation.
    fn child(&self, pivot: [f32; 3], q: [V3; 3]) -> Frame {
        Frame { p: self.at(pivot), m: rot_mul(self.m, q) }
    }
    /// Ray vs a box of the model (centre and full size in this frame, as
    /// models.ts `box`): the entry distance.
    pub fn ray_box(&self, o: V3, d: V3, c: [f32; 3], size: [f32; 3], maxt: f32) -> Option<f32> {
        ray_obb(o, d, self.at(c), self.m, [size[0] * 0.5, size[1] * 0.5, size[2] * 0.5], maxt)
    }
}

/// What the client draws besides the pose: the held weapon and the snapshot
/// flags that move the arms (models.ts emitCharacter).
#[derive(Clone, Copy, Default)]
pub struct Gear {
    pub weapon: u8,
    pub ads: bool,
    pub firing: bool,
    pub healing: bool,
    pub reloading: bool,
    pub vest: bool,
    pub helm: bool,
}

/// The client model's torso boxes in the torso frame: centre and full size
/// (models.ts emitCharacter: shirt, vest (when worn), backpack).
pub const SHIRT_BOX: ([f32; 3], [f32; 3]) = ([0.0, 0.3, 0.0], [0.46, 0.56, 0.26]);
pub const VEST_BOX: ([f32; 3], [f32; 3]) = ([0.0, 0.27, 0.0], [0.5, 0.42, 0.31]);
pub const PACK_BOX: ([f32; 3], [f32; 3]) = ([0.0, 0.32, -0.2], [0.34, 0.42, 0.14]);

/// The client's gun models (models.ts GUNS): centre and full size per box.
const GUN_BOXES: [&[[f32; 6]]; 7] = [
    &[],
    &[[0.0, 0.05, 0.07, 0.05, 0.075, 0.2], [0.0, -0.04, -0.01, 0.045, 0.12, 0.06]],
    &[[0.0, 0.04, 0.1, 0.06, 0.1, 0.34], [0.0, -0.09, 0.13, 0.04, 0.17, 0.05], [0.0, 0.03, -0.15, 0.04, 0.07, 0.17], [0.0, 0.05, 0.32, 0.03, 0.03, 0.12]],
    &[[0.0, 0.04, 0.12, 0.06, 0.1, 0.5], [0.0, -0.1, 0.17, 0.045, 0.18, 0.07], [0.0, 0.02, -0.22, 0.05, 0.11, 0.25], [0.0, 0.06, 0.48, 0.03, 0.03, 0.24], [0.0, 0.04, 0.3, 0.07, 0.08, 0.16]],
    &[[0.0, 0.03, 0.17, 0.06, 0.09, 0.62], [0.0, -0.03, 0.33, 0.07, 0.06, 0.18], [0.0, 0.0, -0.2, 0.05, 0.12, 0.28]],
    &[[0.0, 0.04, 0.15, 0.06, 0.1, 0.6], [0.0, 0.13, 0.1, 0.05, 0.05, 0.22], [0.0, -0.08, 0.14, 0.04, 0.13, 0.06], [0.0, 0.02, -0.22, 0.05, 0.11, 0.24], [0.0, 0.06, 0.55, 0.03, 0.03, 0.22]],
    &[[0.0, 0.04, 0.2, 0.06, 0.1, 0.8], [0.0, 0.14, 0.12, 0.06, 0.06, 0.32], [0.0, 0.14, 0.29, 0.07, 0.07, 0.03], [0.0, -0.07, 0.12, 0.04, 0.1, 0.06], [0.0, 0.01, -0.27, 0.06, 0.13, 0.3], [0.0, 0.06, 0.74, 0.035, 0.035, 0.36]],
];

impl BodyPose {
    /// Ray vs the head: the 0.2 m sphere on the drawn head, and the drawn
    /// head boxes themselves (face, hair or helmet: the helmet and the box's
    /// corners reach beyond the sphere). The entry distance.
    pub fn ray_head(&self, o: V3, d: V3, maxt: f32, helm: bool) -> Option<f32> {
        let head = self.torso.child([0.0, 0.6, 0.0], rot_x(-self.pitch * 0.55 - self.lean * 0.6));
        let top = if helm { ([0.0, 0.25, 0.0], [0.3, 0.13, 0.31]) } else { ([0.0, 0.27, -0.01], [0.26, 0.05, 0.27]) };
        [ray_sphere(o, d, self.head, 0.2, maxt), head.ray_box(o, d, [0.0, 0.14, 0.01], [0.25, 0.27, 0.27], maxt), head.ray_box(o, d, top.0, top.1, maxt)]
            .into_iter()
            .flatten()
            .min_by(|a, b| a.total_cmp(b))
    }

    /// Ray vs the drawn torso (shirt, vest, backpack): the entry distance.
    pub fn ray_torso(&self, o: V3, d: V3, maxt: f32, vest: bool) -> Option<f32> {
        [Some(SHIRT_BOX), vest.then_some(VEST_BOX), Some(PACK_BOX)]
            .iter()
            .flatten()
            .filter_map(|(c, s)| self.torso.ray_box(o, d, *c, *s, maxt))
            .min_by(|a, b| a.total_cmp(b))
    }

    /// Does the ray pass through a drawn part of the upper body before
    /// `maxt`: the torso, the arms and hands, the held weapon or the medkit
    /// (models.ts emitCharacter; the walking arm swing of an unarmed player
    /// is not known here and taken at rest)?
    pub fn ray_upper(&self, o: V3, d: V3, maxt: f32, g: Gear) -> bool {
        if self.ray_torso(o, d, maxt, g.vest).is_some() {
            return true;
        }
        let (fall, para) = (self.mode == MODE_FALL, self.mode == MODE_PARA);
        let gun = (g.weapon as usize) < GUN_BOXES.len() && g.weapon > 0;
        let arm_pitch = -std::f32::consts::FRAC_PI_2 - self.pitch * 0.85 - self.lean;
        for s in [1.0f32, -1.0] {
            let (q, elbow) = if fall {
                (rot_mul(rot_z(s * 1.25), rot_x(-0.3)), -0.2)
            } else if para {
                (rot_mul(rot_z(s * 0.25), rot_x(-std::f32::consts::PI + 0.25)), -0.3)
            } else if g.healing {
                (rot_mul(rot_y(s * -0.4), rot_x(-1.1)), -0.9)
            } else if gun {
                if s > 0.0 {
                    (rot_mul(rot_y(-0.55), rot_x(arm_pitch + if g.reloading { 0.7 } else { 0.05 })), if g.reloading { -0.9 } else { -0.35 })
                } else {
                    (rot_mul(rot_y(-0.1), rot_x(arm_pitch + 0.25)), -0.75)
                }
            } else if g.firing && s < 0.0 {
                (rot_x(arm_pitch), 0.0)
            } else {
                (rot_z(s * 0.08), -0.35)
            };
            let arm = self.torso.child([0.29 * s, 0.5, 0.0], q);
            if arm.ray_box(o, d, [0.0, -0.15, 0.0], [0.13, 0.32, 0.13], maxt).is_some() {
                return true;
            }
            let fore = arm.child([0.0, -0.3, 0.0], rot_x(elbow));
            if fore.ray_box(o, d, [0.0, -0.14, 0.0], [0.12, 0.3, 0.12], maxt).is_some() || fore.ray_box(o, d, [0.0, -0.31, 0.0], [0.1, 0.1, 0.1], maxt).is_some() {
                return true;
            }
        }
        if gun && !fall && !para {
            let gp = if g.ads { [-0.06, 0.52, 0.34] } else { [-0.13, 0.42, 0.36] };
            let gq = if g.reloading || g.healing { rot_mul(rot_x(0.65), rot_z(0.3)) } else { rot_x(-self.pitch - self.lean) };
            let gf = self.torso.child(gp, gq);
            if GUN_BOXES[g.weapon as usize].iter().any(|b| gf.ray_box(o, d, [b[0], b[1], b[2]], [b[3], b[4], b[5]], maxt).is_some()) {
                return true;
            }
        }
        g.healing && self.torso.ray_box(o, d, [0.0, 0.25, 0.42], [0.16, 0.1, 0.1], maxt).is_some()
    }
}

/// Horizontal speed above which the model is drawn sprinting (leaning).
pub const SPRINT_DRAW_SPEED: f32 = 6.2;

pub fn body_pose(pos: V3, yaw: f32, pitch: f32, crouch: bool, sprint: bool, mode: u8) -> BodyPose {
    let ground = mode == MODE_GROUND;
    let crouch = crouch && ground;
    let lean: f32 = if !ground {
        0.0
    } else if crouch {
        0.15
    } else if sprint {
        0.22
    } else {
        0.05
    };
    let hip = if crouch { 0.5 } else { 0.95 };
    // head pivot on top of the torso, then the head box centre (0, 0.14, 0.01)
    // turned by lean + head pitch (-pitch * 0.55 - lean * 0.6)
    let a = lean * 0.4 - pitch.clamp(-1.2, 1.2) * 0.55;
    let head = (hip + 0.02 + 0.6 * lean.cos() + 0.14 * a.cos() - 0.01 * a.sin(), 0.6 * lean.sin() + 0.14 * a.sin() + 0.01 * a.cos());
    let lo = (0.3, 0.0);
    let hi = (head.0 - 0.3, 0.0);
    let tilt = |p: (f32, f32)| {
        if mode == MODE_FALL {
            let (sn, cs) = 1.25f32.sin_cos();
            let (y, z) = (p.0 - 0.95, p.1);
            (0.95 + y * cs - z * sn, y * sn + z * cs)
        } else {
            p
        }
    };
    let f = fwd(yaw);
    let world = |p: (f32, f32)| {
        let p = tilt(p);
        V3::new(pos.x + f.x * p.1, pos.y + p.0, pos.z + f.z * p.1)
    };
    // The model's frames as the client composes them: qY(yaw) (times qX(1.25)
    // about the hips in free fall), the hips, then the torso turned by the
    // lean about its base (hip + 0.02).
    let root = if mode == MODE_FALL {
        let q = rot_mul(rot_y(yaw), rot_x(1.25));
        Frame { p: pos + V3::new(0.0, 0.95, 0.0) + q[1] * -0.95, m: q }
    } else {
        Frame { p: pos, m: rot_y(yaw) }
    };
    BodyPose {
        head: world(head),
        a: world(lo),
        b: world(hi),
        hip: world((hip, 0.0)),
        torso: root.child([0.0, hip + 0.02, 0.0], rot_x(lean)),
        lean,
        pitch: pitch.clamp(-1.2, 1.2),
        mode,
    }
}

pub fn muzzle(pos: V3, yaw: f32, pitch: f32, crouch: bool) -> V3 {
    let head = pos + V3::new(0.0, eye_h(crouch) - 0.12, 0.0);
    head + right(yaw) * 0.22 + dir_from(yaw, pitch) * 0.55
}
