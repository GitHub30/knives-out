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
