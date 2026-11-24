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
            s.crouch = c.has(B_CROUCH) && !s.in_water;
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
                w.collide_body(&mut np, P_RADIUS, hgt);
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
pub fn muzzle(pos: V3, yaw: f32, pitch: f32, crouch: bool) -> V3 {
    let head = pos + V3::new(0.0, eye_h(crouch) - 0.12, 0.0);
    head + right(yaw) * 0.22 + dir_from(yaw, pitch) * 0.55
}
