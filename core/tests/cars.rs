//! Drivable cars: getting in and out, driving, collisions, running people
//! over and bullets against the bodywork.

use zonefall_core::defs::*;
use zonefall_core::math::*;
use zonefall_core::net::*;
use zonefall_core::player::*;
use zonefall_core::sim::*;
use zonefall_core::world::*;

fn two_humans(seed: u32) -> (Match, u8, u8) {
    let mut m = Match::new(seed, 2, FLAG_OFFLINE, 99);
    let a = m.add_human(b"A").unwrap();
    let b = m.add_human(b"B").unwrap();
    m.step();
    assert_eq!(m.phase, PH_PLAY);
    (m, a, b)
}

/// Flat open ground with 75 m of clear room along +x.
fn open_run(m: &Match) -> V3 {
    for gz in (80..940).step_by(11) {
        for gx in (80..860).step_by(11) {
            let (x, z) = (gx as f32, gz as f32);
            let h0 = m.w.height(x, z);
            if h0 < 3.0 || m.w.building_at(x, z).is_some() {
                continue;
            }
            let mut ok = true;
            for k in 0..75 {
                for dz in [-3.0f32, 0.0, 3.0] {
                    let h = m.w.height(x + k as f32, z + dz);
                    if (h - h0).abs() > 1.0 || h < 2.0 {
                        ok = false;
                    }
                }
            }
            for y in [0.4f32, 1.0, 1.6] {
                for dz in [-2.5f32, 0.0, 2.5] {
                    if m.w.raycast(V3::new(x - 4.0, h0 + y + 0.25, z + dz), V3::new(1.0, 0.0, 0.0), 80.0).is_some() {
                        ok = false;
                    }
                }
            }
            if ok {
                return V3::new(x, h0, z);
            }
        }
    }
    panic!("no open ground");
}

fn place(m: &mut Match, pid: u8, pos: V3) {
    let y = m.w.ground_below(pos.x, pos.z, pos.y + 3.0, P_RADIUS);
    m.players[pid as usize].ms = MoveState { pos: V3::new(pos.x, y, pos.z), mode: MODE_GROUND, on_ground: true, ..Default::default() };
}

struct Pad {
    seq: u32,
}
impl Pad {
    /// One command for `pid`, then one tick.
    fn go(&mut self, m: &mut Match, pid: u8, mx: i8, mz: i8, yaw: f32, pitch: f32, buttons: u16) {
        self.seq += 1;
        let mut v = vec![C_INPUT, 1u8];
        v.extend_from_slice(&self.seq.to_le_bytes());
        v.extend_from_slice(&(m.tick * 256).to_le_bytes());
        v.push(mx as u8);
        v.push(mz as u8);
        v.extend_from_slice(&q_yaw(yaw).to_le_bytes());
        v.extend_from_slice(&q_pitch(pitch).to_le_bytes());
        v.extend_from_slice(&buttons.to_le_bytes());
        m.on_message(pid, &v);
        m.step();
    }
}

/// Park car 0 on open ground facing +x, everything else far away.
fn stage(seed: u32) -> (Match, u8, u8, V3) {
    let (mut m, a, b) = two_humans(seed);
    let o = open_run(&m);
    assert!(!m.w.cars.is_empty(), "the world has no cars");
    for k in 1..m.w.cars.len() {
        // keep the other cars out of the way of the test track
        if dist_xz(m.w.cars[k][0], m.w.cars[k][2], o.x + 35.0, o.z) < 60.0 {
            m.w.cars[k][0] = 5.0;
            m.w.cars[k][2] = 5.0;
        }
    }
    m.w.cars[0] = [o.x, m.w.height(o.x, o.z), o.z, std::f32::consts::FRAC_PI_2];
    place(&mut m, b, V3::new(o.x - 30.0, o.y, o.z + 30.0));
    place(&mut m, a, V3::new(o.x, o.y, o.z + 1.6));
    (m, a, b, o)
}

#[test]
fn enter_drive_and_leave() {
    for seed in [7u32, 42] {
        let (mut m, a, _b, o) = stage(seed);
        let mut pad = Pad { seq: 0 };
        pad.go(&mut m, a, 0, 0, 0.0, 0.0, 0);
        pad.go(&mut m, a, 0, 0, 0.0, 0.0, B_USE);
        let p = &m.players[a as usize];
        assert_eq!(p.ms.mode, MODE_DRIVE, "seed {seed}: did not get in");
        assert_eq!(m.car_driver[0], a);
        // holding the button does not hop straight back out
        pad.go(&mut m, a, 0, 0, 0.0, 0.0, B_USE);
        assert_eq!(m.players[a as usize].ms.mode, MODE_DRIVE);
        let mut top = 0.0f32;
        for _ in 0..90 {
            pad.go(&mut m, a, 0, 127, 0.0, 0.0, 0);
            top = top.max(car_speed(&m.players[a as usize].ms));
        }
        let c = m.w.cars[0];
        assert!(c[0] - o.x > 25.0, "seed {seed}: the car only moved {} m in 3 s", c[0] - o.x);
        assert!((c[2] - o.z).abs() < 1.0, "seed {seed}: drifted sideways {}", c[2] - o.z);
        assert!(top > 12.0 && top <= CAR_VMAX + 0.01, "seed {seed}: top speed {top}");
        assert_eq!(m.players[a as usize].ms.pos.x, c[0]);
        // too fast to jump out
        pad.go(&mut m, a, 0, 127, 0.0, 0.0, B_USE);
        assert_eq!(m.players[a as usize].ms.mode, MODE_DRIVE, "seed {seed}: left a speeding car");
        // steering turns the car
        let y0 = m.w.cars[0][3];
        for _ in 0..20 {
            pad.go(&mut m, a, 127, 60, 0.0, 0.0, 0);
        }
        assert!(wrap_angle(m.w.cars[0][3] - y0).abs() > 0.2, "seed {seed}: steering did nothing");
        // brake to a stop and get out
        for _ in 0..90 {
            pad.go(&mut m, a, 0, 0, 0.0, 0.0, B_JUMP);
        }
        assert!(car_speed(&m.players[a as usize].ms).abs() < 0.1);
        pad.go(&mut m, a, 0, 0, 0.0, 0.0, 0);
        pad.go(&mut m, a, 0, 0, 0.0, 0.0, B_USE);
        let p = &m.players[a as usize];
        assert_eq!(p.ms.mode, MODE_GROUND, "seed {seed}: did not get out");
        assert_eq!(m.car_driver[0], 255);
        let mut q = p.ms.pos;
        assert!(!m.w.push_car(0, &mut q, P_RADIUS - 0.01), "seed {seed}: got out inside the car");
        let c = m.w.cars[0];
        assert!(dist_xz(p.ms.pos.x, p.ms.pos.z, c[0], c[2]) < 4.5);
    }
}

#[test]
fn cars_are_solid_for_walkers_and_bullets() {
    let (mut m, a, b, o) = stage(7);
    let mut pad = Pad { seq: 0 };
    // walk at the side of the car for two seconds: never inside it
    place(&mut m, a, V3::new(o.x, o.y, o.z + 4.0));
    let yaw = yaw_to(0.0, -1.0);
    for _ in 0..60 {
        pad.go(&mut m, a, 0, 127, yaw, 0.0, 0);
        let mut q = m.players[a as usize].ms.pos;
        assert!(!m.w.push_car(0, &mut q, P_RADIUS - 0.02), "walked into the car at {:?}", m.players[a as usize].ms.pos);
    }
    assert!(m.players[a as usize].ms.pos.z > o.z + CAR_HW, "walked through the car");
    // a bullet across the bonnet stops at the car
    place(&mut m, a, V3::new(o.x, o.y, o.z + 6.0));
    place(&mut m, b, V3::new(o.x, o.y, o.z - 6.0));
    {
        let p = &mut m.players[a as usize];
        p.slots[0] = (W_AR, 30);
        p.cur = 0;
        p.switch_t = 0.0;
    }
    let from = m.players[a as usize].ms.pos + V3::new(0.0, 0.8, 0.0);
    let to = m.players[b as usize].ms.pos + V3::new(0.0, 0.8, 0.0);
    assert!(m.w.blocked(from, to), "the car does not block the line");
    let hp = m.players[b as usize].health;
    let pitch = -0.12;
    for _ in 0..20 {
        pad.go(&mut m, a, 0, 0, yaw, pitch, B_FIRE | B_ADS);
        pad.go(&mut m, a, 0, 0, yaw, pitch, B_ADS);
    }
    assert!(m.players[a as usize].slots[0].1 < 30, "never fired");
    assert_eq!(m.players[b as usize].health, hp, "shot through the car");
    // the same shots hurt someone sitting in it, but less than a body hit
    place(&mut m, b, V3::new(o.x, o.y, o.z - 1.6));
    let mut padb = Pad { seq: 0 };
    padb.go(&mut m, b, 0, 0, 0.0, 0.0, 0);
    padb.go(&mut m, b, 0, 0, 0.0, 0.0, B_USE);
    assert_eq!(m.players[b as usize].ms.mode, MODE_DRIVE);
    m.players[a as usize].slots[0].1 = 30;
    m.players[a as usize].reload_t = 0.0;
    let hp = m.players[b as usize].health;
    let mut n = 0;
    for _ in 0..12 {
        let before = m.players[a as usize].slots[0].1;
        pad.go(&mut m, a, 0, 0, yaw, pitch, B_FIRE | B_ADS);
        pad.go(&mut m, a, 0, 0, yaw, pitch, B_ADS);
        n += before - m.players[a as usize].slots[0].1;
        if n >= 3 {
            break;
        }
    }
    let lost = hp - m.players[b as usize].health;
    assert!(lost > 5.0, "bullets into an occupied car did nothing");
    assert!(lost < WD[W_AR as usize].dmg * n as f32 * 0.7, "the car gave no protection: {lost} from {n} shots");
}

#[test]
fn running_someone_over_hurts_and_a_dead_driver_frees_the_car() {
    let (mut m, a, b, o) = stage(42);
    let mut pad = Pad { seq: 0 };
    pad.go(&mut m, a, 0, 0, 0.0, 0.0, 0);
    pad.go(&mut m, a, 0, 0, 0.0, 0.0, B_USE);
    assert_eq!(m.players[a as usize].ms.mode, MODE_DRIVE);
    place(&mut m, b, V3::new(o.x + 40.0, o.y, o.z));
    let hp = m.players[b as usize].health;
    for _ in 0..120 {
        pad.go(&mut m, a, 0, 127, 0.0, 0.0, 0);
        let mut q = m.players[b as usize].ms.pos;
        if m.players[b as usize].alive {
            assert!(!m.w.push_car(0, &mut q, P_RADIUS - 0.05), "left standing inside the car");
        }
    }
    let lost = hp - m.players[b as usize].health;
    assert!(lost > 30.0, "run over at speed for only {lost}");
    assert!(m.players[a as usize].dmg_dealt > 30.0, "the driver got no credit");
    // the driver dies at the wheel: the car is free again
    m.damage(a as usize, KILLER_ZONE, 500.0, false, 0, V3::ZERO);
    assert!(!m.players[a as usize].alive);
    assert_eq!(m.car_driver[0], 255);
}

#[test]
fn a_car_stops_at_walls_and_stays_out_of_buildings() {
    let (mut m, a, _b, _o) = stage(7);
    let mut pad = Pad { seq: 0 };
    // aim the car at the nearest building from 25 m out
    let bl = m.w.buildings[0];
    let (bx, bz) = ((bl.x0 + bl.x1) * 0.5, (bl.z0 + bl.z1) * 0.5);
    let (sx, sz) = (bl.x0 - 25.0, bz);
    m.w.cars[0] = [sx, m.w.height(sx, sz), sz, std::f32::consts::FRAC_PI_2];
    place(&mut m, a, V3::new(sx, 0.0, sz + 1.6));
    m.players[a as usize].ms.pos.y = m.w.height(sx, sz + 1.6);
    pad.go(&mut m, a, 0, 0, 0.0, 0.0, 0);
    pad.go(&mut m, a, 0, 0, 0.0, 0.0, B_USE);
    assert_eq!(m.players[a as usize].ms.mode, MODE_DRIVE);
    for _ in 0..240 {
        pad.go(&mut m, a, 0, 127, 0.0, 0.0, 0);
        let c = m.w.cars[0];
        assert!(!bl.contains(c[0], c[2]), "drove into the building at {:?} (centre {bx} {bz})", c);
        assert!(m.players[a as usize].ms.pos.finite());
    }
}

#[test]
fn no_getting_in_or_out_through_a_wall() {
    let (mut m, a, _b, _o) = stage(7);
    let mut pad = Pad { seq: 0 };
    // park the car flush against the west wall of a building, nose north
    let bl = *m.w.buildings.iter().find(|b| b.x1 - b.x0 > 5.0 && b.z1 - b.z0 > 6.0).expect("a building");
    let (cx, cz) = (bl.x0 - CAR_HW - 0.1, (bl.z0 + bl.z1) * 0.5);
    for k in 1..m.w.cars.len() {
        m.w.cars[k][0] = 5.0;
        m.w.cars[k][2] = 5.0;
    }
    m.w.cars[0] = [cx, m.w.height(cx, cz), cz, 0.0];
    // from inside the building, a step from the wall: the car is within reach
    // but behind the wall
    let mut inside = None;
    for dz in [-2.0f32, -1.0, 0.0, 1.0, 2.0] {
        let at = V3::new(bl.x0 + 0.75, bl.y, cz + dz);
        if m.w.blocked_static(at + V3::new(0.0, 1.0, 0.0), V3::new(cx, m.w.cars[0][1] + 1.0, cz)) {
            let mut q = at;
            if m.w.push_car(0, &mut q, 1.5) {
                inside = Some(at);
                break;
            }
        }
    }
    let inside = inside.expect("no spot behind a solid stretch of wall");
    m.players[a as usize].ms = MoveState { pos: inside, mode: MODE_GROUND, on_ground: true, ..Default::default() };
    pad.go(&mut m, a, 0, 0, 0.0, 0.0, 0);
    pad.go(&mut m, a, 0, 0, 0.0, 0.0, B_USE);
    assert_eq!(m.players[a as usize].ms.mode, MODE_GROUND, "got into a car through a wall");
    // from the outside it works, and getting out never lands in the building
    let out = V3::new(cx - 1.6, m.w.height(cx - 1.6, cz), cz);
    m.players[a as usize].ms = MoveState { pos: out, mode: MODE_GROUND, on_ground: true, ..Default::default() };
    pad.go(&mut m, a, 0, 0, 0.0, 0.0, 0);
    pad.go(&mut m, a, 0, 0, 0.0, 0.0, B_USE);
    assert_eq!(m.players[a as usize].ms.mode, MODE_DRIVE);
    pad.go(&mut m, a, 0, 0, 0.0, 0.0, 0);
    pad.go(&mut m, a, 0, 0, 0.0, 0.0, B_USE);
    let p = m.players[a as usize].ms;
    assert_eq!(p.mode, MODE_GROUND);
    assert!(!bl.contains(p.pos.x, p.pos.z), "got out of the car into the building at {:?}", p.pos);
}

#[test]
fn every_town_has_a_fire_truck_and_it_drives() {
    for seed in [7u32, 42, 2024] {
        let w = World::generate(seed);
        let trucks = w.car_kind.iter().filter(|&&k| k == VK_TRUCK).count();
        assert_eq!(w.cars.len(), w.car_kind.len());
        assert!(trucks >= w.towns.len() / 2, "seed {seed}: only {trucks} fire trucks for {} towns", w.towns.len());
        assert_eq!(w.car_kind[0], VK_CAR);
        // nothing is parked inside anything else
        for a in 0..w.cars.len() {
            for b in 0..a {
                let d = dist_xz(w.cars[a][0], w.cars[a][2], w.cars[b][0], w.cars[b][2]);
                assert!(d > 5.0, "seed {seed}: vehicles {a} and {b} are {d} m apart");
            }
        }
    }
    let (mut m, a, _b, o) = stage(7);
    let t = m.w.car_kind.iter().position(|&k| k == VK_TRUCK).expect("a fire truck");
    m.w.cars[0] = [5.0, 0.0, 5.0, 0.0];
    m.w.cars[t] = [o.x, m.w.height(o.x, o.z), o.z, std::f32::consts::FRAC_PI_2];
    // beside the cab: 1.25 m of half width and the body radius away
    place(&mut m, a, V3::new(o.x, o.y, o.z + 1.25 + 0.7));
    let mut pad = Pad { seq: 0 };
    pad.go(&mut m, a, 0, 0, 0.0, 0.0, 0);
    pad.go(&mut m, a, 0, 0, 0.0, 0.0, B_USE);
    assert_eq!(m.players[a as usize].ms.mode, MODE_DRIVE, "did not get into the truck");
    assert_eq!(m.car_driver[t], a);
    let mut top = 0.0f32;
    for _ in 0..120 {
        pad.go(&mut m, a, 0, 127, 0.0, 0.0, 0);
        top = top.max(car_speed(&m.players[a as usize].ms));
    }
    assert!(m.w.cars[t][0] - o.x > 25.0, "the truck only moved {} m in 4 s", m.w.cars[t][0] - o.x);
    assert!(top > 9.0 && top < CAR_VMAX - 3.0, "truck top speed {top}: it should be slower than a car");
    for _ in 0..90 {
        pad.go(&mut m, a, 0, 0, 0.0, 0.0, B_JUMP);
    }
    pad.go(&mut m, a, 0, 0, 0.0, 0.0, 0);
    pad.go(&mut m, a, 0, 0, 0.0, 0.0, B_USE);
    let p = m.players[a as usize].ms;
    assert_eq!(p.mode, MODE_GROUND);
    let mut q = p.pos;
    assert!(!m.w.push_car(t, &mut q, P_RADIUS - 0.01), "got out inside the truck");
    // tall enough to hide behind, and it stops bullets up there
    let c = m.w.cars[t];
    let side = right(c[3]);
    let from = V3::new(c[0] + side.x * 6.0, c[1] + 2.4, c[2] + side.z * 6.0);
    let to = V3::new(c[0] - side.x * 6.0, c[1] + 2.4, c[2] - side.z * 6.0);
    assert!(m.w.blocked(from, to), "the truck body does not block at 2.4 m");
}
