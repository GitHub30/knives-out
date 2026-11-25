//! Doors: shut ones are solid, the use button swings the nearest one, a door
//! is not shut on somebody in the doorway, and bots let themselves in.

use zonefall_core::defs::*;
use zonefall_core::math::*;
use zonefall_core::net::*;
use zonefall_core::player::*;
use zonefall_core::sim::*;

fn game(seed: u32) -> (Match, u8) {
    let mut m = Match::new(seed, 2, FLAG_OFFLINE, 99);
    let a = m.add_human(b"A").unwrap();
    let _b = m.add_human(b"B").unwrap();
    m.step();
    assert_eq!(m.phase, PH_PLAY);
    (m, a)
}

struct Pad {
    seq: u32,
    yaw: f32,
}
impl Pad {
    fn send(&mut self, m: &mut Match, a: u8, mz: i8, b: u16) {
        self.seq += 1;
        let mut v = vec![C_INPUT, 1u8];
        v.extend_from_slice(&self.seq.to_le_bytes());
        v.extend_from_slice(&(m.tick * 256).to_le_bytes());
        v.push(0);
        v.push(mz as u8);
        v.extend_from_slice(&q_yaw(self.yaw).to_le_bytes());
        v.extend_from_slice(&q_pitch(0.0).to_le_bytes());
        v.extend_from_slice(&b.to_le_bytes());
        m.on_message(a, &v);
        m.step();
    }
}

/// A door at ground level (no step up to it): door index, outside and inside points.
fn level_door(m: &Match) -> (usize, (f32, f32), (f32, f32)) {
    for bl in &m.w.buildings {
        let mid = ((bl.door_out.0 + bl.door_in.0) * 0.5, (bl.door_out.1 + bl.door_in.1) * 0.5);
        let Some(k) = (0..m.w.doors.len()).find(|&k| {
            let d = m.w.doors[k];
            let c = d.hinge + fwd(d.shut) * (d.w * 0.5);
            dist_xz(c.x, c.z, mid.0, mid.1) < 0.2 && (d.hinge.y - bl.y).abs() < 0.01
        }) else { continue };
        if (bl.y - m.w.height(bl.door_out.0, bl.door_out.1)).abs() < 0.12 && m.w.cars.iter().all(|c| dist_xz(c[0], c[2], mid.0, mid.1) > 9.0) {
            return (k, bl.door_out, bl.door_in);
        }
    }
    panic!("no level door");
}

#[test]
fn houses_have_doors_and_all_start_shut() {
    for seed in [1, 7, 42] {
        let (m, _) = game(seed);
        assert!(m.w.doors.len() >= 20, "seed {seed}: {} doors", m.w.doors.len());
        assert!(m.w.door_open.iter().all(|o| !o));
        assert!(m.w.doors.len() < 65535);
    }
}

#[test]
fn a_door_opens_for_a_walker_and_can_be_shut_behind() {
    let (mut m, a) = game(7);
    let (k, out, inn) = level_door(&m);
    let i = a as usize;
    let mut pad = Pad { seq: 0, yaw: yaw_to(inn.0 - out.0, inn.1 - out.1) };
    m.players[i].ms = MoveState { pos: V3::new(out.0, m.w.height(out.0, out.1), out.1), mode: MODE_GROUND, on_ground: true, ..Default::default() };
    pad.send(&mut m, a, 0, 0);
    // shut, it stops a shot; standing in front of it does not open it
    let eye = V3::new(out.0, m.players[i].ms.pos.y + 1.4, out.1);
    let d = V3::new(inn.0 - out.0, 0.0, inn.1 - out.1).norm();
    assert!(!m.w.door_open[k] && m.w.raycast(eye, d, 3.0).is_some());
    // walking up to it does
    let c = m.w.doors[k].hinge + fwd(m.w.doors[k].shut) * (m.w.doors[k].w * 0.5);
    for _ in 0..200 {
        let p = m.players[i].ms.pos;
        if dist_xz(p.x, p.z, c.x, c.z) < 0.25 {
            break;
        }
        pad.send(&mut m, a, 127, 0);
    }
    assert!(m.w.door_open[k], "did not open");
    let p = m.players[i].ms.pos;
    assert!(dist_xz(p.x, p.z, c.x, c.z) < 0.3, "stopped at the door: {p:?}");
    // it cannot be shut on the walker standing in it
    pad.send(&mut m, a, 0, 0);
    pad.send(&mut m, a, 0, B_USE);
    assert!(m.w.door_open[k], "shut on the player standing in it");
    pad.send(&mut m, a, 0, 0);
    for _ in 0..9 {
        pad.send(&mut m, a, 127, 0);
    }
    let p = m.players[i].ms.pos;
    assert!(m.w.building_at(p.x, p.z).is_some(), "did not get in: {p:?}");
    // shut behind: walking on (away from it) leaves it shut
    pad.send(&mut m, a, 0, 0);
    pad.send(&mut m, a, 0, B_USE);
    assert!(!m.w.door_open[k]);
    pad.send(&mut m, a, 0, 0);
    for _ in 0..4 {
        pad.send(&mut m, a, 127, 0);
    }
    assert!(!m.w.door_open[k], "opened for a walker going away from it");
}

#[test]
fn a_door_out_of_reach_is_not_worked() {
    let (mut m, a) = game(7);
    let (k, out, inn) = level_door(&m);
    let i = a as usize;
    let far = (out.0 + (out.0 - inn.0) * 1.2, out.1 + (out.1 - inn.1) * 1.2);
    let mut pad = Pad { seq: 0, yaw: 0.0 };
    m.players[i].ms = MoveState { pos: V3::new(far.0, m.w.height(far.0, far.1), far.1), mode: MODE_GROUND, on_ground: true, ..Default::default() };
    pad.send(&mut m, a, 0, 0);
    pad.send(&mut m, a, 0, B_USE);
    assert!(!m.w.door_open[k]);
}

#[test]
fn bots_open_doors_on_their_way() {
    let mut m = Match::new(11, 30, FLAG_OFFLINE, 5);
    m.add_human(b"A").unwrap();
    for _ in 0..30 * 150 {
        m.step();
    }
    let n = m.w.door_open.iter().filter(|&&o| o).count();
    assert!(n >= 1, "no bot opened any door in 150 s");
}
