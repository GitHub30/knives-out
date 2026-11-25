//! Loose crates and containers: cars knock crates flying and shove
//! containers, a walker slides a crate along, blasts scatter them, and
//! everything comes to rest on the ground again.

use zonefall_core::defs::*;
use zonefall_core::math::*;
use zonefall_core::net::*;
use zonefall_core::player::*;
use zonefall_core::sim::*;
use zonefall_core::world::*;

fn game(seed: u32) -> (Match, u8) {
    let mut m = Match::new(seed, 2, FLAG_OFFLINE, 99);
    let a = m.add_human(b"A").unwrap();
    let _b = m.add_human(b"B").unwrap();
    m.step();
    assert_eq!(m.phase, PH_PLAY);
    (m, a)
}

/// A prop of the wanted kind standing on open ground: nothing else within 6 m along +x.
fn open_prop(m: &Match, crate_: bool) -> usize {
    for (k, p) in m.props.iter().enumerate() {
        if p.crate_ != crate_ || (p.pos.y - m.w.height(p.pos.x, p.pos.z)).abs() > 0.3 || m.w.building_at(p.pos.x, p.pos.z).is_some() {
            continue;
        }
        let b = m.w.boxes[p.bx];
        let clear = (1..3).all(|h| {
            let o = V3::new(b.max.x + 0.3, p.pos.y + 0.4 * h as f32, p.pos.z);
            m.w.raycast(o, V3::new(1.0, 0.0, 0.0), 14.0).is_none() && m.w.raycast(V3::new(b.min.x - 0.3, o.y, p.pos.z), V3::new(-1.0, 0.0, 0.0), 14.0).is_none()
        });
        let flat = (0..14).all(|d| (m.w.height(p.pos.x + d as f32, p.pos.z) - p.pos.y).abs() < 0.8 && (m.w.height(p.pos.x - d as f32, p.pos.z) - p.pos.y).abs() < 0.8);
        let stacked = m.props.iter().any(|q| q.bx != p.bx && dist_xz(q.pos.x, q.pos.z, p.pos.x, p.pos.z) < 3.0);
        if clear && flat && !stacked {
            return k;
        }
    }
    panic!("no open prop");
}

fn settle(m: &mut Match, k: usize) -> u32 {
    for t in 0..600 {
        m.step();
        if !m.props[k].moving {
            return t;
        }
    }
    panic!("prop {k} never came to rest: {:?}", m.props[k]);
}

#[test]
fn a_car_sends_crates_flying_and_only_shoves_containers() {
    let (mut m, a) = game(7);
    let mut seq = 0u32;
    for crate_ in [true, false] {
        let k = open_prop(&m, crate_);
        let p0 = m.props[k].pos;
        // a car 11 m back, pointed at it
        m.w.cars[0] = [p0.x - 11.0, m.w.height(p0.x - 11.0, p0.z), p0.z, std::f32::consts::FRAC_PI_2];
        m.players[a as usize].ms = MoveState { pos: V3::new(p0.x - 11.0, m.w.height(p0.x - 11.0, p0.z + 1.6), p0.z + 1.6), mode: MODE_GROUND, on_ground: true, ..Default::default() };
        let mut send = |m: &mut Match, mz: i8, b: u16| {
            seq += 1;
            let mut v = vec![C_INPUT, 1u8];
            v.extend_from_slice(&seq.to_le_bytes());
            v.extend_from_slice(&(m.tick * 256).to_le_bytes());
            v.push(0);
            v.push(mz as u8);
            v.extend_from_slice(&q_yaw(0.0).to_le_bytes());
            v.extend_from_slice(&q_pitch(0.0).to_le_bytes());
            v.extend_from_slice(&b.to_le_bytes());
            m.on_message(a, &v);
            m.step();
        };
        send(&mut m, 0, 0);
        send(&mut m, 0, B_USE);
        assert_eq!(m.players[a as usize].ms.mode, MODE_DRIVE);
        let mut high = 0.0f32;
        for _ in 0..75 {
            send(&mut m, 127, 0);
            high = high.max(m.props[k].pos.y - p0.y);
        }
        let car_x = m.w.cars[0][0];
        send(&mut m, 0, 0);
        send(&mut m, 0, B_JUMP);
        for _ in 0..60 {
            send(&mut m, 0, B_JUMP);
        }
        send(&mut m, 0, 0);
        send(&mut m, 0, B_USE);
        settle(&mut m, k);
        let p1 = m.props[k].pos;
        let went = p1.x - p0.x;
        if crate_ {
            assert!(went > 6.0, "the crate only moved {went} m");
            assert!(high > 0.8, "the crate never left the ground ({high} m)");
            assert!(car_x > p0.x + 2.0, "the car stopped at the crate (x {} vs {})", car_x, p0.x);
        } else {
            assert!(went > 0.2 && went < 6.0, "the container moved {went} m");
            assert!(high < 0.3, "the container flew ({high} m)");
        }
        // it rests on the ground, and the world's collision box went with it
        assert!((p1.y - m.w.ground_below(p1.x, p1.z, p1.y + 0.2, 0.3)).abs() < 0.6, "left hanging at {:?}", p1);
        let b = m.w.boxes[m.props[k].bx];
        assert!(((b.min.x + b.max.x) * 0.5 - p1.x).abs() < 1e-3 && (b.min.y - p1.y).abs() < 1e-3);
        assert!(m.w.raycast(V3::new(p1.x, p1.y + 0.4, p1.z - 6.0), V3::new(0.0, 0.0, 1.0), 12.0).is_some(), "nothing solid where the prop came to rest");
        assert!(m.w.raycast(V3::new(p0.x - 4.0, p0.y + 0.4, p0.z), V3::new(1.0, 0.0, 0.0), 3.5).map_or(true, |h| h.2 == BK_CAR) || went < 4.0, "a ghost of the prop is still where it stood");
    }
}

#[test]
fn a_walker_slides_a_crate_and_a_blast_scatters_it() {
    let (mut m, a) = game(42);
    let k = open_prop(&m, true);
    let p0 = m.props[k].pos;
    let b = m.w.boxes[m.props[k].bx];
    m.players[a as usize].ms = MoveState { pos: V3::new(b.min.x - 0.6, p0.y, p0.z), mode: MODE_GROUND, on_ground: true, ..Default::default() };
    let yaw = yaw_to(1.0, 0.0);
    for s in 1..=90u32 {
        let mut v = vec![C_INPUT, 1u8];
        v.extend_from_slice(&s.to_le_bytes());
        v.extend_from_slice(&(m.tick * 256).to_le_bytes());
        v.push(0);
        v.push(127);
        v.extend_from_slice(&q_yaw(yaw).to_le_bytes());
        v.extend_from_slice(&q_pitch(0.0).to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes());
        m.on_message(a, &v);
        m.step();
    }
    settle(&mut m, k);
    let slid = m.props[k].pos.x - p0.x;
    assert!(slid > 1.0 && slid < 9.0, "pushed for 3 s the crate moved {slid} m");
    assert!(m.players[a as usize].ms.pos.x > b.min.x, "the walker was held back by a crate that should slide");
    // a blast beside it
    let p1 = m.props[k].pos;
    m.players[a as usize].ms.pos = V3::new(p1.x - 60.0, p1.y, p1.z);
    m.hurt_car(0, 1.0, 255);
    m.w.cars[0] = [p1.x - 2.6, p1.y, p1.z, 0.0];
    m.hurt_car(0, 10_000.0, 255);
    assert!(m.props[k].moving, "the blast left the crate standing");
    let mut high = 0.0f32;
    for _ in 0..40 {
        m.step();
        high = high.max(m.props[k].pos.y - p1.y);
    }
    settle(&mut m, k);
    assert!(high > 1.0, "the crate rose only {high} m in the blast");
    assert!(dist_xz(m.props[k].pos.x, m.props[k].pos.z, p1.x, p1.z) > 3.0);
}
