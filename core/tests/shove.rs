//! Vehicles shoving each other: a car that is run into moves, slides and
//! stops; weight decides who moves how much; helicopters; and what the
//! predicting client makes of it.

use zonefall_core::defs::*;
use zonefall_core::math::*;
use zonefall_core::net::*;
use zonefall_core::player::*;
use zonefall_core::sim::*;
use zonefall_core::world::*;

const EAST: f32 = std::f32::consts::FRAC_PI_2;

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
            for y in [0.4f32, 1.0, 1.6, 3.0, 5.0] {
                for dz in [-4.5f32, -2.5, 0.0, 2.5, 4.5] {
                    if m.w.raycast(V3::new(x - 4.0, h0 + y + 0.25, z + dz), V3::new(1.0, 0.0, 0.0), 80.0).is_some() {
                        ok = false;
                    }
                }
            }
            // nothing a wide truck could catch on (thin trunks slip between the rays)
            for k in -4..79 {
                for dz in -4..=4 {
                    let (px, pz) = (x + k as f32, z + dz as f32);
                    let mut q = V3::new(px, m.w.height(px, pz) + 0.45, pz);
                    if ok && m.w.collide_body_ex(&mut q, 0.75, 1.0, false) {
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

fn park(m: &mut Match, k: usize, x: f32, z: f32, yaw: f32) {
    m.w.cars[k] = [x, m.w.height(x, z), z, yaw];
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

/// The first car and the first truck of the world.
fn kinds(m: &Match) -> (usize, usize, usize) {
    let car: Vec<usize> = (0..m.w.cars.len()).filter(|&k| m.w.car_kind[k] != VK_TRUCK).collect();
    let truck = (0..m.w.cars.len()).find(|&k| m.w.car_kind[k] == VK_TRUCK).expect("no truck");
    (car[0], car[1], truck)
}

/// Everything out of the way of the open run; the helicopters too.
fn stage(seed: u32) -> (Match, u8, u8, V3) {
    let (mut m, a, b) = two_humans(seed);
    let o = open_run(&m);
    for k in 0..m.w.cars.len() {
        if dist_xz(m.w.cars[k][0], m.w.cars[k][2], o.x + 35.0, o.z) < 80.0 {
            park(&mut m, k, 8.0 + k as f32 * 9.0, 8.0, 0.0);
        }
    }
    for k in 0..m.w.helis.len() {
        if dist_xz(m.w.helis[k][0], m.w.helis[k][2], o.x + 35.0, o.z) < 90.0 {
            m.w.helis[k] = [30.0 + 40.0 * k as f32, m.w.height(30.0 + 40.0 * k as f32, 30.0).max(0.0), 30.0, 0.0];
        }
    }
    place(&mut m, b, V3::new(o.x - 30.0, o.y, o.z + 30.0));
    (m, a, b, o)
}

/// Sit `pid` in car `k`.
fn board(m: &mut Match, pad: &mut Pad, pid: u8, k: usize) {
    let c = m.w.cars[k];
    let r = right(c[3]);
    let (hw, _, _) = m.w.car_dims(k);
    place(m, pid, V3::new(c[0] - r.x * (hw + 0.6), c[1], c[2] - r.z * (hw + 0.6)));
    pad.go(m, pid, 0, 0, 0.0, 0.0, 0);
    pad.go(m, pid, 0, 0, 0.0, 0.0, B_USE);
    assert_eq!(m.players[pid as usize].ms.mode, MODE_DRIVE, "did not get in");
    assert_eq!(m.car_driver[k], pid, "got into another car");
}

/// How deep the collision circles of two cars overlap (<= 0: apart).
fn overlap(m: &Match, a: usize, b: usize) -> f32 {
    let (ca, cb) = (m.w.cars[a], m.w.cars[b]);
    let (sa, ra, _) = car_shape(m.w.car_kind[a]);
    let (sb, rb, _) = car_shape(m.w.car_kind[b]);
    let (fa, fb) = (fwd(ca[3]), fwd(cb[3]));
    let mut worst = f32::MIN;
    for oa in sa {
        for ob in sb {
            let d = dist_xz(ca[0] + fa.x * oa, ca[2] + fa.z * oa, cb[0] + fb.x * ob, cb[2] + fb.z * ob);
            worst = worst.max(ra + rb - d);
        }
    }
    worst
}

/// How deep car `k` sits in the static world.
fn in_wall(m: &Match, k: usize) -> f32 {
    let c = m.w.cars[k];
    let (s, r, _) = car_shape(m.w.car_kind[k]);
    let f = fwd(c[3]);
    let mut worst = 0.0f32;
    for off in s {
        let base = V3::new(c[0] + f.x * off, m.w.height(c[0] + f.x * off, c[2] + f.z * off) + 0.45, c[2] + f.z * off);
        let mut q = base;
        m.w.collide_body_ex(&mut q, r, 1.0, true);
        worst = worst.max(dist_xz(q.x, q.z, base.x, base.z));
    }
    worst
}

fn sane(m: &Match, what: &str) {
    for k in 0..m.w.cars.len() {
        let c = m.w.cars[k];
        assert!(c.iter().all(|v| v.is_finite()), "{what}: car {k} at {c:?}");
        assert!(c[0] >= 4.0 && c[0] <= MAP - 4.0 && c[2] >= 4.0 && c[2] <= MAP - 4.0, "{what}: car {k} left the map: {c:?}");
        assert!((c[1] - m.w.height(c[0], c[2])).abs() < 1e-3, "{what}: car {k} is not on the ground: {c:?}");
        assert!(m.car_slide[k][0].is_finite() && m.car_slide[k][1].is_finite() && m.car_spin[k].is_finite());
    }
}

/// Ram car `target` (parked `gap` m ahead, turned `tyaw`, `side` m off the
/// line) with `striker` from a standing start, then brake. Returns how fast
/// the striker was just before and just after the hit, the fastest the
/// target went, and how far it ended from where it stood.
fn ram(m: &mut Match, a: u8, o: V3, striker: usize, target: usize, gap: f32, tyaw: f32, side: f32) -> (f32, f32, f32, f32) {
    park(m, striker, o.x, o.z, EAST);
    park(m, target, o.x + gap, o.z + side, tyaw);
    let t0 = m.w.cars[target];
    let mut pad = Pad { seq: 0 };
    board(m, &mut pad, a, striker);
    let (mut before, mut after, mut peak) = (0.0f32, 0.0f32, 0.0f32);
    let mut hit = false;
    for t in 0..420 {
        let v = car_speed(&m.players[a as usize].ms);
        if hit && t % 2 == 0 {
            pad.go(m, a, 0, 0, 0.0, 0.0, B_JUMP);
        } else {
            pad.go(m, a, 0, if hit { 0 } else { 127 }, 0.0, 0.0, 0);
        }
        let s = m.car_slide[target];
        let sp = (s[0] * s[0] + s[1] * s[1]).sqrt();
        if !hit && sp > 0.0 {
            hit = true;
            before = v;
            after = car_speed(&m.players[a as usize].ms);
        }
        peak = peak.max(sp);
        assert!(overlap(m, striker, target) < 0.05, "tick {t}: the cars overlap by {}", overlap(m, striker, target));
        sane(m, "ram");
    }
    assert!(hit, "never touched the target");
    let c = m.w.cars[target];
    (before, after, peak, dist_xz(c[0], c[2], t0[0], t0[2]))
}

#[test]
fn a_parked_car_moves_when_rammed_and_comes_to_rest() {
    for seed in [7u32, 42] {
        let (mut m, a, _b, o) = stage(seed);
        let (c0, c1, _) = kinds(&m);
        let hp = m.car_hp[c1];
        let (before, after, peak, moved) = ram(&mut m, a, o, c0, c1, 34.0, EAST, 0.0);
        assert!(before > 14.0, "seed {seed}: only {before} m/s at the hit");
        // equal weights: each ends up with about half
        assert!(after > before * 0.25 && after < before * 0.6, "seed {seed}: the striker went from {before} to {after}");
        assert!(peak > before * 0.4 && peak < before * 0.75, "seed {seed}: hit at {before}, the target got {peak}");
        assert!(moved > 4.0 && moved < 40.0, "seed {seed}: the target moved {moved} m");
        // at rest, and it stays there
        assert_eq!(m.car_slide[c1], [0.0; 2], "seed {seed}: still sliding");
        assert_eq!(m.car_spin[c1], 0.0);
        let c = m.w.cars[c1];
        for _ in 0..30 {
            m.step();
        }
        assert_eq!(m.w.cars[c1], c);
        // pushed straight along the line it was hit on, still pointing the same way
        assert!((c[2] - o.z).abs() < 0.6 && wrap_angle(c[3] - EAST).abs() < 0.15, "seed {seed}: ended at {c:?}");
        // a hard knock dents it, but does not wreck it
        assert!(m.car_hp[c1] < hp && m.car_hp[c1] > hp * 0.5, "seed {seed}: bodywork {} of {hp}", m.car_hp[c1]);
        assert_eq!(m.players[a as usize].ms.mode, MODE_DRIVE);
    }
}

#[test]
fn weight_decides_who_moves() {
    let (mut m, a, _b, o) = stage(42);
    let (c0, c1, truck) = kinds(&m);
    let (b1, a1, p1, d1) = ram(&mut m, a, o, c0, c1, 34.0, EAST, 0.0);
    let (mut m, a, _b, o) = stage(42);
    let (b2, a2, p2, d2) = ram(&mut m, a, o, c0, truck, 34.5, EAST, 0.0);
    assert!((b1 - b2).abs() < 1.0, "not the same hit: {b1} {b2}");
    assert!(p2 < p1 * 0.65 && p2 > 1.0, "the truck got {p2} m/s, the car {p1}");
    assert!(d2 < d1 * 0.6 && d2 > 0.3, "the truck moved {d2} m, the car {d1}");
    assert!(a2 < a1, "the striker kept {a2} against the truck and {a1} against the car");
    // and a truck sends a car further than a car does at the same speed
    let (mut m, a, _b, o) = stage(42);
    let (b3, a3, p3, _) = ram(&mut m, a, o, truck, c1, 60.0, EAST, 0.0);
    assert!(p3 / b3 > p1 / b1 * 1.3, "truck into car: {p3} from {b3}; car into car: {p1} from {b1}");
    assert!(a3 / b3 > 0.6, "the truck lost too much: {b3} -> {a3}");
}

#[test]
fn a_shoved_car_shoves_the_next_one() {
    let (mut m, a, _b, o) = stage(42);
    let (c0, c1, truck) = kinds(&m);
    let c2 = (0..m.w.cars.len()).find(|&k| k != c0 && k != c1 && k != truck).unwrap();
    park(&mut m, c2, o.x + 42.0, o.z, EAST);
    let third = m.w.cars[c2];
    let (_, _, _, moved) = ram(&mut m, a, o, c0, c1, 34.0, EAST, 0.0);
    let c = m.w.cars[c2];
    assert!(c[0] - third[0] > 1.0, "the third car moved {} m", c[0] - third[0]);
    assert!(moved > 2.0);
    assert!(overlap(&m, c1, c2) < 0.05 && overlap(&m, c0, c2) < 0.05);
    assert_eq!(m.car_slide[c2], [0.0; 2]);
}

#[test]
fn a_hit_off_centre_turns_the_car() {
    let (mut m, a, _b, o) = stage(42);
    let (c0, c1, _) = kinds(&m);
    // across the road, caught on its tail
    let (_, _, _, moved) = ram(&mut m, a, o, c0, c1, 34.0, 0.0, 1.6);
    let c = m.w.cars[c1];
    assert!(moved > 1.0);
    assert!(wrap_angle(c[3]).abs() > 0.15, "it did not turn: yaw {}", c[3]);
}

/// A solid wall across x at `x`, `half` wide each way along z, `h` tall.
fn add_wall(m: &mut Match, x: f32, z: f32, half: f32, y: f32, h: f32) {
    m.w.boxes.push(Bx { min: V3::new(x - 0.3, y - 1.0, z - half), max: V3::new(x + 0.3, y + h, z + half), kind: BK_CONCRETE, col: 0x808080 });
    let i = m.w.boxes.len() - 1;
    m.w.link_box(i);
}

#[test]
fn a_shoved_car_stays_out_of_walls() {
    // pushed against a wall with the throttle held
    let (mut m, a, _b, o) = stage(42);
    let (c0, c1, _) = kinds(&m);
    let wx = o.x + 41.0;
    add_wall(&mut m, wx, o.z, 9.0, o.y, 3.0);
    park(&mut m, c0, o.x, o.z, EAST);
    park(&mut m, c1, o.x + 30.0, o.z + 0.4, EAST);
    let mut pad = Pad { seq: 0 };
    board(&mut m, &mut pad, a, c0);
    for t in 0..400 {
        pad.go(&mut m, a, if t > 250 { 60 } else { 0 }, 127, 0.0, 0.0, 0);
        for k in [c0, c1] {
            assert!(in_wall(&m, k) < 0.06, "tick {t}: car {k} is {} m inside the wall", in_wall(&m, k));
            assert!(m.w.cars[k][0] < wx - 0.3, "tick {t}: car {k} is through the wall: {:?}", m.w.cars[k]);
        }
        assert!(overlap(&m, c0, c1) < 0.08, "tick {t}: overlap {}", overlap(&m, c0, c1));
        sane(&m, "wall");
    }
    assert!(m.w.cars[c1][0] > o.x + 33.0, "the target was not pushed up to the wall: {:?}", m.w.cars[c1]);
    // thrown at a wall as fast as a car can be thrown, from every distance
    for gap in 0..24 {
        let (mut m, _a, _b, o) = stage(42);
        add_wall(&mut m, wx, o.z, 9.0, o.y, 3.0);
        let x0 = wx - 3.0 - gap as f32 * 0.37;
        park(&mut m, c1, x0, o.z, if gap % 2 == 0 { EAST } else { 0.4 });
        m.shove_car(c1, 200.0, 3.0, x0 - 1.0, o.z + 0.2 * (gap % 3) as f32, 255);
        let s = m.car_slide[c1];
        assert!((s[0] * s[0] + s[1] * s[1]).sqrt() <= CAR_SLIDE_MAX + 0.01);
        for t in 0..200 {
            m.step();
            assert!(in_wall(&m, c1) < 0.06 && m.w.cars[c1][0] < wx - 0.3, "gap {gap} tick {t}: {:?} (wall at {wx})", m.w.cars[c1]);
        }
        assert_eq!(m.car_slide[c1], [0.0; 2], "gap {gap}: never came to rest");
        sane(&m, "thrown");
    }
    // and off the edge of the map
    let (mut m, _a, _b, _o) = stage(42);
    park(&mut m, c1, 12.0, 300.0, EAST);
    m.shove_car(c1, -40.0, 0.0, 14.0, 300.0, 255);
    for _ in 0..300 {
        m.step();
        sane(&m, "edge");
    }
    assert_eq!(m.car_slide[c1], [0.0; 2]);
}

#[test]
fn a_car_with_a_driver_is_shoved_too() {
    for (tyaw, side) in [(0.0f32, 0.0f32), (EAST, 0.0), (0.0, 1.5)] {
        let (mut m, a, b, o) = stage(42);
        let (c0, c1, _) = kinds(&m);
        park(&mut m, c1, o.x + 34.0, o.z + side, tyaw);
        let mut padb = Pad { seq: 0 };
        board(&mut m, &mut padb, b, c1);
        park(&mut m, c0, o.x, o.z, EAST);
        let mut pad = Pad { seq: 0 };
        board(&mut m, &mut pad, a, c0);
        let start = m.w.cars[c1];
        let mut fastest = 0.0f32;
        for t in 0..300 {
            pad.go(&mut m, a, 0, if t < 150 { 127 } else { 0 }, 0.0, 0.0, if t < 150 { 0 } else { B_JUMP });
            let (p, c) = (m.players[b as usize].ms, m.w.cars[c1]);
            assert_eq!(p.mode, MODE_DRIVE, "tick {t}: thrown out");
            assert_eq!(m.car_driver[c1], b);
            assert_eq!((p.pos.x, p.pos.y, p.pos.z, p.airtime), (c[0], c[1], c[2], c[3]), "tick {t}: the driver and the car came apart");
            // (their own speed is along the car, as the driving code expects)
            let f = fwd(c[3]);
            assert!((p.vel.x * f.z - p.vel.z * f.x).abs() < 1e-3 && p.vel.y == 0.0, "tick {t}: velocity {:?} at yaw {}", p.vel, c[3]);
            fastest = fastest.max(car_speed(&p).abs());
            assert!(overlap(&m, c0, c1) < 0.08, "tick {t}: overlap {}", overlap(&m, c0, c1));
            sane(&m, "occupied");
        }
        let c = m.w.cars[c1];
        assert!(c[0] - start[0] > 2.0, "yaw {tyaw}: the occupied car moved {} m", c[0] - start[0]);
        if tyaw == EAST {
            // hit from behind: the driver feels it as speed
            assert!(fastest > 5.0, "rear-ended: its speed only reached {fastest}");
        }
        // and it still drives
        // (wherever the knock left it pointing: it gets up to speed and away)
        let mut top = 0.0f32;
        for _ in 0..90 {
            padb.go(&mut m, b, 0, 127, 0.0, 0.0, 0);
            top = top.max(car_speed(&m.players[b as usize].ms));
            assert_eq!((m.car_slide[c1], m.car_spin[c1]), ([0.0; 2], 0.0));
        }
        let d = m.w.cars[c1];
        assert!(top > 5.0 && dist_xz(c[0], c[2], d[0], d[2]) > 5.0, "yaw {tyaw}: it no longer drives (top {top})");
    }
}

#[test]
fn a_shoved_car_runs_people_over_only_when_fast() {
    // fast: somebody standing behind the car that is rammed
    let (mut m, a, b, o) = stage(42);
    let (c0, c1, _) = kinds(&m);
    place(&mut m, b, V3::new(o.x + 34.0 + 4.0, o.y, o.z));
    let _ = ram(&mut m, a, o, c0, c1, 34.0, EAST, 0.0);
    let q = &m.players[b as usize];
    assert!(q.health < 95.0 && q.health > 20.0, "run over by the shoved car: health {}", q.health);
    assert_eq!(q.hurt_by, a, "the driver who shoved it is not credited");
    let mut at = q.ms.pos;
    assert!(!m.w.push_car(c1, &mut at, P_RADIUS - 0.02), "left standing inside the car");
    // slow: pushed aside, not hurt
    let (mut m, _a, b, o) = stage(42);
    park(&mut m, c1, o.x + 10.0, o.z, EAST);
    place(&mut m, b, V3::new(o.x + 10.0 + 3.2, o.y, o.z));
    let p0 = m.players[b as usize].ms.pos;
    m.shove_car(c1, 4.5, 0.0, o.x + 8.0, o.z, 255);
    for _ in 0..120 {
        m.step();
    }
    let q = &m.players[b as usize];
    assert_eq!(q.health, 100.0, "hurt by a car at a walking pace");
    assert!(q.ms.pos.x > p0.x + 0.2 || (q.ms.pos.z - p0.z).abs() > 0.2, "not pushed aside: {:?} -> {:?}", p0, q.ms.pos);
    let mut at = q.ms.pos;
    assert!(!m.w.push_car(c1, &mut at, P_RADIUS - 0.02));
}

#[test]
fn a_sliding_car_is_on_the_wire_while_it_moves() {
    let (mut m, a, b, o) = stage(42);
    let (c0, c1, _) = kinds(&m);
    park(&mut m, c0, o.x, o.z, EAST);
    park(&mut m, c1, o.x + 34.0, o.z, EAST);
    let mut pad = Pad { seq: 0 };
    board(&mut m, &mut pad, a, c0);
    let mut seen = 0;
    for _ in 0..400 {
        pad.go(&mut m, a, 0, 127, 0.0, 0.0, 0);
        if m.car_slide[c1] == [0.0; 2] && seen == 0 {
            continue;
        }
        // every snapshot carries where it is now, with nobody at the wheel
        let c = m.w.cars[c1];
        let mut pat = vec![c1 as u8];
        for v in [c[0], c[1], c[2]] {
            pat.extend_from_slice(&v.to_le_bytes());
        }
        pat.extend_from_slice(&q_yaw(c[3]).to_le_bytes());
        pat.push(255);
        for pid in [a, b] {
            let buf = m.snapshot(pid).to_vec();
            let at = buf.windows(pat.len()).position(|w| w == &pat[..]).expect("the sliding car is missing from a snapshot");
            // ... and, after its bodywork, how fast it moves and turns (for
            // the client that predicts where a knock sends it)
            let v = &buf[at + pat.len() + 1..];
            let (vx, vz, spin) = (i16::from_le_bytes([v[0], v[1]]) as f32 / 100.0, i16::from_le_bytes([v[2], v[3]]) as f32 / 100.0, v[4] as i8 as f32 / 50.0);
            let sl = m.car_slide[c1];
            assert!((vx - sl[0]).abs() < 0.006 && (vz - sl[1]).abs() < 0.006 && (spin - m.car_spin[c1]).abs() < 0.011, "speed on the wire {vx} {vz} {spin}, really {sl:?} {}", m.car_spin[c1]);
        }
        m.flush();
        seen += 1;
        if seen > 60 {
            break;
        }
    }
    assert!(seen > 60);
}

/// How deep car `k` reaches into the hull of helicopter `h`.
fn in_heli(m: &Match, k: usize, h: usize) -> f32 {
    let c = m.w.cars[k];
    let (s, r, _) = car_shape(m.w.car_kind[k]);
    let f = fwd(c[3]);
    let mut worst = 0.0f32;
    for off in s {
        let base = V3::new(c[0] + f.x * off, c[1] + 0.45, c[2] + f.z * off);
        let mut q = base;
        m.w.push_heli(h, &mut q, r, 1.0);
        worst = worst.max(dist_xz(q.x, q.z, base.x, base.z));
    }
    worst
}

#[test]
fn a_landed_helicopter_hardly_moves() {
    let (m, _a, _b, _o) = stage(42);
    let (c0, _, truck) = kinds(&m);
    let mut moved = [0.0f32; 2];
    for (n, striker) in [c0, truck].into_iter().enumerate() {
        let (mut m, a, _b, o) = stage(42);
        // side on to the road
        let at = V3::new(o.x + 36.0, o.y, o.z);
        let y = heli_floor(&m.w, at, 0.0, o.y + 2.0);
        m.w.helis[0] = [at.x, y, at.z, 0.0];
        let h0 = m.w.helis[0];
        let hp = m.helis[0].hp;
        park(&mut m, striker, o.x - 20.0 * n as f32, o.z, EAST);
        let mut pad = Pad { seq: 0 };
        board(&mut m, &mut pad, a, striker);
        let mut top = 0.0f32;
        for t in 0..330 {
            pad.go(&mut m, a, 0, 127, 0.0, 0.0, 0);
            top = top.max(car_speed(&m.players[a as usize].ms));
            assert!(in_heli(&m, striker, 0) < 0.06, "tick {t}: the car is {} m inside the hull", in_heli(&m, striker, 0));
        }
        let h = m.w.helis[0];
        moved[n] = dist_xz(h[0], h[2], h0[0], h0[2]);
        assert!(top > 13.0, "only reached {top} m/s");
        assert!(moved[n] > 0.01 && moved[n] < 1.2, "kind {n}: the helicopter moved {} m", moved[n]);
        assert!((h[1] - h0[1]).abs() < 0.3 && h[3] == h0[3] && !m.helis[0].air);
        assert!(m.helis[0].hp > hp - 250.0 && m.helis[0].hp <= hp, "hull {} of {hp}", m.helis[0].hp);
        assert!(m.w.cars[striker][0] < h[0], "the car went through it");
        // it stands again
        let h1 = m.w.helis[0];
        for _ in 0..60 {
            pad.go(&mut m, a, 0, 0, 0.0, 0.0, B_JUMP);
        }
        assert!(dist_xz(m.w.helis[0][0], m.w.helis[0][2], h1[0], h1[2]) < 0.05, "it keeps sliding");
        assert_eq!(m.helis[0].vel, V3::ZERO);
    }
    assert!(m.w.helis.len() >= 1 && moved[1] >= moved[0], "the truck moved it {} m, the car {}", moved[1], moved[0]);
}

#[test]
fn a_flying_helicopter_shoves_cars_out_of_its_way() {
    let (mut m, a, _b, o) = stage(42);
    let (_, c1, _) = kinds(&m);
    let at = V3::new(o.x + 8.0, o.y, o.z);
    let y = heli_floor(&m.w, at, EAST, o.y + 2.0);
    m.w.helis[0] = [at.x, y, at.z, EAST];
    park(&mut m, c1, o.x + 50.0, o.z + 0.8, 0.0);
    let c0 = m.w.cars[c1];
    place(&mut m, a, V3::new(at.x + 4.0, o.y, o.z + HELI_HW + 0.9));
    let mut pad = Pad { seq: 0 };
    pad.go(&mut m, a, 0, 0, EAST, 0.0, 0);
    pad.go(&mut m, a, 0, 0, EAST, 0.0, B_USE);
    assert_eq!(m.players[a as usize].ms.mode, MODE_FLY, "did not get in");
    // skimming the ground, nose first into the car
    let hp = m.helis[0].hp;
    let (mut before, mut after) = (0.0f32, 0.0f32);
    let mut hit = false;
    for t in 0..150 {
        let low = m.players[a as usize].ms.pos.y - y < 0.25;
        let v = m.players[a as usize].ms.vel.len();
        pad.go(&mut m, a, 0, 127, EAST, 0.0, if low && !hit { B_JUMP } else { 0 });
        if !hit && m.car_slide[c1] != [0.0; 2] {
            hit = true;
            before = v;
            after = m.players[a as usize].ms.vel.len();
            assert!(m.players[a as usize].ms.pos.y - y < 1.2, "too high to touch it");
        }
        if hit {
            // never left inside the hull's circles
            let h = m.w.helis[0];
            let f = fwd(h[3]);
            for off in HELI_CIRCLES {
                let mut q = V3::new(h[0] + f.x * off, h[1] + 0.2, h[2] + f.z * off);
                let base = q;
                m.w.push_car(c1, &mut q, HELI_CR);
                assert!(dist_xz(q.x, q.z, base.x, base.z) < 0.9, "tick {t}: the helicopter is {} m inside the car", dist_xz(q.x, q.z, base.x, base.z));
            }
        }
        sane(&m, "heli");
    }
    assert!(hit, "never touched the car");
    assert!(before > 8.0 && after > before * 0.8, "the helicopter went from {before} to {after} m/s");
    let c = m.w.cars[c1];
    assert!(c[0] - c0[0] > 5.0, "the car was moved {} m", c[0] - c0[0]);
    assert!(m.helis[0].hp > hp - 200.0, "hull {} of {hp}", m.helis[0].hp);
}

#[test]
fn a_helicopter_settling_on_a_car_pushes_it_out() {
    let (mut m, _a, _b, o) = stage(42);
    let (_, c1, _) = kinds(&m);
    let at = V3::new(o.x + 30.0, o.y, o.z);
    let y = heli_floor(&m.w, at, EAST, o.y + 2.0);
    // left in the air with nobody aboard, a car under its side
    m.w.helis[0] = [at.x, y + 5.0, at.z, EAST];
    m.helis[0].air = true;
    park(&mut m, c1, at.x + 1.0, at.z + 1.2, EAST);
    let c0 = m.w.cars[c1];
    for _ in 0..600 {
        m.step();
        sane(&m, "settle");
    }
    assert!(!m.helis[0].air, "it never landed");
    let c = m.w.cars[c1];
    assert!(dist_xz(c[0], c[2], c0[0], c0[2]) > 0.1, "the car stayed where it was");
    assert!(in_heli(&m, c1, 0) < 0.06, "the car is {} m inside the landed hull", in_heli(&m, c1, 0));
    assert_eq!(m.car_slide[c1], [0.0; 2]);
    assert!(m.car_hp[c1] > 0.0);
}

/// What the predicting client does: its own copy of the world, in which the
/// other cars stand where the last snapshot (`lag` ticks old) put them and
/// its own car is parked out of the way; every snapshot resets the state to
/// the server's and replays the commands sent since.
/// Returns (the largest jump a snapshot made in the predicted position, the
/// largest such jump in the last second, how far the prediction is from the
/// server at the end, the furthest it ever was from the server).
fn predict_into_parked_car(lag: usize) -> (f32, f32, f32, f32) {
    let (mut m, a, _b, o) = stage(42);
    let (c0, c1, _) = kinds(&m);
    park(&mut m, c0, o.x, o.z, EAST);
    park(&mut m, c1, o.x + 30.0, o.z + 0.3, EAST);
    let mut pad = Pad { seq: 0 };
    board(&mut m, &mut pad, a, c0);
    let mut w = World::generate(42);
    w.cars = m.w.cars.clone();
    w.helis = m.w.helis.clone();
    w.cars[c0] = [0.0, -1000.0, 0.0, 0.0];
    let mut s = m.players[a as usize].ms;
    let mut hist: Vec<(MoveState, Vec<[f32; 4]>)> = vec![(s, m.w.cars.clone())];
    let mut cmds: Vec<Cmd> = vec![Cmd::default()];
    let (mut worst, mut late, mut off) = (0.0f32, 0.0f32, 0.0f32);
    let total = 520usize;
    for t in 1..=total {
        // hold the throttle into the car and keep pushing, then steer off and brake
        let (mx, mz, b) = if t < 260 { (0i8, 127i8, 0u16) } else if t < 330 { (70, 127, 0) } else { (0, 0, B_JUMP) };
        let c = Cmd::from_quant(0, 0, mx, mz, q_yaw(0.0), q_pitch(0.0), b);
        cmds.push(c);
        step_move(&w, &mut s, &c, 1.0, DT);
        pad.go(&mut m, a, mx, mz, 0.0, 0.0, b);
        hist.push((m.players[a as usize].ms, m.w.cars.clone()));
        assert_eq!(m.players[a as usize].ms.mode, MODE_DRIVE);
        if t >= lag {
            let before = s;
            let (ms, cars) = &hist[t - lag];
            w.cars = cars.clone();
            w.cars[c0] = [0.0, -1000.0, 0.0, 0.0];
            s = *ms;
            for c in &cmds[t - lag + 1..=t] {
                step_move(&w, &mut s, c, 1.0, DT);
            }
            let jump = (s.pos - before.pos).len();
            worst = worst.max(jump);
            if t > total - 30 {
                late = late.max(jump);
            }
        }
        off = off.max((s.pos - m.players[a as usize].ms.pos).len());
    }
    assert!(m.car_slide[c1] == [0.0; 2] && m.w.cars[c1][0] > o.x + 40.0, "the car was not pushed along: {:?}", m.w.cars[c1]);
    (worst, late, (s.pos - m.players[a as usize].ms.pos).len(), off)
}

#[test]
fn prediction_does_not_fight_the_server_over_a_shoved_car() {
    // With the server's answer in hand at once the client is exact: the
    // hit itself is predicted with the same arithmetic.
    let (worst, late, end, off) = predict_into_parked_car(0);
    assert!(worst < 1e-4 && late == 0.0 && end == 0.0 && off < 1e-4, "no latency: {worst} {late} {end} {off}");
    // With latency it sees the shoved car where it was a moment ago, so
    // while it pushes it expects to be held up more than it is: it trails
    // the server by about what the car travels in that time. Each snapshot
    // moves it on by a tick's travel; nothing grows, and it is exact again
    // once the cars are at rest.
    for lag in [2usize, 4, 6, 9] {
        let (worst, late, end, off) = predict_into_parked_car(lag);
        eprintln!("lag {lag}: worst correction {worst:.3} m, furthest from the server {off:.3} m, at rest {late:.5}, end error {end:.5}");
        assert!(worst < 0.6, "lag {lag}: a snapshot moved the prediction {worst} m");
        assert!(off < 0.3 + 0.35 * lag as f32, "lag {lag}: the prediction was {off} m from the server");
        assert!(late < 1e-3 && end < 1e-3, "lag {lag}: still correcting at rest: {late} {end}");
    }
}

/// The same drive as `predict_into_parked_car`, but the client predicts
/// the cars it shoves as well (as `cli_pred_step` does): each snapshot
/// (every `every` ticks, `lag` ticks old) gives it the poses and how fast
/// each car moves and turns, as finely as the wire tells. Returns (largest correction of its
/// own car, furthest its car was from the server's, the lowest speed it
/// predicted while pushing against the server's at that moment, furthest
/// the predicted pushed car was from the server's, end error).
fn predict_with_cars(lag: usize, every: usize) -> (f32, f32, (f32, f32), f32, f32) {
    let (mut m, a, _b, o) = stage(42);
    let (c0, c1, _) = kinds(&m);
    park(&mut m, c0, o.x, o.z, EAST);
    park(&mut m, c1, o.x + 30.0, o.z + 0.3, EAST);
    let mut pad = Pad { seq: 0 };
    board(&mut m, &mut pad, a, c0);
    let mut w = World::generate(42);
    w.cars = m.w.cars.clone();
    w.helis = m.w.helis.clone();
    w.cars[c0] = [0.0, -1000.0, 0.0, 0.0];
    let n = w.cars.len();
    let mut cp = CarsPred { spin: vec![0.0; n], free: vec![true; n] };
    cp.free[c0] = false;
    let kind = w.car_kind[c0];
    let mut s = m.players[a as usize].ms;
    let wire = |m: &Match| -> Vec<[f32; 3]> {
        (0..n).map(|k| { let (x, z) = m.car_velocity(k); [(x * 100.0).round() / 100.0, (z * 100.0).round() / 100.0, (m.car_spin[k] * 50.0).round() / 50.0] }).collect()
    };
    let mut hist: Vec<(MoveState, Vec<[f32; 4]>, Vec<[f32; 3]>)> = vec![(s, m.w.cars.clone(), wire(&m))];
    let mut cmds: Vec<Cmd> = vec![Cmd::default()];
    let step = |w: &mut World, cp: &mut CarsPred, s: &mut MoveState, c: &Cmd| {
        let mut hits = DriveHits::default();
        step_move_hits(w, s, c, 1.0, DT, &mut hits);
        predict_shoves(w, cp, kind, &hits);
        predict_slides(w, cp, c0, DT);
    };
    let (mut worst, mut off, mut car_off) = (0.0f32, 0.0f32, 0.0f32);
    let mut slow = (f32::MAX, 0.0f32);
    let total = 520usize;
    for t in 1..=total {
        let (mx, mz, b) = if t < 260 { (0i8, 127i8, 0u16) } else if t < 330 { (70, 127, 0) } else { (0, 0, B_JUMP) };
        let c = Cmd::from_quant(0, 0, mx, mz, q_yaw(0.0), q_pitch(0.0), b);
        cmds.push(c);
        step(&mut w, &mut cp, &mut s, &c);
        pad.go(&mut m, a, mx, mz, 0.0, 0.0, b);
        hist.push((m.players[a as usize].ms, m.w.cars.clone(), wire(&m)));
        if t >= lag && t % every == 0 {
            let before = s;
            let (ms, cars, vels) = &hist[t - lag];
            for k in 0..n {
                w.cars[k] = cars[k];
                w.car_vel[k] = [vels[k][0], vels[k][1]];
                cp.spin[k] = vels[k][2];
            }
            w.cars[c0] = [0.0, -1000.0, 0.0, 0.0];
            w.car_vel[c0] = [0.0; 2];
            s = *ms;
            for c in &cmds[t - lag + 1..=t] {
                step(&mut w, &mut cp, &mut s, c);
            }
            worst = worst.max((s.pos - before.pos).len());
        }
        let srv = m.players[a as usize].ms;
        off = off.max((s.pos - srv.pos).len());
        let (pc, sc) = (w.cars[c1], m.w.cars[c1]);
        car_off = car_off.max(dist_xz(pc[0], pc[2], sc[0], sc[2]));
        if (140..250).contains(&t) && car_speed(&s) < slow.0 {
            slow = (car_speed(&s), car_speed(&srv));
        }
    }
    assert!(m.car_slide[c1] == [0.0; 2] && m.w.cars[c1][0] > o.x + 40.0, "the car was not pushed along: {:?}", m.w.cars[c1]);
    (worst, off, slow, car_off, (s.pos - m.players[a as usize].ms.pos).len())
}

#[test]
fn the_client_predicts_the_car_it_shoves() {
    // The client moves the car it runs into the way the server will, so
    // with latency its own car is not held up by a car that has long gone:
    // the speed it predicts while pushing is the server's, and what a
    // snapshot corrects is small.
    for (lag, every) in [(2usize, 1usize), (4, 2), (6, 2), (9, 2), (14, 2)] {
        let (worst, off, slow, car_off, end) = predict_with_cars(lag, every);
        eprintln!("lag {lag} (a snapshot every {every}): worst correction {worst:.3} m, furthest from the server {off:.3} m, slowest predicted push {:.2} m/s (server {:.2}), pushed car off by up to {car_off:.3} m, end error {end:.5}", slow.0, slow.1);
        assert!(worst < 0.2, "lag {lag}: a snapshot moved the prediction {worst} m");
        assert!(off < 0.3, "lag {lag}: the prediction was {off} m from the server");
        assert!((slow.0 - slow.1).abs() < 1.0 && slow.0 > 3.0, "lag {lag}: pushing at {} m/s in the prediction, {} on the server", slow.0, slow.1);
        assert!(car_off < 0.4, "lag {lag}: the pushed car was predicted {car_off} m from where it was");
        assert!(end < 1e-3, "lag {lag}: still off at rest: {end}");
    }
}

#[test]
fn pushing_a_car_keeps_the_engine_working() {
    // Nose to tail from a standing start, throttle held: what the pusher
    // loses the pushed car gets, so the pair gets up to a fair speed (a
    // car cannot get a fire truck going from rest, but a truck sweeps a
    // car along).
    for (truck_pushes, truck_pushed, min_v, min_d) in [(false, false, 6.0f32, 35.0f32), (true, false, 9.0, 50.0), (false, true, -1.0, -1.0)] {
        let (mut m, a, _b, o) = stage(42);
        let (c0, c1, tr) = kinds(&m);
        let (striker, target) = if truck_pushes { (tr, c0) } else if truck_pushed { (c0, tr) } else { (c0, c1) };
        park(&mut m, striker, o.x, o.z, EAST);
        let gap = car_shape(m.w.car_kind[striker]).0[2] + car_shape(m.w.car_kind[striker]).1 + car_shape(m.w.car_kind[target]).0[2] + car_shape(m.w.car_kind[target]).1 + 0.05;
        park(&mut m, target, o.x + gap, o.z, EAST);
        let mut pad = Pad { seq: 0 };
        board(&mut m, &mut pad, a, striker);
        let x0 = m.w.cars[striker][0];
        let mut energy_ok = true;
        for t in 0..240 {
            pad.go(&mut m, a, 0, 127, 0.0, 0.0, 0);
            assert!(overlap(&m, striker, target) < 0.08, "tick {t}: overlap {}", overlap(&m, striker, target));
            // (the pushed car never runs away faster than a bounce allows)
            let (tv, _) = m.car_velocity(target);
            energy_ok &= tv < car_speed(&m.players[a as usize].ms) * 1.3 + 1.0;
            sane(&m, "push");
        }
        let v = car_speed(&m.players[a as usize].ms);
        let d = m.w.cars[striker][0] - x0;
        eprintln!("truck pushes {truck_pushes}, truck pushed {truck_pushed}: {v:.2} m/s after 8 s, {d:.1} m covered");
        assert!(energy_ok, "the pushed car ran away");
        assert!(v > min_v && d > min_d, "pushing: only {v} m/s and {d} m after 8 s");
        assert!(v < car_shape(m.w.car_kind[striker]).2, "pushing made it faster than driving");
    }
}

/// How deep two cars of a match overlap and how deep one sits in a wall,
/// with cars thrown hard at their neighbours all match long.
#[test]
fn cars_thrown_at_their_neighbours_never_rest_in_a_wall_or_in_each_other() {
    for seed in [1u32, 2, 7, 8] {
        let mut m = Match::new(seed, 30, FLAG_OFFLINE, 7);
        m.add_human(b"A").unwrap();
        m.step();
        let n = m.w.cars.len();
        // (what the world was generated with is not this test's business)
        let mut init = vec![0.0f32; n];
        for a in 0..n {
            init[a] = in_wall(&m, a);
        }
        let (mut wall_rest, mut over_rest) = (0.0f32, 0.0f32);
        let mut desc = String::new();
        let mut thrown = vec![false; n];
        let mut deep = vec![0u32; n];
        for t in 0..3000u32 {
            if m.phase != PH_PLAY {
                break;
            }
            if t % 5 == 0 {
                let k = (t as usize / 5 * 7) % n;
                let c = m.w.cars[k];
                let mut best = (f32::MAX, k);
                for j in 0..n {
                    let d = dist_xz(c[0], c[2], m.w.cars[j][0], m.w.cars[j][2]);
                    if j != k && d < best.0 {
                        best = (d, j);
                    }
                }
                let o = m.w.cars[best.1];
                let d = best.0.max(0.1);
                let sp = 10.0 + (t % 7) as f32 * 5.0;
                m.shove_car(k, (o[0] - c[0]) / d * sp, (o[2] - c[2]) / d * sp, c[0] + 0.5, c[2] - 0.3, 255);
                thrown[k] = true;
                if best.0 < 12.0 {
                    thrown[best.1] = true;
                }
            }
            m.step();
            sane_moving(&m);
            if t % 5 != 4 {
                continue;
            }
            let rest = |k: usize| m.car_slide[k] == [0.0; 2] && m.car_spin[k] == 0.0 && m.car_driver[k] == 255;
            for a in 0..n {
                if !rest(a) || m.car_hp[a] <= 0.0 {
                    deep[a] = 0;
                    continue;
                }
                // (three looks in a row: a container still falling onto
                // it is not what it rests in; where that lands the car is
                // moved out from under it)
                let iw = in_wall(&m, a);
                deep[a] = if thrown[a] && iw > init[a] + 0.01 { deep[a] + 1 } else { 0 };
                if deep[a] >= 3 && iw > wall_rest {
                    wall_rest = iw;
                    desc = format!("t{t} car {a} at {:?}", m.w.cars[a]);
                }
                for b in a + 1..n {
                    if rest(b) && (thrown[a] || thrown[b]) && dist_xz(m.w.cars[a][0], m.w.cars[a][2], m.w.cars[b][0], m.w.cars[b][2]) < 9.0 && (m.w.cars[a][1] - m.w.cars[b][1]).abs() < 3.0 {
                        over_rest = over_rest.max(overlap(&m, a, b));
                    }
                }
            }
        }
        eprintln!("seed {seed}: at rest at most {wall_rest:.3} m in a wall ({desc}), {over_rest:.3} m in another car");
        assert!(wall_rest < 0.12, "seed {seed}: a car rests {wall_rest} m inside a wall: {desc}");
        assert!(over_rest < 0.25, "seed {seed}: two cars rest {over_rest} m inside each other");
    }
}

fn sane_moving(m: &Match) {
    for k in 0..m.w.cars.len() {
        let c = m.w.cars[k];
        assert!(c.iter().all(|v| v.is_finite()) && c[0] >= 4.0 && c[0] <= MAP - 4.0 && c[2] >= 4.0 && c[2] <= MAP - 4.0, "car {k} at {c:?}");
    }
}

#[test]
fn a_whole_match_with_cars_thrown_about_stays_sane_and_repeats() {
    // bots fight (and may drive) while every car in turn is shoved hard
    let run = || {
        let mut m = Match::new(11, 24, FLAG_OFFLINE, 5);
        m.add_human(b"A").unwrap();
        m.step();
        assert_eq!(m.phase, PH_PLAY);
        let n = m.w.cars.len();
        for t in 0..2400u32 {
            if m.phase != PH_PLAY {
                break;
            }
            if t % 7 == 0 {
                let k = (t as usize / 7 * 5) % n;
                let c = m.w.cars[k];
                let a = t as f32 * 0.37;
                m.shove_car(k, a.sin() * 30.0, a.cos() * 30.0, c[0] + a.cos(), c[2] - a.sin(), 255);
            }
            m.step();
            sane(&m, "match");
            if t % 40 == 0 {
                for k in 0..n {
                    assert!(in_wall(&m, k) < 0.12, "tick {t}: car {k} is {} m inside the world at {:?}", in_wall(&m, k), m.w.cars[k]);
                }
            }
        }
        (m.w.cars.clone(), m.car_hp.clone(), m.tick)
    };
    let (a, b) = (run(), run());
    assert!(a.0 == b.0 && a.1 == b.1 && a.2 == b.2, "two runs of the same match differ");
}
