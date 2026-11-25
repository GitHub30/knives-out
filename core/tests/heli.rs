//! The helicopters: where they are parked, boarding, flying and landing,
//! missiles, damage, and what goes over the wire.

use zonefall_core::defs::*;
use zonefall_core::math::*;
use zonefall_core::net::*;
use zonefall_core::player::*;
use zonefall_core::sim::*;
use zonefall_core::world::*;

const EAST: f32 = std::f32::consts::FRAC_PI_2;

fn humans(seed: u32, n: usize) -> (Match, Vec<u8>) {
    let mut m = Match::new(seed, n, FLAG_OFFLINE, 99);
    let ids: Vec<u8> = (0..n).map(|k| m.add_human(format!("P{k}").as_bytes()).unwrap()).collect();
    m.step();
    assert_eq!(m.phase, PH_PLAY);
    (m, ids)
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

/// Helicopter 0 parked on open ground with its nose to +x and the first
/// player standing beside it; everything else out of the way. Returns the
/// match, the players and the start of the open run (the hull's tail).
fn stage(seed: u32, n: usize) -> (Match, Vec<u8>, V3) {
    let (mut m, ids) = humans(seed, n);
    let o = open_run(&m);
    for k in 0..m.w.cars.len() {
        if dist_xz(m.w.cars[k][0], m.w.cars[k][2], o.x + 35.0, o.z) < 70.0 {
            m.w.cars[k][0] = 5.0;
            m.w.cars[k][2] = 5.0;
        }
    }
    assert_eq!(m.w.helis.len(), 2);
    m.w.helis[1] = [20.0, m.w.height(20.0, 20.0).max(0.0), 20.0, 0.0];
    let at = V3::new(o.x + 8.0, o.y, o.z);
    let y = heli_floor(&m.w, at, EAST, o.y + 2.0);
    m.w.helis[0] = [at.x, y, at.z, EAST];
    for (k, &id) in ids.iter().enumerate().skip(1) {
        place(&mut m, id, V3::new(o.x - 30.0 - 3.0 * k as f32, o.y, o.z + 30.0));
    }
    place(&mut m, ids[0], V3::new(at.x + 4.0, o.y, o.z + HELI_HW + 0.9));
    (m, ids, o)
}

/// Get in, and climb for `ticks`.
fn board_and_climb(m: &mut Match, pad: &mut Pad, a: u8, ticks: usize) {
    pad.go(m, a, 0, 0, EAST, 0.0, 0);
    pad.go(m, a, 0, 0, EAST, 0.0, B_USE);
    assert_eq!(m.players[a as usize].ms.mode, MODE_FLY, "did not get in");
    for _ in 0..ticks {
        pad.go(m, a, 0, 0, EAST, 0.0, B_JUMP);
    }
}

/// A solid wall across x at `x`, `half` wide each way along z, `h` tall.
fn add_wall(m: &mut Match, x: f32, z: f32, half: f32, y: f32, h: f32) {
    m.w.boxes.push(Bx { min: V3::new(x - 0.3, y - 1.0, z - half), max: V3::new(x + 0.3, y + h, z + half), kind: BK_CONCRETE, col: 0x808080 });
    let i = m.w.boxes.len() - 1;
    m.w.link_box(i);
}

/// The helicopter and missile blocks of a snapshot (mirrors `Match::snapshot`).
struct Wire {
    /// id, x, y, z, yaw, pitch, roll, rotor, pilot, hp, ammo, flags
    helis: Vec<(u8, V3, f32, f32, f32, u8, u8, u8, u8, u8)>,
    missiles: Vec<(u16, V3, f32, f32)>,
    /// Ids of the other players sent.
    players: Vec<u8>,
    events: usize,
}

fn wire(buf: &[u8]) -> Wire {
    struct R<'a>(&'a [u8], usize);
    impl R<'_> {
        fn u8(&mut self) -> u8 {
            self.1 += 1;
            self.0[self.1 - 1]
        }
        fn u16(&mut self) -> u16 {
            self.1 += 2;
            u16::from_le_bytes([self.0[self.1 - 2], self.0[self.1 - 1]])
        }
        fn f32(&mut self) -> f32 {
            self.1 += 4;
            f32::from_le_bytes(self.0[self.1 - 4..self.1].try_into().unwrap())
        }
        fn v3(&mut self) -> V3 {
            V3::new(self.f32(), self.f32(), self.f32())
        }
    }
    let mut r = R(buf, 0);
    assert_eq!(r.u8(), S_SNAP);
    r.1 += 4 + 4 + 1 + 4 + 1 + 1 + 1;
    r.1 += 1 + 1 + 12 + 12 + 4 + 4 + 4 + 1 + 6 + 8 + 3 + 1 + 4 + 4 + 1 + 4 + 4 + 2 + 2 + 3 + 1;
    r.1 += 2 + 4 * 8;
    r.1 += 1 + 4 * 7;
    let mut players = vec![];
    for _ in 0..r.u8() {
        players.push(r.u8());
        r.1 += 21;
    }
    let n = r.u8() as usize;
    r.1 += n * 14;
    let n = r.u8() as usize;
    r.1 += n * 22;
    let n = r.u8() as usize;
    r.1 += n * 15;
    let mut helis = vec![];
    for _ in 0..r.u8() {
        let id = r.u8();
        let pos = r.v3();
        let yaw = dq_yaw(r.u16());
        let pitch = r.u8() as i8 as f32 / 100.0;
        let roll = r.u8() as i8 as f32 / 100.0;
        helis.push((id, pos, yaw, pitch, roll, r.u8(), r.u8(), r.u8(), r.u8(), r.u8()));
    }
    let mut missiles = vec![];
    for _ in 0..r.u8() {
        missiles.push((r.u16(), r.v3(), dq_yaw(r.u16()), dq_pitch(r.u16() as i16)));
    }
    let events = r.u16() as usize;
    assert!(r.1 <= buf.len());
    Wire { helis, missiles, players, events }
}

#[test]
fn two_helicopters_parked_on_sane_ground() {
    for seed in [1u32, 3, 7, 11, 42, 58, 777, 2024, 31337, 900001] {
        let w = World::generate(seed);
        assert_eq!(w.helis.len(), 2, "seed {seed}");
        assert_eq!(World::generate(seed).helis, w.helis, "seed {seed}: not deterministic");
        for (k, h) in w.helis.iter().enumerate() {
            let (x, y, z) = (h[0], h[1], h[2]);
            assert!(y > 2.0, "seed {seed} heli {k}: at the waterline ({y})");
            assert!((w.height(x, z) - y).abs() < 0.01, "seed {seed} heli {k}: not on the ground");
            // a level pad under the whole hull, whichever way it points
            for a in 0..16 {
                let (dx, dz) = ((a as f32 * 0.4).cos() * 8.5, (a as f32 * 0.4).sin() * 8.5);
                assert!((w.height(x + dx, z + dz) - y).abs() < 0.35, "seed {seed} heli {k}: pad not level ({})", w.height(x + dx, z + dz) - y);
            }
            assert!(!w.cyls.iter().any(|c| dist_xz(c.x, c.z, x, z) < 14.0), "seed {seed} heli {k}: a tree or rock on the pad");
            assert!(!w.boxes.iter().any(|b| x > b.min.x - 12.0 && x < b.max.x + 12.0 && z > b.min.z - 12.0 && z < b.max.z + 12.0), "seed {seed} heli {k}: something built on the pad");
            assert!(!w.towns.iter().any(|t| dist_xz(t.x, t.z, x, z) < t.r + 10.0), "seed {seed} heli {k}: in a town");
            assert!(!w.cars.iter().any(|c| dist_xz(c[0], c[2], x, z) < 12.0), "seed {seed} heli {k}: a car on the pad");
            // nothing in the way of the rotors or of walking around it
            for a in 0..12 {
                let d = fwd(a as f32 * 0.5236);
                for hgt in [2.5f32, 5.0] {
                    assert!(w.raycast_static(V3::new(x, y + hgt, z), d, 12.0).is_none(), "seed {seed} heli {k}: blocked at {hgt} m");
                }
            }
            // the grass is worn away under it
            let g = w.grass_mask();
            assert_eq!(g[z as usize * 1024 + x as usize], 0, "seed {seed} heli {k}: grass on the pad");
        }
        assert!(dist_xz(w.helis[0][0], w.helis[0][2], w.helis[1][0], w.helis[1][2]) > 60.0, "seed {seed}: parked together");
        // the hull is solid for rays and for walkers
        let h = w.helis[0];
        let r = right(h[3]);
        let from = V3::new(h[0], h[1] + 1.5, h[2]) + r * 6.0;
        let hit = w.raycast(from, r * -1.0, 20.0).expect("the hull stops nothing");
        assert_eq!(hit.2, BK_HELI, "seed {seed}");
        assert!((hit.0 - (6.0 - HELI_HW)).abs() < 0.01, "seed {seed}: hull side at {}", hit.0);
        let mut s = MoveState { pos: V3::new(from.x, h[1], from.z), mode: MODE_GROUND, on_ground: true, ..Default::default() };
        let c = Cmd { mz: 1.0, yaw: yaw_to(-r.x, -r.z), ..Default::default() };
        for _ in 0..120 {
            step_move(&w, &mut s, &c, 1.0, DT);
        }
        let side = (s.pos.x - h[0]) * r.x + (s.pos.z - h[2]) * r.z;
        assert!(side > HELI_HW + P_RADIUS - 0.02, "seed {seed}: walked into the hull ({side})");
    }
}

#[test]
fn board_lift_off_fly_and_land() {
    for seed in [7u32, 42] {
        let (mut m, ids, o) = stage(seed, 2);
        let a = ids[0];
        let mut pad = Pad { seq: 0 };
        let y0 = m.w.helis[0][1];
        pad.go(&mut m, a, 0, 0, EAST, 0.0, 0);
        pad.go(&mut m, a, 0, 0, EAST, 0.0, B_USE);
        assert_eq!(m.players[a as usize].ms.mode, MODE_FLY, "seed {seed}: did not get in");
        assert_eq!(m.helis[0].pilot, a);
        assert!(m.players[a as usize].ms.on_ground);
        // holding the button does not hop straight back out; the stick alone
        // does not move a landed helicopter
        for _ in 0..30 {
            pad.go(&mut m, a, 60, 127, EAST + 1.0, 0.0, B_USE);
        }
        assert_eq!(m.players[a as usize].ms.mode, MODE_FLY);
        assert_eq!(m.w.helis[0], [o.x + 8.0, y0, o.z, EAST], "seed {seed}: moved on the ground");
        assert!(m.helis[0].rotor > 0.2 && !m.helis[0].air);
        // climb
        for _ in 0..150 {
            pad.go(&mut m, a, 0, 0, EAST, 0.0, B_JUMP);
        }
        let p = m.players[a as usize].ms;
        assert!(p.pos.y - y0 > 35.0 && p.pos.y - y0 < 46.0, "seed {seed}: climbed {} m in 5 s", p.pos.y - y0);
        assert!(!p.on_ground && m.helis[0].air);
        assert!((p.vel.y - HELI_CLIMB).abs() < 0.01);
        assert!(m.helis[0].rotor > 0.99, "seed {seed}: rotor at {}", m.helis[0].rotor);
        // cannot get out in mid-air
        pad.go(&mut m, a, 0, 0, EAST, 0.0, B_USE);
        assert_eq!(m.players[a as usize].ms.mode, MODE_FLY, "seed {seed}: left in mid-air");
        // hover: it holds its height
        for _ in 0..60 {
            pad.go(&mut m, a, 0, 0, EAST, 0.0, 0);
        }
        let hover = m.players[a as usize].ms;
        assert!(hover.vel.len() < 0.01, "seed {seed}: drifting in the hover");
        // forward
        let mut top = 0.0f32;
        let mut nose_down = 0.0f32;
        for _ in 0..180 {
            pad.go(&mut m, a, 0, 127, EAST, 0.0, 0);
            top = top.max(m.players[a as usize].ms.vel.len());
            nose_down = nose_down.max(m.helis[0].pitch);
        }
        let p = m.players[a as usize].ms;
        assert!(p.pos.x - hover.pos.x > 150.0, "seed {seed}: flew {} m in 6 s", p.pos.x - hover.pos.x);
        assert!((p.pos.z - hover.pos.z).abs() < 0.5 && (p.pos.y - hover.pos.y).abs() < 0.01, "seed {seed}: wandered off course");
        assert!(top > 44.0 && top <= HELI_VMAX + 0.01, "seed {seed}: top speed {top}");
        assert!(nose_down > 0.15 && nose_down <= 0.3, "seed {seed}: nose down {nose_down}");
        assert_eq!(m.w.helis[0], [p.pos.x, p.pos.y, p.pos.z, p.airtime]);
        // the heading follows the view at a limited rate, banking
        for _ in 0..30 {
            pad.go(&mut m, a, 0, 127, EAST + 2.0, 0.0, 0);
        }
        let turned = wrap_angle(m.players[a as usize].ms.airtime - EAST);
        assert!((turned - HELI_TURN).abs() < 0.02, "seed {seed}: turned {turned} in a second");
        assert!(m.helis[0].roll < -0.05, "seed {seed}: no bank in a left turn ({})", m.helis[0].roll);
        // back over the open run, stop, and down
        let (bx, by, bz) = (o.x + 30.0, m.players[a as usize].ms.pos.y, o.z);
        m.players[a as usize].ms.pos = V3::new(bx, by, bz);
        m.players[a as usize].ms.vel = V3::ZERO;
        m.players[a as usize].ms.airtime = EAST;
        let mut landed = 0;
        for t in 0..1500 {
            pad.go(&mut m, a, 0, 0, EAST, 0.0, B_CROUCH);
            if m.players[a as usize].ms.on_ground {
                landed = t;
                break;
            }
        }
        assert!(landed > 100, "seed {seed}: did not come down (or fell)");
        let p = m.players[a as usize].ms;
        assert!((p.pos.y - m.w.height(p.pos.x, p.pos.z)).abs() < 1.0, "seed {seed}: landed at {} over ground {}", p.pos.y, m.w.height(p.pos.x, p.pos.z));
        assert_eq!(m.helis[0].hp, HELI_HP, "seed {seed}: a gentle landing dented it");
        assert!(!m.helis[0].air);
        pad.go(&mut m, a, 0, 0, EAST, 0.0, 0);
        // out, beside the hull
        pad.go(&mut m, a, 0, 0, EAST, 0.0, B_USE);
        let q = m.players[a as usize].ms;
        assert_eq!(q.mode, MODE_GROUND, "seed {seed}: could not get out after landing");
        assert_eq!(m.helis[0].pilot, 255);
        let h = m.w.helis[0];
        let (lx, lz) = ((q.pos.x - h[0]) * right(h[3]).x + (q.pos.z - h[2]) * right(h[3]).z, (q.pos.x - h[0]) * fwd(h[3]).x + (q.pos.z - h[2]) * fwd(h[3]).z);
        assert!(lx.abs() > HELI_HW + P_RADIUS - 0.01 && lx.abs() < HELI_HW + 1.5 && lz.abs() < HELI_HL, "seed {seed}: stepped out at ({lx}, {lz})");
        assert!((q.pos.y - h[1]).abs() < 1.0 && q.on_ground);
        // the rotor runs down, the helicopter stays
        for _ in 0..300 {
            pad.go(&mut m, a, 0, 0, EAST, 0.0, 0);
        }
        assert!(m.helis[0].rotor < 0.01);
        assert_eq!(m.w.helis[0], h);
        // and it can be boarded again
        pad.go(&mut m, a, 0, 0, EAST, 0.0, B_USE);
        assert_eq!(m.players[a as usize].ms.mode, MODE_FLY, "seed {seed}: could not board it again");
    }
}

#[test]
fn prediction_matches_the_server() {
    // The client runs step_move on the pilot's state against its own copy of
    // the world: it must land on the server's result exactly.
    let (mut m, ids, _o) = stage(42, 2);
    let a = ids[0];
    let mut w = World::generate(42);
    w.cars = m.w.cars.clone();
    w.helis = m.w.helis.clone();
    let mut pad = Pad { seq: 0 };
    pad.go(&mut m, a, 0, 0, EAST, 0.0, 0);
    pad.go(&mut m, a, 0, 0, EAST, 0.0, B_USE);
    let mut s = m.players[a as usize].ms;
    assert_eq!(s.mode, MODE_FLY);
    // The client parks its copy of the helicopter it flies far below the
    // map (so the camera and the aiming ray do not hit it): the flight must
    // not depend on where the world says that helicopter is.
    w.helis[s.car as usize] = [0.0, -1000.0, 0.0, 0.0];
    for t in 0..400u32 {
        let (mx, mz) = (((t / 50) % 3) as i8 * 60 - 60, if t > 60 { 127 } else { 0 });
        let yaw = EAST + (t as f32 * 0.01).sin();
        let b = if t < 90 { B_JUMP } else if t > 330 { B_CROUCH } else { 0 };
        let c = Cmd::from_quant(0, 0, mx, mz, q_yaw(yaw), q_pitch(0.0), b);
        step_move(&w, &mut s, &c, 1.0, DT);
        pad.go(&mut m, a, mx, mz, yaw, 0.0, b);
        let p = m.players[a as usize].ms;
        assert_eq!((s.pos, s.vel, s.airtime, s.on_ground), (p.pos, p.vel, p.airtime, p.on_ground), "tick {t}");
    }
}

#[test]
fn missile_hits_what_it_is_aimed_at() {
    let (mut m, ids, o) = stage(42, 4);
    let (a, near, hidden, far) = (ids[0], ids[1], ids[2], ids[3]);
    // the target: a spot on the ground 45 m ahead of the nose, with a
    // bystander 5 m to one side, a wall 3 m to the other with somebody
    // behind it, and a car just short of it
    let t = V3::new(o.x + 60.0, m.w.height(o.x + 60.0, o.z), o.z);
    add_wall(&mut m, t.x + 3.0, t.z, 4.0, t.y, 3.0);
    place(&mut m, near, V3::new(t.x - 2.0, t.y, t.z + 4.5));
    place(&mut m, hidden, V3::new(t.x + 5.0, t.y, t.z));
    place(&mut m, far, V3::new(t.x - 2.0, t.y, t.z - 30.0));
    m.w.cars[0] = [t.x - 3.0, m.w.height(t.x - 3.0, t.z - 4.5), t.z - 4.5, 0.0];
    let car_hp = m.car_hp[0];
    let mut pad = Pad { seq: 0 };
    board_and_climb(&mut m, &mut pad, a, 60);
    for _ in 0..40 {
        pad.go(&mut m, a, 0, 0, EAST, 0.0, 0);
    }
    let p = m.players[a as usize].ms;
    assert!(p.pos.y - o.y > 8.0 && p.vel.len() < 0.01);
    let mid = p.pos + V3::new(0.0, 2.0, 0.0);
    let pitch = (t.y - mid.y).atan2(t.x - mid.x);
    // not while only looking
    pad.go(&mut m, a, 0, 0, EAST, pitch, 0);
    assert!(m.missiles.is_empty());
    pad.go(&mut m, a, 0, 0, EAST, pitch, B_FIRE);
    assert_eq!(m.missiles.len(), 1);
    assert_eq!(m.helis[0].ammo, HELI_AMMO - 1);
    let ms = m.missiles[0];
    assert_eq!((ms.owner, ms.heli), (a, 0));
    // it leaves from a pod on the side of the hull, not from inside it
    assert!((ms.pos.z - p.pos.z).abs() > HELI_HW && (ms.pos - p.pos).len() < 6.0, "launched from {:?}", ms.pos - p.pos);
    // on the wire
    let wr = wire(m.snapshot(a));
    assert_eq!(wr.missiles.len(), 1);
    assert_eq!(wr.missiles[0].0, ms.id);
    assert!((wr.missiles[0].1 - ms.pos).len() < 1e-3);
    assert!(wrap_angle(wr.missiles[0].2 - EAST).abs() < 0.1 && (wr.missiles[0].3 - pitch).abs() < 0.1, "direction on the wire {} {}", wr.missiles[0].2, wr.missiles[0].3);
    m.flush();
    let mut last = ms.pos;
    let mut flew = 0;
    while !m.missiles.is_empty() {
        last = m.missiles[0].pos;
        let step = (last - ms.pos).len();
        assert!(step <= MISSILE_SPEED * DT * (flew as f32 + 1.0) + 0.01);
        pad.go(&mut m, a, 0, 0, EAST, pitch, 0);
        flew += 1;
        assert!(flew < 60, "the missile never arrived");
    }
    assert!((last - t).len() < MISSILE_SPEED * DT + 1.5, "went off {} m from the target", (last - t).len());
    assert!(flew >= 12 && flew <= 20, "took {flew} ticks for 45 m");
    // the blast is an event for everyone near
    let buf = m.snapshot(near).to_vec();
    let boom = buf.windows(13).any(|w| w[0] == E_BOOM && (V3::new(f32::from_le_bytes(w[1..5].try_into().unwrap()), f32::from_le_bytes(w[5..9].try_into().unwrap()), f32::from_le_bytes(w[9..13].try_into().unwrap())) - t).len() < 2.0);
    assert!(boom, "no explosion event at the target");
    assert!(wire(&buf).events > 0);
    let hn = m.players[near as usize].health;
    assert!(hn < 80.0 && hn > 0.0, "the bystander has {hn}");
    assert_eq!(m.players[near as usize].hurt_by, a, "the pilot is not credited");
    assert_eq!(m.players[hidden as usize].health, 100.0, "hurt through the wall");
    assert_eq!(m.players[far as usize].health, 100.0);
    assert!(m.car_hp[0] < car_hp - 150.0, "the car lost {}", car_hp - m.car_hp[0]);
    assert_eq!(m.players[a as usize].health, 100.0);
    assert_eq!(m.helis[0].hp, HELI_HP);
    // a second one finishes the bystander: the kill is the pilot's
    m.players[near as usize].health = 20.0;
    for _ in 0..30 {
        pad.go(&mut m, a, 0, 0, EAST, pitch, 0);
    }
    pad.go(&mut m, a, 0, 0, EAST, pitch, B_FIRE);
    for _ in 0..30 {
        pad.go(&mut m, a, 0, 0, EAST, pitch, 0);
    }
    let v = &m.players[near as usize];
    assert!(!v.alive, "the bystander survived a second missile");
    assert_eq!(v.killer, a);
    assert_eq!(m.players[a as usize].kills, 1);
    assert!(m.players[hidden as usize].alive && m.players[hidden as usize].health == 100.0);
    // a direct hit on a parked car wrecks it
    let (mut m, ids, o) = stage(42, 2);
    let a = ids[0];
    m.w.cars[0] = [o.x + 55.0, m.w.height(o.x + 55.0, o.z), o.z, 0.0];
    let mut pad = Pad { seq: 0 };
    board_and_climb(&mut m, &mut pad, a, 60);
    for _ in 0..40 {
        pad.go(&mut m, a, 0, 0, EAST, 0.0, 0);
    }
    let mid = m.players[a as usize].ms.pos + V3::new(0.0, 2.0, 0.0);
    let c = m.w.cars[0];
    let pitch = (c[1] + 0.8 - mid.y).atan2(c[0] - mid.x);
    pad.go(&mut m, a, 0, 0, EAST, pitch, B_FIRE);
    for _ in 0..40 {
        pad.go(&mut m, a, 0, 0, EAST, pitch, 0);
    }
    assert!(m.car_hp[0] <= 0.0, "a direct hit left the car {}", m.car_hp[0]);
}

#[test]
fn ammunition_and_cooldown() {
    let (mut m, ids, _o) = stage(7, 2);
    let a = ids[0];
    let mut pad = Pad { seq: 0 };
    // somebody on foot has no missiles
    pad.go(&mut m, a, 0, 0, EAST, 0.4, B_FIRE);
    assert!(m.missiles.is_empty());
    board_and_climb(&mut m, &mut pad, a, 90);
    let mut launches: Vec<u32> = vec![];
    let mut seen = 0u16;
    let mut sides = vec![];
    // the trigger held for half a minute, aiming up and away
    for _ in 0..900 {
        pad.go(&mut m, a, 0, 0, EAST, 0.5, B_FIRE);
        for ms in &m.missiles {
            if ms.id > seen {
                seen = ms.id;
                launches.push(m.tick);
                sides.push((ms.pos.z - m.players[a as usize].ms.pos.z).signum());
            }
        }
        assert_eq!(m.players[a as usize].health, 100.0);
    }
    assert_eq!(launches.len(), HELI_AMMO as usize, "launched {}", launches.len());
    assert_eq!(m.helis[0].ammo, 0);
    for k in 1..launches.len() {
        let gap = launches[k] - launches[k - 1];
        assert!((3..=4).contains(&gap), "launches {gap} ticks apart");
        assert!(sides[k] != sides[k - 1], "the pods do not take turns");
    }
    assert!(m.missiles.is_empty(), "missiles still flying after their run");
    assert_eq!(m.helis[0].hp, HELI_HP);
    let wr = wire(m.snapshot(a));
    assert_eq!(wr.helis[0].8, 0, "ammunition on the wire");
}

#[test]
fn missiles_fired_off_the_map_do_not_come_back_on_the_helicopter() {
    // Hovering against the edge of the map (flight is clamped 10 m inside
    // it) and firing outwards: the missiles leave the map a few metres from
    // the pods. They must vanish there, not go off beside the hull.
    let (mut m, ids, _o) = stage(42, 2);
    let a = ids[0];
    let mut pad = Pad { seq: 0 };
    board_and_climb(&mut m, &mut pad, a, 60);
    let y = m.players[a as usize].ms.pos.y + 40.0;
    let z = m.players[a as usize].ms.pos.z;
    m.players[a as usize].ms.pos = V3::new(MAP - 10.0, y, z);
    m.players[a as usize].ms.vel = V3::ZERO;
    m.w.helis[0] = [MAP - 10.0, y, z, EAST];
    let mut fired = 0;
    let mut seen = 0u16;
    for _ in 0..120 {
        pad.go(&mut m, a, 0, 0, EAST, 0.0, B_FIRE);
        for ms in &m.missiles {
            if ms.id > seen {
                seen = ms.id;
                fired += 1;
            }
        }
    }
    assert!(fired >= 4, "fired {fired}");
    assert!(m.missiles.is_empty() || m.missiles.len() == 1);
    assert_eq!(m.helis[0].hp, HELI_HP, "its own missiles hurt it");
    assert!(m.players[a as usize].alive);
    assert_eq!(m.players[a as usize].ms.mode, MODE_FLY);
}

#[test]
fn bullets_wear_the_hull_down_and_it_blows_up() {
    let (mut m, ids, o) = stage(42, 3);
    let (a, b, c) = (ids[0], ids[1], ids[2]);
    let mut pad = Pad { seq: 0 };
    pad.go(&mut m, a, 0, 0, EAST, 0.0, 0);
    pad.go(&mut m, a, 0, 0, EAST, 0.0, B_USE);
    assert_eq!(m.players[a as usize].ms.mode, MODE_FLY);
    // b shoots the nose from 10 m; c stands behind the tail
    place(&mut m, b, V3::new(o.x + 26.0, o.y, o.z));
    place(&mut m, c, V3::new(o.x - 3.0, o.y, o.z));
    {
        let p = &mut m.players[b as usize];
        p.slots[0] = (W_AR, 30);
        p.cur = 0;
    }
    let west = -EAST;
    for _ in 0..100 {
        pad.go(&mut m, b, 0, 0, west, 0.0, B_FIRE | B_ADS);
    }
    let hp = m.helis[0].hp;
    assert!((hp - (HELI_HP - 30.0 * WD[W_AR as usize].dmg)).abs() < 0.5, "30 rifle rounds left the hull at {hp}");
    assert_eq!(m.helis[0].hit_by, b);
    assert_eq!(m.players[a as usize].health, 100.0, "the pilot was hit through the hull");
    assert_eq!(m.players[c as usize].health, 100.0, "shot through the hull");
    let wr = wire(m.snapshot(b));
    let hw = wr.helis.iter().find(|h| h.0 == 0).expect("helicopter not sent");
    assert_eq!(hw.7, ((hp / HELI_HP) * 255.0).ceil() as u8);
    assert_eq!(hw.6, a, "a pilot in plain view is named");
    // grenades and other blasts reach it too
    let before = m.helis[0].hp;
    m.nades.push(Nade { pos: V3::new(o.x + 18.0, o.y + 1.3, o.z), vel: V3::ZERO, fuse: 0.01, owner: b, id: 999 });
    m.players[b as usize].ms.pos.x += 20.0;
    pad.go(&mut m, b, 0, 0, west, 0.0, 0);
    assert!(m.helis[0].hp < before - 100.0, "a grenade beside it did {}", before - m.helis[0].hp);
    assert_eq!(m.players[a as usize].health, 100.0);
    // the last of it: the pilot dies with it, b gets the kill, a wreck stays
    m.players[c as usize].ms.pos.x -= 30.0;
    let left = m.helis[0].hp;
    m.hurt_heli(0, left + 1.0, b);
    assert_eq!(m.helis[0].hp, 0.0);
    assert_eq!(m.helis[0].pilot, 255);
    let p = &m.players[a as usize];
    assert!(!p.alive);
    assert_eq!(p.killer, b);
    assert_eq!(m.players[b as usize].kills, 1);
    let buf = m.snapshot(c).to_vec();
    assert!(buf.windows(2).any(|w| w[0] == E_BOOM), "no explosion event");
    let wr = wire(&buf);
    let hw = wr.helis.iter().find(|h| h.0 == 0).unwrap();
    assert_eq!((hw.7, hw.6), (0, 255), "the wreck on the wire");
    m.flush();
    // the wreck cannot be boarded, still stops bullets and walkers
    place(&mut m, c, V3::new(o.x + 12.0, o.y, o.z - HELI_HW - 0.9));
    pad.go(&mut m, c, 0, 0, 0.0, 0.0, 0);
    pad.go(&mut m, c, 0, 0, 0.0, 0.0, B_USE);
    assert_eq!(m.players[c as usize].ms.mode, MODE_GROUND, "boarded a wreck");
    assert!(m.w.raycast(V3::new(o.x + 30.0, o.y + 2.0, o.z), V3::new(-1.0, 0.0, 0.0), 20.0).map_or(false, |h| h.2 == BK_HELI));
}

#[test]
fn a_crash_wrecks_it() {
    for (speed_ticks, dies) in [(45usize, false), (200, true)] {
        let (mut m, ids, o) = stage(7, 3);
        let a = ids[0];
        let mut pad = Pad { seq: 0 };
        // (well above the trees beyond the open run)
        board_and_climb(&mut m, &mut pad, a, 150);
        let p = m.players[a as usize].ms;
        // a wall across the way
        let x0 = p.pos.x;
        // far enough to get up to speed, close enough to still be building it
        let gap = if dies { 160.0 } else { 26.0 };
        add_wall(&mut m, x0 + gap, o.z, 30.0, o.y, 120.0);
        let mut v_hit = 0.0f32;
        for _ in 0..speed_ticks + 200 {
            let v = m.players[a as usize].ms.vel.len();
            pad.go(&mut m, a, 0, 127, EAST, 0.0, 0);
            if m.helis[0].hp < HELI_HP {
                v_hit = v;
                break;
            }
        }
        assert!(v_hit > 5.0, "never reached the wall");
        let hp = m.helis[0].hp;
        if dies {
            assert!(v_hit > 40.0, "hit at {v_hit}");
            assert_eq!(hp, 0.0, "survived a wall at {v_hit} m/s with {hp}");
            let p = &m.players[a as usize];
            assert!(!p.alive, "the pilot survived the crash");
            assert_eq!(p.killer, a);
            assert_eq!(m.helis[0].pilot, 255);
            // the wreck falls out of the sky and comes to rest
            assert!(m.helis[0].air);
            for _ in 0..300 {
                m.step();
            }
            let h = m.w.helis[0];
            assert!(!m.helis[0].air, "the wreck is still up at {}", h[1]);
            assert!((h[1] - m.w.height(h[0], h[2])).abs() < 1.5, "the wreck rests at {} over {}", h[1], m.w.height(h[0], h[2]));
            assert!(h[0] < x0 + gap, "the wreck passed through the wall");
        } else {
            assert!(v_hit < 25.0, "hit at {v_hit}");
            assert!(hp > 0.0 && hp < HELI_HP - 100.0, "a wall at {v_hit} m/s left {hp}");
            assert!(m.players[a as usize].alive);
            // it stopped at the wall
            for _ in 0..30 {
                pad.go(&mut m, a, 0, 127, EAST, 0.0, 0);
            }
            assert!(m.players[a as usize].ms.pos.x + HELI_HL < x0 + gap, "flew through the wall");
        }
    }
}

#[test]
fn a_pilot_who_disconnects_bails_out_and_the_helicopter_comes_down() {
    let (mut m, ids, o) = stage(42, 3);
    let a = ids[0];
    let mut pad = Pad { seq: 0 };
    board_and_climb(&mut m, &mut pad, a, 150);
    for _ in 0..60 {
        pad.go(&mut m, a, 0, 127, EAST, 0.0, 0);
    }
    let up = m.w.helis[0];
    assert!(up[1] - o.y > 35.0);
    m.remove_human(a);
    let p = &m.players[a as usize];
    assert!(p.bot && p.alive);
    assert_eq!(p.ms.mode, MODE_PARA);
    assert_eq!(m.helis[0].pilot, 255);
    let mut down = 0;
    for t in 0..1800 {
        m.step();
        if !m.helis[0].air {
            down = t;
            break;
        }
    }
    assert!(down > 150, "the empty helicopter fell (or never landed): {down}");
    let h = m.w.helis[0];
    assert_eq!(m.helis[0].hp, HELI_HP, "the autorotation ended in a crash");
    assert!((h[1] - m.w.height(h[0], h[2]).max(WATER - HELI_DRAUGHT)).abs() < 1.5);
    assert!(dist_xz(h[0], h[2], up[0], up[2]) < 80.0, "it flew on for {} m", dist_xz(h[0], h[2], up[0], up[2]));
    for _ in 0..600 {
        m.step();
    }
    let p = &m.players[a as usize];
    assert!(p.alive && p.ms.mode == MODE_GROUND, "the bot did not land ({})", p.ms.mode);
    assert_eq!(m.w.helis[0], h, "the landed helicopter moved");
}

#[test]
fn snapshot_carries_helicopters() {
    assert_eq!(PROTO, 16);
    let (mut m, ids, o) = stage(42, 3);
    let (a, b) = (ids[0], ids[1]);
    // b watches from the far end of the open run, the third from the far corner
    place(&mut m, b, V3::new(o.x + 73.0, o.y, o.z));
    m.players[ids[2] as usize].ms.pos = V3::new(1000.0, m.w.height(1000.0, 1000.0), 1000.0);
    let mut pad = Pad { seq: 0 };
    let wr = wire(m.snapshot(b));
    let h = wr.helis.iter().find(|h| h.0 == 0).expect("a parked helicopter in range is sent");
    assert!((h.1 - V3::new(m.w.helis[0][0], m.w.helis[0][1], m.w.helis[0][2])).len() < 1e-3);
    assert!(wrap_angle(h.2 - EAST).abs() < 1e-3);
    assert_eq!((h.3, h.4, h.5, h.6, h.7, h.8, h.9), (0.0, 0.0, 0, 255, 255, HELI_AMMO, 0));
    m.flush();
    board_and_climb(&mut m, &mut pad, a, 90);
    for _ in 0..60 {
        pad.go(&mut m, a, 0, 127, EAST, 0.0, 0);
    }
    for _ in 0..4 {
        pad.go(&mut m, a, 0, 127, EAST, 0.0, 0);
        let _ = m.snapshot(b).len();
        m.flush();
    }
    for pid in [a, b] {
        let buf = m.snapshot(pid).to_vec();
        let wr = wire(&buf);
        let h = wr.helis.iter().find(|h| h.0 == 0).expect("the flying helicopter is sent");
        let p = m.players[a as usize].ms;
        assert!((h.1 - p.pos).len() < 1e-3);
        assert!(h.3 > 0.05, "pitch on the wire {}", h.3);
        assert_eq!(h.5, 255, "rotor at full speed");
        assert_eq!(h.9 & 1, 1, "airborne");
        // its pilot is in the open sky: named, and sent
        assert_eq!(h.6, a);
        assert!(pid == a || wr.players.contains(&a));
        if pid == a {
            // the pilot's own state is the helicopter's
            assert_eq!(buf[17], MODE_FLY);
            let x = f32::from_le_bytes(buf[19..23].try_into().unwrap());
            assert_eq!(x, p.pos.x);
        }
    }
    m.flush();
    // out of range: not sent, nor is anything about it
    let wr = wire(m.snapshot(ids[2]));
    assert!(wr.helis.iter().all(|h| h.0 != 0), "sent across the map");
    // a pilot behind a hill or a building stays anonymous: the viewer is
    // put behind a wall 60 m away
    m.players[a as usize].ms.pos.x = o.x + 10.0;
    m.players[a as usize].ms.vel = V3::ZERO;
    add_wall(&mut m, o.x + 72.0, o.z, 6.0, o.y, 120.0);
    for _ in 0..90 {
        pad.go(&mut m, a, 0, 0, EAST, 0.0, 0);
        let _ = m.snapshot(b).len();
        m.flush();
    }
    let wr = wire(m.snapshot(b));
    let h = wr.helis.iter().find(|h| h.0 == 0).expect("a helicopter in range is sent, seen or not");
    assert_eq!(h.6, 254, "an unseen pilot was named");
    assert!(dist_xz(h.1.x, h.1.z, o.x + 73.0, o.z) > 45.0);
    // and the pilot itself is not among the players sent
    assert!(!wr.players.contains(&a), "the hidden pilot is sent as a player");
}
