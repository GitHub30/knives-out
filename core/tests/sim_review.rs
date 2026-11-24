//! Regression tests for the simulation findings of the second adversarial
//! review: window-sill phantom ground (mid-air jumps, ejection through the
//! wall), torso / shoulder hit registration, items spawned inside solids and
//! bots freezing at door waypoints.

use zonefall_core::defs::*;
use zonefall_core::math::*;
use zonefall_core::player::*;
use zonefall_core::rng::Rng;
use zonefall_core::sim::*;
use zonefall_core::world::*;

/// Window openings: (building, sill box) for sills of the outer walls.
fn windows(w: &World) -> Vec<(usize, Bx)> {
    let mut out = vec![];
    for (bi, b) in w.buildings.iter().enumerate() {
        for bx in &w.boxes {
            if bx.kind != BK_WALL || (bx.min.y - b.y).abs() > 1e-3 || (bx.max.y - b.y - 1.0).abs() > 0.05 {
                continue;
            }
            let on_x = (bx.min.z - b.z0).abs() < 1e-3 || (bx.max.z - b.z1).abs() < 1e-3;
            let on_z = (bx.min.x - b.x0).abs() < 1e-3 || (bx.max.x - b.x1).abs() < 1e-3;
            if on_x || on_z {
                out.push((bi, *bx));
            }
        }
    }
    out
}

#[test]
fn window_sills_give_no_mid_air_jump_or_exit() {
    let mut runs = 0;
    for seed in [7u32, 42] {
        let w = World::generate(seed);
        let wins = windows(&w);
        assert!(wins.len() > 50, "seed {seed}: only {} windows found", wins.len());
        let mut rng = Rng::new(seed as u64 * 31 + 1);
        for (bi, sill) in wins.iter().take(160) {
            let b = w.buildings[*bi];
            let cx = (sill.min.x + sill.max.x) * 0.5;
            let cz = (sill.min.z + sill.max.z) * 0.5;
            // inward normal of this wall and the yaw that faces the window
            let (nx, nz) = if (sill.min.z - b.z0).abs() < 1e-3 {
                (0.0, 1.0)
            } else if (sill.max.z - b.z1).abs() < 1e-3 {
                (0.0, -1.0)
            } else if (sill.min.x - b.x0).abs() < 1e-3 {
                (1.0, 0.0)
            } else {
                (-1.0, 0.0)
            };
            let face = yaw_to(-nx, -nz);
            for _ in 0..12 {
                runs += 1;
                let start = V3::new(cx + nx * 0.75, b.y, cz + nz * 0.75);
                let mut s = MoveState { pos: start, mode: MODE_GROUND, on_ground: true, crouch: true, ..Default::default() };
                let mut crouch = true;
                let mut near_door = false;
                for t in 0..90 {
                    if rng.chance(0.12) {
                        crouch = !crouch;
                    }
                    let mut buttons = if crouch { B_CROUCH } else { 0 };
                    if rng.chance(0.35) {
                        buttons |= B_JUMP;
                    }
                    if rng.chance(0.2) {
                        buttons |= B_SPRINT;
                    }
                    let c = Cmd { seq: t, view: 0, mx: rng.range(-0.3, 0.3), mz: rng.range(0.2, 1.0), yaw: face + rng.range(-0.5, 0.5), pitch: 0.0, buttons };
                    let g = w.ground_below(s.pos.x, s.pos.z, s.pos.y, P_RADIUS);
                    let lifted = s.pos.y - g;
                    let ev = step_move(&w, &mut s, &c, 1.0, DT);
                    assert!(!(ev & EV_JUMPED != 0 && lifted > 0.25), "seed {seed} building {bi}: jumped from {lifted:.2} m in mid-air at {:?}", s.pos);
                    near_door |= dist_xz(s.pos.x, s.pos.z, b.door_in.0, b.door_in.1) < 2.0 || dist_xz(s.pos.x, s.pos.z, b.door_out.0, b.door_out.1) < 2.0;
                    if !near_door {
                        assert!(b.contains(s.pos.x, s.pos.z), "seed {seed} building {bi}: left the building through a window wall at {:?} (tick {t})", s.pos);
                    }
                }
            }
        }
    }
    assert!(runs > 2000);
}

#[test]
fn standing_up_needs_headroom() {
    // a crouched player under a 3.2 m roof cannot stand up into the roof slab
    let w = World::generate(7);
    let b = w.buildings[0];
    let c = ((b.x0 + b.x1) * 0.5, (b.z0 + b.z1) * 0.5);
    let roof = w.boxes.iter().find(|x| x.kind == BK_ROOF && x.min.x < c.0 && x.max.x > c.0 && x.min.z < c.1 && x.max.z > c.1).unwrap();
    let feet = roof.min.y - 1.3; // crouched body fits, standing would not
    let mut s = MoveState { pos: V3::new(c.0, feet, c.1), mode: MODE_GROUND, on_ground: true, crouch: true, ..Default::default() };
    let cmd = Cmd { buttons: 0, ..Default::default() };
    let before = s.pos;
    step_move(&w, &mut s, &cmd, 1.0, DT);
    assert!(s.crouch, "stood up into the roof");
    assert!(dist_xz(s.pos.x, s.pos.z, before.x, before.z) < 0.2, "pushed {:?} -> {:?}", before, s.pos);
}

/// Classify a ray aimed horizontally at (dx, y) in B's frame: 0 miss, 1 body, 2 head.
fn ray_class(m: &Match, shooter: usize, target: V3, dx: f32, y: f32) -> u8 {
    let o = V3::new(target.x + dx, target.y + y, target.z - 10.0);
    let d = V3::new(0.0, 0.0, 1.0);
    match m.ray_players(shooter, o, d, 30.0, 0) {
        None => 0,
        Some((_, _, head)) => {
            if head {
                2
            } else {
                1
            }
        }
    }
}

#[test]
fn torso_shoulders_and_heads_register() {
    let mut m = Match::new(42, 2, FLAG_OFFLINE, 3);
    let a = m.add_human(b"A").unwrap() as usize;
    let b = m.add_human(b"B").unwrap() as usize;
    m.step();
    m.players[a].bot = true; // live positions, no rewind
    let t = V3::new(500.0, 30.0, 500.0);
    for crouch in [false, true] {
        m.players[b].ms = MoveState { pos: t, mode: MODE_GROUND, on_ground: true, crouch, ..Default::default() };
        m.players[a].ms = MoveState { pos: t - V3::new(0.0, 0.0, 10.0), mode: MODE_GROUND, on_ground: true, ..Default::default() };
        // rendered torso: hips (0.95 / 0.5) to shoulders (+0.56), 0.23 m half-width
        let (t0, t1, h0, h1) = if crouch { (0.6, 1.05, 1.15, 1.32) } else { (1.0, 1.52, 1.6, 1.8) };
        let mut y = t0;
        while y <= t1 {
            for dx in [-0.22f32, -0.1, 0.0, 0.1, 0.22] {
                assert_eq!(ray_class(&m, a, t, dx, y), 1, "crouch={crouch}: torso ray at x {dx} y {y:.2} not a body hit");
            }
            y += 0.04;
        }
        let mut y = h0;
        while y <= h1 {
            for dx in [-0.08f32, 0.0, 0.08] {
                assert_eq!(ray_class(&m, a, t, dx, y), 2, "crouch={crouch}: head ray at x {dx} y {y:.2} not a head hit");
            }
            y += 0.03;
        }
    }
}

#[test]
fn no_item_spawns_inside_solids() {
    // crates of a later town used to cover loot spots of earlier ones
    // (seeds 25, 55, 132, 162)
    for seed in (1..=40u32).chain([55, 132, 162]) {
        let m = Match::new(seed, 8, FLAG_OFFLINE, 1);
        for it in m.items.iter().filter(|it| it.alive) {
            assert!(m.w.item_spot_clear(it.pos.x, it.pos.y + 0.2, it.pos.z, 0.0), "seed {seed}: item kind {} inside a solid at {:?}", it.kind, it.pos);
        }
    }
}

#[test]
fn bots_do_not_freeze_at_waypoints() {
    let mut worst = 0.0f32;
    for seed in 1..=16u32 {
        let mut m = Match::new(seed, 30, FLAG_OFFLINE, 0x5150 + seed as u64);
        m.add_human(b"H").unwrap();
        m.step();
        // the human just floats in the plane
        let mut still = vec![0.0f32; MAX_PLAYERS];
        let mut last = vec![V3::ZERO; MAX_PLAYERS];
        for _ in 0..(30 * 240) {
            m.step();
            if m.phase != PH_PLAY {
                break;
            }
            for i in 0..MAX_PLAYERS {
                let p = &m.players[i];
                if !p.used || !p.bot || !p.alive || p.ms.mode != MODE_GROUND {
                    still[i] = 0.0;
                    continue;
                }
                let busy = p.heal_t > 0.0 || (p.brain.target as usize) < MAX_PLAYERS;
                if dist_xz(p.ms.pos.x, p.ms.pos.z, last[i].x, last[i].z) > 0.05 || busy {
                    still[i] = 0.0;
                    last[i] = p.ms.pos;
                } else {
                    still[i] += DT;
                    worst = worst.max(still[i]);
                }
            }
        }
    }
    eprintln!("longest idle bot freeze: {worst:.1} s");
    assert!(worst < 12.0, "a bot stood still for {worst:.1} s without fighting or healing");
}

/// Third review: the head sphere sat on the body axis while the drawn head
/// is ahead of it (crouched 0.11 m, more when looking down), so from the side
/// a quarter of the drawn head missed and air behind it counted as head.
/// The head box here is rebuilt from src/render/models.ts (emitCharacter).
#[test]
fn side_shots_hit_the_drawn_head() {
    let mut m = Match::new(42, 2, FLAG_OFFLINE, 3);
    let a = m.add_human(b"A").unwrap() as usize;
    let b = m.add_human(b"B").unwrap() as usize;
    m.step();
    m.players[a].bot = true; // live positions, no rewind
    let t = V3::new(500.0, 30.0, 500.0);
    let rot = |ang: f32, y: f32, z: f32| (y * ang.cos() - z * ang.sin(), y * ang.sin() + z * ang.cos());
    let mut worst = 1.0f32;
    for crouch in [false, true] {
        for pitch in [-0.6f32, 0.0, 0.5] {
            for yaw in [0.0f32, 1.2, -2.5] {
                {
                    let q = &mut m.players[b];
                    q.ms = MoveState { pos: t, mode: MODE_GROUND, on_ground: true, crouch, ..Default::default() };
                    q.yaw = yaw;
                    q.pitch = pitch;
                }
                // models.ts: hips at hipH, torso pivot +0.02 leaning `lean`, head pivot
                // +0.6 up the torso, head box centre (0, 0.14, 0.01) turned by
                // lean + headPitch, size 0.25 x 0.27 x 0.27
                let lean = if crouch { 0.15f32 } else { 0.05 };
                let hip = if crouch { 0.5f32 } else { 0.95 };
                let head_pitch = -pitch * 0.55 - lean * 0.6;
                let (py, pz) = rot(lean, 0.6, 0.0);
                let (cy, cz) = rot(lean + head_pitch, 0.14, 0.01);
                let (hy, hz) = (hip + 0.02 + py + cy, pz + cz);
                let f = fwd(yaw);
                let side = right(yaw);
                let at = |up: f32, ahead: f32| t + V3::new(f.x * ahead, up, f.z * ahead);
                // rays from the side through the central 80% of the head's side silhouette
                let (mut n, mut head) = (0, 0);
                for iy in -4..=4 {
                    for iz in -4..=4 {
                        let p = at(hy + iy as f32 * 0.027, hz + iz as f32 * 0.027);
                        m.players[a].ms = MoveState { pos: p + side * 10.0 - V3::new(0.0, eye_h(false), 0.0), mode: MODE_GROUND, on_ground: true, ..Default::default() };
                        let o = p + side * 10.0;
                        n += 1;
                        if let Some((j, _, true)) = m.ray_players(a, o, side * -1.0, 30.0, 0) {
                            if j == b {
                                head += 1;
                            }
                        }
                    }
                }
                let frac = head as f32 / n as f32;
                worst = worst.min(frac);
                assert!(frac >= 0.97, "crouch={crouch} pitch={pitch} yaw={yaw}: only {:.0}% of the drawn head registers as head", frac * 100.0);
                // air 0.25 m behind the head centre is not a headshot
                let p = at(hy, hz - 0.25);
                let o = p + side * 10.0;
                let r = m.ray_players(a, o, side * -1.0, 30.0, 0);
                assert!(!matches!(r, Some((_, _, true))), "crouch={crouch} pitch={pitch}: air behind the head counts as a headshot");
            }
        }
    }
    eprintln!("worst head coverage from the side: {:.0}%", worst * 100.0);
}

/// A free-falling body is drawn face down around the hips: shots at the drawn
/// torso hit, shots where an upright body would be miss.
#[test]
fn free_fall_hitbox_follows_the_drawn_body() {
    let mut m = Match::new(42, 2, FLAG_OFFLINE, 3);
    let a = m.add_human(b"A").unwrap() as usize;
    let b = m.add_human(b"B").unwrap() as usize;
    m.step();
    m.players[a].bot = true;
    let t = V3::new(500.0, 200.0, 500.0);
    for yaw in [0.0f32, 2.0] {
        {
            let q = &mut m.players[b];
            q.ms = MoveState { pos: t, mode: MODE_FALL, ..Default::default() };
            q.yaw = yaw;
            q.pitch = 0.0;
        }
        let f = fwd(yaw);
        let side = right(yaw);
        let shot = |m: &Match, up: f32, ahead: f32| {
            let p = t + V3::new(f.x * ahead, up, f.z * ahead);
            m.ray_players(a, p + side * 10.0, side * -1.0, 30.0, 0).map(|h| h.2)
        };
        // the drawn torso and head lie ahead of the hips (0.95 m up)
        assert_eq!(shot(&m, 0.95 + 0.12, 0.35), Some(false), "yaw {yaw}: drawn torso not hit");
        assert_eq!(shot(&m, 0.95 + 0.2, 0.73), Some(true), "yaw {yaw}: drawn head not hit");
        // where an upright head would be: empty air
        assert_eq!(shot(&m, 1.7, 0.0), None, "yaw {yaw}: upright head position still hit");
    }
}

// ---------------------------------------------------------------------------
// Fourth adversarial review

type Q = (u32, u32, i8, i8, u16, i16, u16);

fn cmd_bytes(cmds: &[Q]) -> Vec<u8> {
    let mut v = vec![C_INPUT, cmds.len() as u8];
    for &(seq, view, mx, mz, yaw, pitch, b) in cmds {
        v.extend_from_slice(&seq.to_le_bytes());
        v.extend_from_slice(&view.to_le_bytes());
        v.push(mx as u8);
        v.push(mz as u8);
        v.extend_from_slice(&yaw.to_le_bytes());
        v.extend_from_slice(&pitch.to_le_bytes());
        v.extend_from_slice(&b.to_le_bytes());
    }
    v
}

/// Flat open ground with 30 m of clear view along +z.
fn open_ground(m: &Match) -> V3 {
    for gz in (80..940).step_by(11) {
        for gx in (80..860).step_by(11) {
            let (x, z) = (gx as f32, gz as f32);
            let h0 = m.w.height(x, z);
            if h0 < 3.0 || m.w.building_at(x, z).is_some() {
                continue;
            }
            let flat = (0..30).all(|k| {
                let h = m.w.height(x, z + k as f32);
                (h - h0).abs() < 0.6 && h > 2.0
            });
            let clear = [0.4f32, 1.0, 1.6, 2.0].iter().all(|&y| m.w.raycast(V3::new(x, h0 + y, z), V3::new(0.0, 0.0, 1.0), 30.0).is_none());
            if flat && clear {
                return V3::new(x, m.w.ground_below(x, z, h0 + 5.0, P_RADIUS), z);
            }
        }
    }
    panic!("no open ground");
}

/// Online snapshots go out every 2 ticks, and the client draws the stance of
/// the newer snapshot around its render time. A target crouching only on odd
/// ticks is always drawn standing (crouching only on even ticks, always
/// crouched): shots at the head it is drawn with must hit that head.
#[test]
fn stance_flicker_between_snapshots_is_judged_as_drawn() {
    for odd in [true, false] {
        let mut m = Match::new(42, 2, 0, 99);
        let a = m.add_human(b"A").unwrap() as usize;
        let b = m.add_human(b"B").unwrap() as usize;
        m.step();
        assert_eq!((m.phase, m.snap_every), (PH_PLAY, 2));
        let g = open_ground(&m);
        let gb = V3::new(g.x, m.w.ground_below(g.x, g.z + 20.0, g.y + 6.0, P_RADIUS), g.z + 20.0);
        m.players[a].ms = MoveState { pos: g, mode: MODE_GROUND, on_ground: true, ..Default::default() };
        m.players[b].ms = MoveState { pos: gb, mode: MODE_GROUND, on_ground: true, ..Default::default() };
        let byaw = yaw_to(0.0, -1.0);
        // what B's snapshots carry, per tick
        let mut hist = vec![(V3::ZERO, false, 0.0f32, 0.0f32, 0u8); 1000];
        let (mut sa, mut sb) = (m.players[a].last_seq, m.players[b].last_seq);
        let mut rng = Rng::new(5);
        let (mut n, mut heads, mut drawn_crouched) = (0, 0, 0);
        for _ in 0..400 {
            let next = m.tick + 1;
            let crouch = (next % 2 == 1) == odd;
            sb += 1;
            m.on_message(b as u8, &cmd_bytes(&[(sb, m.tick.saturating_sub(5) * 256, 0, 0, net_yaw(byaw), 0, if crouch { B_CROUCH } else { 0 })]));
            sa += 1;
            m.on_message(a as u8, &cmd_bytes(&[(sa, m.tick.saturating_sub(5) * 256, 0, 0, 0, 0, 0)]));
            m.step();
            let q = &m.players[b];
            hist[m.tick as usize] = (q.ms.pos, q.ms.crouch, q.yaw, q.pitch, q.ms.mode);
            if m.tick < 60 {
                continue;
            }
            // A renders 4-10 ticks back, between the even-tick snapshots
            let eye = m.players[a].ms.pos + V3::new(0.0, eye_h(false), 0.0);
            for _ in 0..6 {
                let rt = m.tick as f32 - rng.range(4.0, 10.0);
                let t0 = (rt / 2.0).floor() as usize * 2;
                let (x, y) = (hist[t0], hist[t0 + 2]);
                let f = (rt - t0 as f32) / 2.0;
                let head = body_pose(x.0.lerp(y.0, f), x.2 + wrap_angle(y.2 - x.2) * f, x.3 + (y.3 - x.3) * f, y.1, false, y.4).head;
                drawn_crouched += y.1 as u32;
                n += 1;
                if let Some((j, _, true)) = m.ray_players(a, eye, (head - eye).norm(), 60.0, (rt * 256.0) as u32) {
                    if j == b {
                        heads += 1;
                    }
                }
            }
        }
        let drawn = drawn_crouched as f32 / n as f32;
        assert!(if odd { drawn == 0.0 } else { drawn == 1.0 }, "odd={odd}: drawn crouched {drawn:.2} of the time");
        let frac = heads as f32 / n as f32;
        eprintln!("crouch on {} ticks: {:.1}% of shots at the drawn head are headshots", if odd { "odd" } else { "even" }, frac * 100.0);
        assert!(frac >= 0.99, "crouch on {} ticks: only {:.1}% headshots at the drawn head", if odd { "odd" } else { "even" }, frac * 100.0);
    }
}

fn net_yaw(y: f32) -> u16 {
    zonefall_core::net::q_yaw(y)
}

/// The client model's torso frame (models.ts emitCharacter), composed the way
/// the client composes its rotations: qY(yaw) (times qX(1.25) about the hips
/// in free fall), the hips, then the torso turned by qX(lean).
fn drawn_torso(pos: V3, yaw: f32, crouch: bool, sprint: bool, mode: u8) -> (V3, [V3; 3]) {
    let ry = |v: V3| V3::new(v.x * yaw.cos() + v.z * yaw.sin(), v.y, -v.x * yaw.sin() + v.z * yaw.cos());
    let rx = |a: f32, v: V3| V3::new(v.x, v.y * a.cos() - v.z * a.sin(), v.y * a.sin() + v.z * a.cos());
    let ground = mode == MODE_GROUND;
    let crouch = crouch && ground;
    let hip_h = if crouch { 0.5 } else { 0.95 };
    let lean = if !ground { 0.0 } else if crouch { 0.15 } else if sprint { 0.22 } else { 0.05 };
    let fall = mode == MODE_FALL;
    let r0 = |v: V3| if fall { ry(rx(1.25, v)) } else { ry(v) };
    let p0 = if fall { pos + V3::new(0.0, 0.95, 0.0) + r0(V3::new(0.0, -0.95, 0.0)) } else { pos };
    let origin = p0 + r0(V3::new(0.0, hip_h + 0.02, 0.0));
    let rt = |v: V3| r0(rx(lean, v));
    (origin, [rt(V3::new(1.0, 0.0, 0.0)), rt(V3::new(0.0, 1.0, 0.0)), rt(V3::new(0.0, 0.0, 1.0))])
}

/// Does the ray pass through the drawn shirt, vest or backpack before `maxt`?
fn through_drawn_torso(o: V3, d: V3, frame: (V3, [V3; 3]), maxt: f32) -> bool {
    let (org, ax) = frame;
    [([0.0, 0.3, 0.0], [0.23, 0.28, 0.13]), ([0.0, 0.27, 0.0], [0.25, 0.21, 0.155]), ([0.0, 0.32, -0.2], [0.17, 0.21, 0.07])]
        .iter()
        .any(|(c, h)| ray_obb(o, d, org + ax[0] * c[0] + ax[1] * c[1] + ax[2] * c[2], ax, *h, maxt).is_some())
}

/// Shots at the drawn head that pass no part of the drawn torso are
/// headshots: from below (the fat body capsule used to shadow the chin) and
/// at a free-falling body (whose tilted capsule wrapped the back of the
/// head). Shots through the drawn torso first stay body hits.
#[test]
fn unobstructed_shots_at_the_drawn_head_are_headshots() {
    let mut m = Match::new(42, 2, FLAG_OFFLINE, 3);
    let a = m.add_human(b"A").unwrap() as usize;
    let b = m.add_human(b"B").unwrap() as usize;
    m.step();
    m.players[a].bot = true; // live positions, no rewind
    let t = V3::new(500.0, 200.0, 500.0);
    let mut rng = Rng::new(9);
    for (mode, crouch, name) in [(MODE_GROUND, false, "standing"), (MODE_GROUND, true, "crouched"), (MODE_PARA, false, "parachute"), (MODE_FALL, false, "free fall")] {
        let (mut free, mut free_head, mut blocked, mut blocked_body) = (0, 0, 0, 0);
        for yaw in [0.0f32, 1.3, 2.6, -1.9] {
            for pitch in [-0.5f32, 0.0, 0.6] {
                {
                    let q = &mut m.players[b];
                    q.ms = MoveState { pos: t, mode, on_ground: mode == MODE_GROUND, crouch, ..Default::default() };
                    q.yaw = yaw;
                    q.pitch = pitch;
                    q.draw_speed = 0.0;
                }
                let pose = body_pose(t, yaw, pitch, crouch, false, mode);
                let frame = drawn_torso(t, yaw, crouch, false, mode);
                // shooters all around, level up to 60 degrees below the target
                for k in 0..12 {
                    let az = k as f32 * TAU / 12.0;
                    for elev in [0.0f32, 0.3, 0.6, 1.05] {
                        for _ in 0..6 {
                            let aim = pose.head + V3::new(rng.range(-0.07, 0.07), rng.range(-0.07, 0.07), rng.range(-0.07, 0.07));
                            let back = V3::new(az.sin() * elev.cos(), -elev.sin(), az.cos() * elev.cos());
                            let o = aim + back * 20.0;
                            let d = (aim - o).norm();
                            let r = m.ray_players(a, o, d, 40.0, 0);
                            if through_drawn_torso(o, d, frame, 20.0) {
                                blocked += 1;
                                blocked_body += matches!(r, Some((j, _, false)) if j == b) as u32;
                            } else {
                                free += 1;
                                free_head += matches!(r, Some((j, _, true)) if j == b) as u32;
                            }
                        }
                    }
                }
            }
        }
        let fh = free_head as f32 / free.max(1) as f32;
        let bb = blocked_body as f32 / blocked.max(1) as f32;
        eprintln!("{name}: {free} clear shots at the head, {:.1}% headshots; {blocked} through the torso, {:.1}% body", fh * 100.0, bb * 100.0);
        assert!(fh >= 0.99, "{name}: only {:.1}% of clear shots at the drawn head are headshots", fh * 100.0);
        if blocked >= 20 {
            assert!(bb >= 0.9, "{name}: only {:.1}% of shots through the drawn torso are body hits", bb * 100.0);
        }
    }
}

/// Running into a wall keeps the velocity up while the player does not move.
/// Clients animate from the change of position and draw them standing, so
/// the hit volume must not lean the head forward as if sprinting.
#[test]
fn running_into_a_wall_is_not_judged_as_sprinting() {
    let mut m = Match::new(42, 2, FLAG_OFFLINE, 3);
    let a = m.add_human(b"A").unwrap() as usize;
    let b = m.add_human(b"B").unwrap() as usize;
    m.step();
    // an outer wall with open ground in front of it
    let (pos, yaw) = m
        .w
        .buildings
        .iter()
        .find_map(|bl| {
            let (x, z) = (bl.x0 - 0.6, (bl.z0 + bl.z1) * 0.5);
            let y = m.w.ground_below(x, z, bl.y + 3.0, P_RADIUS);
            let open = m.w.building_at(x - 3.0, z).is_none() && (y - bl.y).abs() < 0.5 && m.w.raycast(V3::new(x - 4.0, y + 1.0, z), V3::new(1.0, 0.0, 0.0), 3.0).is_none();
            open.then_some((V3::new(x - 1.5, y, z), yaw_to(1.0, 0.0)))
        })
        .expect("no wall");
    m.players[b].ms = MoveState { pos, mode: MODE_GROUND, on_ground: true, ..Default::default() };
    let mut seq = m.players[b].last_seq;
    for _ in 0..45 {
        seq += 1;
        m.on_message(b as u8, &cmd_bytes(&[(seq, m.tick.saturating_sub(3) * 256, 0, 127, net_yaw(yaw), 0, B_SPRINT)]));
        m.step();
    }
    let q = &m.players[b];
    assert!(q.ms.vel.len_xz() > SPRINT_DRAW_SPEED, "not pressing into the wall at sprint speed ({:.2} m/s)", q.ms.vel.len_xz());
    assert!(q.draw_speed < 1.0, "drawn speed {:.2} m/s while standing at the wall", q.draw_speed);
    // side shots at the head as drawn (standing, not leaning)
    let (p, qy, qp) = (q.ms.pos, q.yaw, q.pitch);
    let head = body_pose(p, qy, qp, false, false, MODE_GROUND).head;
    m.players[a].bot = true;
    let side = right(qy);
    let mut rng = Rng::new(3);
    let (mut n, mut heads) = (0, 0);
    for _ in 0..200 {
        let aim = head + V3::new(0.0, rng.range(-0.1, 0.1), 0.0) + fwd(qy) * rng.range(-0.1, 0.1);
        let o = aim + side * 10.0;
        n += 1;
        heads += matches!(m.ray_players(a, o, side * -1.0, 30.0, 0), Some((j, _, true)) if j == b) as u32;
    }
    let frac = heads as f32 / n as f32;
    assert!(frac >= 0.97, "only {:.1}% of side shots at the drawn head are headshots", frac * 100.0);
}
