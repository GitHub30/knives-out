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
