//! Multi-storey houses and apartment blocks: they exist, the stairs can be
//! climbed all the way up, and the upper floors hold up.

use zonefall_core::defs::*;
use zonefall_core::math::*;
use zonefall_core::player::*;
use zonefall_core::world::*;

/// Walk from `from` holding forward along `yaw` for `ticks`.
fn walk(w: &World, from: V3, yaw: f32, ticks: usize) -> MoveState {
    let mut s = MoveState { pos: from, mode: MODE_GROUND, on_ground: true, ..Default::default() };
    let c = Cmd { mz: 1.0, yaw, ..Default::default() };
    for _ in 0..ticks {
        step_move(w, &mut s, &c, 1.0, DT);
    }
    s
}

#[test]
fn towns_have_two_storey_houses_and_apartment_blocks() {
    let mut houses = 0;
    let mut blocks = 0;
    for seed in [7u32, 42, 2024, 99] {
        let w = World::generate(seed);
        houses += w.buildings.iter().filter(|b| b.floors == 2).count();
        blocks += w.buildings.iter().filter(|b| b.floors >= 3).count();
        assert!(!w.deco.is_empty());
    }
    assert!(houses >= 12, "only {houses} two-storey houses in four worlds");
    assert!(blocks >= 3, "only {blocks} apartment blocks in four worlds");
}

#[test]
fn every_staircase_leads_to_every_floor() {
    let mut climbed = 0;
    for seed in [7u32, 42, 2024] {
        let w = World::generate(seed);
        for (bi, b) in w.buildings.iter().enumerate() {
            if b.floors < 2 {
                continue;
            }
            let along = b.x1 - b.x0 >= b.z1 - b.z0;
            let (a_lo, p_lo, p_hi) = if along { (b.x0 + 0.3, b.z0 + 0.3, b.z1 - 0.3) } else { (b.z0 + 0.3, b.x0 + 0.3, b.x1 - 0.3) };
            let at = |a: f32, p: f32, y: f32| if along { V3::new(a, y, p) } else { V3::new(p, y, a) };
            let (yaw_up, yaw_down) = if along { (yaw_to(1.0, 0.0), yaw_to(-1.0, 0.0)) } else { (yaw_to(0.0, 1.0), yaw_to(0.0, -1.0)) };
            // the stairs hug one of the two long walls: find the lane that climbs
            let mut ok = false;
            for wall in 0..2 {
                let lane = |k: usize| {
                    let off = 0.575 + 1.15 * k as f32;
                    if wall == 0 { p_lo + off } else { p_hi - off }
                };
                let mut y = b.y;
                let mut reached = 1;
                for f in 1..b.floors as usize {
                    // flights alternate lane and direction; start a step before the first tread
                    let up = (f - 1) % 2 == 0;
                    let start = if up { a_lo + 0.9 } else { a_lo + 1.35 + 4.2 + 0.45 };
                    let s = walk(&w, at(start, lane((f - 1) % 2), y), if up { yaw_up } else { yaw_down }, 36);
                    let want = b.y + STOREY * f as f32;
                    if (s.pos.y - want).abs() > 0.02 || !s.on_ground {
                        break;
                    }
                    y = want;
                    reached += 1;
                }
                if reached == b.floors as usize {
                    ok = true;
                    // and the top floor is a floor: stand in the middle of the building
                    let (mx, mz) = ((b.x0 + b.x1) * 0.5, (b.z0 + b.z1) * 0.5);
                    let g = w.ground_below(mx, mz, y + 1.0, P_RADIUS);
                    let far = if along { (b.x1 - 1.2, mz) } else { (mx, b.z1 - 1.2) };
                    let g2 = w.ground_below(far.0, far.1, y + 1.0, P_RADIUS);
                    assert!((g - y).abs() < 0.02 || (g2 - y).abs() < 0.02, "seed {seed} building {bi}: no floor upstairs ({g} / {g2}, want {y})");
                    break;
                }
            }
            assert!(ok, "seed {seed} building {bi} ({} floors): the stairs do not reach the top", b.floors);
            climbed += 1;
        }
    }
    assert!(climbed >= 10, "only {climbed} staircases tested");
}
