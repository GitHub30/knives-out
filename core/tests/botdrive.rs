//! The few bots that drive: who they are, that they get somewhere, and
//! that they leave a car that burns or will not move.

use zonefall_core::bot::*;
use zonefall_core::defs::*;
use zonefall_core::math::*;
use zonefall_core::player::*;
use zonefall_core::sim::*;
use zonefall_core::world::*;

fn full_match(seed: u32) -> Match {
    let mut m = Match::new(seed, 30, FLAG_OFFLINE, 7);
    m.add_human(b"A").unwrap();
    m.step();
    assert_eq!(m.phase, PH_PLAY);
    m
}

#[derive(Default, Clone, Copy)]
struct Trip {
    ticks: u32,
    dist: f32,
    trips: u32,
    /// Ticks it was alive at all.
    alive: u32,
    /// Longest time at the wheel without leaving a 6 m circle.
    idle: u32,
    anchor: (f32, f32),
    since: u32,
}

/// Run `ticks` of a full match; per player what it did at the wheel.
fn watch(m: &mut Match, ticks: u32) -> Vec<Trip> {
    let mut out = vec![Trip::default(); MAX_PLAYERS];
    let mut last: Vec<(u8, V3)> = m.players.iter().map(|p| (p.ms.mode, p.ms.pos)).collect();
    for _ in 0..ticks {
        if m.phase != PH_PLAY {
            break;
        }
        m.step();
        for i in 0..MAX_PLAYERS {
            let p = &m.players[i];
            out[i].alive += (p.used && p.alive) as u32;
            if p.used && p.alive && p.ms.mode == MODE_DRIVE {
                out[i].ticks += 1;
                if last[i].0 == MODE_DRIVE {
                    out[i].dist += dist_xz(p.ms.pos.x, p.ms.pos.z, last[i].1.x, last[i].1.z);
                } else {
                    out[i].trips += 1;
                    out[i].anchor = (p.ms.pos.x, p.ms.pos.z);
                    out[i].since = 0;
                }
                if dist_xz(p.ms.pos.x, p.ms.pos.z, out[i].anchor.0, out[i].anchor.1) > 6.0 {
                    out[i].anchor = (p.ms.pos.x, p.ms.pos.z);
                    out[i].since = 0;
                }
                out[i].since += 1;
                out[i].idle = out[i].idle.max(out[i].since);
            }
            if last[i].0 == MODE_DRIVE && p.ms.mode != MODE_DRIVE && std::env::var("DRV_DBG").is_ok() {
                // (set DRV_DBG to see how each trip ended)
                let b = &p.brain;
                let k = b.drv_car as usize;
                eprintln!(
                    "  t{} bot {i} out: alive {} trip {:.1}s stuck {} wet {:.1} noprog {:.1} dest {:.0} m carhp {:.0}/{:.0} fire {} flee {:.1} cd {:.1} pos {:.0},{:.0}",
                    m.tick, p.alive, b.drv_t, b.drv_stuck, b.drv_wet_t, b.drv_best_t,
                    dist_xz(p.ms.pos.x, p.ms.pos.z, b.drv_dest.0, b.drv_dest.1), m.car_hp[k], m.car_max_hp(k), m.car_fire[k], b.drv_flee_t, b.drv_cd, p.ms.pos.x, p.ms.pos.z
                );
            }
            last[i] = (p.ms.mode, p.ms.pos);
        }
    }
    out
}


#[test]
fn about_a_tenth_of_the_bots_drive_and_only_the_drivers_do() {
    assert_eq!(driver_count(29), 4);
    assert_eq!(driver_count(23), 3);
    assert_eq!(driver_count(3), 0);
    // Over twenty matches of four minutes: every match has its four
    // drivers, nobody else ever sits in a car, and the share of bots that
    // have driven is 10 % give or take (band 8 % .. 13 %: at most the 4 of
    // 29 that are drivers, less those killed before they reached a car).
    let (mut bots, mut flagged, mut drove, mut far) = (0u32, 0u32, 0u32, 0u32);
    let (mut alive_t, mut wheel_t) = (0u64, 0u64);
    let mut idle = 0u32;
    let mut none = 0u32;
    for seed in 1..=20u32 {
        let mut m = full_match(seed);
        let tr = watch(&mut m, 30 * 240);
        let (mut here, mut drove_here) = (0, 0);
        for i in 0..MAX_PLAYERS {
            let p = &m.players[i];
            if !p.used || !p.bot {
                assert!(!p.brain.driver);
                continue;
            }
            bots += 1;
            alive_t += tr[i].alive as u64;
            wheel_t += tr[i].ticks as u64;
            if p.brain.driver {
                here += 1;
                eprintln!("seed {seed} bot {i}: {:.1} s at the wheel, {:.0} m in {} trips, longest wait {:.1} s", tr[i].ticks as f32 / 30.0, tr[i].dist, tr[i].trips, tr[i].idle as f32 / 30.0);
            } else {
                assert_eq!(tr[i].ticks, 0, "seed {seed}: bot {i} is no driver but drove");
            }
            if tr[i].ticks > 0 {
                drove += 1;
                drove_here += 1;
                // (the wheel time the bot counts itself agrees)
                assert!((p.brain.drove_t - tr[i].ticks as f32 / 30.0).abs() < 1.0);
            }
            if tr[i].dist > 100.0 {
                far += 1;
            }
            idle = idle.max(tr[i].idle);
        }
        assert_eq!(here, 4, "seed {seed}: {here} drivers");
        flagged += here;
        none += (drove_here == 0) as u32;
    }
    let share = drove as f32 / bots as f32;
    eprintln!(
        "{bots} bots, {flagged} drivers, {drove} drove ({:.1} %), {far} of them over 100 m; {:.1} % of all the time bots were alive was spent at a wheel; {none} matches without a bot at a wheel; longest wait in a car {:.1} s",
        share * 100.0, wheel_t as f32 / alive_t as f32 * 100.0, idle as f32 / 30.0
    );
    assert_eq!(bots, 580);
    assert!((0.08..=0.13).contains(&share), "share of bots that drove: {share}");
    assert!(far * 2 >= drove, "only {far} of {drove} drivers got anywhere");
    assert!(none <= 1, "{none} of 20 matches had no bot at a wheel");
    // nobody sits in a car that goes nowhere
    assert!(idle < 30 * 20, "a bot sat {} s in a car without moving 6 m", idle as f32 / 30.0);
}

#[test]
fn a_match_with_drivers_repeats() {
    let run = || {
        let mut m = full_match(21);
        for _ in 0..30 * 150 {
            m.step();
        }
        (m.w.cars.clone(), m.players.iter().map(|p| (p.ms.pos, p.health, p.ms.mode)).collect::<Vec<_>>())
    };
    let (a, b) = (run(), run());
    assert!(a.0 == b.0 && a.1 == b.1);
}

/// One human (parked far away) and one bot, which is made a driver and
/// stood `off` beside car `k`.
fn lone_driver(seed: u32, k: usize, off: f32) -> (Match, usize) {
    let mut m = Match::new(seed, 2, FLAG_OFFLINE, 7);
    let h = m.add_human(b"A").unwrap() as usize;
    m.step();
    assert_eq!(m.phase, PH_PLAY);
    let bot = (0..MAX_PLAYERS).find(|&i| m.players[i].used && m.players[i].bot).unwrap();
    assert!(!m.players[bot].brain.driver);
    let c = m.w.cars[k];
    let r = right(c[3]);
    let (x, z) = (c[0] + r.x * off, c[2] + r.z * off);
    let y = m.w.ground_below(x, z, c[1] + 3.0, P_RADIUS);
    m.players[bot].ms = MoveState { pos: V3::new(x, y, z), mode: MODE_GROUND, on_ground: true, ..Default::default() };
    m.players[bot].brain.driver = true;
    m.players[bot].brain.ground_t = 60.0;
    // the human: on the ground a long way off, inside the zone
    let mut far = (0.0f32, V3::ZERO);
    for gx in (200..840).step_by(40) {
        for gz in (200..840).step_by(40) {
            let (hx, hz) = (gx as f32, gz as f32);
            let d = dist_xz(hx, hz, c[0], c[2]);
            if m.w.height(hx, hz) > 2.0 && m.w.building_at(hx, hz).is_none() && d > far.0 {
                far = (d, V3::new(hx, m.w.height(hx, hz), hz));
            }
        }
    }
    assert!(far.0 > 300.0);
    m.players[h].ms = MoveState { pos: far.1, mode: MODE_GROUND, on_ground: true, ..Default::default() };
    (m, bot)
}

/// Step until the bot is at the wheel (it walks over and gets in by itself).
fn until_driving(m: &mut Match, bot: usize, max: u32) -> u32 {
    for t in 0..max {
        m.step();
        if m.players[bot].ms.mode == MODE_DRIVE {
            return t;
        }
        assert_eq!(m.players[bot].ms.mode, MODE_GROUND);
    }
    panic!("the bot never got into a car, it stands at {:?}", m.players[bot].ms.pos);
}

#[test]
fn a_driver_walks_to_a_car_gets_in_and_covers_ground() {
    let mut went = 0;
    let mut total = 0;
    for seed in [1u32, 2, 3, 4, 5, 6] {
        for k in [0usize, 3] {
            let (mut m, bot) = lone_driver(seed, k, 9.0);
            let c0 = m.w.cars[k];
            let t = until_driving(&mut m, bot, 30 * 20);
            let car = m.players[bot].ms.car as usize;
            assert_eq!(m.car_driver[car], bot as u8);
            // it got in where a player could: from beside the car, having walked there
            assert!(car != k || t > 15, "seed {seed} car {k}: in the car after {t} ticks from 9 m away");
            let mut dist = 0.0;
            let mut vmax = 0.0f32;
            let mut last = m.players[bot].ms.pos;
            for _ in 0..30 * 45 {
                m.step();
                let p = &m.players[bot];
                if p.ms.mode != MODE_DRIVE {
                    break;
                }
                dist += dist_xz(p.ms.pos.x, p.ms.pos.z, last.x, last.z);
                vmax = vmax.max(car_speed(&p.ms).abs());
                last = p.ms.pos;
            }
            let p = &m.players[bot];
            let gone = dist_xz(m.w.cars[car][0], m.w.cars[car][2], c0[0], c0[2]);
            eprintln!("seed {seed} car {k}: in after {:.1} s, drove {dist:.0} m (now {gone:.0} m away), top {vmax:.1} m/s, car {:.0}/{:.0}", t as f32 / 30.0, m.car_hp[car], m.car_max_hp(car));
            assert!(p.alive);
            // bots keep to their cruising speed (some slack for downhill)
            assert!(vmax < DRIVER_CRUISE + 2.0, "seed {seed} car {k}: {vmax} m/s");
            assert!(m.car_hp[car] > m.car_max_hp(car) * 0.5, "seed {seed} car {k}: the car is wrecked");
            total += 1;
            if dist > 80.0 && gone > 50.0 {
                went += 1;
            }
        }
    }
    // (a trip can end early in a jam; most must not)
    assert!(went * 4 >= total * 3, "{went} of {total} trips got anywhere");
}

#[test]
fn a_driver_leaves_a_burning_car_and_gets_clear() {
    for seed in [1u32, 2, 3, 4] {
        let (mut m, bot) = lone_driver(seed, 0, 5.0);
        until_driving(&mut m, bot, 30 * 20);
        let car = m.players[bot].ms.car as usize;
        // let it get going
        for _ in 0..75 {
            m.step();
        }
        assert_eq!(m.players[bot].ms.mode, MODE_DRIVE, "seed {seed}");
        // the engine catches fire
        let hp = m.car_hp[car];
        m.hurt_car(car, hp - m.car_max_hp(car) * (CAR_FIRE_AT - 0.02), 255);
        assert!(m.car_fire[car] != 0 && m.car_hp[car] > 0.0);
        let mut out = None;
        for t in 0..CAR_FUSE_TICKS + 30 {
            m.step();
            if out.is_none() && m.players[bot].ms.mode == MODE_GROUND {
                out = Some(t);
            }
            if m.car_hp[car] <= 0.0 {
                break;
            }
        }
        let p = &m.players[bot];
        let c = m.w.cars[car];
        let d = dist_xz(p.ms.pos.x, p.ms.pos.z, c[0], c[2]);
        eprintln!("seed {seed}: out after {:?} ticks, {d:.1} m away at the blast, health {}", out, p.health);
        assert!(m.car_hp[car] <= 0.0, "seed {seed}: the car never blew up");
        assert!(out.map_or(false, |t| t < 45), "seed {seed}: still in the burning car ({out:?})");
        assert!(p.alive && p.health > 60.0 && d > 8.0, "seed {seed}: caught by the blast at {d} m, health {}", p.health);
        // and it does not go back to the wreck
        for _ in 0..30 * 20 {
            m.step();
            let p = &m.players[bot];
            assert!(p.ms.mode == MODE_GROUND || p.ms.car as usize != car);
        }
    }
}

#[test]
fn a_driver_also_leaves_a_badly_damaged_car() {
    let (mut m, bot) = lone_driver(5, 1, 5.0);
    until_driving(&mut m, bot, 30 * 20);
    let car = m.players[bot].ms.car as usize;
    for _ in 0..60 {
        m.step();
    }
    let hp = m.car_hp[car];
    m.hurt_car(car, hp - m.car_max_hp(car) * (DRIVER_BAIL_HP - 0.05), 255);
    assert!(m.car_fire[car] == 0);
    for _ in 0..45 {
        m.step();
    }
    assert_eq!(m.players[bot].ms.mode, MODE_GROUND);
    assert!(m.players[bot].alive && m.car_driver[car] == 255);
}

#[test]
fn a_driver_facing_a_wall_backs_out_or_gets_out() {
    // The car is put nose-on to a building with the destination straight
    // through it. Within 40 s the bot must have driven clear of the spot
    // (25 m) or left the car: never sit there pushing.
    let (mut cases, mut cleared, mut walked) = (0, 0, 0);
    let mut worst = 0u32;
    for seed in [1u32, 2, 3] {
        let w = World::generate(seed);
        let mut done = 0;
        for bi in 0..w.buildings.len() {
            if done >= 5 {
                break;
            }
            let bl = w.buildings[bi];
            let (cx, cz) = ((bl.x0 + bl.x1) * 0.5, (bl.z0 + bl.z1) * 0.5);
            // from the -x side, 5 m off the wall, on level free ground
            let (sx, sz) = (bl.x0 - 5.0 - 2.5, cz);
            let h = w.height(sx, sz);
            if h < 1.5 || (w.height(sx - 3.0, sz) - h).abs() > 0.4 || (w.height(sx + 3.0, sz) - h).abs() > 0.4 {
                continue;
            }
            let mut free = true;
            for a in 0..8 {
                let d = fwd(a as f32 * 0.785);
                if w.raycast_static(V3::new(sx, h + 0.7, sz), d, 4.0).map_or(false, |x| x.2 != HIT_TERRAIN) {
                    free = false;
                }
            }
            if !free || w.cars.iter().skip(1).any(|c| dist_xz(c[0], c[2], sx, sz) < 9.0) {
                continue;
            }
            done += 1;
            cases += 1;
            let (mut m, bot) = lone_driver(seed, 0, 5.0);
            m.w.cars[0] = [sx, h, sz, std::f32::consts::FRAC_PI_2];
            let r = right(m.w.cars[0][3]);
            let (px, pz) = (sx + r.x * 2.5, sz + r.z * 2.5);
            m.players[bot].ms.pos = V3::new(px, m.w.height(px, pz), pz);
            until_driving(&mut m, bot, 30 * 10);
            assert_eq!(m.players[bot].ms.car, 0);
            m.players[bot].brain.drv_dest = (cx + (bl.x1 - bl.x0) * 0.5 + 60.0, cz);
            m.players[bot].brain.drv_best = f32::MAX;
            let mut t = 0;
            let mut how = "";
            while t < 30 * 40 {
                m.step();
                t += 1;
                let c = m.w.cars[0];
                if m.players[bot].ms.mode != MODE_DRIVE {
                    how = "walked";
                    walked += 1;
                    break;
                }
                if dist_xz(c[0], c[2], sx, sz) > 25.0 {
                    how = "cleared";
                    cleared += 1;
                    break;
                }
            }
            worst = worst.max(t);
            eprintln!("seed {seed} building {bi}: {how} after {:.1} s, car {:.0}/{:.0}", t as f32 / 30.0, m.car_hp[0], m.car_max_hp(0));
            assert!(!how.is_empty(), "seed {seed} building {bi}: still pushing at the wall after 40 s");
            assert!(m.players[bot].alive && m.car_hp[0] > m.car_max_hp(0) * 0.6);
        }
    }
    eprintln!("{cases} walls: {cleared} driven round, {walked} left on foot, slowest {:.1} s", worst as f32 / 30.0);
    assert!(cases >= 10);
    // steering round is the rule, giving up the exception
    assert!(cleared * 3 >= cases * 2, "{cleared} of {cases}");
}

#[test]
fn bots_do_not_fly() {
    // put into a helicopter by force, a bot just gets out again
    let (mut m, bot) = lone_driver(2, 0, 5.0);
    if m.w.helis.is_empty() {
        return;
    }
    let hpos = m.w.helis[0];
    m.players[bot].brain.driver = false;
    m.players[bot].ms = MoveState { pos: V3::new(hpos[0], hpos[1], hpos[2]), mode: MODE_FLY, on_ground: true, airtime: hpos[3], car: 0, ..Default::default() };
    for _ in 0..10 {
        m.step();
    }
    assert_ne!(m.players[bot].ms.mode, MODE_FLY);
}
