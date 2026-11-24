//! Regression tests for exploits found during the adversarial security review.
//! Each test sets up the exploit scenario and asserts that it no longer works.

use zonefall_core::defs::*;
use zonefall_core::math::*;
use zonefall_core::net::*;
use zonefall_core::player::*;
use zonefall_core::sim::*;
use zonefall_core::world::*;

fn cmd_bytes(cmds: &[(u32, u32, i8, i8, u16, i16, u16)]) -> Vec<u8> {
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

fn two_humans(seed: u32) -> (Match, u8, u8) {
    let mut m = Match::new(seed, 2, FLAG_OFFLINE, 99);
    let a = m.add_human(b"A").unwrap();
    let b = m.add_human(b"B").unwrap();
    m.step();
    assert_eq!(m.phase, PH_PLAY);
    (m, a, b)
}

/// A full-height wall on the -z side of a building with walkable ground outside.
fn find_wall(w: &World) -> Option<(usize, usize)> {
    for (bi, b) in w.buildings.iter().enumerate() {
        for (k, bx) in w.boxes.iter().enumerate() {
            if bx.kind != BK_WALL {
                continue;
            }
            if (bx.min.z - b.z0).abs() > 1e-3 || (bx.max.z - (b.z0 + 0.3)).abs() > 1e-3 {
                continue;
            }
            if (bx.min.y - b.y).abs() > 1e-3 || bx.max.y - bx.min.y < 3.0 || bx.max.x - bx.min.x < 3.0 {
                continue;
            }
            let x = (bx.min.x + bx.max.x) * 0.5;
            let g = w.height(x, b.z0 - 0.4);
            if g < 1.0 || b.y - g > 0.9 {
                continue;
            }
            return Some((bi, k));
        }
    }
    None
}

fn place(m: &mut Match, pid: u8, pos: V3) {
    m.players[pid as usize].ms = MoveState { pos, mode: MODE_GROUND, on_ground: true, ..Default::default() };
}

/// Enqueue one command and simulate one tick.
fn run_cmd(m: &mut Match, pid: u8, seq: u32, yaw: u16, pitch: i16, buttons: u16) {
    let msg = cmd_bytes(&[(seq, m.tick * 256, 0, 0, yaw, pitch, buttons)]);
    m.on_message(pid, &msg);
    m.step();
}

const SEEDS: [u32; 7] = [3, 11, 42, 58, 77, 123, 999];

#[test]
fn no_wallbang_when_hugging_a_wall() {
    let mut walled_cases = 0;
    for seed in SEEDS {
        let (m0, _, _) = two_humans(seed);
        let Some((bi, k)) = find_wall(&m0.w) else { continue };
        drop(m0);
        for deg in [5.0f32, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0] {
            let (mut m, a, b) = two_humans(seed);
            let bl = m.w.buildings[bi];
            let bx = m.w.boxes[k];
            let x = (bx.min.x + bx.max.x) * 0.5;
            let z = bl.z0 - P_RADIUS - 0.002;
            let gy = m.w.ground_below(x, z, bl.y, P_RADIUS);
            place(&mut m, a, V3::new(x, gy, z));
            // right shoulder into the wall, aiming at a point 3 m inside the building
            let yaw = PI * 0.5 - deg.to_radians();
            let yq = q_yaw(yaw);
            let yawd = dq_yaw(yq);
            let apos = m.players[a as usize].ms.pos;
            let mut pitch = 0.0f32;
            let mut bpos = V3::ZERO;
            for _ in 0..4 {
                let (o, d) = aim_ray(apos, yawd, dq_pitch(q_pitch(pitch)), false, false, false);
                let tz = (bl.z0 + 3.3 - o.z) / d.z;
                let p = o + d * tz;
                bpos = V3::new(p.x, bl.y, p.z);
                pitch = ((bl.y + 1.0 - o.y) / tz).atan();
            }
            if !bl.contains(bpos.x, bpos.z) {
                continue;
            }
            place(&mut m, b, bpos);
            {
                let p = &mut m.players[a as usize];
                p.slots[0] = (W_DMR, 10);
                p.cur = 0;
                p.switch_t = 0.0;
                p.fire_cd = 0.0;
                p.yaw = yawd;
            }
            m.step();
            let eye = m.players[a as usize].ms.pos + V3::new(0.0, eye_h(false), 0.0);
            let chest = m.players[b as usize].ms.pos + V3::new(0.0, 1.0, 0.0);
            if !m.w.blocked(eye, chest) {
                continue;
            }
            walled_cases += 1;
            run_cmd(&mut m, a, 1, yq, q_pitch(pitch), B_FIRE);
            let hp = m.players[b as usize].health;
            assert_eq!(hp, 100.0, "seed {seed} angle {deg}: target behind a wall took damage");
        }
    }
    assert!(walled_cases >= 5, "too few walled scenarios exercised ({walled_cases})");
}

#[test]
fn grenades_cannot_spawn_inside_walls() {
    let mut checked = 0;
    for seed in SEEDS {
        let (m0, _, _) = two_humans(seed);
        let Some((bi, _)) = find_wall(&m0.w) else { continue };
        let bl = m0.w.buildings[bi];
        drop(m0);
        let (mut m, a, b) = two_humans(seed);
        let bx = find_wall(&m.w).unwrap().1;
        let wall = m.w.boxes[bx];
        let x = (wall.min.x + wall.max.x) * 0.5;
        let z = bl.z0 - P_RADIUS - 0.002;
        let gy = m.w.ground_below(x, z, bl.y, P_RADIUS);
        place(&mut m, a, V3::new(x, gy, z));
        place(&mut m, b, V3::new(10.0, 50.0, 10.0));
        m.players[a as usize].nades = 2;
        m.step();
        run_cmd(&mut m, a, 1, q_yaw(0.0), q_pitch(-0.25), B_NADE);
        assert!(!m.nades.is_empty(), "grenade was not thrown");
        let mut last = m.nades[0].pos;
        for _ in 0..200 {
            if m.nades.is_empty() {
                break;
            }
            last = m.nades[0].pos;
            assert!(!bl.contains(last.x, last.z) || last.y > bl.y + WALL_H, "seed {seed}: grenade got inside the building through the wall at {last:?}");
            m.step();
        }
        let _ = last;
        checked += 1;
    }
    assert!(checked >= 3);
}

#[test]
fn no_pickup_through_walls() {
    let mut checked = 0;
    for seed in SEEDS {
        let (m0, _, _) = two_humans(seed);
        let Some((bi, k)) = find_wall(&m0.w) else { continue };
        drop(m0);
        let (mut m, a, _b) = two_humans(seed);
        let bl = m.w.buildings[bi];
        let bx = m.w.boxes[k];
        let x = (bx.min.x + bx.max.x) * 0.5;
        let z = bl.z0 - P_RADIUS - 0.002;
        let gy = m.w.ground_below(x, z, bl.y, P_RADIUS);
        place(&mut m, a, V3::new(x, gy, z));
        let id = m.items.iter().position(|it| it.alive && is_weapon(it.kind)).unwrap();
        m.items[id].pos = V3::new(x, bl.y + 0.05, bl.z0 + 0.3 + 0.9);
        m.step();
        run_cmd(&mut m, a, 1, q_yaw(0.0), 0, B_PICKUP);
        assert!(m.items[id].alive, "seed {seed}: picked up an item through a wall");
        checked += 1;
    }
    assert!(checked >= 3);
}

#[test]
fn no_melee_through_walls() {
    let mut checked = 0;
    for seed in SEEDS {
        let (m0, _, _) = two_humans(seed);
        let Some((bi, k)) = find_wall(&m0.w) else { continue };
        drop(m0);
        let (mut m, a, b) = two_humans(seed);
        let bl = m.w.buildings[bi];
        let bx = m.w.boxes[k];
        let x = (bx.min.x + bx.max.x) * 0.5;
        let za = bl.z0 - P_RADIUS - 0.002;
        let ga = m.w.ground_below(x, za, bl.y, P_RADIUS);
        place(&mut m, a, V3::new(x, ga, za));
        place(&mut m, b, V3::new(x, bl.y, bl.z0 + 0.3 + P_RADIUS + 0.002));
        m.players[a as usize].cur = 2;
        m.step();
        let mut seq = 0;
        for _ in 0..6 {
            seq += 1;
            run_cmd(&mut m, a, seq, q_yaw(0.0), q_pitch(-0.3), B_FIRE);
            for _ in 0..16 {
                m.step();
            }
        }
        assert_eq!(m.players[b as usize].health, 100.0, "seed {seed}: punched through a wall");
        checked += 1;
    }
    assert!(checked >= 3);
}

#[test]
fn command_bursts_cannot_teleport_or_speed_up() {
    let (mut m, a, _b) = two_humans(42);
    let t = m.w.towns[0];
    let gy = m.w.ground_below(t.x + 3.0, t.z, 200.0, P_RADIUS);
    place(&mut m, a, V3::new(t.x + 3.0, gy, t.z));
    for _ in 0..12 {
        m.step();
    }
    let p0 = m.players[a as usize].ms.pos;
    let mut seq = 0u32;
    let mut burst = Vec::new();
    for _ in 0..16 {
        seq += 1;
        burst.push((seq, m.tick * 256, 0i8, 127i8, q_yaw(0.7), 0i16, B_SPRINT));
    }
    m.on_message(a, &cmd_bytes(&burst[..8]));
    m.on_message(a, &cmd_bytes(&burst[8..]));
    let p1 = m.players[a as usize].ms.pos;
    assert_eq!(p0, p1, "commands must not be simulated on arrival");
    let before = m.players[a as usize].cmds_applied;
    m.step();
    assert!(m.players[a as usize].cmds_applied - before <= 2, "more than two commands simulated in one tick");

    // flood: 3 commands every tick for 30 s
    let start = m.players[a as usize].cmds_applied;
    for _ in 0..900 {
        let mut v = Vec::new();
        for _ in 0..3 {
            seq += 1;
            v.push((seq, m.tick * 256, 0i8, 127i8, q_yaw(0.7), 0i16, 0u16));
        }
        m.on_message(a, &cmd_bytes(&v));
        m.step();
    }
    let applied = m.players[a as usize].cmds_applied - start;
    assert!(applied <= 903, "simulated {applied} commands in 900 ticks (speed hack)");
}

#[test]
fn starving_the_server_does_not_hover() {
    let (mut m, a, _b) = two_humans(42);
    let t = m.w.towns[0];
    let gy = m.w.ground_below(t.x + 3.0, t.z, 200.0, P_RADIUS);
    place(&mut m, a, V3::new(t.x + 3.0, gy, t.z));
    m.step();
    let mut seq = 1;
    run_cmd(&mut m, a, seq, 0, 0, B_JUMP);
    let mut airborne = 0;
    for k in 0..300 {
        if k % 8 == 7 {
            seq += 1;
            m.on_message(a, &cmd_bytes(&[(seq, m.tick * 256, 0, 0, 0, 0, 0)]));
        }
        m.step();
        let p = &m.players[a as usize];
        let h = p.ms.pos.y - m.w.ground_below(p.ms.pos.x, p.ms.pos.z, p.ms.pos.y, P_RADIUS);
        if h > 0.3 {
            airborne += 1;
        }
    }
    assert!(airborne <= 30, "airborne for {airborne} ticks after a single jump");
}

fn max_marks_per_base(name: &str) -> usize {
    let (mut run, mut worst) = (0, 0);
    for c in name.chars() {
        let c = c as u32;
        let mark = matches!(c, 0x0300..=0x036F | 0x0483..=0x0489 | 0x0591..=0x05C7 | 0x064B..=0x065F | 0x0E31..=0x0E4E);
        if mark {
            run += 1;
            worst = worst.max(run);
        } else {
            run = 0;
        }
    }
    worst
}

#[test]
fn names_are_sanitized_and_unique() {
    let mut m = Match::new(1, 16, FLAG_OFFLINE, 7);
    let p1 = m.add_human("\u{202E}gnaK\u{200B} \u{2066}x\u{0301}\u{0301}\u{0301}\u{0301}".as_bytes()).unwrap();
    let n1 = m.players[p1 as usize].name.clone();
    assert!(n1.chars().all(|c| !matches!(c as u32, 0x200B..=0x200F | 0x202A..=0x202E | 0x2066..=0x2069)), "{n1:?}");
    assert!(max_marks_per_base(&n1) <= 2, "stacked marks kept: {n1:?}");
    let p2 = m.add_human(n1.as_bytes()).unwrap();
    assert_ne!(m.players[p2 as usize].name, n1, "duplicate names must be disambiguated");
    let p3 = m.add_human(b"   ").unwrap();
    assert_eq!(m.players[p3 as usize].name, "Player");
    // invisible format characters are dropped, so look-alike names collide
    let a = m.add_human(b"ab").unwrap();
    let b = m.add_human("a\u{180B}\u{180C}\u{180D}\u{180F}b\u{1D173}\u{1BCA0}\u{110BD}\u{13430}".as_bytes()).unwrap();
    assert_eq!(m.players[a as usize].name, "ab");
    assert_eq!(m.players[b as usize].name, "ab#2");
    // zalgo from blocks outside the basic combining range
    let z = m.add_human("a\u{0489}\u{0489}\u{0489}\u{0489}\u{0489}\u{0489}\u{0489}\u{0489}\u{0489}\u{0489}\u{0489}\u{0489}\u{0489}".as_bytes()).unwrap();
    assert!(max_marks_per_base(&m.players[z as usize].name) <= 2);
    // homoglyph and fullwidth duplicates are disambiguated
    let k1 = m.add_human(b"Kaze").unwrap();
    let k2 = m.add_human("K\u{0430}ze".as_bytes()).unwrap();
    let k3 = m.add_human("\u{FF2B}\u{FF41}\u{FF5A}\u{FF45}".as_bytes()).unwrap();
    assert_eq!(m.players[k1 as usize].name, "Kaze");
    assert!(m.players[k2 as usize].name.ends_with("#2"), "{:?}", m.players[k2 as usize].name);
    assert!(m.players[k3 as usize].name.ends_with("#3"), "{:?}", m.players[k3 as usize].name);
    // ordinary Japanese names survive untouched
    let j = m.add_human("ゆうき＿Ｖ３".as_bytes()).unwrap();
    assert_eq!(m.players[j as usize].name, "ゆうき＿Ｖ３");
}

/// Offsets of the players block in a snapshot (mirrors `Match::snapshot`).
fn snapshot_player_ids(buf: &[u8]) -> Vec<u8> {
    let header = 1 + 4 + 4 + 1 + 4 + 1 + 1 + 1;
    let selfb = 1 + 1 + 12 + 12 + 4 + 4 + 4 + 1 + 6 + 8 + 3 + 1 + 4 + 4 + 1 + 4 + 4 + 2 + 2 + 3;
    let zone = 2 + 4 * 8;
    let plane = 1 + 4 * 7;
    let off = header + selfb + zone + plane;
    let n = buf[off] as usize;
    (0..n).map(|k| buf[off + 1 + k * 22]).collect()
}

#[test]
fn hidden_players_are_not_sent() {
    // Place B at many spots ~120 m from A; B must appear in A's snapshot
    // exactly when A has line of sight to B's head or chest.
    let (mut m, a, b) = two_humans(42);
    let t = m.w.towns[1];
    let ga = m.w.ground_below(t.x, t.z, 200.0, P_RADIUS);
    place(&mut m, a, V3::new(t.x, ga, t.z));
    let (mut seen, mut hidden) = (0, 0);
    for k in 0..48 {
        let ang = k as f32 * TAU / 48.0;
        let x = clampf(t.x + ang.cos() * 120.0, 20.0, MAP - 20.0);
        let z = clampf(t.z + ang.sin() * 120.0, 20.0, MAP - 20.0);
        let gb = m.w.ground_below(x, z, 200.0, P_RADIUS);
        place(&mut m, b, V3::new(x, gb, z));
        // fresh visibility state
        m.players[a as usize].seen_tick = [0; MAX_PLAYERS];
        m.players[a as usize].vis_check = [0; MAX_PLAYERS];
        m.step();
        let eye = m.players[a as usize].ms.pos + V3::new(0.0, eye_h(false), 0.0);
        let bp = m.players[b as usize].ms.pos;
        let visible = !m.w.blocked(eye, bp + V3::new(0.0, eye_h(false) + 0.1, 0.0)) || !m.w.blocked(eye, bp + V3::new(0.0, 0.9, 0.0));
        let snap = m.snapshot(a).to_vec();
        m.flush();
        let sent = snapshot_player_ids(&snap).contains(&b);
        assert_eq!(sent, visible, "angle {k}: sent={sent} visible={visible}");
        if visible {
            seen += 1;
        } else {
            hidden += 1;
        }
    }
    assert!(seen > 0 && hidden > 0, "need both visible ({seen}) and hidden ({hidden}) cases");
}
