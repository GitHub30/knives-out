//! Regression tests for information leaks found in the second adversarial
//! review: item changes revealing hidden looters, hidden shooters' ids and
//! exact positions, grenades / throws / landings of hidden players, and
//! spectator cycling.

use zonefall_core::defs::*;
use zonefall_core::math::*;
use zonefall_core::net::*;
use zonefall_core::player::*;
use zonefall_core::sim::*;
use zonefall_core::world::*;

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

#[allow(dead_code)]
#[derive(Debug, Clone)]
enum Ev {
    Shot { shooter: u8, from: V3 },
    Item { id: u16 },
    Idel { id: u16 },
    Chunk { chunk: u8, ids: Vec<u16> },
    Throw(u8),
    Land(u8),
    Other(u8),
}

struct Snap {
    players: Vec<u8>,
    nades: Vec<V3>,
    events: Vec<Ev>,
    spect: u8,
}

struct Rd<'a> {
    b: &'a [u8],
    o: usize,
}
impl<'a> Rd<'a> {
    fn u8(&mut self) -> u8 {
        let v = self.b[self.o];
        self.o += 1;
        v
    }
    fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes([self.b[self.o], self.b[self.o + 1]]);
        self.o += 2;
        v
    }
    fn f32(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.b[self.o..self.o + 4].try_into().unwrap());
        self.o += 4;
        v
    }
    fn skip(&mut self, n: usize) {
        self.o += n;
    }
}

/// Full parse of a snapshot (mirrors `Match::snapshot`); panics on any mismatch.
fn parse(buf: &[u8]) -> Snap {
    let mut r = Rd { b: buf, o: 0 };
    assert_eq!(r.u8(), S_SNAP);
    r.skip(4 + 4 + 1 + 4 + 1 + 1 + 1);
    // self block; spect is the third byte from the end
    let self_len = 1 + 1 + 12 + 12 + 4 + 4 + 4 + 1 + 6 + 8 + 3 + 1 + 4 + 4 + 1 + 4 + 4 + 2 + 2 + 3;
    let spect = buf[r.o + self_len - 3];
    r.skip(self_len);
    r.skip(2 + 4 * 8); // zone
    r.skip(1 + 4 * 7); // plane
    let n = r.u8() as usize;
    let mut players = vec![];
    for _ in 0..n {
        players.push(r.u8());
        r.skip(21);
    }
    let nn = r.u8() as usize;
    let mut nades = vec![];
    for _ in 0..nn {
        r.u16();
        nades.push(V3::new(r.f32(), r.f32(), r.f32()));
    }
    let ne = r.u16() as usize;
    let mut events = vec![];
    for _ in 0..ne {
        let t = r.u8();
        events.push(match t {
            1 => {
                let shooter = r.u8();
                r.u8();
                let from = V3::new(r.f32(), r.f32(), r.f32());
                r.skip(13);
                Ev::Shot { shooter, from }
            }
            2 => {
                r.skip(3);
                Ev::Other(t)
            }
            3 => {
                r.skip(10);
                Ev::Other(t)
            }
            4 => {
                r.skip(4);
                Ev::Other(t)
            }
            5 => {
                let id = r.u16();
                r.skip(15);
                Ev::Item { id }
            }
            6 => Ev::Idel { id: r.u16() },
            7 => {
                r.skip(3);
                let l = r.u8() as usize;
                r.skip(l);
                Ev::Other(t)
            }
            8 => {
                r.skip(12);
                Ev::Other(t)
            }
            9 => {
                r.skip(1);
                Ev::Other(t)
            }
            10 => {
                r.skip(3);
                Ev::Other(t)
            }
            11 => Ev::Throw(r.u8()),
            12 => Ev::Land(r.u8()),
            13 => {
                let chunk = r.u8();
                let k = r.u16() as usize;
                let mut ids = vec![];
                for _ in 0..k {
                    ids.push(r.u16());
                    r.skip(15);
                }
                Ev::Chunk { chunk, ids }
            }
            _ => panic!("unknown event {t}"),
        });
    }
    assert_eq!(r.o, buf.len(), "snapshot not fully consumed");
    Snap { players, nades, events, spect }
}

fn setup(seed: u32, n: usize) -> (Match, Vec<u8>) {
    let mut m = Match::new(seed, n, FLAG_OFFLINE, 5);
    let ids: Vec<u8> = (0..n).map(|k| m.add_human(format!("P{k}").as_bytes()).unwrap()).collect();
    for &i in &ids {
        let _ = m.welcome(i);
    }
    m.step();
    assert_eq!(m.phase, PH_PLAY);
    (m, ids)
}

fn place(m: &mut Match, pid: u8, pos: V3) {
    m.players[pid as usize].ms = MoveState { pos, mode: MODE_GROUND, on_ground: true, ..Default::default() };
}

fn eye(p: V3) -> V3 {
    p + V3::new(0.0, eye_h(false), 0.0)
}

fn hidden(m: &Match, viewer: V3, target: V3) -> bool {
    m.w.blocked(eye(viewer), target + V3::new(0.0, eye_h(false) + 0.1, 0.0)) && m.w.blocked(eye(viewer), target + V3::new(0.0, 0.9, 0.0))
}

/// An item inside a building, and a spot `dist` m away outdoors that cannot see it.
fn indoor_case(m: &Match, dist: f32) -> Option<(u16, V3, V3)> {
    for (id, it) in m.items.iter().enumerate() {
        if !it.alive {
            continue;
        }
        let Some(bi) = m.w.building_at(it.pos.x, it.pos.z) else { continue };
        let bl = m.w.buildings[bi];
        let loot = V3::new(it.pos.x, m.w.ground_below(it.pos.x, it.pos.z, it.pos.y + 0.5, P_RADIUS), it.pos.z);
        if (loot.y - bl.y).abs() > 0.3 {
            continue;
        }
        for k in 0..24 {
            let a = k as f32 * TAU / 24.0;
            let (x, z) = (it.pos.x + a.cos() * dist, it.pos.z + a.sin() * dist);
            if !(30.0..MAP - 30.0).contains(&x) || !(30.0..MAP - 30.0).contains(&z) {
                continue;
            }
            if m.w.height(x, z) < 2.0 || m.w.building_at(x, z).is_some() {
                continue;
            }
            let v = V3::new(x, m.w.ground_below(x, z, 300.0, P_RADIUS), z);
            if hidden(m, v, loot) && m.w.blocked(eye(v), it.pos + V3::new(0.0, 0.25, 0.0)) {
                return Some((id as u16, loot, v));
            }
        }
    }
    None
}

fn snap(m: &mut Match, pid: u8) -> Snap {
    let s = parse(&m.snapshot(pid).to_vec());
    s
}

fn tick(m: &mut Match, viewers: &[u8]) -> Vec<Snap> {
    m.step();
    let out = viewers.iter().map(|&v| snap(m, v)).collect();
    m.flush();
    out
}

fn mentions_item(s: &Snap, id: u16, chunk: u8) -> bool {
    s.events.iter().any(|e| match e {
        Ev::Item { id: k } | Ev::Idel { id: k } => *k == id,
        Ev::Chunk { chunk: c, .. } => *c == chunk,
        _ => false,
    })
}

#[test]
fn hidden_looter_is_not_revealed() {
    let mut cases = 0;
    for seed in [3u32, 11, 42, 77, 123] {
        for dist in [60.0f32, 100.0, 150.0] {
            let (mut m, ids) = setup(seed, 2);
            let (a, b) = (ids[0], ids[1]);
            let Some((id, loot, va)) = indoor_case(&m, dist) else { continue };
            let it = m.items[id as usize];
            let chunk = item_chunk(it.pos.x, it.pos.z) as u8;
            place(&mut m, a, va);
            place(&mut m, b, loot);
            // B auto-picks the item up within a few ticks
            let mut taken_at = None;
            for t in 0..300u32 {
                let s = tick(&mut m, &[a]);
                if taken_at.is_none() && !m.items[id as usize].alive {
                    taken_at = Some(t);
                }
                let s = &s[0];
                assert!(!s.players.contains(&b), "seed {seed}: hidden looter sent in the players block");
                if let Some(t0) = taken_at {
                    if t - t0 < (ITEM_QUIET_TEST - 2) {
                        assert!(!mentions_item(s, id, chunk), "seed {seed} dist {dist}: pickup by a hidden player revealed {} ticks later", t - t0);
                    }
                }
            }
            if taken_at.is_none() {
                continue;
            }
            cases += 1;
            // walking up to the spot shows the item is gone
            place(&mut m, a, loot + V3::new(0.0, 0.0, 0.0));
            m.players[b as usize].ms.pos = V3::new(30.0, 60.0, 30.0);
            let s = tick(&mut m, &[a]);
            assert!(s[0].events.iter().any(|e| matches!(e, Ev::Idel { id: k } if *k == id)) || s[0].events.iter().any(|e| matches!(e, Ev::Chunk { chunk: c, ids } if *c == chunk && !ids.contains(&id))), "seed {seed}: the change never reached a viewer standing on the spot");
        }
    }
    assert!(cases >= 4, "too few hidden-looter cases ({cases})");
}

const ITEM_QUIET_TEST: u32 = 300;

#[test]
fn quiet_chunks_are_resynced_eventually() {
    for seed in [3u32, 42, 123] {
        let (mut m, ids) = setup(seed, 2);
        let (a, b) = (ids[0], ids[1]);
        let Some((id, loot, va)) = indoor_case(&m, 100.0) else { continue };
        let chunk = item_chunk(loot.x, loot.z) as u8;
        place(&mut m, a, va);
        place(&mut m, b, loot);
        let mut got = None;
        let mut taken = None;
        for t in 0..420u32 {
            let s = tick(&mut m, &[a]);
            if taken.is_none() && !m.items[id as usize].alive {
                taken = Some(t);
                // B leaves the area (no further changes in the chunk)
                m.players[b as usize].ms.pos = V3::new(30.0, 60.0, 30.0);
            }
            if s[0].events.iter().any(|e| matches!(e, Ev::Chunk { chunk: c, ids } if *c == chunk && !ids.contains(&id))) {
                got = Some(t);
                break;
            }
        }
        let (t0, t1) = (taken.unwrap(), got.expect("chunk never resynced"));
        assert!(t1 - t0 >= ITEM_QUIET_TEST - 2, "resync after {} ticks: too early", t1 - t0);
        return;
    }
    panic!("no case");
}

#[test]
fn visible_looter_is_shown_at_once() {
    let (mut m, ids) = setup(42, 2);
    let (a, b) = (ids[0], ids[1]);
    // an outdoor item with a clear view from 80 m
    for (id, it) in m.items.clone().iter().enumerate() {
        if !it.alive || m.w.building_at(it.pos.x, it.pos.z).is_some() {
            continue;
        }
        let spot = V3::new(it.pos.x, m.w.ground_below(it.pos.x, it.pos.z, it.pos.y + 0.5, P_RADIUS), it.pos.z);
        for k in 0..16 {
            let ang = k as f32 * TAU / 16.0;
            let (x, z) = (it.pos.x + ang.cos() * 80.0, it.pos.z + ang.sin() * 80.0);
            if !(30.0..MAP - 30.0).contains(&x) || !(30.0..MAP - 30.0).contains(&z) || m.w.height(x, z) < 2.0 {
                continue;
            }
            let v = V3::new(x, m.w.ground_below(x, z, 300.0, P_RADIUS), z);
            if hidden(&m, v, spot) || m.w.blocked(eye(v), spot + V3::new(0.0, 1.6, 0.0)) {
                continue;
            }
            place(&mut m, a, v);
            place(&mut m, b, spot);
            for _ in 0..12 {
                let s = tick(&mut m, &[a]);
                if !m.items[id].alive {
                    // removal must be in this very snapshot or the next
                    let s2 = tick(&mut m, &[a]);
                    let seen = s[0].events.iter().chain(s2[0].events.iter()).any(|e| matches!(e, Ev::Idel { id: k } if *k as usize == id));
                    assert!(seen, "a visible looter's pickup was not shown");
                    return;
                }
            }
        }
    }
    panic!("no visible-looter case found");
}

#[test]
fn hidden_shooter_is_anonymous_and_coarse() {
    for seed in [3u32, 42, 123] {
        let (mut m, ids) = setup(seed, 2);
        let (a, b) = (ids[0], ids[1]);
        let Some((_, spot, va)) = indoor_case(&m, 60.0) else { continue };
        place(&mut m, a, va);
        place(&mut m, b, spot);
        {
            let p = &mut m.players[b as usize];
            p.slots[0] = (W_AR, 30);
            p.cur = 0;
            p.switch_t = 0.0;
        }
        let mut seq = m.players[b as usize].last_seq;
        let mut shots = 0;
        for _ in 0..20 {
            seq += 1;
            m.on_message(b, &cmd_bytes(&[(seq, m.tick * 256, 0, 0, 0, 0, B_FIRE)]));
            let s = tick(&mut m, &[a]);
            for e in &s[0].events {
                if let Ev::Shot { shooter, from } = e {
                    shots += 1;
                    assert_eq!(*shooter, 255, "hidden shooter's id was sent");
                    let pb = &m.players[b as usize];
                    let muz = muzzle(pb.ms.pos, pb.yaw, pb.pitch, pb.ms.crouch);
                    let g = (dist_xz(muz.x, muz.z, va.x, va.z) / 8.0).max(4.0);
                    // snapped to the centre of a >= 4 m grid cell around the muzzle
                    let on_grid = |v: f32| {
                        let k = v / g - 0.5;
                        (k - k.round()).abs() < 1e-3
                    };
                    assert!(on_grid(from.x) && on_grid(from.z), "{from:?} not on the {g} m grid");
                    assert!(dist_xz(from.x, from.z, muz.x, muz.z) <= g * 0.75);
                }
                assert!(!matches!(e, Ev::Throw(_) | Ev::Land(_)));
            }
        }
        assert!(shots > 0, "no shots heard at all");
        return;
    }
    panic!("no case");
}

#[test]
fn visible_shooter_keeps_id() {
    let (mut m, ids) = setup(42, 2);
    let (a, b) = (ids[0], ids[1]);
    let pa = V3::new(500.0, m.w.ground_below(500.0, 500.0, 300.0, P_RADIUS), 500.0);
    place(&mut m, a, pa);
    let pb = V3::new(510.0, m.w.ground_below(510.0, 500.0, 300.0, P_RADIUS), 500.0);
    place(&mut m, b, pb);
    {
        let p = &mut m.players[b as usize];
        p.slots[0] = (W_AR, 30);
        p.cur = 0;
    }
    let mut seq = m.players[b as usize].last_seq;
    let mut ok = false;
    for _ in 0..10 {
        seq += 1;
        m.on_message(b, &cmd_bytes(&[(seq, m.tick * 256, 0, 0, 0, 0, B_FIRE)]));
        let s = tick(&mut m, &[a]);
        for e in &s[0].events {
            if let Ev::Shot { shooter, .. } = e {
                assert_eq!(*shooter, b);
                ok = true;
            }
        }
    }
    assert!(ok);
}

#[test]
fn hidden_grenades_and_throws_are_not_sent() {
    for seed in [3u32, 42, 123] {
        let (mut m, ids) = setup(seed, 2);
        let (a, b) = (ids[0], ids[1]);
        let Some((_, spot, va)) = indoor_case(&m, 100.0) else { continue };
        place(&mut m, a, va);
        place(&mut m, b, spot);
        m.players[b as usize].nades = 2;
        let mut seq = m.players[b as usize].last_seq;
        seq += 1;
        // thrown at the floor inside the building
        m.on_message(b, &cmd_bytes(&[(seq, m.tick * 256, 0, 0, 0, q_pitch(-1.2), B_NADE)]));
        let mut thrown = false;
        for _ in 0..60 {
            let s = tick(&mut m, &[a]);
            thrown |= !m.nades.is_empty();
            for n in &s[0].nades {
                let ok = !m.w.blocked(eye(va), *n + V3::new(0.0, 0.1, 0.0));
                assert!(ok, "seed {seed}: a grenade the viewer cannot see was sent");
            }
            assert!(!s[0].events.iter().any(|e| matches!(e, Ev::Throw(_))), "E_THROW of another player was sent");
        }
        assert!(thrown);
        return;
    }
    panic!("no case");
}

#[test]
fn spectating_switches_are_rate_limited() {
    let (mut m, ids) = setup(9, 8);
    let a = ids[0];
    {
        let p = &mut m.players[a as usize];
        p.alive = false;
        p.ms.mode = MODE_DEAD;
        p.death_tick = m.tick;
    }
    let mut seq = m.players[a as usize].last_seq;
    let mut last = 255u8;
    let mut changes = 0;
    for k in 0..90u32 {
        seq += 1;
        // tap N every other tick
        let b = if k % 2 == 0 { B_SPECT } else { 0 };
        m.on_message(a, &cmd_bytes(&[(seq, m.tick * 256, 0, 0, 0, 0, b)]));
        let s = tick(&mut m, &[a]);
        if s[0].spect != last {
            changes += 1;
            last = s[0].spect;
        }
    }
    assert!(changes <= 3, "spectated target changed {changes} times in 3 s");
    assert!(changes >= 1);
}
