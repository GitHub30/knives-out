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
    Shot { shooter: u8, weapon: u8, from: V3, end: V3, kind: u8 },
    Hurt { attacker: u8, sx: f32, sz: f32 },
    Item { id: u16, pos: V3 },
    Idel { id: u16 },
    Chunk { chunk: u8, ids: Vec<u16>, pos: Vec<V3> },
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
    // self block; spect is the fourth byte from the end (then place, killer, debt)
    let self_len = 1 + 1 + 12 + 12 + 4 + 4 + 4 + 1 + 6 + 8 + 3 + 1 + 4 + 4 + 1 + 4 + 4 + 2 + 2 + 3 + 1;
    let spect = buf[r.o + self_len - 4];
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
                let weapon = r.u8();
                let from = V3::new(r.f32(), r.f32(), r.f32());
                let end = V3::new(r.f32(), r.f32(), r.f32());
                let kind = r.u8();
                Ev::Shot { shooter, weapon, from, end, kind }
            }
            2 => {
                r.skip(3);
                Ev::Other(t)
            }
            3 => {
                let attacker = r.u8();
                let (sx, sz) = (r.f32(), r.f32());
                r.u8();
                Ev::Hurt { attacker, sx, sz }
            }
            4 => {
                r.skip(4);
                Ev::Other(t)
            }
            5 => {
                let id = r.u16();
                r.skip(3);
                let pos = V3::new(r.f32(), r.f32(), r.f32());
                Ev::Item { id, pos }
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
                let mut pos = vec![];
                for _ in 0..k {
                    ids.push(r.u16());
                    r.skip(3);
                    pos.push(V3::new(r.f32(), r.f32(), r.f32()));
                }
                Ev::Chunk { chunk, ids, pos }
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

/// Centre of the `g`-sized world-aligned cell holding `v`.
fn cell_c(v: f32, g: f32) -> f32 {
    ((v / g).floor() + 0.5) * g
}

/// Cell size for a hidden shot at `muz` heard at `viewer`: 32 m, refined to
/// 16 / 8 m from the centres of the enclosing coarser cells only.
fn hidden_cell(muz: V3, viewer: V3) -> f32 {
    let mut g = 32.0f32;
    while g > 8.0 && dist_xz(cell_c(muz.x, g), cell_c(muz.z, g), viewer.x, viewer.z) < g * 6.0 {
        g *= 0.5;
    }
    g
}

fn mentions_item(s: &Snap, id: u16, chunk: u8) -> bool {
    s.events.iter().any(|e| match e {
        Ev::Item { id: k, .. } | Ev::Idel { id: k } => *k == id,
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
            assert!(s[0].events.iter().any(|e| matches!(e, Ev::Idel { id: k } if *k == id)) || s[0].events.iter().any(|e| matches!(e, Ev::Chunk { chunk: c, ids, .. } if *c == chunk && !ids.contains(&id))), "seed {seed}: the change never reached a viewer standing on the spot");
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
            if s[0].events.iter().any(|e| matches!(e, Ev::Chunk { chunk: c, ids, .. } if *c == chunk && !ids.contains(&id))) {
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
        let mut misses = 0;
        for k in 0..40 {
            seq += 1;
            // level shots (walls), then into the sky (misses)
            let pitch = if k < 20 { 0 } else { q_pitch(0.7) };
            m.players[b as usize].ammo = [200; NUM_AMMO];
            m.on_message(b, &cmd_bytes(&[(seq, m.tick * 256, 0, 0, (k as u16) * 1500, pitch, B_FIRE)]));
            let s = tick(&mut m, &[a]);
            for e in &s[0].events {
                if let Ev::Shot { shooter, weapon, from, end, kind } = e {
                    shots += 1;
                    assert_eq!(*shooter, 255, "hidden shooter's id was sent");
                    let pb = &m.players[b as usize];
                    let muz = muzzle(pb.ms.pos, pb.yaw, pb.pitch, pb.ms.crouch);
                    // the centre of the world-aligned 8, 16 or 32 m cell holding
                    // the muzzle, chosen from cell centres alone
                    let g = hidden_cell(muz, va);
                    assert_eq!((from.x, from.z), (cell_c(muz.x, g), cell_c(muz.z, g)), "not the {g} m cell of {muz:?}");
                    assert!(g >= dist_xz(muz.x, muz.z, va.x, va.z) / 16.0, "{g} m cell too fine");
                    if *kind == 0 {
                        misses += 1;
                        // a miss: only a direction from the coarse origin
                        assert!(((*end - *from).len() - WD[*weapon as usize].range).abs() < 0.05, "miss end {end:?} not range from the coarse origin {from:?}");
                    }
                }
                assert!(!matches!(e, Ev::Throw(_) | Ev::Land(_)));
            }
        }
        assert!(shots > 0, "no shots heard at all");
        let _ = misses;
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


// ---------------------------------------------------------------------------
// Third adversarial review

/// Aim `pid` (standing at its position) at `target` and fire `n` times with
/// the current weapon, collecting every viewer's snapshots.
fn fire_at(m: &mut Match, pid: u8, target: V3, n: usize, viewers: &[u8]) -> Vec<Snap> {
    let me = m.players[pid as usize].ms.pos;
    let eye = me + V3::new(0.0, eye_h(false), 0.0);
    let d = target - eye;
    let yaw = q_yaw(yaw_to(d.x, d.z));
    let pitch = q_pitch((d.y / d.len_xz()).atan());
    let mut seq = m.players[pid as usize].last_seq;
    let mut out = vec![];
    for k in 0..(n + 10) {
        seq += 1;
        let b = if k >= 10 { B_ADS | B_FIRE } else { B_ADS };
        {
            let p = &mut m.players[pid as usize];
            p.fire_cd = 0.0;
            p.switch_t = 0.0;
            p.ammo = [200; NUM_AMMO];
        }
        m.on_message(pid, &cmd_bytes(&[(seq, m.tick * 256, 0, 0, yaw, pitch, b)]));
        out.extend(tick(m, viewers));
    }
    out
}

/// A visible shooter hitting a player the viewer is not sent: the impact
/// point would locate the victim, so the viewer sees the shot fly on as if
/// it had missed (to the wall behind, or full range).
#[test]
fn hits_on_hidden_victims_do_not_locate_them() {
    let mut cases = 0;
    'seeds: for seed in [3u32, 42, 77, 123, 999] {
        let (mut m, ids) = setup(seed, 3);
        let (a, b, c) = (ids[0], ids[1], ids[2]);
        let ground = |m: &Match, x: f32, z: f32| V3::new(x, m.w.ground_below(x, z, 300.0, P_RADIUS), z);
        let open = |m: &Match, x: f32, z: f32| (30.0..MAP - 30.0).contains(&x) && (30.0..MAP - 30.0).contains(&z) && m.w.height(x, z) > 2.5 && m.w.building_at(x, z).is_none();
        for bl in m.w.buildings.clone() {
            let zm = (bl.z0 + bl.z1) * 0.5;
            // C right behind the building, A in front of it, B off to the side
            let (cx, ax) = (bl.x1 + 1.5, bl.x0 - 50.0);
            let (bx, bz) = (bl.x1 + 12.0, bl.z1 + 25.0);
            if !open(&m, cx, zm) || !open(&m, ax, zm) || !open(&m, bx, bz) {
                continue;
            }
            let (pc, pa, pb) = (ground(&m, cx, zm), ground(&m, ax, zm), ground(&m, bx, bz));
            let chest = |p: V3| p + V3::new(0.0, 1.2, 0.0);
            if !hidden(&m, pa, pc) || m.w.blocked(eye(pa), chest(pb)) || m.w.blocked(eye(pb), chest(pc)) || m.w.blocked(eye(pb), eye(pc)) {
                continue;
            }
            place(&mut m, a, pa);
            place(&mut m, b, pb);
            place(&mut m, c, pc);
            {
                let p = &mut m.players[b as usize];
                p.slots[0] = (W_AR, 30);
                p.cur = 0;
            }
            for _ in 0..4 {
                tick(&mut m, &[a]);
            }
            let s0 = snap(&mut m, a);
            if !s0.players.contains(&b) || s0.players.contains(&c) {
                continue;
            }
            let hp0 = m.players[c as usize].health;
            let snaps = fire_at(&mut m, b, chest(pc), 8, &[a]);
            if m.players[c as usize].health >= hp0 {
                continue;
            }
            let mut shots = 0;
            for s in &snaps {
                assert!(!s.players.contains(&c), "victim became visible");
                for e in &s.events {
                    if let Ev::Shot { shooter, end, kind, .. } = e {
                        if *shooter != b {
                            continue;
                        }
                        shots += 1;
                        assert!(*kind == 0 || *kind == 3, "hit kind {kind} on a hidden victim sent");
                        assert!(dist_xz(end.x, end.z, pc.x, pc.z) > 2.0, "shot end {end:?} locates the hidden victim at {pc:?}");
                    }
                }
            }
            assert!(shots > 0, "the viewer heard none of the visible shooter's shots");
            cases += 1;
            if cases >= 2 {
                break 'seeds;
            }
            continue 'seeds;
        }
    }
    assert!(cases > 0, "no case found");
}

/// A sniper beyond the interest radius hits the viewer: the hurt event only
/// gives a direction for the damage indicator, never the sniper's id or spot.
#[test]
fn hurt_by_unseen_sniper_gives_only_a_direction() {
    for seed in [42u32, 7, 123, 2024] {
        let (mut m, ids) = setup(seed, 2);
        let (a, b) = (ids[0], ids[1]);
        let mut found = None;
        'search: for gz in (40..990).step_by(37) {
            for gx in (40..990).step_by(37) {
                let (x, z) = (gx as f32, gz as f32);
                if m.w.height(x, z) < 12.0 || m.w.building_at(x, z).is_some() {
                    continue;
                }
                let pb = V3::new(x, m.w.ground_below(x, z, 300.0, P_RADIUS), z);
                for k in 0..16 {
                    let ang = k as f32 * TAU / 16.0;
                    let (ax, az) = (x + ang.cos() * 580.0, z + ang.sin() * 580.0);
                    if !(30.0..MAP - 30.0).contains(&ax) || !(30.0..MAP - 30.0).contains(&az) || m.w.height(ax, az) < 2.5 || m.w.building_at(ax, az).is_some() {
                        continue;
                    }
                    let pa = V3::new(ax, m.w.ground_below(ax, az, 300.0, P_RADIUS), az);
                    if !m.w.blocked(eye(pb), pa + V3::new(0.0, 1.2, 0.0)) && !m.w.blocked(eye(pb), pa + V3::new(0.0, 0.9, 0.0)) {
                        found = Some((pa, pb));
                        break 'search;
                    }
                }
            }
        }
        let Some((pa, pb)) = found else { continue };
        place(&mut m, a, pa);
        place(&mut m, b, pb);
        {
            let p = &mut m.players[b as usize];
            p.slots[0] = (W_SNIPER, 5);
            p.cur = 0;
        }
        let hp0 = m.players[a as usize].health;
        // one shot (~78 damage): a dead viewer would spectate its killer
        let snaps = fire_at(&mut m, b, pa + V3::new(0.0, 1.2, 0.0), 1, &[a]);
        if m.players[a as usize].health >= hp0 {
            continue;
        }
        let truth = (pb.x - pa.x).atan2(pb.z - pa.z);
        let mut hurts = 0;
        for s in &snaps {
            assert!(!s.players.contains(&b));
            for e in &s.events {
                if let Ev::Hurt { attacker, sx, sz } = e {
                    hurts += 1;
                    assert_eq!(*attacker, HURT_HIDDEN, "the unseen sniper's id was sent");
                    let pa = m.players[a as usize].ms.pos;
                    let r = dist_xz(*sx, *sz, pa.x, pa.z);
                    assert!((r - 20.0).abs() < 0.5, "hurt source {r:.2} m away, not a bare direction");
                    let got = (sx - pa.x).atan2(sz - pa.z);
                    let err = ((got - truth + PI).rem_euclid(TAU) - PI).abs();
                    assert!(err <= TAU / 64.0 + 0.01, "damage direction off by {err:.3} rad");
                }
                if let Ev::Shot { .. } = e {
                    panic!("a shot from beyond the interest radius was sent");
                }
            }
        }
        assert!(hurts > 0);
        return;
    }
    panic!("no sniper case found");
}


/// A hidden shooter outdoors firing into the sky: misses end exactly `range`
/// from the aim origin, so a few of them used to give the shooter's exact
/// spot. They must carry only a direction from the coarse cell centre.
#[test]
fn hidden_misses_carry_no_range() {
    let mut cases = 0;
    for seed in [3u32, 42, 77, 123] {
        let (mut m, ids) = setup(seed, 2);
        let (a, b) = (ids[0], ids[1]);
        let open = |m: &Match, x: f32, z: f32| (30.0..MAP - 30.0).contains(&x) && (30.0..MAP - 30.0).contains(&z) && m.w.height(x, z) > 2.5 && m.w.building_at(x, z).is_none();
        let ground = |m: &Match, x: f32, z: f32| V3::new(x, m.w.ground_below(x, z, 300.0, P_RADIUS), z);
        let Some((pa, pb)) = m.w.buildings.iter().find_map(|bl| {
            let zm = (bl.z0 + bl.z1) * 0.5;
            let (bx, ax) = (bl.x1 + 2.5, bl.x0 - 70.0);
            if !open(&m, bx, zm) || !open(&m, ax, zm) {
                return None;
            }
            let (pa, pb) = (ground(&m, ax, zm), ground(&m, bx, zm));
            // hidden, and open sky above the shooter
            let up = V3::new(0.6, 0.8, 0.0).norm();
            (hidden(&m, pa, pb) && m.w.raycast(eye(pb), up, 200.0).is_none()).then_some((pa, pb))
        }) else {
            continue;
        };
        place(&mut m, a, pa);
        place(&mut m, b, pb);
        {
            let p = &mut m.players[b as usize];
            p.slots[0] = (W_AR, 30);
            p.cur = 0;
        }
        for _ in 0..4 {
            tick(&mut m, &[a]);
        }
        let mut seq = m.players[b as usize].last_seq;
        let mut misses = vec![];
        for _ in 0..24 {
            seq += 1;
            {
                let p = &mut m.players[b as usize];
                p.fire_cd = 0.0;
                p.ammo = [200; NUM_AMMO];
            }
            m.on_message(b, &cmd_bytes(&[(seq, m.tick * 256, 0, 0, q_yaw(yaw_to(1.0, 0.0)), q_pitch(0.9), B_FIRE)]));
            for s in tick(&mut m, &[a]) {
                for e in &s.events {
                    if let Ev::Shot { shooter, weapon, from, end, kind } = e {
                        assert_eq!(*shooter, 255);
                        if *kind == 0 {
                            let r = (*end - *from).len();
                            assert!((r - WD[*weapon as usize].range).abs() < 0.05, "miss end {r:.3} m from the coarse origin");
                            misses.push(*end);
                        }
                    }
                }
            }
        }
        assert!(misses.len() >= 5, "only {} misses heard", misses.len());
        // all ends lie on one sphere around the cell centre, not the shooter
        cases += 1;
    }
    assert!(cases > 0, "no case found");
}


/// The client's item map, updated exactly like `Game.onEvent` in src/game.ts.
struct ClientItems(std::collections::HashMap<u16, V3>);
impl ClientItems {
    fn apply(&mut self, s: &Snap) {
        for e in &s.events {
            match e {
                Ev::Item { id, pos } => {
                    self.0.insert(*id, *pos);
                }
                Ev::Idel { id } => {
                    self.0.remove(id);
                }
                Ev::Chunk { chunk, ids, pos } => {
                    self.0.retain(|_, p| item_chunk(p.x, p.z) != *chunk as usize);
                    for (id, p) in ids.iter().zip(pos) {
                        self.0.insert(*id, *p);
                    }
                }
                _ => {}
            }
        }
    }
}

/// A player dies indoors, out of the viewer's sight; the looter strips the
/// pile there and then walks out into view; the viewer walks up to the pile.
/// The pile's (withheld) creation must never be replayed after its looting:
/// that brought looted items back as ghosts with a dead pickup prompt.
#[test]
fn looted_pile_never_comes_back_as_ghosts() {
    let mut runs = 0;
    for seed in [3u32, 11, 42, 77, 123, 5, 7, 9] {
        for busy in [false, true] {
            let (mut m, ids) = setup(seed, 4);
            let (v, p, l, h) = (ids[0], ids[1], ids[2], ids[3]);
            let Some((_, loot, va)) = indoor_case(&m, 60.0) else { continue };
            let chunk = item_chunk(loot.x, loot.z);
            let dir = V3::new(loot.x - va.x, 0.0, loot.z - va.z);
            let total = dir.len_xz();
            let dir = dir * (1.0 / total);
            let path: Vec<V3> = (0..=20).map(|k| va + dir * (total * k as f32 / 20.0)).collect();
            // busy: a second hidden death in the same chunk keeps it from resyncing
            let mut hspot = None;
            if busy {
                hspot = m.items.iter().filter(|it| it.alive && m.w.building_at(it.pos.x, it.pos.z).is_some() && item_chunk(it.pos.x, it.pos.z) == chunk).find_map(|it| {
                    let sp = V3::new(it.pos.x, m.w.ground_below(it.pos.x, it.pos.z, it.pos.y + 0.5, P_RADIUS), it.pos.z);
                    let far = dist_xz(sp.x, sp.z, loot.x, loot.z) >= 50.0;
                    let unseen = path.iter().all(|&q| dist_xz(q.x, q.z, sp.x, sp.z) > 45.0 && hidden(&m, V3::new(q.x, m.w.ground_below(q.x, q.z, 300.0, P_RADIUS), q.z), sp));
                    (far && unseen).then_some(sp)
                });
                if hspot.is_none() {
                    continue;
                }
            }
            // what the welcome gave the viewer
            let mut cv = ClientItems(m.items.iter().enumerate().filter(|(_, it)| it.alive).map(|(id, it)| (id as u16, it.pos)).collect());
            place(&mut m, v, va);
            place(&mut m, p, loot);
            m.players[l as usize].ms.pos = V3::new(30.0, 60.0, 30.0);
            place(&mut m, h, hspot.unwrap_or(V3::new(MAP - 40.0, 60.0, MAP - 40.0)));
            for q in [p, h] {
                let pl = &mut m.players[q as usize];
                pl.slots[0] = (W_AR, 30);
                pl.ammo = [40, 60, 20, 10];
                pl.heals = [4, 1, 1];
                pl.vest = 2;
                pl.vest_dur = 80.0;
                pl.nades = 2;
            }
            let mut pile: Vec<usize> = vec![];
            let mut deleted = std::collections::HashSet::new();
            let (mut resurrected, mut ghosts) = (0, 0);
            let mut vpos = va;
            for k in 0..900u32 {
                if k == 2 {
                    let n0 = m.items.len();
                    m.damage(p as usize, l, 1000.0, false, W_AR, loot);
                    pile = (n0..m.items.len()).collect();
                }
                // the looter walks over the pile, hidden indoors
                if (3..39).contains(&k) && !pile.is_empty() {
                    let it = m.items[pile[((k - 3) / 3) as usize % pile.len()]];
                    place(&mut m, l, it.pos);
                }
                if k == 45 {
                    place(&mut m, l, va + V3::new(3.0, 0.0, 0.0));
                }
                if busy && k == 50 {
                    m.damage(h as usize, l, 1000.0, false, W_AR, hspot.unwrap());
                }
                if k >= 55 {
                    let d = ((k - 55) as f32 * 0.2).min(total - 3.0);
                    vpos = va + dir * d;
                    vpos.y = m.w.ground_below(vpos.x, vpos.z, vpos.y.max(loot.y) + 1.0, P_RADIUS);
                    place(&mut m, v, vpos);
                }
                for q in [v, l] {
                    m.players[q as usize].health = 100.0;
                }
                let s = tick(&mut m, &[v, l]).swap_remove(0);
                for e in &s.events {
                    match e {
                        Ev::Idel { id } => {
                            deleted.insert(*id);
                        }
                        Ev::Item { id, .. } if deleted.contains(id) && !m.items[*id as usize].alive => resurrected += 1,
                        _ => {}
                    }
                }
                cv.apply(&s);
                ghosts += cv.0.iter().filter(|(id, p)| !m.items[**id as usize].alive && dist_xz(p.x, p.z, vpos.x, vpos.z) < 38.0).count();
            }
            assert!(pile.iter().any(|&i| !m.items[i].alive), "nothing was looted");
            assert_eq!(resurrected, 0, "seed {seed} busy {busy}: looted items re-sent after their deletion");
            assert_eq!(ghosts, 0, "seed {seed} busy {busy}: ghost items next to the viewer");
            runs += 1;
        }
    }
    assert!(runs >= 4, "only {runs} runs");
}

/// Probing with one's own item changes (a weapon swap at one's feet) must get
/// the same answer whether or not an unseen player is looting in the same
/// chunk. A chunk resync used to follow the probe at once unless a hidden
/// change was pending there, which told the viewer about the hidden looter in
/// real time; and the withheld resync came exactly 10 s after the hidden
/// change, which dated it.
#[test]
fn own_item_changes_do_not_reveal_hidden_looting() {
    let mut delays = vec![];
    for seed in [3u32, 11, 42, 77, 123, 5, 7, 9, 13, 21, 31, 55] {
        let (mut m, ids) = setup(seed, 2);
        let (v, l) = (ids[0], ids[1]);
        let Some((xid, loot, _)) = indoor_case(&m, 60.0) else { continue };
        let c = item_chunk(loot.x, loot.z);
        let cw = MAP / ICH as f32;
        let (x0, z0) = ((c % ICH) as f32 * cw, (c / ICH) as f32 * cw);
        // the viewer: elsewhere in the same chunk, unable to see the loot
        let mut vpos = None;
        'find: for gz in 0..15 {
            for gx in 0..15 {
                let (x, z) = (x0 + 8.0 + gx as f32 * 8.0, z0 + 8.0 + gz as f32 * 8.0);
                if dist_xz(x, z, loot.x, loot.z) < 50.0 || m.w.height(x, z) < 2.5 || m.w.building_at(x, z).is_some() {
                    continue;
                }
                let p = V3::new(x, m.w.ground_below(x, z, 300.0, P_RADIUS), z);
                if hidden(&m, p, loot) && m.w.blocked(eye(p), m.items[xid as usize].pos + V3::new(0.0, 0.25, 0.0)) {
                    vpos = Some(p);
                    break 'find;
                }
            }
        }
        let Some(vpos) = vpos else { continue };
        // far beyond the interest radius, so the viewer never sees the looter
        let park = V3::new(if vpos.x < MAP * 0.5 { MAP - 20.0 } else { 20.0 }, 200.0, if vpos.z < MAP * 0.5 { MAP - 20.0 } else { 20.0 });
        place(&mut m, v, vpos);
        place(&mut m, l, park);
        {
            let p = &mut m.players[v as usize];
            p.slots = [(W_PISTOL, 10), (W_SMG, 10)];
            p.cur = 0;
        }
        // a gun at the viewer's feet to swap with
        let gun = m.items.iter().position(|it| it.alive && it.kind == W_AR && dist_xz(it.pos.x, it.pos.z, loot.x, loot.z) > 60.0).unwrap();
        m.items[gun].pos = vpos + V3::new(0.4, 0.05, 0.0);
        let mut seq_v = m.players[v as usize].last_seq;
        for _ in 0..60 {
            m.players[l as usize].ms.pos = park;
            tick(&mut m, &[v]);
        }
        let mut probe = |m: &mut Match| -> Snap {
            seq_v += 1;
            m.on_message(v, &cmd_bytes(&[(seq_v, m.tick * 256, 0, 0, 0, 0, 0)]));
            tick(m, &[v]);
            seq_v += 1;
            m.on_message(v, &cmd_bytes(&[(seq_v, m.tick * 256, 0, 0, 0, 0, B_PICKUP)]));
            tick(m, &[v]).swap_remove(0)
        };
        let own = |s: &Snap| s.events.iter().any(|e| matches!(e, Ev::Item { .. } | Ev::Idel { .. }));
        let resync = |s: &Snap| s.events.iter().any(|e| matches!(e, Ev::Chunk { chunk, .. } if *chunk as usize == c));
        // no hidden change pending: the probe's own change, nothing else
        for _ in 0..3 {
            let s = probe(&mut m);
            assert!(own(&s), "seed {seed}: the swap was not sent");
            assert!(!resync(&s), "seed {seed}: own change answered with a resync");
            for _ in 0..20 {
                tick(&mut m, &[v]);
            }
        }
        // an unseen player loots indoors in the same chunk
        let before = m.items[xid as usize];
        place(&mut m, l, loot);
        let mut seq_l = m.players[l as usize].last_seq;
        let mut looted_at = None;
        for _ in 0..12 {
            seq_l += 1;
            m.on_message(l, &cmd_bytes(&[(seq_l, m.tick * 256, 0, 0, 0, 0, if m.tick % 2 == 0 { B_PICKUP } else { 0 })]));
            for s in tick(&mut m, &[v]) {
                assert!(!resync(&s));
            }
            let now = m.items[xid as usize];
            if now.alive != before.alive || now.amount != before.amount {
                looted_at = Some(m.tick);
                break;
            }
        }
        let Some(looted_at) = looted_at else { continue };
        m.players[l as usize].ms.pos = park;
        // the same probes now get the same answer
        for _ in 0..3 {
            let s = probe(&mut m);
            assert!(own(&s), "seed {seed}: the swap was not sent");
            assert!(!resync(&s), "seed {seed}: own change answered with a resync");
            for _ in 0..20 {
                m.players[l as usize].ms.pos = park;
                tick(&mut m, &[v]);
            }
        }
        // the hidden change is resynced once the chunk has been quiet long enough
        let mut got = None;
        for _ in 0..600 {
            m.players[l as usize].ms.pos = park;
            if tick(&mut m, &[v]).iter().any(|s| resync(s)) {
                got = Some(m.tick - looted_at);
                break;
            }
        }
        let d = got.unwrap_or_else(|| panic!("seed {seed}: the hidden change was never resynced"));
        assert!((300..=452).contains(&d), "seed {seed}: resync {d} ticks after the hidden change");
        delays.push(d);
    }
    eprintln!("resync delays after a hidden change: {delays:?}");
    assert!(delays.len() >= 3, "only {} cases", delays.len());
    assert!(delays.iter().any(|&d| d != delays[0]), "the resync always comes the same time after the hidden change");
}

// ---------------------------------------------------------------------------
// Fourth adversarial review

/// A shooter the viewer cannot see, firing level into the walls of the
/// building it stands in: the wall hits right next to it would locate it to
/// about a metre, so they must arrive like misses (a coarse direction from
/// the cell centre), never as exact impacts near the muzzle.
#[test]
fn hidden_wall_hits_near_the_shooter_are_coarse() {
    let mut cases = 0;
    for seed in [3u32, 42, 123, 77] {
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
        let (mut shots, mut near_walls) = (0, 0);
        for k in 0..24u16 {
            seq += 1;
            {
                let p = &mut m.players[b as usize];
                p.fire_cd = 0.0;
                p.ammo = [200; NUM_AMMO];
            }
            m.on_message(b, &cmd_bytes(&[(seq, m.tick * 256, 0, 0, k * 2730, 0, B_FIRE)]));
            let s = tick(&mut m, &[a]);
            let pb = &m.players[b as usize];
            let muz = muzzle(pb.ms.pos, pb.yaw, pb.pitch, pb.ms.crouch);
            for e in &s[0].events {
                if let Ev::Shot { shooter, weapon, from, end, kind } = e {
                    assert_eq!(*shooter, 255);
                    shots += 1;
                    let g = hidden_cell(muz, va);
                    assert!((*end - muz).len() >= 8.0, "seed {seed}: an anonymised shot ends {:.2} m from the hidden muzzle (kind {kind})", (*end - muz).len());
                    match kind {
                        0 => assert!(((*end - *from).len() - WD[*weapon as usize].range).abs() < 0.05, "a coarse end not range from the cell centre"),
                        3 => assert!((*end - *from).len() >= g + 8.0, "an exact wall hit {:.2} m from the cell centre", (*end - *from).len()),
                        _ => {}
                    }
                }
            }
            // the walls really are next to the shooter
            let d = dir_from(dq_yaw(k * 2730), 0.0);
            if m.w.raycast(eye(spot), d, 8.0).is_some() {
                near_walls += 1;
            }
        }
        assert!(shots >= 20, "seed {seed}: only {shots} shots heard");
        if near_walls >= 6 {
            cases += 1;
        }
    }
    assert!(cases >= 2, "only {cases} cases with walls near the shooter");
}

/// Resync delay after a hidden looter's change, with the viewer idle or
/// changing items in the same chunk every `probe` ticks.
fn resync_delay(seed: u32, probe: Option<u32>) -> Option<u32> {
    let (mut m, ids) = setup(seed, 2);
    let (v, l) = (ids[0], ids[1]);
    let (xid, loot, _) = indoor_case(&m, 60.0)?;
    let c = item_chunk(loot.x, loot.z);
    let cw = MAP / ICH as f32;
    let (x0, z0) = ((c % ICH) as f32 * cw, (c / ICH) as f32 * cw);
    let mut vpos = None;
    'find: for gz in 0..15 {
        for gx in 0..15 {
            let (x, z) = (x0 + 8.0 + gx as f32 * 8.0, z0 + 8.0 + gz as f32 * 8.0);
            if dist_xz(x, z, loot.x, loot.z) < 50.0 || m.w.height(x, z) < 2.5 || m.w.building_at(x, z).is_some() {
                continue;
            }
            let p = V3::new(x, m.w.ground_below(x, z, 300.0, P_RADIUS), z);
            if hidden(&m, p, loot) && m.w.blocked(eye(p), m.items[xid as usize].pos + V3::new(0.0, 0.25, 0.0)) {
                vpos = Some(p);
                break 'find;
            }
        }
    }
    let vpos = vpos?;
    let park = V3::new(if vpos.x < MAP * 0.5 { MAP - 20.0 } else { 20.0 }, 200.0, if vpos.z < MAP * 0.5 { MAP - 20.0 } else { 20.0 });
    place(&mut m, v, vpos);
    place(&mut m, l, park);
    {
        let p = &mut m.players[v as usize];
        p.slots = [(W_PISTOL, 10), (W_SMG, 10)];
        p.cur = 0;
    }
    let gun = m.items.iter().position(|it| it.alive && it.kind == W_AR && dist_xz(it.pos.x, it.pos.z, loot.x, loot.z) > 60.0)?;
    m.items[gun].pos = vpos + V3::new(0.4, 0.05, 0.0);
    for _ in 0..60 {
        m.players[l as usize].ms.pos = park;
        tick(&mut m, &[v]);
    }
    let resync = |s: &Snap| s.events.iter().any(|e| matches!(e, Ev::Chunk { chunk, .. } if *chunk as usize == c));
    let before = m.items[xid as usize];
    place(&mut m, l, loot);
    let mut seq_l = m.players[l as usize].last_seq;
    let mut looted_at = None;
    for _ in 0..12 {
        seq_l += 1;
        m.on_message(l, &cmd_bytes(&[(seq_l, m.tick * 256, 0, 0, 0, 0, if m.tick % 2 == 0 { B_PICKUP } else { 0 })]));
        tick(&mut m, &[v]);
        let now = m.items[xid as usize];
        if now.alive != before.alive || now.amount != before.amount {
            looted_at = Some(m.tick);
            break;
        }
    }
    let looted_at = looted_at?;
    m.players[l as usize].ms.pos = park;
    let mut seq_v = m.players[v as usize].last_seq;
    let mut swaps = 0;
    for k in 0..700u32 {
        m.players[l as usize].ms.pos = park;
        seq_v += 1;
        let b = match probe {
            Some(p) if k % p == 0 => B_PICKUP,
            _ => 0,
        };
        m.on_message(v, &cmd_bytes(&[(seq_v, m.tick * 256, 0, 0, 0, 0, b)]));
        let s = tick(&mut m, &[v]).swap_remove(0);
        swaps += s.events.iter().filter(|e| matches!(e, Ev::Item { .. })).count();
        if resync(&s) {
            if probe.is_some() {
                assert!(swaps >= 20, "seed {seed}: the viewer's swaps did not happen ({swaps})");
            }
            return Some(m.tick - looted_at);
        }
    }
    panic!("seed {seed}: the hidden change was never resynced");
}

/// The resync of a hidden change comes after a secret, jittered quiet
/// period. A viewer changing items of its own in the same chunk must not be
/// able to re-draw that period (and so bring the resync forward, dating the
/// hidden change more precisely).
#[test]
fn own_item_changes_do_not_redraw_the_quiet_period() {
    let mut cases = 0;
    for seed in [3u32, 11, 42, 77, 123, 5, 7, 9, 13, 21, 31, 55] {
        let Some(idle) = resync_delay(seed, None) else { continue };
        let probed = resync_delay(seed, Some(10)).unwrap();
        assert_eq!(idle, probed, "seed {seed}: the viewer's own swaps moved the resync ({idle} vs {probed} ticks)");
        cases += 1;
    }
    assert!(cases >= 3, "only {cases} cases");
}
