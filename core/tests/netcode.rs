//! Regression tests for the command-timing exploits found in the second
//! adversarial review: free fallback steps (speed hack), late pairs, hover by
//! starving, stale rewind slots, selective rewind and queue backlog.

use std::collections::VecDeque;
use zonefall_core::defs::*;
use zonefall_core::math::*;
use zonefall_core::net::*;
use zonefall_core::player::*;
use zonefall_core::rng::Rng;
use zonefall_core::sim::*;

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

fn two_humans(seed: u32) -> (Match, u8, u8) {
    let mut m = Match::new(seed, 2, FLAG_OFFLINE, 99);
    let a = m.add_human(b"A").unwrap();
    let b = m.add_human(b"B").unwrap();
    m.step();
    assert_eq!(m.phase, PH_PLAY);
    (m, a, b)
}

/// Flat open ground with 70 m of clear running room along +x.
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
                let h = m.w.height(x + k as f32, z);
                if (h - h0).abs() > 1.2 || h < 2.0 {
                    ok = false;
                    break;
                }
            }
            for y in [0.4f32, 1.0, 1.6] {
                if m.w.raycast(V3::new(x, h0 + y, z), V3::new(1.0, 0.0, 0.0), 75.0).is_some() {
                    ok = false;
                }
            }
            if ok {
                return V3::new(x, m.w.ground_below(x, z, h0 + 5.0, P_RADIUS), z);
            }
        }
    }
    panic!("no open ground");
}

fn place(m: &mut Match, pid: u8, pos: V3) {
    m.players[pid as usize].ms = MoveState { pos, mode: MODE_GROUND, on_ground: true, ..Default::default() };
}

fn steps(m: &Match, pid: u8) -> u32 {
    let p = &m.players[pid as usize];
    p.cmds_applied + p.idle_steps
}

/// A client that makes `gen(k)` commands at tick k and delivers `send(k, pending)` of them.
struct Client {
    seq: u32,
    pending: VecDeque<Q>,
}

impl Client {
    fn new(m: &Match, pid: u8) -> Client {
        Client { seq: m.players[pid as usize].last_seq, pending: VecDeque::new() }
    }
    fn tick(&mut self, m: &mut Match, pid: u8, k: u32, gen: usize, send: usize, mz: i8, buttons: u16, pitch: f32) {
        for _ in 0..gen {
            self.seq += 1;
            let view = m.tick.saturating_sub(4) * 256;
            self.pending.push_back((self.seq, view, 0, mz, q_yaw(yaw_to(1.0, 0.0)), q_pitch(pitch), buttons));
        }
        let n = send.min(self.pending.len());
        if n > 0 {
            let batch: Vec<Q> = self.pending.drain(..n).collect();
            for c in batch.chunks(8) {
                m.on_message(pid, &cmd_bytes(c));
            }
        }
        let _ = k;
        m.step();
    }
}

#[derive(Clone, Copy, Debug)]
enum Pat {
    Honest,
    Pairs,
    PairsOneBatchLate,
    Cycle0050,
    Flood3,
    EveryThird,
    RandomJitter,
}

const PATS: [Pat; 7] = [Pat::Honest, Pat::Pairs, Pat::PairsOneBatchLate, Pat::Cycle0050, Pat::Flood3, Pat::EveryThird, Pat::RandomJitter];

/// (commands made, commands delivered) at tick k.
fn pattern(p: Pat, k: u32, pending: usize, rng: &mut Rng) -> (usize, usize) {
    match p {
        Pat::Honest => (1, 1),
        Pat::Pairs => (1, if k % 2 == 1 { 2 } else { 0 }),
        // pairs, but one batch is held back once: from then on every
        // command arrives one batch late
        Pat::PairsOneBatchLate => (1, if k % 2 == 1 && k != 21 { 2 } else { 0 }),
        // more commands than ticks, delivered in bursts
        Pat::Cycle0050 => (if k % 4 == 2 { 5 } else { 0 }, usize::MAX),
        Pat::Flood3 => (3, usize::MAX),
        Pat::EveryThird => (if k % 3 == 0 { 1 } else { 0 }, usize::MAX),
        Pat::RandomJitter => (1, if rng.chance(0.3) { 0 } else { pending.max(1) }),
    }
}

#[test]
fn no_arrival_pattern_runs_faster_than_honest() {
    let mut honest = 0.0;
    for (pi, pat) in PATS.iter().enumerate() {
        let (mut m, a, _) = two_humans(42);
        let start = open_run(&m);
        place(&mut m, a, start);
        let mut cl = Client::new(&m, a);
        let mut rng = Rng::new(7 + pi as u64);
        let t0 = m.tick;
        let s0 = steps(&m, a);
        for k in 0..240u32 {
            let (g, s) = pattern(*pat, k, cl.pending.len(), &mut rng);
            cl.tick(&mut m, a, k, g, s, 127, B_SPRINT, 0.0);
            let elapsed = m.tick - t0;
            let st = steps(&m, a) - s0;
            assert!(st <= elapsed, "{pat:?}: {st} steps in {elapsed} ticks");
        }
        let dx = m.players[a as usize].ms.pos.x - start.x;
        if pi == 0 {
            honest = dx;
        }
        eprintln!("{pat:?}: ran {dx:.2} m (honest {honest:.2}), idle steps {}", m.players[a as usize].idle_steps);
        assert!(dx <= honest + 0.3, "{pat:?}: {dx:.2} m vs honest {honest:.2} m (speed hack)");
    }
}

#[test]
fn free_fall_takes_honest_time_for_every_pattern() {
    let mut honest = 0;
    for (pi, pat) in PATS.iter().enumerate() {
        let (mut m, a, _) = two_humans(42);
        let g = open_run(&m);
        m.players[a as usize].ms = MoveState { pos: g + V3::new(0.0, 260.0, 0.0), vel: V3::new(0.0, -5.0, 0.0), mode: MODE_FALL, ..Default::default() };
        let mut cl = Client::new(&m, a);
        let mut rng = Rng::new(70 + pi as u64);
        let mut landed = 0;
        for k in 0..2000u32 {
            let (gn, s) = pattern(*pat, k, cl.pending.len(), &mut rng);
            // dive (pitch down) without opening the chute early
            cl.tick(&mut m, a, k, gn, s, 0, 0, -1.2);
            if m.players[a as usize].ms.mode == MODE_GROUND {
                landed = k;
                break;
            }
        }
        assert!(landed > 0, "{pat:?}: never landed");
        if pi == 0 {
            honest = landed;
        }
        eprintln!("{pat:?}: landed after {landed} ticks (honest {honest})");
        let lo = honest as i64 - 2;
        let hi = honest as i64 + LAG_GRACE as i64 + 3;
        assert!((lo..=hi).contains(&(landed as i64)), "{pat:?}: landed after {landed} ticks, honest {honest}");
    }
}

#[test]
fn starving_never_freezes_or_slows_a_jump() {
    for every in [1u32, 2, 3, 4, 8, 30] {
        let (mut m, a, _) = two_humans(42);
        let start = open_run(&m);
        place(&mut m, a, start);
        let mut seq = m.players[a as usize].last_seq;
        seq += 1;
        m.on_message(a, &cmd_bytes(&[(seq, m.tick * 256, 0, 0, 0, 0, B_JUMP)]));
        let mut air = 0;
        let mut was_air = false;
        for k in 1..200u32 {
            if k % every == 0 {
                seq += 1;
                m.on_message(a, &cmd_bytes(&[(seq, m.tick * 256, 0, 0, 0, 0, 0)]));
            }
            m.step();
            let p = &m.players[a as usize];
            let h = p.ms.pos.y - m.w.ground_below(p.ms.pos.x, p.ms.pos.z, p.ms.pos.y, P_RADIUS);
            if h > 0.05 {
                air += 1;
                was_air = true;
            } else if was_air {
                break;
            }
        }
        eprintln!("one command every {every} ticks: airborne {air} ticks");
        // a jump lasts ~20 ticks; starving may delay it by at most the grace
        assert!(air <= 21 + LAG_GRACE as u32 + 1, "every {every}: airborne {air} ticks (hover)");
    }
}

#[test]
fn stall_burst_leaves_no_backlog_and_keeps_taps() {
    let (mut m, a, _) = two_humans(42);
    let start = open_run(&m);
    place(&mut m, a, start);
    let mut cl = Client::new(&m, a);
    for k in 0..30 {
        cl.tick(&mut m, a, k, 1, 1, 0, 0, 0.0);
    }
    // 12-tick network stall; the player taps jump during it
    for k in 0..12u32 {
        cl.seq += 1;
        let b = if k == 6 { B_JUMP } else { 0 };
        cl.pending.push_back((cl.seq, m.tick.saturating_sub(4) * 256, 0, 0, 0, 0, b));
        m.step();
    }
    let mut jumped = false;
    let sent = cl.seq;
    // the whole backlog arrives at once, then normal traffic
    for k in 0..6 {
        cl.tick(&mut m, a, k, 1, usize::MAX, 0, 0, 0.0);
        jumped |= m.players[a as usize].ms.vel.y > 0.5;
    }
    let p = &m.players[a as usize];
    assert!(jumped, "the jump tap inside the stalled commands was lost");
    assert!(p.cmdq.len() <= 1, "backlog of {} commands left after the stall", p.cmdq.len());
    assert!(cl.seq - p.acked_seq <= 2, "ack is {} commands behind", cl.seq - p.acked_seq);
    assert!(p.acked_seq >= sent);
    // and it stays drained
    for k in 0..120 {
        cl.tick(&mut m, a, k, 1, 1, 0, 0, 0.0);
        assert!(m.players[a as usize].cmdq.len() <= 1);
    }
}

#[test]
fn client_time_skip_recovers_grace() {
    let (mut m, a, _) = two_humans(42);
    let start = open_run(&m);
    place(&mut m, a, start);
    let mut cl = Client::new(&m, a);
    for k in 0..30 {
        cl.tick(&mut m, a, k, 1, 1, 0, 0, 0.0);
    }
    // background tab: nothing for 10 s, then commands resume (no burst)
    for _ in 0..300 {
        m.step();
    }
    let before = m.players[a as usize].cmds_applied;
    for k in 0..400 {
        cl.tick(&mut m, a, k, 1, 1, 0, 0, 0.0);
    }
    let p = &m.players[a as usize];
    assert_eq!(p.cmds_applied - before, 400, "fresh commands were dropped after the skip");
    assert_eq!(p.lag, 0, "grace never recovered (lag {})", p.lag);
    assert_eq!(p.debt, 0);
}

/// B stands at `p0` for a while, then moves 10 m sideways; A (a scoped
/// sniper 12 m away) fires at p0 `k` ticks after the move with `view`.
fn rewind_case(k: u32, view_of: impl Fn(u32) -> u32, prime_r: u32) -> f32 {
    let (mut m, a, b) = two_humans(42);
    let start = open_run(&m);
    place(&mut m, a, start);
    let p0 = V3::new(start.x + 12.0, m.w.ground_below(start.x + 12.0, start.z, start.y + 5.0, P_RADIUS), start.z);
    place(&mut m, b, p0);
    {
        let p = &mut m.players[a as usize];
        p.slots[0] = (W_SNIPER, 5);
        p.cur = 0;
        p.ammo = [200; NUM_AMMO];
    }
    let eye = start + V3::new(0.0, eye_h(false), 0.0);
    let chest = p0 + V3::new(0.0, 1.2, 0.0);
    let d = chest - eye;
    let yaw = q_yaw(yaw_to(d.x, d.z));
    let pitch = q_pitch((d.y / d.len_xz()).atan());
    let mut seq = m.players[a as usize].last_seq;
    // A plays normally for 2 s (establishes its usual rewind), aiming (ADS)
    for _ in 0..60 {
        seq += 1;
        let v = m.tick.saturating_sub(prime_r) * 256;
        m.on_message(a, &cmd_bytes(&[(seq, v, 0, 0, yaw, pitch, B_ADS)]));
        m.step();
        m.players[b as usize].ms.pos = p0;
    }
    // B leaves
    let p1 = p0 + V3::new(0.0, 0.0, 10.0);
    for t in 0..k {
        m.players[b as usize].ms.pos = p1;
        seq += 1;
        let v = m.tick.saturating_sub(prime_r) * 256;
        m.on_message(a, &cmd_bytes(&[(seq, v, 0, 0, yaw, pitch, B_ADS)]));
        m.step();
        let _ = t;
    }
    m.players[a as usize].fire_cd = 0.0;
    seq += 1;
    // the shot is simulated on tick m.tick + 1
    let view = view_of(m.tick + 1);
    m.on_message(a, &cmd_bytes(&[(seq, view, 0, 0, yaw, pitch, B_ADS | B_FIRE)]));
    m.step();
    m.players[b as usize].health
}

#[test]
fn rewind_never_reads_unrecorded_or_stale_history() {
    // control: an honest shot right after B left still hits where B was
    assert!(rewind_case(1, |t| (t - 2) * 256, 2) < 100.0, "control shot should hit");
    for k in [2u32, 8, 20, 31, 32, 33] {
        for (name, f) in [("u32::MAX", Box::new(|_t: u32| u32::MAX) as Box<dyn Fn(u32) -> u32>), ("now", Box::new(|t: u32| t * 256)), ("now-0.5", Box::new(|t: u32| t * 256 - 128))] {
            let hp = rewind_case(k, |t| f(t), 2);
            assert_eq!(hp, 100.0, "view {name}, {k} ticks after B left: hit a stale history slot");
        }
    }
}

#[test]
fn selective_rewind_is_clamped() {
    // A normally claims a 2-tick rewind; a single 12-tick claim must not reach
    // a position B left 10 ticks ago
    let hp = rewind_case(10, |t| (t - 12) * 256, 2);
    assert_eq!(hp, 100.0, "a low-latency client rewound 12 ticks for one shot");
    // a client that always renders 12 ticks behind (high ping) keeps its rewind
    let hp = rewind_case(10, |t| (t - 12) * 256, 12);
    assert!(hp < 100.0, "a steady high-latency client lost its lag compensation");
}

#[test]
fn holding_pickup_does_not_repeat_swaps() {
    let (mut m, a, _) = two_humans(42);
    let start = open_run(&m);
    place(&mut m, a, start);
    {
        let p = &mut m.players[a as usize];
        p.slots = [(W_PISTOL, 5), (W_SMG, 5)];
        p.cur = 0;
    }
    // a better gun on the floor
    let id = m.items.iter().position(|it| it.alive && it.kind == W_AR).unwrap_or(0);
    m.items[id].pos = start + V3::new(0.5, 0.05, 0.0);
    m.items[id].kind = W_AR;
    let mut seq = m.players[a as usize].last_seq;
    let kinds0 = m.players[a as usize].slots;
    let mut changes = 0;
    let mut last = kinds0;
    for _ in 0..90 {
        seq += 1;
        m.on_message(a, &cmd_bytes(&[(seq, m.tick * 256, 0, 0, q_yaw(yaw_to(1.0, 0.0)), 0, B_PICKUP)]));
        m.step();
        let s = m.players[a as usize].slots;
        if s != last {
            changes += 1;
            last = s;
        }
    }
    assert!(changes <= 1, "holding pickup swapped weapons {changes} times in 3 s");
}

#[test]
fn queue_overflow_keeps_taps() {
    let (mut m, a, _) = two_humans(42);
    let start = open_run(&m);
    place(&mut m, a, start);
    let mut seq = m.players[a as usize].last_seq;
    // 16 commands in one go (overflows the 8-command queue); the jump is early
    let mut v = Vec::new();
    for k in 0..16u32 {
        seq += 1;
        v.push((seq, m.tick * 256, 0i8, 0i8, 0u16, 0i16, if k == 1 { B_JUMP } else { 0 }));
    }
    m.on_message(a, &cmd_bytes(&v[..8]));
    m.on_message(a, &cmd_bytes(&v[8..]));
    let mut jumped = false;
    for _ in 0..12 {
        m.step();
        jumped |= m.players[a as usize].ms.vel.y > 0.5;
    }
    assert!(jumped, "a jump tap was lost when the queue overflowed");
}
