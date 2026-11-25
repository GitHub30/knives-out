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
    // up to 4 s: TCP retransmit backoff, a radio handover
    for stall in [12u32, 20, 30, 45, 60, 120] {
        let (mut m, a, _) = two_humans(42);
        let start = open_run(&m);
        place(&mut m, a, start);
        let mut cl = Client::new(&m, a);
        for k in 0..30 {
            cl.tick(&mut m, a, k, 1, 1, 0, 0, 0.0);
        }
        // network stall; the player taps jump during it
        for k in 0..stall {
            cl.seq += 1;
            let b = if k == stall / 2 { B_JUMP } else { 0 };
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
        assert!(jumped, "{stall}-tick stall: the jump tap inside the stalled commands was lost");
        assert!(p.cmdq.len() <= 1, "{stall}-tick stall: backlog of {} commands left", p.cmdq.len());
        assert!(cl.seq - p.acked_seq <= 2, "{stall}-tick stall: ack is {} commands behind", cl.seq - p.acked_seq);
        assert!(p.acked_seq >= sent);
        assert_eq!(p.debt, 0, "{stall}-tick stall: debt left after the burst");
        // and it stays drained
        for k in 0..120 {
            cl.tick(&mut m, a, k, 1, 1, 0, 0, 0.0);
            assert!(m.players[a as usize].cmdq.len() <= 1);
        }
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
    // A view at or ahead of the server tick is simulated at the client's
    // floor (1 tick after prime 0, 3 ticks after prime 2): never at an
    // unrecorded or stale slot.
    for (prime, ks) in [(0u32, [2u32, 8, 20, 31, 32, 33]), (2, [5, 8, 20, 31, 32, 33])] {
        for k in ks {
            for (name, f) in [("u32::MAX", Box::new(|_t: u32| u32::MAX) as Box<dyn Fn(u32) -> u32>), ("now", Box::new(|t: u32| t * 256)), ("now-0.5", Box::new(|t: u32| t * 256 - 128))] {
                let hp = rewind_case(k, |t| f(t), prime);
                assert_eq!(hp, 100.0, "view {name}, prime {prime}, {k} ticks after B left: hit a stale history slot");
            }
        }
    }
    // ... and a future view still gets the floor's lag compensation
    assert!(rewind_case(2, |_t| u32::MAX, 2) < 100.0, "a future view lost the 3-tick floor");
}

/// Views ahead of the server tick on most commands leave the floor where it
/// is; they must then also be rewound by the floor, or a client gets a 1-tick
/// rewind on every shot and a 15-tick one on the shots it chooses.
#[test]
fn future_views_do_not_reopen_selective_rewind() {
    for ahead in [u32::MAX, 40 * 256] {
        let (mut m, a, _) = two_humans(42);
        let start = open_run(&m);
        place(&mut m, a, start);
        for _ in 0..20 {
            m.step();
        }
        let mut seq = m.players[a as usize].last_seq;
        let mut applied = vec![];
        for i in 0..150u32 {
            seq += 1;
            // the command is simulated on tick m.tick + 1
            let now = (m.tick + 1) * 256;
            let v = if i % 5 == 4 { now - 15 * 256 } else { now.saturating_add(ahead) };
            m.on_message(a, &cmd_bytes(&[(seq, v, 0, 0, 0, 0, 0)]));
            m.step();
            applied.push(m.tick as f32 - m.players[a as usize].last_cmd.view as f32 / 256.0);
        }
        let lo = applied[60..].iter().cloned().fold(f32::MAX, f32::min);
        let hi = applied[60..].iter().cloned().fold(f32::MIN, f32::max);
        assert!(hi - lo <= 4.1, "applied rewinds span {lo:.2}..{hi:.2} ticks");
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

// ---------------------------------------------------------------------------
// Third adversarial review

/// The uplink delay drops (bufferbloat clears): the commands that were in
/// flight arrive early. Lag accounting can't simulate them faster, so they
/// must be trimmed, not left as a standing queue (input latency) for the rest
/// of the match.
#[test]
fn delay_drop_leaves_no_standing_backlog() {
    for (hi, lo) in [(8u32, 2u32), (5, 2), (4, 1)] {
        let (mut m, a, _) = two_humans(42);
        let start = open_run(&m);
        place(&mut m, a, start);
        let mut seq = m.players[a as usize].last_seq;
        // (seq, tick made)
        let mut flight: VecDeque<(u32, u32)> = VecDeque::new();
        let (mut qsum, mut lsum, mut n) = (0usize, 0u32, 0u32);
        for k in 0..900u32 {
            seq += 1;
            flight.push_back((seq, k));
            let delay = if k < 150 { hi } else { lo };
            // online client: two commands per message, sent every other tick
            if k % 2 == 1 {
                let mut batch = vec![];
                while let Some(&(s, made)) = flight.front() {
                    if made + delay > k {
                        break;
                    }
                    flight.pop_front();
                    batch.push((s, m.tick.saturating_sub(4) * 256, 0i8, 0i8, 0u16, 0i16, 0u16));
                }
                for c in batch.chunks(8) {
                    m.on_message(a, &cmd_bytes(c));
                }
            }
            m.step();
            if k >= 600 {
                let p = &m.players[a as usize];
                qsum += p.cmdq.len();
                lsum += seq - p.acked_seq;
                n += 1;
            }
        }
        let q = qsum as f32 / n as f32;
        let behind = lsum as f32 / n as f32;
        eprintln!("delay {hi} -> {lo} ticks: queue {q:.2}, ack {behind:.2} commands behind (network delay {lo})");
        assert!(q <= 1.0, "delay {hi} -> {lo}: standing queue of {q:.2} commands");
        assert!(behind <= lo as f32 + 1.5, "delay {hi} -> {lo}: ack {behind:.2} commands behind");
    }
}

/// Commands pushed out of a full queue are the late ones that idle steps
/// already covered: they pay the debt, so no stale debt is left to drop (and
/// to tell the client to skip) commands that are not late.
#[test]
fn queue_overflow_pays_debt() {
    let (mut m, a, _) = two_humans(42);
    let start = open_run(&m);
    place(&mut m, a, start);
    let mut cl = Client::new(&m, a);
    for k in 0..30 {
        cl.tick(&mut m, a, k, 1, 1, 0, 0, 0.0);
    }
    // 20-tick uplink stall, then the whole burst lands at once
    for _ in 0..20 {
        cl.seq += 1;
        cl.pending.push_back((cl.seq, m.tick.saturating_sub(4) * 256, 0, 0, 0, 0, 0));
        m.step();
    }
    assert!(m.players[a as usize].debt >= 12, "debt {}", m.players[a as usize].debt);
    cl.tick(&mut m, a, 0, 1, usize::MAX, 0, 0, 0.0);
    for k in 0..3 {
        cl.tick(&mut m, a, k, 1, 1, 0, 0, 0.0);
    }
    let p = &m.players[a as usize];
    assert!(p.debt <= 1, "stale debt of {} after the late commands were dropped", p.debt);
    assert!(p.cmds_applied + p.idle_steps <= m.tick, "more steps than ticks");
}

/// After a server hitch the catch-up ticks simulate commands whose view is
/// ahead of the server tick. They say nothing about the client's latency and
/// must not cut an honest high-latency player's lag compensation.
#[test]
fn server_hitch_keeps_rewind_floor() {
    let (mut m, a, _) = two_humans(42);
    let start = open_run(&m);
    place(&mut m, a, start);
    let mut seq = m.players[a as usize].last_seq;
    fn send(m: &mut Match, a: u8, seq: &mut u32, view: u32) {
        *seq += 1;
        m.on_message(a, &cmd_bytes(&[(*seq, view, 0, 0, 0, 0, 0)]));
        m.step();
    }
    let rew = |m: &Match| m.tick as f32 - m.players[a as usize].last_cmd.view as f32 / 256.0;
    for _ in 0..20 {
        m.step();
    }
    // a high-latency client: renders 9 ticks behind the tick that simulates it
    for _ in 0..120 {
        let v = (m.tick + 1 - 9) * 256;
        send(&mut m, a, &mut seq, v);
    }
    assert!((rew(&m) - 9.0).abs() < 0.01, "rewind {}", rew(&m));
    // server hitch: commands made while it stood still are simulated in
    // catch-up ticks, with views up to a few ticks ahead; they keep the
    // player's lag compensation
    for _ in 0..6 {
        let v = (m.tick + 1 + 3) * 256;
        send(&mut m, a, &mut seq, v);
        assert!((rew(&m) - 9.0).abs() < 0.01, "a catch-up command was rewound {:.2} ticks", rew(&m));
    }
    let v = (m.tick + 1 - 9) * 256;
    send(&mut m, a, &mut seq, v);
    assert!((rew(&m) - 9.0).abs() < 0.01, "rewind cut to {:.2} ticks after a server hitch", rew(&m));
}

/// Client prediction across uplink stalls (TCP retransmits, radio hiccups).
/// The server stands in for late commands with idle steps and drops as many
/// late commands when they arrive; the snapshot says how many (debt), so the
/// client must not replay them. Mirrors `Game.reconcile` in src/game.ts.
fn stall_run(mode: u8, seed: u32, hold_ms: f64) -> (u32, u32, f32) {
    let mut m = Match::new(42 + seed, 2, 0, 99);
    let a = m.add_human(b"A").unwrap();
    let _b = m.add_human(b"B").unwrap();
    m.step();
    assert_eq!(m.phase, PH_PLAY);
    let g = open_run(&m);
    m.players[a as usize].ms = match mode {
        MODE_FALL => MoveState { pos: g + V3::new(0.0, 700.0, 0.0), vel: V3::new(0.0, -5.0, 0.0), mode: MODE_FALL, ..Default::default() },
        MODE_PARA => MoveState { pos: g + V3::new(0.0, 125.0, 0.0), vel: V3::new(0.0, -6.0, 0.0), mode: MODE_PARA, ..Default::default() },
        _ => MoveState { pos: g, mode: MODE_GROUND, on_ground: true, ..Default::default() },
    };
    let yaw = q_yaw(yaw_to(1.0, 0.0));
    let mut rng = Rng::new(1000 + seed as u64 * 7 + mode as u64);
    let (up, down, jit) = (25.0f64, 25.0f64, 10.0f64);
    let tick_ms = 1000.0 / 30.0;
    let frame_ms = 1000.0 / 60.0;
    let (mut next_tick, mut next_frame) = (tick_ms, 0.0f64);
    // uplink messages and downlink snapshots (tick, ack, debt, state, speed)
    let mut up_q: VecDeque<(f64, Vec<u8>)> = VecDeque::new();
    let mut down_q: VecDeque<(f64, (u32, u32, u8, MoveState, f32))> = VecDeque::new();
    let (mut last_up, mut last_down) = (0.0f64, 0.0f64);
    let mut seq = m.players[a as usize].last_seq;
    // pending (seq, command, made at ms)
    let mut pending: VecDeque<(u32, Q, f64)> = VecDeque::new();
    let mut outbox: Vec<Q> = Vec::new();
    let mut pred: Option<MoveState> = None;
    let mut mult = 1.0f32;
    let (mut cmd_acc, mut last_frame, mut latest_tick) = (0.0f64, 0.0f64, 0u32);
    let rtt = up + down + jit;
    let holds: Vec<f64> = (0..3).map(|k| 2500.0 + k as f64 * (hold_ms + 2100.0)).collect();
    let end = 2500.0 + 3.0 * (hold_ms + 2100.0);
    let (mut corrections, mut hard, mut worst) = (0u32, 0u32, 0.0f32);
    loop {
        let t;
        if next_tick <= next_frame {
            t = next_tick;
            next_tick += tick_ms;
            while up_q.front().map_or(false, |(at, _)| *at <= t) {
                let (_, msg) = up_q.pop_front().unwrap();
                m.on_message(a, &msg);
            }
            if m.step() & 1 != 0 {
                // the client reads ack and debt off the wire
                let buf = m.snapshot(a).to_vec();
                let ack = u32::from_le_bytes([buf[5], buf[6], buf[7], buf[8]]);
                let debt = buf[17 + 81];
                let p = &m.players[a as usize];
                assert_eq!((ack, debt), (p.acked_seq, p.debt), "snapshot ack/debt offsets");
                let at = (t + down + rng.f() as f64 * jit).max(last_down);
                last_down = at;
                down_q.push_back((at, (m.tick, ack, debt, p.ms, p.speed_mult())));
            }
        } else {
            t = next_frame;
            next_frame += frame_ms;
            while down_q.front().map_or(false, |(at, _)| *at <= t) {
                let (_, (tick, ack, debt, ms, sm)) = down_q.pop_front().unwrap();
                if tick <= latest_tick {
                    continue;
                }
                latest_tick = tick;
                while pending.front().map_or(false, |q| q.0 <= ack) {
                    pending.pop_front();
                }
                let old = pred.map(|p| p.pos);
                let mut s = ms;
                mult = sm;
                let late = t - (rtt + 100.0);
                let mut skip = 0;
                while skip < debt as usize && skip < pending.len() && pending[skip].2 < late {
                    skip += 1;
                }
                for (_, q, _) in pending.iter().skip(skip) {
                    step_move(&m.w, &mut s, &Cmd::from_quant(0, 0, q.2, q.3, q.4, q.5, q.6), mult, DT);
                }
                if let (Some(o), true) = (old, ms.mode == mode) {
                    let d = (o - s.pos).len();
                    if d > 1e-3 {
                        corrections += 1;
                    }
                    if d >= 3.0 {
                        hard += 1;
                    }
                    worst = worst.max(d);
                }
                pred = Some(s);
            }
            // commands at 30 Hz, two per message
            cmd_acc += t - last_frame;
            last_frame = t;
            let mut n = 0;
            while cmd_acc >= tick_ms && n < 4 {
                cmd_acc -= tick_ms;
                n += 1;
                seq += 1;
                let q: Q = (seq, latest_tick.saturating_sub(4) * 256, 0, 127, yaw, 0, 0);
                pending.push_back((seq, q, t));
                outbox.push(q);
                if let Some(p) = pred.as_mut() {
                    step_move(&m.w, p, &Cmd::from_quant(0, 0, q.2, q.3, q.4, q.5, q.6), mult, DT);
                }
            }
            if outbox.len() >= 2 {
                let mut at = t + up + rng.f() as f64 * jit;
                for &h in &holds {
                    if t >= h && t < h + hold_ms {
                        at = at.max(h + hold_ms);
                    }
                }
                at = at.max(last_up);
                last_up = at;
                up_q.push_back((at, cmd_bytes(&outbox)));
                outbox.clear();
            }
        }
        if t > end || (m.players[a as usize].ms.mode != mode && t > 1000.0) {
            break;
        }
    }
    (corrections, hard, worst)
}

#[test]
fn uplink_stalls_do_not_snap_the_prediction() {
    // While the uplink is stalled the server holds the player still (after a
    // few repeated steps) and the client is pulled back a snapshot's worth of
    // movement at a time; a longer stall adds no larger correction.
    for hold in [400.0, 1000.0, 2500.0] {
        let ground = if hold <= 400.0 { 0.6 } else { 0.8 };
        for (mode, name, limit) in [(MODE_FALL, "freefall", 3.0f32), (MODE_PARA, "parachute", 1.2), (MODE_GROUND, "ground", ground)] {
            for seed in 0..2 {
                let (corr, hard, worst) = stall_run(mode, seed, hold);
                eprintln!("{hold} ms {name} seed {seed}: {corr} corrections, {hard} hard snaps, worst {worst:.2} m");
                assert_eq!(hard, 0, "{hold} ms {name}: {hard} hard snaps (worst {worst:.2} m)");
                assert!(worst < limit, "{name}: prediction corrected by {worst:.2} m after {hold} ms uplink stalls");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fifth adversarial review

/// A server hitch: the Room loop runs the missed ticks late, up to 5 per
/// callback. The commands simulated in those catch-up steps carry views made
/// in real time, so they look less rewound than they are; they must not pull
/// an honest high-latency player's rewind floor down.
#[test]
fn server_hitch_catch_up_keeps_rewind() {
    for hitch in [5u32, 10, 12, 15, 18] {
        let (mut m, a, _) = two_humans(42);
        let start = open_run(&m);
        place(&mut m, a, start);
        let mut w = m.tick; // wall-clock tick
        let mut seq = m.players[a as usize].last_seq;
        let (mut out, mut held): (Vec<Q>, Vec<Vec<u8>>) = (vec![], vec![]);
        let h0 = w + 120;
        let (mut rmin, mut fmin) = (f32::MAX, f32::MAX);
        let mut applied = m.players[a as usize].cmds_applied;
        for _ in 0..400 {
            w += 1;
            seq += 1;
            // the client renders 9 ticks behind real time
            out.push((seq, w.saturating_sub(9) * 256, 0, 0, 0, 0, 0));
            if out.len() == 2 {
                held.push(cmd_bytes(&out));
                out.clear();
            }
            if w >= h0 && w < h0 + hitch {
                continue; // the server stands still
            }
            for msg in held.drain(..) {
                m.on_message(a, &msg);
            }
            let mut steps = 0;
            while m.tick < w && steps < 5 {
                m.catch_up = w - m.tick > 1;
                m.step();
                steps += 1;
                let p = &m.players[a as usize];
                if p.cmds_applied != applied {
                    applied = p.cmds_applied;
                    if w >= h0 {
                        rmin = rmin.min(m.tick as f32 - p.last_cmd.view as f32 / 256.0);
                        fmin = fmin.min(p.rew_floor);
                    }
                }
            }
        }
        m.catch_up = false;
        eprintln!("{} ms hitch: min applied rewind {rmin:.2}, min floor {fmin:.2}", hitch * 100 / 3);
        assert!(rmin >= 8.5 && fmin >= 8.5, "{} ms hitch: rewind cut to {rmin:.2} ticks (floor {fmin:.2})", hitch * 100 / 3);
    }
}

/// The floor falls at most one tick per command: a client that keeps a high
/// floor cannot take a 15-tick shot and a 1-tick shot back to back (a target
/// behind cover and a fresh peeker), and an honest latency drop still
/// converges within a few hundred milliseconds.
#[test]
fn rewind_cannot_drop_from_floor_to_one_tick() {
    let (mut m, a, _) = two_humans(42);
    let start = open_run(&m);
    place(&mut m, a, start);
    for _ in 0..20 {
        m.step();
    }
    let mut seq = m.players[a as usize].last_seq;
    let mut claim = |m: &mut Match, ticks: u32| -> f32 {
        seq += 1;
        let v = (m.tick + 1 - ticks) * 256;
        m.on_message(a, &cmd_bytes(&[(seq, v, 0, 0, 0, 0, 0)]));
        m.step();
        m.tick as f32 - m.players[a as usize].last_cmd.view as f32 / 256.0
    };
    for _ in 0..40 {
        claim(&mut m, 15);
    }
    assert!((claim(&mut m, 15) - 15.0).abs() < 0.01);
    let r = claim(&mut m, 1);
    assert!(r >= 13.9, "a 1-tick claim right after 15-tick ones was applied at {r:.2} ticks");
    // an honest latency drop from 15 to 3 ticks converges in 12 commands
    let mut last = 0.0;
    for _ in 0..12 {
        last = claim(&mut m, 3);
    }
    assert!((last - 3.0).abs() < 0.01, "still rewound {last:.2} ticks after the drop");
}
