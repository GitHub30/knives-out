//! Bot AI. Bots produce the same `Cmd`s humans send, so they obey identical rules.

use crate::defs::*;
use crate::math::*;
use crate::player::*;
use crate::rng::Rng;
use crate::sim::{Match, Plane, Player, SCOPE_ZOOM};
use crate::world::*;

#[derive(Clone, Default)]
pub struct Brain {
    pub rng: Rng,
    pub skill: f32,
    pub jump_t: f32,
    pub land: (f32, f32),
    pub goal: (f32, f32),
    pub goal_kind: u8, // 0 none, 1 item, 2 roam, 3 zone
    pub goal_item: u16,
    pub goal_t: f32,
    pub path: [(f32, f32); 5],
    pub path_n: u8,
    pub path_i: u8,
    pub target: u8,
    pub seen_t: f32,
    pub last_seen: (f32, f32),
    pub aim_ex: f32,
    pub aim_ey: f32,
    pub aim_tx: f32,
    pub aim_ty: f32,
    pub aim_t: f32,
    pub strafe: f32,
    pub strafe_t: f32,
    pub last_pos: (f32, f32),
    pub check_t: f32,
    pub unstuck_t: f32,
    pub unstuck_dir: f32,
    pub stuck_n: u8,
    pub yaw: f32,
    pub pitch: f32,
    pub burst_t: f32,
    pub pause_t: f32,
    pub heard: (f32, f32),
    pub heard_t: f32,
    pub crouch: bool,
    pub jump_cd: f32,
    pub think_t: f32,
    pub avoid: f32,
    pub avoid_t: f32,
    pub slot: u8,
    /// Items this bot gave up on (unreachable), ring buffer.
    pub ignore: [u16; 8],
    pub ignore_i: u8,
    /// Time spent standing next to the goal item without getting it.
    pub near_t: f32,
    /// Closest the bot has got to its goal item, and time since it improved.
    pub prog_d: f32,
    pub prog_t: f32,
    /// Time since landing: fresh bots loot first and only take close fights.
    pub ground_t: f32,
}

impl Brain {
    pub fn new(seed: u32, skill: f32) -> Brain {
        Brain { rng: Rng::new(seed as u64 | 1), skill, target: 255, slot: 2, ignore: [u16::MAX; 8], ..Default::default() }
    }
}

#[inline]
fn turn_toward(cur: f32, target: f32, max: f32) -> f32 {
    wrap_angle(cur + clampf(wrap_angle(target - cur), -max, max))
}

/// Choose a landing target and when to leave the plane. Spots far from the
/// landings other bots already chose are preferred, so the island does not
/// empty in a few landing brawls.
pub fn pick_landing(w: &World, pl: &Plane, b: &mut Brain, taken: &[(f32, f32)]) -> (f32, f32, f32) {
    let mut best: Option<(f32, (f32, f32, f32))> = None;
    let spread = 70.0;
    for _ in 0..14 {
        let (x, z) = if !w.buildings.is_empty() && b.rng.chance(0.85) {
            let k = b.rng.below(w.buildings.len() as u32) as usize;
            let bl = &w.buildings[k];
            // aim for the ground outside the door (never over the roof)
            let (ox, oz) = (bl.door_out.0 - bl.door_in.0, bl.door_out.1 - bl.door_in.1);
            let l = (ox * ox + oz * oz).sqrt().max(1e-3);
            let (nx, nz) = (ox / l, oz / l);
            let out = b.rng.range(0.5, 4.0);
            let side = b.rng.range(-3.0, 3.0);
            (bl.door_out.0 + nx * out - nz * side, bl.door_out.1 + nz * out + nx * side)
        } else {
            (b.rng.range(150.0, MAP - 150.0), b.rng.range(150.0, MAP - 150.0))
        };
        let rx = x - pl.sx;
        let rz = z - pl.sz;
        let along = rx * pl.dx + rz * pl.dz;
        let perp = (rx * pl.dz - rz * pl.dx).abs();
        if perp > 260.0 || w.height(x, z) < 1.0 {
            continue;
        }
        let lead = (260.0f32 * 260.0 - perp * perp).max(0.0).sqrt() * b.rng.range(0.15, 0.5);
        let jt = clampf(along - lead, pl.enter_t + 5.0, pl.exit_t - 5.0);
        let near = taken.iter().map(|&(tx, tz)| dist_xz(tx, tz, x, z)).fold(f32::MAX, f32::min);
        if near >= spread {
            return (x, z, jt);
        }
        if best.map_or(true, |(d, _)| near > d) {
            best = Some((near, (x, z, jt)));
        }
    }
    if let Some((_, r)) = best {
        return r;
    }
    let t = b.rng.range(pl.enter_t + 20.0, pl.exit_t - 20.0);
    let side = if b.rng.chance(0.5) { 1.0 } else { -1.0 } * b.rng.range(40.0, 180.0);
    let x = clampf(pl.sx + pl.dx * t + pl.dz * side, 80.0, MAP - 80.0);
    let z = clampf(pl.sz + pl.dz * t - pl.dx * side, 80.0, MAP - 80.0);
    (x, z, t)
}

pub fn think(m: &mut Match, i: usize) -> Cmd {
    let mut b = std::mem::take(&mut m.players[i].brain);
    let c = logic(m, i, &mut b);
    m.players[i].brain = b;
    c
}

/// Set mx/mz so the bot walks toward world direction (dx, dz) while facing `yaw`.
fn move_dir(c: &mut Cmd, yaw: f32, dx: f32, dz: f32, amount: f32) {
    let l = (dx * dx + dz * dz).sqrt();
    if l < 1e-4 {
        return;
    }
    let (dx, dz) = (dx / l, dz / l);
    let f = fwd(yaw);
    let r = right(yaw);
    c.mx = (dx * r.x + dz * r.z) * amount;
    c.mz = (dx * f.x + dz * f.z) * amount;
}

fn logic(m: &Match, i: usize, b: &mut Brain) -> Cmd {
    let p = &m.players[i];
    let mut c = Cmd { view: m.tick.wrapping_mul(256), yaw: b.yaw, pitch: b.pitch, ..Default::default() };
    c.buttons |= ((b.slot as u16 + 1) & 3) << 8;
    match p.ms.mode {
        MODE_PLANE => {
            b.yaw = p.yaw;
            c.yaw = b.yaw;
            if m.plane.can_jump() && m.plane.t >= b.jump_t {
                c.buttons |= B_JUMP;
            }
            c
        }
        MODE_FALL | MODE_PARA => {
            let dx = b.land.0 - p.ms.pos.x;
            let dz = b.land.1 - p.ms.pos.z;
            let d = (dx * dx + dz * dz).sqrt();
            b.yaw = turn_toward(b.yaw, yaw_to(dx, dz), 2.5 * DT);
            c.yaw = b.yaw;
            let alt = p.ms.pos.y - m.w.height(p.ms.pos.x, p.ms.pos.z);
            let glide_reach = alt * if p.ms.mode == MODE_FALL { 0.9 } else { 1.4 };
            c.pitch = if p.ms.mode == MODE_FALL && d < glide_reach * 0.5 { -0.9 } else { -0.2 };
            if d > 3.0 {
                c.mz = 1.0;
            }
            b.pitch = c.pitch;
            c
        }
        MODE_GROUND => {
            ground(m, i, b, &mut c);
            c
        }
        _ => c,
    }
}

fn weapon_rank(wk: u8) -> u8 {
    WD[wk as usize].rank
}

fn slot_usable(p: &Player, s: usize) -> bool {
    let (wk, mag) = p.slots[s];
    wk != 0 && (mag > 0 || p.ammo[WD[wk as usize].ammo as usize] > 0)
}

fn pick_slot(p: &Player, dist: f32) -> u8 {
    let mut best = 2u8;
    let mut best_score = -1.0f32;
    for s in 0..2 {
        if !slot_usable(p, s) {
            continue;
        }
        let wk = p.slots[s].0;
        let mut score = weapon_rank(wk) as f32;
        match wk {
            W_SHOTGUN | W_SMG if dist < 18.0 => score += 4.0,
            W_SHOTGUN if dist > 35.0 => score -= 4.0,
            W_SNIPER | W_DMR if dist > 70.0 => score += 3.0,
            W_SNIPER if dist < 20.0 => score -= 3.0,
            _ => {}
        }
        if p.slots[s].1 == 0 {
            score -= 1.5;
        }
        if score > best_score {
            best_score = score;
            best = s as u8;
        }
    }
    best
}

fn find_target(m: &Match, i: usize, b: &Brain) -> u8 {
    let p = &m.players[i];
    let eye = p.ms.pos + V3::new(0.0, eye_h(p.ms.crouch), 0.0);
    let view = 80.0 + 110.0 * b.skill;
    let f = fwd(b.yaw);
    let mut best = (255u8, f32::MAX);
    for j in 0..MAX_PLAYERS {
        if j == i {
            continue;
        }
        let q = &m.players[j];
        if !q.used || !q.alive || q.ms.mode == MODE_PLANE {
            continue;
        }
        let tp = q.ms.pos + V3::new(0.0, if q.ms.crouch { 0.9 } else { 1.25 }, 0.0);
        let to = tp - eye;
        let d = to.len();
        let mut range = if q.firing_t > 0.0 { view + 90.0 } else { view };
        if q.ms.crouch {
            range *= 0.75;
        }
        if q.ms.mode != MODE_GROUND {
            range = range.min(130.0);
        }
        if d > range || d >= best.1 {
            continue;
        }
        let lxz = to.len_xz().max(0.01);
        let cosang = (to.x * f.x + to.z * f.z) / lxz;
        let current = j as u8 == b.target;
        let attacker = p.hurt_by as usize == j && p.hurt_t > 0.0;
        // right after landing, only close threats are worth a fight
        if b.ground_t < 45.0 && !current && !attacker && d > 30.0 + b.ground_t * 2.0 {
            continue;
        }
        if d > 10.0 && cosang < 0.2 && !current && !attacker {
            continue;
        }
        if m.w.blocked(eye, tp) {
            continue;
        }
        best = (j as u8, d);
    }
    best.0
}

fn useful(p: &Player, kind: u8, amount: u16) -> bool {
    if is_weapon(kind) {
        let worst = weapon_rank(p.slots[0].0).min(weapon_rank(p.slots[1].0));
        let empty = p.slots[0].0 == 0 || p.slots[1].0 == 0;
        let dup = p.slots.iter().any(|s| s.0 == kind);
        return (empty && !dup) || weapon_rank(kind) > worst + 1;
    }
    if (IK_AMMO..IK_AMMO + NUM_AMMO as u8).contains(&kind) {
        let a = (kind - IK_AMMO) as usize;
        let uses = p.slots.iter().any(|s| s.0 != 0 && WD[s.0 as usize].ammo as usize == a);
        return uses && p.ammo[a] < AMMO_CAP[a] * 2 / 3;
    }
    match kind {
        IK_BANDAGE => p.heals[0] < 8,
        IK_MEDKIT => p.heals[1] < 2,
        IK_DRINK => p.heals[2] < 2,
        k if (IK_VEST..IK_VEST + 3).contains(&k) => k - IK_VEST + 1 > p.vest,
        k if (IK_HELM..IK_HELM + 3).contains(&k) => k - IK_HELM + 1 > p.helm,
        IK_NADE => p.nades < 2 && amount > 0,
        _ => false,
    }
}

fn find_loot(m: &Match, i: usize, radius: f32, ignore: &[u16]) -> Option<u16> {
    let p = &m.players[i];
    let pos = p.ms.pos;
    let unarmed = p.slots[0].0 == 0 && p.slots[1].0 == 0;
    let mut best: Option<(u16, f32)> = None;
    for (id, it) in m.items.iter().enumerate() {
        if !it.alive || ignore.contains(&(id as u16)) {
            continue;
        }
        let d = dist_xz(it.pos.x, it.pos.z, pos.x, pos.z);
        if d > radius || (it.pos.y - pos.y).abs() > 6.0 {
            continue;
        }
        if !useful(p, it.kind, it.amount) {
            continue;
        }
        let zone_ok = dist_xz(it.pos.x, it.pos.z, m.zone.cx, m.zone.cz) < m.zone.r;
        if !zone_ok {
            continue;
        }
        let mut score = d;
        if is_weapon(it.kind) && unarmed {
            score -= 40.0;
        }
        if best.map_or(true, |b| score < b.1) {
            best = Some((id as u16, score));
        }
    }
    best.map(|b| b.0)
}

fn plan_path(m: &Match, b: &mut Brain, from: V3, to: (f32, f32), kind: u8) {
    let same = kind == b.goal_kind && dist_xz(to.0, to.1, b.goal.0, b.goal.1) < 2.0;
    if !same {
        b.goal_t = 0.0;
        b.near_t = 0.0;
        b.prog_d = f32::MAX;
        b.prog_t = 0.0;
    }
    b.goal = to;
    b.goal_kind = kind;
    b.path_n = 0;
    b.path_i = 0;
    let push = |b: &mut Brain, pt: (f32, f32)| {
        if (b.path_n as usize) < b.path.len() {
            b.path[b.path_n as usize] = pt;
            b.path_n += 1;
        }
    };
    let bf = m.w.building_at(from.x, from.z);
    let bt = m.w.building_at(to.0, to.1);
    if bf != bt {
        if let Some(k) = bf {
            let bl = m.w.buildings[k];
            push(b, bl.door_in);
            push(b, bl.door_out);
        }
        if let Some(k) = bt {
            let bl = m.w.buildings[k];
            push(b, bl.door_out);
            push(b, bl.door_in);
        }
    }
    push(b, to);
}

fn choose_goal(m: &Match, i: usize, b: &mut Brain) {
    let p = &m.players[i];
    let pos = p.ms.pos;
    let z = &m.zone;
    let in_cur = dist_xz(pos.x, pos.z, z.cx, z.cz) < z.r - 4.0;
    let in_next = dist_xz(pos.x, pos.z, z.nx, z.nz) < z.nr - 4.0;
    let urgent = !in_cur || (!in_next && (z.shrinking || z.timer < 35.0 + 20.0 * b.skill));
    if urgent {
        let a = b.rng.range(0.0, TAU);
        let rr = b.rng.f().sqrt() * z.nr * 0.55;
        let tx = clampf(z.nx + a.cos() * rr, 10.0, MAP - 10.0);
        let tz = clampf(z.nz + a.sin() * rr, 10.0, MAP - 10.0);
        plan_path(m, b, pos, (tx, tz), 3);
        return;
    }
    // stranded on a roof: walk off the nearest edge first
    if let Some(k) = m.w.building_at(pos.x, pos.z) {
        let bl = m.w.buildings[k];
        if pos.y > bl.y + 2.0 {
            let cands = [(bl.x0 - 2.0, pos.z), (bl.x1 + 2.0, pos.z), (pos.x, bl.z0 - 2.0), (pos.x, bl.z1 + 2.0)];
            let mut best = cands[0];
            for c in cands {
                if dist_xz(c.0, c.1, pos.x, pos.z) < dist_xz(best.0, best.1, pos.x, pos.z) {
                    best = c;
                }
            }
            b.goal_kind = 0;
            b.path_n = 1;
            b.path_i = 0;
            b.path[0] = (clampf(best.0, 5.0, MAP - 5.0), clampf(best.1, 5.0, MAP - 5.0));
            b.goal = b.path[0];
            return;
        }
    }
    let ignore = b.ignore;
    // the first minute on the ground: loot around the landing spot
    let loot_r = if b.ground_t < 60.0 { 30.0 } else { 65.0 };
    if let Some(id) = find_loot(m, i, loot_r, &ignore) {
        let it = m.items[id as usize];
        b.goal_item = id;
        plan_path(m, b, pos, (it.pos.x, it.pos.z), 1);
        return;
    }
    // roam to a building inside the next circle, or toward the circle
    let mut picked = None;
    for _ in 0..8 {
        if m.w.buildings.is_empty() {
            break;
        }
        let k = b.rng.below(m.w.buildings.len() as u32) as usize;
        let bl = &m.w.buildings[k];
        let cx = (bl.x0 + bl.x1) * 0.5;
        let cz = (bl.z0 + bl.z1) * 0.5;
        let d = dist_xz(cx, cz, pos.x, pos.z);
        if d < 260.0 && dist_xz(cx, cz, z.nx, z.nz) < z.nr * 0.9 {
            picked = Some((cx, cz));
            break;
        }
    }
    let target = picked.unwrap_or_else(|| {
        let a = b.rng.range(0.0, TAU);
        let rr = b.rng.f().sqrt() * z.nr * 0.6;
        (clampf(z.nx + a.cos() * rr, 10.0, MAP - 10.0), clampf(z.nz + a.sin() * rr, 10.0, MAP - 10.0))
    });
    plan_path(m, b, pos, target, 2);
}

/// Returns true when the final waypoint is reached.
fn follow(m: &Match, i: usize, b: &mut Brain, c: &mut Cmd, face: Option<f32>) -> bool {
    let p = &m.players[i];
    let pos = p.ms.pos;
    // skip every waypoint already reached and steer toward the next one in
    // the same tick (a re-plan must never produce a standing-still tick)
    let (dx, dz, d) = loop {
        if b.path_i >= b.path_n {
            return true;
        }
        let (wx, wz) = b.path[b.path_i as usize];
        let (dx, dz) = (wx - pos.x, wz - pos.z);
        let d = (dx * dx + dz * dz).sqrt();
        let last = b.path_i + 1 == b.path_n;
        if d < if last { 1.0 } else { 0.8 } {
            b.path_i += 1;
            continue;
        }
        break (dx, dz, d);
    };
    let mut want = yaw_to(dx, dz);
    if b.avoid_t > 0.0 {
        want += b.avoid;
    } else if (m.tick + i as u32) % 4 == 0 {
        let o = pos + V3::new(0.0, 0.7, 0.0);
        let probe = |a: f32| m.w.raycast(o, fwd(a), 1.6).map_or(false, |h| h.2 != HIT_TERRAIN);
        if probe(want) {
            let l = !probe(want + 0.9);
            let r = !probe(want - 0.9);
            b.avoid = if l && !r {
                0.9
            } else if r && !l {
                -0.9
            } else if b.rng.chance(0.5) {
                1.4
            } else {
                -1.4
            };
            b.avoid_t = 0.7;
            want += b.avoid;
        }
    }
    match face {
        Some(fy) => {
            c.yaw = fy;
            let f = fwd(want);
            move_dir(c, fy, f.x, f.z, 1.0);
        }
        None => {
            b.yaw = turn_toward(b.yaw, want, 7.0 * DT);
            c.yaw = b.yaw;
            c.mz = 1.0;
            if d > 14.0 && !p.ms.in_water {
                c.buttons |= B_SPRINT;
            }
        }
    }
    false
}

fn stuck_logic(m: &Match, i: usize, b: &mut Brain, c: &mut Cmd) {
    let p = &m.players[i];
    let pos = p.ms.pos;
    b.check_t -= DT;
    if b.check_t <= 0.0 {
        b.check_t = 1.0;
        let moved = dist_xz(pos.x, pos.z, b.last_pos.0, b.last_pos.1);
        let wants = c.mx.abs() + c.mz.abs() > 0.3;
        if wants && moved < 0.6 {
            b.stuck_n += 1;
            b.unstuck_t = 0.9;
            b.unstuck_dir = if b.rng.chance(0.5) { 1.0 } else { -1.0 };
            if b.stuck_n >= 3 {
                b.think_t = 0.0;
                b.path_i = b.path_n;
                b.stuck_n = 0;
            }
        } else {
            b.stuck_n = 0;
        }
        b.last_pos = (pos.x, pos.z);
    }
    if b.unstuck_t > 0.0 {
        b.unstuck_t -= DT;
        c.mx = b.unstuck_dir;
        c.mz = -0.4;
        if b.jump_cd <= 0.0 {
            c.buttons |= B_JUMP;
            b.jump_cd = 1.2;
        }
    }
}

fn ground(m: &Match, i: usize, b: &mut Brain, c: &mut Cmd) {
    let p = &m.players[i];
    let pos = p.ms.pos;
    b.ground_t += DT;
    b.jump_cd -= DT;
    b.strafe_t -= DT;
    b.aim_t -= DT;
    b.heard_t -= DT;
    b.think_t -= DT;
    b.avoid_t -= DT;

    // ---- perception
    if (m.tick + i as u32) % 6 == 0 {
        let t = find_target(m, i, b);
        if t != b.target {
            if t == 255 && (b.target as usize) < MAX_PLAYERS {
                let q = &m.players[b.target as usize];
                b.heard = (q.ms.pos.x, q.ms.pos.z);
                b.heard_t = 4.0;
            }
            b.seen_t = 0.0;
            b.target = t;
        }
    }
    if (b.target as usize) < MAX_PLAYERS && !m.players[b.target as usize].alive {
        b.target = 255;
    }
    if b.target == 255 && p.hurt_t > 3.6 && (p.hurt_by as usize) < MAX_PLAYERS {
        let a = &m.players[p.hurt_by as usize];
        if a.alive {
            b.heard = (a.ms.pos.x, a.ms.pos.z);
            b.heard_t = 3.0;
        }
    }
    let armed = slot_usable(p, 0) || slot_usable(p, 1);
    if (b.target as usize) < MAX_PLAYERS && !p.ms.in_water {
        let q = &m.players[b.target as usize];
        let d = dist_xz(q.ms.pos.x, q.ms.pos.z, pos.x, pos.z);
        // Unarmed: keep looting for a gun; only punch back at close range
        // when attacked or against another unarmed player who is right there.
        let q_armed = slot_usable(q, 0) || slot_usable(q, 1);
        let attacked = p.hurt_t > 0.0 && p.hurt_by == b.target;
        if armed || (d < 4.0 && (attacked || !q_armed)) {
            fight(m, i, b, c);
            stuck_logic(m, i, b, c);
            return;
        }
    }
    b.seen_t = 0.0;
    b.crouch = false;

    // ---- healing / reloading when calm (only inside the safe zone)
    let in_zone = dist_xz(pos.x, pos.z, m.zone.cx, m.zone.cz) < m.zone.r - 3.0;
    if p.heal_t > 0.0 && in_zone {
        c.buttons |= B_CROUCH;
        look_around(m, i, b, c);
        return;
    }
    if p.hurt_t <= 0.0 && in_zone && p.heal_t <= 0.0 {
        if p.health < 50.0 && p.heals[1] > 0 {
            c.buttons |= 2 << 10;
        } else if p.health < 72.0 && p.heals[0] > 0 {
            c.buttons |= 1 << 10;
        } else if p.health < 90.0 && p.heals[2] > 0 && p.boost <= 0.0 {
            c.buttons |= 3 << 10;
        }
    }
    let best_slot = pick_slot(p, 40.0);
    b.slot = best_slot;
    c.buttons = (c.buttons & !(3 << 8)) | ((b.slot as u16 + 1) << 8);
    if p.cur < 2 {
        let (wk, mag) = p.slots[p.cur as usize];
        if wk != 0 && mag < WD[wk as usize].mag / 2 && p.ammo[WD[wk as usize].ammo as usize] > 0 {
            c.buttons |= B_RELOAD;
        }
    }

    // ---- goals
    b.goal_t += DT;
    // an item that stays out of reach (no progress for 6 s, or 30 s in all)
    // is given up once, instead of being re-planned every tick
    if b.goal_kind == 1 && b.path_i < b.path_n {
        // remaining path length (walking around to a door is progress too)
        let k0 = b.path_i as usize;
        let mut d = dist_xz(b.path[k0].0, b.path[k0].1, pos.x, pos.z);
        for k in k0 + 1..b.path_n as usize {
            d += dist_xz(b.path[k].0, b.path[k].1, b.path[k - 1].0, b.path[k - 1].1);
        }
        if d < b.prog_d - 0.5 {
            b.prog_d = d;
            b.prog_t = 0.0;
        } else {
            b.prog_t += DT;
        }
    }
    if b.goal_t > 30.0 || (b.goal_kind == 1 && b.prog_t > 6.0) {
        if b.goal_kind == 1 {
            b.ignore[b.ignore_i as usize % 8] = b.goal_item;
            b.ignore_i = b.ignore_i.wrapping_add(1);
        }
        b.goal_kind = 0;
        b.goal_t = 0.0;
        b.near_t = 0.0;
        b.prog_t = 0.0;
        b.think_t = 0.0;
    }
    if b.think_t <= 0.0 || b.path_i >= b.path_n {
        b.think_t = 1.5 + b.rng.f();
        choose_goal(m, i, b);
    } else if b.goal_kind == 1 && !m.items.get(b.goal_item as usize).map_or(false, |it| it.alive) {
        b.think_t = 0.0;
    }
    // weapon swap when standing on a better gun with full slots
    if b.goal_kind == 1 {
        if let Some(it) = m.items.get(b.goal_item as usize) {
            if it.alive && dist_xz(it.pos.x, it.pos.z, pos.x, pos.z) < 1.8 {
                if is_weapon(it.kind) && p.slots[0].0 != 0 && p.slots[1].0 != 0 {
                    let worst = if weapon_rank(p.slots[0].0) <= weapon_rank(p.slots[1].0) { 0 } else { 1 };
                    b.slot = worst;
                    c.buttons = (c.buttons & !(3 << 8)) | ((worst as u16 + 1) << 8);
                    if p.cur == worst && p.switch_t <= 0.0 {
                        c.buttons |= B_PICKUP;
                    }
                } else {
                    c.buttons |= B_PICKUP;
                }
                b.near_t += DT;
                if b.near_t > 2.5 {
                    // unreachable (inside a crate / behind a wall): forget it
                    b.ignore[b.ignore_i as usize % 8] = b.goal_item;
                    b.ignore_i = b.ignore_i.wrapping_add(1);
                    b.near_t = 0.0;
                    b.goal_kind = 0;
                    b.path_i = b.path_n;
                    b.think_t = 0.0;
                }
            }
        }
    }
    let face = if b.heard_t > 0.0 && armed {
        let want = yaw_to(b.heard.0 - pos.x, b.heard.1 - pos.z);
        b.yaw = turn_toward(b.yaw, want, 4.0 * DT);
        Some(b.yaw)
    } else {
        None
    };
    follow(m, i, b, c, face);
    if face.is_some() {
        b.pitch = approach(b.pitch, 0.0, DT);
        c.pitch = b.pitch;
    } else {
        b.pitch = approach(b.pitch, -0.05, DT);
        c.pitch = b.pitch;
    }
    stuck_logic(m, i, b, c);
}

fn look_around(m: &Match, i: usize, b: &mut Brain, c: &mut Cmd) {
    let phase = (m.tick as f32 * DT * 0.6 + i as f32).sin();
    b.yaw = wrap_angle(b.yaw + phase * 0.8 * DT);
    c.yaw = b.yaw;
    c.pitch = b.pitch;
}

fn fight(m: &Match, i: usize, b: &mut Brain, c: &mut Cmd) {
    let p = &m.players[i];
    let q = &m.players[b.target as usize];
    let pos = p.ms.pos;
    let eye = pos + V3::new(0.0, eye_h(p.ms.crouch), 0.0);
    let aim = if b.skill > 0.7 && b.rng.chance(0.01) {
        body_pose(q.ms.pos, q.yaw, q.pitch, q.ms.crouch, q.draw_speed > SPRINT_DRAW_SPEED, q.ms.mode).head
    } else {
        q.ms.pos + V3::new(0.0, if q.ms.crouch { 0.85 } else { 1.2 }, 0.0)
    };
    // lead the target a little based on skill
    let lead = q.ms.vel * (0.03 * b.skill);
    let tp = aim + lead;
    let dist = (tp - eye).len();
    // Shots leave from the shoulder camera ray (or the eye when scoped), so
    // aim from that origin, not from the eye.
    let wk_now = p.weapon();
    let zoom = WD[wk_now as usize].zoom;
    let ads_planned = wk_now != W_FISTS && dist > 22.0 && zoom > 1.5 && p.ms.on_ground;
    let scoped = ads_planned && zoom >= SCOPE_ZOOM;
    let (origin, _) = aim_ray_safe(&m.w, pos, b.yaw, b.pitch, p.ms.crouch, ads_planned, scoped);
    let to = tp - origin;
    b.seen_t += DT;
    b.last_seen = (q.ms.pos.x, q.ms.pos.z);

    b.slot = pick_slot(p, dist);
    c.buttons = (c.buttons & !(3 << 8)) | ((b.slot as u16 + 1) << 8);

    if b.aim_t <= 0.0 {
        // aim wander: wide enough that a moving player is missed most of the time
        let base = (1.15 - b.skill) * 0.17 + q.ms.vel.len_xz() * 0.008 + if p.ms.vel.len_xz() > 1.0 { 0.02 } else { 0.0 };
        b.aim_tx = b.rng.range(-base, base);
        b.aim_ty = b.rng.range(-base * 0.6, base * 0.6);
        b.aim_t = b.rng.range(0.25, 0.6);
    }
    b.aim_ex = lerpf(b.aim_ex, b.aim_tx, 0.12);
    b.aim_ey = lerpf(b.aim_ey, b.aim_ty, 0.12);
    let want_yaw = yaw_to(to.x, to.z) + b.aim_ex;
    let want_pitch = (to.y / to.len_xz().max(0.1)).atan() + b.aim_ey;
    let turn = (1.8 + 3.6 * b.skill) * DT;
    b.yaw = turn_toward(b.yaw, want_yaw, turn);
    b.pitch = approach(b.pitch, want_pitch, turn);
    c.yaw = b.yaw;
    c.pitch = clampf(b.pitch, -1.4, 1.4);
    let aim_err = wrap_angle(want_yaw - b.yaw).abs() + (want_pitch - b.pitch).abs();

    let wk = p.weapon();
    let def = WD[wk as usize];
    let mag = if p.cur < 2 { p.slots[p.cur as usize].1 } else { 0 };
    let react = 1.3 - 0.5 * b.skill;
    let in_range = if wk == W_FISTS { dist < 1.9 } else { dist < def.range * 0.8 };
    if wk != W_FISTS && dist > 22.0 && def.zoom > 1.5 && p.ms.on_ground {
        c.buttons |= B_ADS;
    }
    if wk != W_FISTS && mag == 0 {
        c.buttons |= B_RELOAD;
    }
    if b.seen_t > react && aim_err < 0.14 && in_range && p.reload_t <= 0.0 && p.switch_t <= 0.0 {
        if b.pause_t > 0.0 {
            b.pause_t -= DT;
        } else {
            c.buttons |= B_FIRE;
            b.burst_t += DT;
            let burst_len = if dist < 12.0 { 0.8 } else { 0.25 + b.skill * 0.4 };
            if b.burst_t > burst_len {
                b.burst_t = 0.0;
                b.pause_t = b.rng.range(0.25, 0.7) * (1.4 - b.skill);
            }
        }
    }
    if b.strafe_t <= 0.0 {
        b.strafe = if b.rng.chance(0.5) { 1.0 } else { -1.0 } * b.rng.range(0.5, 1.0);
        if b.rng.chance(0.2) {
            b.strafe = 0.0;
        }
        b.strafe_t = b.rng.range(0.5, 1.4);
        b.crouch = dist > 45.0 && b.rng.chance(0.45);
    }
    let pref = match wk {
        W_SHOTGUN => 6.0,
        W_SMG | W_PISTOL => 14.0,
        W_FISTS => 1.2,
        W_SNIPER | W_DMR => 55.0,
        _ => 28.0,
    };
    let fwd_in = if dist > pref * 1.4 {
        0.85
    } else if dist < pref * 0.45 {
        -0.6
    } else {
        0.0
    };
    c.mx = b.strafe;
    c.mz = fwd_in;
    if b.crouch && fwd_in == 0.0 {
        c.buttons |= B_CROUCH;
        c.mx *= 0.4;
    }
    // badly hurt with something to heal: break off the fight (fall back away
    // from the target, still shooting) instead of trading to the death
    let can_heal = p.heals[0] > 0 || p.heals[1] > 0;
    if p.health < 35.0 && can_heal && dist > 6.0 {
        move_dir(c, c.yaw, pos.x - q.ms.pos.x, pos.z - q.ms.pos.z, 1.0);
        c.mx = clampf(c.mx + b.strafe * 0.5, -1.0, 1.0);
        c.buttons &= !B_CROUCH;
    }
    if dist < 25.0 && b.jump_cd <= 0.0 && b.rng.chance(0.012) {
        c.buttons |= B_JUMP;
        b.jump_cd = 2.5;
    }
    // stay inside the zone even while fighting
    let z = &m.zone;
    if dist_xz(pos.x, pos.z, z.cx, z.cz) > z.r - 2.0 {
        move_dir(c, c.yaw, z.cx - pos.x, z.cz - pos.z, 1.0);
    }
    // throw a grenade at targets behind light cover
    let dh = dist_xz(q.ms.pos.x, q.ms.pos.z, pos.x, pos.z);
    if p.nades > 0 && (9.0..23.0).contains(&dh) && b.rng.chance(0.006) {
        c.pitch = nade_pitch(dh);
        c.buttons &= !B_FIRE;
        c.buttons |= B_NADE;
    }
}

/// Where a grenade explodes (after its bounce and roll) by aim pitch on flat
/// ground, measured with the simulation's own grenade physics.
const NADE_TABLE: [(f32, f32); 16] = [
    (8.0, -0.50),
    (8.7, -0.40),
    (10.1, -0.30),
    (11.2, -0.25),
    (11.9, -0.20),
    (13.3, -0.15),
    (14.6, -0.10),
    (15.5, -0.05),
    (16.8, 0.00),
    (17.7, 0.05),
    (19.1, 0.10),
    (19.8, 0.15),
    (20.7, 0.20),
    (22.3, 0.25),
    (23.4, 0.35),
    (23.7, 0.40),
];

/// Aim pitch that makes a grenade explode `dist` metres away.
pub fn nade_pitch(dist: f32) -> f32 {
    let t = &NADE_TABLE;
    if dist <= t[0].0 {
        return t[0].1;
    }
    for k in 1..t.len() {
        if dist <= t[k].0 {
            let (d0, p0) = t[k - 1];
            let (d1, p1) = t[k];
            return p0 + (p1 - p0) * (dist - d0) / (d1 - d0);
        }
    }
    t[t.len() - 1].1
}
