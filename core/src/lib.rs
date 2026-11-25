//! ZONEFALL core: deterministic battle-royale simulation compiled to WebAssembly.
//!
//! One binary serves three hosts:
//! * the browser main thread (`cli_*`: world meshes, collision queries, prediction),
//! * an offline Web Worker running a local server (`srv_*`),
//! * a Cloudflare Durable Object running the online server (`srv_*`).
//!
//! The host passes bytes through a fixed input buffer (`in_ptr`) and reads
//! output from `srv_out`/`cli_mesh_ptr`. All exports are panic-free on valid use.

#![allow(clippy::missing_safety_doc)]

pub mod bot;
pub mod defs;
pub mod math;
pub mod mesh;
pub mod names;
pub mod net;
pub mod player;
pub mod rng;
pub mod sim;
pub mod world;

use std::ptr::addr_of_mut;

use defs::*;
use math::V3;
use player::*;

const IN_CAP: usize = 65536;
static mut INBUF: [u8; IN_CAP] = [0; IN_CAP];
static mut SERVER: Option<Box<sim::Match>> = None;
static mut CLIENT: Option<Box<ClientState>> = None;

struct ClientState {
    world: world::World,
    meshes: mesh::Meshes,
    pred: [f32; 32],
    wdefs: Vec<f32>,
    towns: Vec<f32>,
    ray_kind: u32,
}

#[inline]
fn inbuf(len: u32) -> &'static [u8] {
    let n = (len as usize).min(IN_CAP);
    unsafe { std::slice::from_raw_parts(addr_of_mut!(INBUF) as *const u8, n) }
}

#[inline]
fn server() -> Option<&'static mut sim::Match> {
    unsafe { (*addr_of_mut!(SERVER)).as_deref_mut() }
}

#[inline]
fn client() -> Option<&'static mut ClientState> {
    unsafe { (*addr_of_mut!(CLIENT)).as_deref_mut() }
}

#[no_mangle]
pub extern "C" fn in_ptr() -> *mut u8 {
    addr_of_mut!(INBUF) as *mut u8
}

#[no_mangle]
pub extern "C" fn in_cap() -> u32 {
    IN_CAP as u32
}

#[no_mangle]
pub extern "C" fn proto_version() -> u32 {
    sim::PROTO as u32
}

// ----------------------------------------------------------------------------
// Server
// ----------------------------------------------------------------------------

/// `seed` is sent to clients; the four secret words (128 bits of server
/// entropy: match RNG + combat RNG) must stay on the server.
#[no_mangle]
pub extern "C" fn srv_init(seed: u32, slots: u32, flags: u32, s0: u32, s1: u32, s2: u32, s3: u32) {
    let secret = (s0 as u64) | ((s1 as u64) << 32);
    let combat = (s2 as u64) | ((s3 as u64) << 32);
    let m = sim::Match::with_secrets(seed, slots as usize, flags, secret, combat);
    unsafe {
        *addr_of_mut!(SERVER) = Some(Box::new(m));
    }
}

/// Name bytes in the input buffer. Returns the player id or -1.
#[no_mangle]
pub extern "C" fn srv_add(name_len: u32) -> i32 {
    let name = inbuf(name_len.min(64)).to_vec();
    match server() {
        Some(s) => s.add_human(&name).map_or(-1, |p| p as i32),
        None => -1,
    }
}

#[no_mangle]
pub extern "C" fn srv_remove(pid: u32) {
    if let Some(s) = server() {
        if pid < MAX_PLAYERS as u32 {
            s.remove_human(pid as u8);
        }
    }
}

#[no_mangle]
pub extern "C" fn srv_msg(pid: u32, len: u32) {
    if pid >= MAX_PLAYERS as u32 || len as usize > 1024 {
        return;
    }
    let data = inbuf(len).to_vec();
    if let Some(s) = server() {
        s.on_message(pid as u8, &data);
    }
}

#[no_mangle]
pub extern "C" fn srv_step(behind: u32) -> u32 {
    server().map_or(0, |s| {
        // set by the host when this step catches up on ticks it fell behind
        s.catch_up = behind != 0;
        s.step()
    })
}

#[no_mangle]
pub extern "C" fn srv_welcome(pid: u32) -> u32 {
    match server() {
        Some(s) if pid < MAX_PLAYERS as u32 => s.welcome(pid as u8).len() as u32,
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn srv_snap(pid: u32) -> u32 {
    match server() {
        Some(s) if pid < MAX_PLAYERS as u32 => s.snapshot(pid as u8).len() as u32,
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn srv_out() -> *const u8 {
    server().map_or(std::ptr::null(), |s| s.out.as_ptr())
}

#[no_mangle]
pub extern "C" fn srv_flush() {
    if let Some(s) = server() {
        s.flush();
    }
}

#[no_mangle]
pub extern "C" fn srv_phase() -> u32 {
    server().map_or(255, |s| s.phase as u32)
}

#[no_mangle]
pub extern "C" fn srv_humans() -> u32 {
    server().map_or(0, |s| s.human_count() as u32)
}

#[no_mangle]
pub extern "C" fn srv_alive() -> u32 {
    server().map_or(0, |s| s.alive_count() as u32)
}

#[no_mangle]
pub extern "C" fn srv_lobby_left() -> f32 {
    server().map_or(0.0, |s| s.lobby_t)
}

// ----------------------------------------------------------------------------
// Client
// ----------------------------------------------------------------------------

/// Generate the world for `seed`; returns its checksum.
#[no_mangle]
pub extern "C" fn cli_init(seed: u32) -> u32 {
    let world = world::World::generate(seed);
    let meshes = mesh::build(&world);
    let mut wdefs = Vec::with_capacity(NUM_WEAPONS * 18);
    for d in WD.iter() {
        wdefs.extend_from_slice(&[
            d.ammo as f32, d.dmg, d.interval, d.mag as f32, d.reload, d.hip, d.ads, d.f0, d.f1, d.fmin,
            d.pellets as f32, d.head, d.range, d.speed, d.zoom, d.rank as f32, d.recoil, 0.0,
        ]);
    }
    let mut towns = Vec::new();
    for t in &world.towns {
        towns.extend_from_slice(&[t.x, t.z, t.r, t.name as f32, t.kind as f32, t.y]);
    }
    let checksum = world.checksum;
    unsafe {
        *addr_of_mut!(CLIENT) = Some(Box::new(ClientState { world, meshes, pred: [0.0; 32], wdefs, towns, ray_kind: 0 }));
    }
    checksum
}

/// kind: 0 terrain verts(u32) 1 terrain indices(u32) 2 terrain chunks(f32)
///       3 prop verts(u32) 4 prop chunks(f32) 5 minimap rgba(u8) 6 weapon defs(f32) 7 towns(f32) 8 cars(f32 x4: x y z yaw)
#[no_mangle]
pub extern "C" fn cli_mesh_ptr(kind: u32) -> *const u8 {
    let Some(c) = client() else { return std::ptr::null() };
    match kind {
        0 => c.meshes.tv.as_ptr() as *const u8,
        1 => c.meshes.ti.as_ptr() as *const u8,
        2 => c.meshes.tchunks.as_ptr() as *const u8,
        3 => c.meshes.pv.as_ptr() as *const u8,
        4 => c.meshes.pchunks.as_ptr() as *const u8,
        5 => c.meshes.map.as_ptr(),
        6 => c.wdefs.as_ptr() as *const u8,
        7 => c.towns.as_ptr() as *const u8,
        8 => c.world.cars.as_ptr() as *const u8,
        _ => std::ptr::null(),
    }
}

/// Length in bytes.
#[no_mangle]
pub extern "C" fn cli_mesh_len(kind: u32) -> u32 {
    let Some(c) = client() else { return 0 };
    (match kind {
        0 => c.meshes.tv.len() * 4,
        1 => c.meshes.ti.len() * 4,
        2 => c.meshes.tchunks.len() * 4,
        3 => c.meshes.pv.len() * 4,
        4 => c.meshes.pchunks.len() * 4,
        5 => c.meshes.map.len(),
        6 => c.wdefs.len() * 4,
        7 => c.towns.len() * 4,
        8 => c.world.cars.len() * 16,
        _ => 0,
    }) as u32
}

/// Free the large mesh buffers once uploaded to the GPU.
#[no_mangle]
pub extern "C" fn cli_free_meshes() {
    if let Some(c) = client() {
        c.meshes.tv = Vec::new();
        c.meshes.ti = Vec::new();
        c.meshes.pv = Vec::new();
    }
}

#[no_mangle]
pub extern "C" fn cli_height(x: f32, z: f32) -> f32 {
    client().map_or(0.0, |c| c.world.height(x, z))
}

#[no_mangle]
pub extern "C" fn cli_ground(x: f32, z: f32, feet: f32) -> f32 {
    client().map_or(0.0, |c| c.world.ground_below(x, z, feet, P_RADIUS))
}

/// Returns hit distance or -1. Kind of the last hit via `cli_ray_kind`.
#[no_mangle]
pub extern "C" fn cli_ray(ox: f32, oy: f32, oz: f32, dx: f32, dy: f32, dz: f32, maxt: f32) -> f32 {
    let Some(c) = client() else { return -1.0 };
    let d = V3::new(dx, dy, dz).norm();
    let o = V3::new(ox, oy, oz);
    if !o.finite() || d.len2() < 0.5 || !maxt.is_finite() {
        return -1.0;
    }
    match c.world.raycast(o, d, maxt.clamp(0.0, 2000.0)) {
        Some((t, _, k)) => {
            c.ray_kind = k as u32;
            t
        }
        None => -1.0,
    }
}

#[no_mangle]
pub extern "C" fn cli_ray_kind() -> u32 {
    client().map_or(0, |c| c.ray_kind)
}

#[no_mangle]
pub extern "C" fn cli_building_at(x: f32, z: f32) -> i32 {
    client().and_then(|c| c.world.building_at(x, z)).map_or(-1, |k| k as i32)
}

/// Where car `k` is now (cars move: the server sends their poses).
#[no_mangle]
pub extern "C" fn cli_set_car(k: u32, x: f32, y: f32, z: f32, yaw: f32) {
    let Some(c) = client() else { return };
    if let Some(car) = c.world.cars.get_mut(k as usize) {
        if x.is_finite() && y.is_finite() && z.is_finite() && yaw.is_finite() {
            *car = [x, y, z, yaw];
        }
    }
}

#[no_mangle]
pub extern "C" fn cli_pred_ptr() -> *mut f32 {
    client().map_or(std::ptr::null_mut(), |c| c.pred.as_mut_ptr())
}

/// Prediction IO layout (f32):
///  0-2 pos, 3-5 vel, 6 mode, 7 on_ground, 8 crouch, 9 in_water, 10 airtime, 11 speed_mult,
///  12 mx_q, 13 mz_q, 14 yaw_q, 15 pitch_q, 16 buttons, 17 out events, 18 car index
/// While driving (mode 5) pos/vel are the car's and airtime is its heading.
#[no_mangle]
pub extern "C" fn cli_pred_step() {
    let Some(c) = client() else { return };
    let p = &mut c.pred;
    let mut s = MoveState {
        pos: V3::new(p[0], p[1], p[2]),
        vel: V3::new(p[3], p[4], p[5]),
        mode: p[6] as u8,
        on_ground: p[7] != 0.0,
        crouch: p[8] != 0.0,
        in_water: p[9] != 0.0,
        airtime: p[10],
        car: if p[18] >= 0.0 && p[18] < 255.0 { p[18] as u8 } else { 255 },
    };
    if !s.pos.finite() || !s.vel.finite() {
        return;
    }
    let cmd = Cmd::from_quant(0, 0, p[12] as i32 as i8, p[13] as i32 as i8, p[14] as u32 as u16, p[15] as i32 as i16, p[16] as u32 as u16);
    let mult = if p[11].is_finite() { p[11].clamp(0.3, 1.5) } else { 1.0 };
    let ev = step_move(&c.world, &mut s, &cmd, mult, DT);
    p[0] = s.pos.x;
    p[1] = s.pos.y;
    p[2] = s.pos.z;
    p[3] = s.vel.x;
    p[4] = s.vel.y;
    p[5] = s.vel.z;
    p[6] = s.mode as f32;
    p[7] = s.on_ground as u8 as f32;
    p[8] = s.crouch as u8 as f32;
    p[9] = s.in_water as u8 as f32;
    p[10] = s.airtime;
    p[17] = ev as f32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::*;

    #[test]
    fn world_is_deterministic() {
        let a = world::World::generate(1234);
        let b = world::World::generate(1234);
        assert_eq!(a.checksum, b.checksum);
        let c = world::World::generate(999);
        assert_ne!(a.checksum, c.checksum);
        assert!(a.towns.len() >= 5, "towns {}", a.towns.len());
        assert!(a.buildings.len() > 40, "buildings {}", a.buildings.len());
    }

    #[test]
    fn land_fraction_reasonable() {
        for seed in [1u32, 7, 42, 2024, 777] {
            let w = world::World::generate(seed);
            let land = w.h.iter().filter(|&&h| h > world::WATER).count() as f32 / w.h.len() as f32;
            assert!(land > 0.35 && land < 0.8, "seed {} land {}", seed, land);
            assert!(w.towns.len() >= 5, "seed {} towns {}", seed, w.towns.len());
        }
    }

    #[test]
    fn walking_stays_on_ground() {
        let w = world::World::generate(42);
        let t = w.towns[0];
        let mut s = MoveState { pos: V3::new(t.x + 3.0, w.height(t.x + 3.0, t.z) + 0.5, t.z), mode: MODE_GROUND, ..Default::default() };
        let c = Cmd { mz: 1.0, yaw: 0.7, ..Default::default() };
        for _ in 0..300 {
            step_move(&w, &mut s, &c, 1.0, DT);
            let g = w.ground_below(s.pos.x, s.pos.z, s.pos.y, P_RADIUS);
            assert!(s.pos.y >= g - 0.05, "sank below ground {} < {}", s.pos.y, g);
        }
    }

    #[test]
    fn full_bot_match_finishes() {
        for seed in [3u32, 11, 58] {
            let mut m = Match::new(seed, 24, FLAG_OFFLINE, (seed ^ 0xABCD) as u64);
            let pid = m.add_human(b"Tester").unwrap();
            let mut ticks = 0u32;
            let mut ended = false;
            // the human idles in the plane; gets auto-ejected and AFK -> dies eventually
            while ticks < 30 * 60 * 12 {
                let r = m.step();
                if r & 1 != 0 {
                    let _ = m.snapshot(pid).len();
                    m.flush();
                }
                ticks += 1;
                if r & 2 != 0 {
                    ended = true;
                    break;
                }
            }
            assert!(ended, "seed {} match did not end, alive {}", seed, m.alive_count());
            let kills: u32 = m.players.iter().map(|p| p.kills as u32).sum();
            assert!(kills > 5, "seed {} too few kills {}", seed, kills);
            eprintln!("seed {} ended after {:.0}s, kills {}, winner {}", seed, ticks as f32 / 30.0, kills, m.winner);
        }
    }

    #[test]
    fn rejects_garbage_input() {
        let mut m = Match::new(5, 8, FLAG_OFFLINE, 99);
        let pid = m.add_human(b"<script>alert(1)</script>\x00\x01").unwrap();
        assert!(!m.players[pid as usize].name.contains('<'));
        m.step();
        for len in 0..64 {
            let data: Vec<u8> = (0..len).map(|i| (i * 37 + len) as u8).collect();
            m.on_message(pid, &data);
            let mut d2 = data.clone();
            if !d2.is_empty() {
                d2[0] = C_INPUT;
            }
            m.on_message(pid, &d2);
        }
        for _ in 0..100 {
            m.step();
        }
        let p = &m.players[pid as usize];
        assert!(p.ms.pos.finite());
    }
}
