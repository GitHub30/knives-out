//! Authoritative match simulation + wire protocol. One `Match` per room.

use crate::defs::*;
use crate::math::*;
use crate::net::*;
use crate::player::*;
use crate::rng::Rng;
use crate::world::*;
use crate::names::{name_key, sanitize_name};

pub const PROTO: u16 = 15;
/// A car catches fire below this share of its bodywork, and then explodes
/// after the fuse (5 s) whatever happens.
pub const CAR_FIRE_AT: f32 = 0.19;
pub const CAR_FUSE_TICKS: u32 = 150;
/// Helicopters: hull strength, missiles carried, seconds between launches,
/// missile speed (m/s) and life (ticks).
pub const HELI_HP: f32 = 1500.0;
/// Vehicles shoving each other: how close counts as touching, the closing
/// speed above which a knock dents (and bodywork lost per m/s above it).
/// (How a shoved car slides and stops is in `player.rs`: the predicting
/// client does the same sums.)
pub const CAR_TOUCH: f32 = 0.03;
pub const CAR_CRASH_V: f32 = 6.0;
pub const CAR_CRASH_DMG: f32 = 9.0;
/// A car sliding faster than this hurts whoever it runs into.
pub const CAR_SLIDE_HURT_V: f32 = 5.0;
/// The closing speed from which a car nudges a helicopter, the speed from
/// which a helicopter skidding on the ground shoves cars (in the air it
/// always does), and how fast it shoves one it merely settles on.
pub const HELI_NUDGE_V: f32 = 3.0;
pub const HELI_SHOVE_V: f32 = HELI_STAND_V;
pub const HELI_SHOVE_MIN: f32 = 2.5;
pub const HELI_AMMO: u8 = 16;
pub const MISSILE_CD: f32 = 0.8;
pub const MISSILE_SPEED: f32 = 95.0;
pub const MISSILE_LIFE: u32 = 120;
/// A missile's blast: radius, damage to a player / a vehicle at its centre.
pub const MISSILE_R: f32 = 9.5;
pub const MISSILE_DMG: f32 = 160.0;
pub const MISSILE_CAR_DMG: f32 = 900.0;
/// Where the missiles leave the hull, in its frame (x = right, y up, z = nose).
pub const MISSILE_POD: [f32; 3] = [2.16, 1.29, 2.45];
pub const HIST: usize = 32;
pub const MAX_ITEMS: usize = 4000;
pub const INTEREST: f32 = 520.0;
pub const CMDQ_CAP: usize = 8;
/// Ticks of late input absorbed before the server simulates a player without it.
pub const LAG_GRACE: u8 = 3;
/// Late commands covered by idle steps must arrive within this many ticks.
const DEBT_TTL: u8 = 10;
/// Late commands an uplink stall can leave to drop: about 8 s at 30 Hz, so
/// the burst after any stall a connection survives leaves no backlog. A u8
/// on the wire; at most 254 so `debt + 1` cannot overflow.
const DEBT_MAX: u8 = 250;
/// Idle steps keep the last movement input for this many ticks, then stop.
const IDLE_REPEAT: u8 = 6;
/// If the lag never returns to zero for this long, take one idle step.
const DRIFT_T: u16 = 60;
/// A queue that has not emptied for BACKLOG_T ticks is pure input latency
/// (lag accounting already absorbs late input): drop its oldest command.
const BACKLOG_MAX: usize = 0;
const BACKLOG_T: u8 = 30;
/// Lag compensation: the most a shot may be rewound, and how far above the
/// player's recent minimum rewind a single command may claim.
pub const REWIND_MAX: f32 = 15.0;
const REWIND_MARGIN: f32 = 4.0;
/// Buttons that act on a single command (carried over when one is dropped).
const EDGE_BUTTONS: u16 = B_JUMP | B_RELOAD | B_PICKUP | B_SPECT | B_NADE | B_FIRE | B_USE;
/// ADS with at least this zoom is a first-person scope.
pub const SCOPE_ZOOM: f32 = 3.5;

pub const C_INPUT: u8 = 2;
pub const C_START: u8 = 4;

pub const S_WELCOME: u8 = 0x81;
pub const S_SNAP: u8 = 0x82;

pub const PH_LOBBY: u8 = 0;
pub const PH_PLAY: u8 = 1;
pub const PH_END: u8 = 2;

pub const E_SHOT: u8 = 1;
pub const E_HIT: u8 = 2;
pub const E_HURT: u8 = 3;
pub const E_KILL: u8 = 4;
pub const E_ITEM: u8 = 5;
pub const E_IDEL: u8 = 6;
pub const E_PINFO: u8 = 7;
pub const E_BOOM: u8 = 8;
pub const E_END: u8 = 9;
pub const E_PICK: u8 = 10;
pub const E_THROW: u8 = 11;
pub const E_LAND: u8 = 12;
/// Per-player resync of every item in one 128 m chunk.
pub const E_ICHUNK: u8 = 13;
pub const ICH: usize = 8;
const ICW: f32 = MAP / ICH as f32;
/// Items are only synced for chunks this close to the viewer.
pub const ITEM_SYNC_R: f32 = 190.0;
/// Players closer than this are always sent; farther ones need line of sight.
const ALWAYS_SEND_R: f32 = 40.0;
/// Item changes are logged; each viewer gets a change once it is near it or
/// can see the player who made it.
const ILOG: usize = 2048;
/// A chunk is only resynced to a viewer when it holds no change younger than
/// this that the viewer was not allowed to see.
const ITEM_QUIET: u32 = 300;
/// Hidden shooters: shots farther than this are not sent, nearer ones only
/// with a coarse position and no shooter id.
const HIDDEN_SHOT_R: f32 = 330.0;
/// E_HURT attacker id for a player the victim is not sent.
pub const HURT_HIDDEN: u8 = 254;

/// splitmix64 finaliser.
fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Item ids on the wire: a secret permutation of the internal ones (a
/// four-round Feistel network on 16 bits keyed by the match salt). Internal
/// ids count up as items are created, so ids a viewer is sent (its own
/// drops, a chunk resync) would otherwise date the changes it is not sent.
fn item_wire(salt: u64, k: u16) -> u16 {
    let (mut l, mut r) = ((k >> 8) as u8, k as u8);
    for i in 0..4u64 {
        let f = mix64(salt ^ 0x17E4_1D00 ^ (i << 8) ^ r as u64) as u8;
        (l, r) = (r, l ^ f);
    }
    ((l as u16) << 8) | r as u16
}
fn item_unwire(salt: u64, w: u16) -> u16 {
    let (mut l, mut r) = ((w >> 8) as u8, w as u8);
    for i in (0..4u64).rev() {
        let f = mix64(salt ^ 0x17E4_1D00 ^ (i << 8) ^ l as u64) as u8;
        (l, r) = (r ^ f, l);
    }
    ((l as u16) << 8) | r as u16
}

/// Centre of the `g`-sized world-aligned cell holding `v`.
#[inline]
fn cell_c(v: f32, g: f32) -> f32 {
    ((v / g).floor() + 0.5) * g
}
fn get_v3(b: &[u8], k: usize) -> V3 {
    let f = |k: usize| f32::from_le_bytes([b[k], b[k + 1], b[k + 2], b[k + 3]]);
    V3::new(f(k), f(k + 4), f(k + 8))
}
fn put_v3(b: &mut [u8], k: usize, v: V3) {
    b[k..k + 4].copy_from_slice(&v.x.to_le_bytes());
    b[k + 4..k + 8].copy_from_slice(&v.y.to_le_bytes());
    b[k + 8..k + 12].copy_from_slice(&v.z.to_le_bytes());
}
/// E_HURT from a player that viewer `i` is not sent.
fn hidden_sh_id(ev: &[u8], i: usize, send: &[bool]) -> bool {
    let a = ev[1] as usize;
    a < MAX_PLAYERS && a != i && !send[a]
}

#[inline]
pub fn item_chunk(x: f32, z: f32) -> usize {
    let cx = clampf((x / ICW).floor(), 0.0, ICH as f32 - 1.0) as usize;
    let cz = clampf((z / ICW).floor(), 0.0, ICH as f32 - 1.0) as usize;
    cz * ICH + cx
}

pub const KILLER_ZONE: u8 = 255;

pub const FLAG_OFFLINE: u32 = 1;
pub const FLAG_PRIVATE: u32 = 2;

#[derive(Clone, Copy, Default)]
pub struct HistE {
    pub pos: V3,
    pub crouch: bool,
    pub alive: bool,
    /// What the model's pose depends on (see `body_pose`).
    pub yaw: f32,
    pub pitch: f32,
    pub sprint: bool,
    pub mode: u8,
    /// The held weapon and the snapshot flags that shape the drawn model:
    /// bit 0 ADS, 1 firing, 2 healing, 3 reloading, 4 vest, 5 helmet.
    pub weapon: u8,
    pub gear: u8,
}

impl HistE {
    fn of(p: &Player) -> HistE {
        HistE {
            pos: p.ms.pos,
            crouch: p.ms.crouch,
            alive: p.used && p.alive,
            yaw: p.yaw,
            pitch: p.pitch,
            sprint: p.draw_speed > SPRINT_DRAW_SPEED,
            mode: p.ms.mode,
            weapon: p.weapon(),
            gear: (p.ads as u8) | (((p.firing_t > 0.0) as u8) << 1) | (((p.heal_t > 0.0) as u8) << 2) | (((p.reload_t > 0.0) as u8) << 3) | (((p.vest > 0) as u8) << 4) | (((p.helm > 0) as u8) << 5),
        }
    }
    fn gear(&self) -> Gear {
        let g = self.gear;
        Gear { weapon: self.weapon, ads: g & 1 != 0, firing: g & 2 != 0, healing: g & 4 != 0, reloading: g & 8 != 0, vest: g & 16 != 0, helm: g & 32 != 0 }
    }
}

#[derive(Clone, Copy, Default)]
pub struct Item {
    pub kind: u8,
    pub amount: u16,
    pub pos: V3,
    pub alive: bool,
}

/// One item change (state after the change), with who caused it.
#[derive(Clone, Copy, Default)]
struct ILogE {
    tick: u32,
    id: u16,
    actor: u8,
    chunk: u8,
    /// The chunk's version after this change.
    ver: u32,
    it: Item,
}

#[derive(Clone, Copy, Default)]
pub struct Nade {
    pub pos: V3,
    pub vel: V3,
    pub fuse: f32,
    pub owner: u8,
    pub id: u16,
}

/// What a match knows of a helicopter besides its pose (`World::helis`).
#[derive(Clone, Copy, Debug)]
pub struct Heli {
    /// Who flies it (255 = nobody).
    pub pilot: u8,
    /// Hull left (0 = a wreck) and who damaged it last.
    pub hp: f32,
    pub hit_by: u8,
    pub ammo: u8,
    /// Seconds until the next missile may leave.
    pub cd: f32,
    /// Rotor speed 0..1 (for show: it winds up with a pilot aboard).
    pub rotor: f32,
    pub vel: V3,
    /// Off the ground.
    pub air: bool,
    /// Attitude for show, radians: pitch > 0 is nose down, roll > 0 is
    /// right side down.
    pub pitch: f32,
    pub roll: f32,
    /// Which pod fires next.
    side: bool,
}

impl Heli {
    fn parked() -> Heli {
        Heli { pilot: 255, hp: HELI_HP, hit_by: 255, ammo: HELI_AMMO, cd: 0.0, rotor: 0.0, vel: V3::ZERO, air: false, pitch: 0.0, roll: 0.0, side: false }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Missile {
    pub pos: V3,
    pub dir: V3,
    /// Ticks left to fly.
    pub life: u32,
    pub owner: u8,
    /// The helicopter that launched it (it cannot hit that one).
    pub heli: u8,
    pub id: u16,
}

#[derive(Clone, Default)]
pub struct Player {
    pub used: bool,
    pub bot: bool,
    pub human: bool,
    pub connected: bool,
    pub name: String,
    pub color: u8,
    pub ms: MoveState,
    pub yaw: f32,
    pub pitch: f32,
    pub alive: bool,
    pub health: f32,
    pub vest: u8,
    pub vest_dur: f32,
    pub helm: u8,
    pub helm_dur: f32,
    pub slots: [(u8, u16); 2],
    pub cur: u8,
    pub ammo: [u16; NUM_AMMO],
    pub heals: [u8; 3],
    pub nades: u8,
    pub reload_t: f32,
    pub heal_t: f32,
    pub heal_kind: u8,
    pub fire_cd: f32,
    pub switch_t: f32,
    pub boost: f32,
    pub nade_cd: f32,
    pub ads: bool,
    pub firing_t: f32,
    pub kills: u16,
    pub dmg_dealt: f32,
    pub place: u8,
    pub killer: u8,
    pub death_tick: u32,
    /// Highest command sequence received (ordering / dedupe).
    pub last_seq: u32,
    /// Highest command sequence actually simulated (sent back as the ack).
    pub acked_seq: u32,
    /// Commands waiting to be simulated, one per tick.
    pub cmdq: std::collections::VecDeque<Cmd>,
    /// Server ticks not yet simulated for this player (see `step_humans`).
    /// Every simulated step costs one, so steps always stay within
    /// [elapsed ticks - LAG_GRACE - 1, elapsed ticks].
    pub lag: u8,
    /// Idle steps taken while input was late: that many late commands that
    /// arrive in the following burst are dropped instead of simulated.
    pub debt: u8,
    pub debt_ttl: u8,
    /// Edge buttons of dropped commands, merged into the next simulated one.
    pub carry: u16,
    /// Consecutive idle steps (movement is only extrapolated briefly).
    pub idle_run: u8,
    /// Ticks the lag has stayed above zero (slow drift back to real time).
    pub lag_hi_t: u16,
    /// Ticks the queue has not emptied (more than BACKLOG_MAX commands left).
    pub backlog_t: u8,
    /// Recent minimum of the rewind (ticks) this client claims for its view.
    pub rew_floor: f32,
    /// Horizontal speed as clients see it (smoothed from the change of
    /// position, like the model's animation speed): what decides whether the
    /// model is drawn sprinting. Running into a wall keeps the velocity up
    /// but not this.
    pub draw_speed: f32,
    draw_at: (f32, f32),
    pub cmds_applied: u32,
    pub idle_steps: u32,
    pub last_cmd: Cmd,
    /// Buttons of the previous simulated command (press-edge detection).
    pub prev_buttons: u16,
    /// Tick a car last ran this player over.
    pub ram_tick: u32,
    pub pick_cd: f32,
    pub spect_cd: f32,
    pub spect: u8,
    pub brain: crate::bot::Brain,
    pub hurt_by: u8,
    pub hurt_t: f32,
    pub zone_acc: f32,
    pub need_welcome: bool,
    pub snap_round: u32,
    /// Item chunk versions this player's client has been sent.
    pub item_ver: Vec<u32>,
    /// Item log entries already sent (bit per ring slot); entries before
    /// `ilog_from` were covered by the welcome.
    ilog_sent: [u64; ILOG / 64],
    ilog_from: u32,
    /// Last tick each other player was visible to this viewer / was checked.
    pub seen_tick: [u32; MAX_PLAYERS],
    pub vis_check: [u32; MAX_PLAYERS],
    /// Per other player: the snapshot tick since which this viewer has been
    /// sent it without a gap (0 = not in the last snapshot).
    pub sent_from: [u32; MAX_PLAYERS],
}

impl Player {
    pub fn weapon(&self) -> u8 {
        if self.cur < 2 {
            self.slots[self.cur as usize].0
        } else {
            W_FISTS
        }
    }
    pub fn speed_mult(&self) -> f32 {
        let mut m = WD[self.weapon() as usize].speed;
        if self.heal_t > 0.0 {
            m *= 0.55;
        }
        if self.boost > 0.0 {
            m *= 1.06;
        }
        m
    }
    fn reset_combat(&mut self) {
        self.health = 100.0;
        self.alive = true;
        self.vest = 0;
        self.vest_dur = 0.0;
        self.helm = 0;
        self.helm_dur = 0.0;
        self.slots = [(0, 0); 2];
        self.cur = 2;
        self.ammo = [0; NUM_AMMO];
        self.heals = [0; 3];
        self.nades = 0;
        self.reload_t = 0.0;
        self.heal_t = 0.0;
        self.heal_kind = 0;
        self.fire_cd = 0.0;
        self.switch_t = 0.0;
        self.boost = 0.0;
        self.kills = 0;
        self.dmg_dealt = 0.0;
        self.place = 0;
        self.killer = 255;
        self.spect = 255;
    }
}

#[derive(Clone, Copy, Default)]
pub struct Zone {
    pub stage: u8,
    pub shrinking: bool,
    pub done: bool,
    pub timer: f32,
    pub cx: f32,
    pub cz: f32,
    pub r: f32,
    pub sx: f32,
    pub sz: f32,
    pub sr: f32,
    pub nx: f32,
    pub nz: f32,
    pub nr: f32,
}

impl Zone {
    pub fn dps(&self) -> f32 {
        ZONES[(self.stage as usize).min(ZONES.len() - 1)].dps
    }
}

#[derive(Clone, Copy, Default)]
pub struct Plane {
    pub sx: f32,
    pub sz: f32,
    pub dx: f32,
    pub dz: f32,
    pub t: f32,
    pub enter_t: f32,
    pub exit_t: f32,
    pub active: bool,
}

impl Plane {
    pub fn pos(&self) -> V3 {
        V3::new(self.sx + self.dx * self.t, PLANE_ALT, self.sz + self.dz * self.t)
    }
    pub fn can_jump(&self) -> bool {
        self.active && self.t >= self.enter_t && self.t <= self.exit_t
    }
}

struct EvRec {
    to: u8,
    start: u32,
    len: u16,
    x: f32,
    z: f32,
    filt: bool,
    /// E_SHOT: the shooter (hidden shooters are anonymised per viewer).
    shooter: u8,
    /// E_SHOT: the player hit (255 = none), and where the shot would have
    /// ended without them (`alt`, `alt_kind`): viewers that are not sent the
    /// victim see that instead, as the impact would locate the victim.
    victim: u8,
    alt: V3,
    alt_kind: u8,
    /// E_SHOT: the pellet's direction (all a viewer gets of a hidden
    /// shooter's shot that ends near it, quantized).
    dir: V3,
}

/// A loose crate or container: an axis-aligned box that can be knocked about.
#[derive(Clone, Copy, Debug)]
pub struct Prop {
    /// Its collision box in the world.
    pub bx: usize,
    pub crate_: bool,
    /// Middle of the underside.
    pub pos: V3,
    pub vel: V3,
    pub moving: bool,
    moved: u32,
    home: V3,
}

pub struct Match {
    pub w: World,
    pub rng: Rng,
    /// Shot spread only (see `with_secrets`).
    crng: Rng,
    /// Server secret for per-viewer jitter (see `quiet_ticks`).
    salt: u64,
    pub tick: u32,
    pub phase: u8,
    pub lobby_t: f32,
    pub end_t: f32,
    pub match_t: f32,
    pub flags: u32,
    pub slots: usize,
    pub snap_every: u32,
    /// The host is running this step late, catching up on ticks it fell
    /// behind (a hitch): commands carry views made in real time, ahead of
    /// the simulation, so they say nothing about the clients' latency.
    pub catch_up: bool,
    pub players: Vec<Player>,
    pub items: Vec<Item>,
    free_items: Vec<u16>,
    pub nades: Vec<Nade>,
    nade_seq: u16,
    pub zone: Zone,
    pub plane: Plane,
    ev: Vec<u8>,
    evs: Vec<EvRec>,
    pub hist: Vec<[HistE; MAX_PLAYERS]>,
    pub winner: u8,
    pub out: Vec<u8>,
    pub ended_flag: bool,
    pub chunk_ver: Vec<u32>,
    /// Tick of the last item change per chunk.
    chunk_tick: Vec<u32>,
    ilog: Vec<ILogE>,
    ilog_n: u32,
    /// Player whose action is changing items right now (255 = none).
    item_actor: u8,
    /// Alive count at the start of the current tick (shared placement).
    pub alive_start: usize,
    /// Cars: who drives each (255 = nobody), the tick each last moved, and
    /// where they are parked at the start of a match. Poses are in `w.cars`.
    pub car_driver: Vec<u8>,
    car_moved: Vec<u32>,
    /// Shoved cars: the velocity (x, z) each slides with on top of what its
    /// driver does, how fast it turns (rad/s), and who set it going.
    pub car_slide: Vec<[f32; 2]>,
    pub car_spin: Vec<f32>,
    car_pusher: Vec<u8>,
    /// Bodywork left (0 = a burnt-out wreck) and who damaged it last.
    pub car_hp: Vec<f32>,
    car_hit_by: Vec<u8>,
    /// Tick each car caught fire (0 = not burning): it blows up a few seconds later.
    pub car_fire: Vec<u32>,
    pub props: Vec<Prop>,
    car0: Vec<[f32; 4]>,
    /// Helicopters (poses in `w.helis`), where they are parked at the start
    /// of a match, and the missiles in flight.
    pub helis: Vec<Heli>,
    heli0: Vec<[f32; 4]>,
    pub missiles: Vec<Missile>,
    missile_seq: u16,
}

impl Match {
    /// Offline / tests: both secret streams derived from one value.
    pub fn new(seed: u32, slots: usize, flags: u32, secret: u64) -> Match {
        let combat = secret.rotate_left(29) ^ 0xC2B2_AE3D_27D4_EB4F;
        Match::with_secrets(seed, slots, flags, secret, combat)
    }

    /// `seed` is public (clients regenerate the island from it). `secret`
    /// (64 bits of server entropy) seeds the match RNG: loot, plane, zone,
    /// bots. `combat` seeds a separate stream for shot spread that never
    /// feeds anything a client can observe directly.
    pub fn with_secrets(seed: u32, slots: usize, flags: u32, secret: u64, combat: u64) -> Match {
        let w = World::generate(seed);
        let slots = slots.clamp(2, MAX_PLAYERS);
        let ncar = w.cars.len();
        let car0 = w.cars.clone();
        let heli0 = w.helis.clone();
        let mut m = Match {
            w,
            rng: Rng::new(secret ^ (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5EED),
            crng: Rng::new(combat ^ 0x6A09_E667_F3BC_C908),
            salt: mix64(mix64(secret ^ 0xA076_1D64_78BD_642F) ^ combat),
            tick: 0,
            phase: PH_LOBBY,
            lobby_t: if flags & FLAG_OFFLINE != 0 {
                0.0
            } else if flags & FLAG_PRIVATE != 0 {
                90.0
            } else {
                25.0
            },
            end_t: 0.0,
            match_t: 0.0,
            flags,
            slots,
            snap_every: if flags & FLAG_OFFLINE != 0 { 1 } else { 2 },
            catch_up: false,
            players: vec![Player::default(); MAX_PLAYERS],
            items: Vec::new(),
            free_items: Vec::new(),
            nades: Vec::new(),
            nade_seq: 0,
            zone: Zone::default(),
            plane: Plane::default(),
            ev: Vec::with_capacity(8192),
            evs: Vec::with_capacity(512),
            hist: vec![[HistE::default(); MAX_PLAYERS]; HIST],
            winner: 255,
            out: Vec::with_capacity(16384),
            ended_flag: false,
            chunk_ver: vec![1; ICH * ICH],
            chunk_tick: vec![0; ICH * ICH],
            ilog: vec![ILogE::default(); ILOG],
            ilog_n: 0,
            item_actor: 255,
            alive_start: 0,
            car_driver: vec![255; ncar],
            car_moved: vec![0; ncar],
            car_slide: vec![[0.0; 2]; ncar],
            car_spin: vec![0.0; ncar],
            car_pusher: vec![255; ncar],
            car_hp: Vec::new(),
            car_hit_by: vec![255; ncar],
            car_fire: vec![0; ncar],
            props: Vec::new(),
            car0,
            helis: vec![Heli::parked(); heli0.len()],
            heli0,
            missiles: Vec::new(),
            missile_seq: 0,
        };
        m.car_hp = (0..ncar).map(|k| m.car_max_hp(k)).collect();
        m.props = m.w.props.iter().map(|&i| {
            let b = m.w.boxes[i as usize];
            let pos = V3::new((b.min.x + b.max.x) * 0.5, b.min.y, (b.min.z + b.max.z) * 0.5);
            Prop { bx: i as usize, crate_: b.kind == BK_CRATE, pos, vel: V3::ZERO, moving: false, moved: 0, home: pos }
        }).collect();
        m.spawn_loot();
        m.zone = Zone { cx: MAP * 0.5, cz: MAP * 0.5, r: 780.0, nx: MAP * 0.5, nz: MAP * 0.5, nr: 780.0, timer: ZONES[0].wait, ..Default::default() };
        m
    }

    // ------------------------------------------------------------------
    // Items
    // ------------------------------------------------------------------
    fn spawn_loot(&mut self) {
        let spots = self.w.spots.clone();
        for s in spots {
            if !self.rng.chance(0.72) {
                continue;
            }
            let tier = s.tier.min(2) as usize;
            let cat = self.rng.weighted(&[30, 24, 20, 11, 9, 6]);
            match cat {
                0 => {
                    let wt: [[u32; 6]; 3] = [[34, 30, 10, 24, 2, 0], [8, 26, 30, 18, 13, 5], [2, 12, 34, 10, 24, 18]];
                    let wk = 1 + self.rng.weighted(&wt[tier]) as u8;
                    self.add_item(wk, 0, s.pos);
                    let a = WD[wk as usize].ammo as usize;
                    let n = 1 + self.rng.below(2);
                    for k in 0..n {
                        // next to the weapon, on the first side that is clear of
                        // walls / crates and reachable from the spot
                        let d = 0.5 + 0.35 * k as f32;
                        let mut pos = s.pos;
                        for (ox, oz) in [(d, 0.35), (-d, 0.35), (d, -0.35), (-d, -0.35), (0.35, d), (-0.35, -d)] {
                            let gy = self.w.ground_below(s.pos.x + ox, s.pos.z + oz, s.pos.y + 0.5, 0.2).max(WATER - 1.2);
                            let q = V3::new(s.pos.x + ox, gy + 0.05, s.pos.z + oz);
                            let up = V3::new(0.0, 0.25, 0.0);
                            if (q.y - s.pos.y).abs() < 0.5 && self.w.item_spot_clear(q.x, q.y + 0.3, q.z, 0.3) && !self.w.blocked(s.pos + up, q + up) {
                                pos = q;
                                break;
                            }
                        }
                        self.add_item(IK_AMMO + a as u8, AMMO_STACK[a], pos);
                    }
                }
                1 => {
                    let a = self.rng.weighted(&[30, 34, 18, 10]);
                    self.add_item(IK_AMMO + a as u8, AMMO_STACK[a], s.pos);
                }
                2 => {
                    let k = self.rng.weighted(&[60, 22, 18]);
                    let amt = if k == 0 { 3 + self.rng.below(3) as u16 } else { 1 };
                    self.add_item(IK_BANDAGE + k as u8, amt, s.pos);
                }
                3 => {
                    let wt: [[u32; 3]; 3] = [[70, 26, 4], [45, 42, 13], [25, 45, 30]];
                    let lv = self.rng.weighted(&wt[tier]) as u8;
                    self.add_item(IK_VEST + lv, 100, s.pos);
                }
                4 => {
                    let wt: [[u32; 3]; 3] = [[70, 26, 4], [45, 42, 13], [25, 45, 30]];
                    let lv = self.rng.weighted(&wt[tier]) as u8;
                    self.add_item(IK_HELM + lv, 100, s.pos);
                }
                _ => {
                    let amt = 1 + self.rng.below(2) as u16;
                    self.add_item(IK_NADE, amt, s.pos);
                }
            }
        }
    }

    fn add_item(&mut self, kind: u8, amount: u16, pos: V3) -> Option<u16> {
        let it = Item { kind, amount, pos, alive: true };
        // Fresh ids while there is room: a reused id would tie a change far
        // away (hidden) to a new item near the viewer.
        let id = if self.items.len() < MAX_ITEMS {
            self.items.push(it);
            (self.items.len() - 1) as u16
        } else if !self.free_items.is_empty() {
            let id = self.free_items.remove(0);
            self.items[id as usize] = it;
            id
        } else {
            return None;
        };
        self.ev_item(id);
        Some(id)
    }

    /// Item changes are never broadcast (that would leak where players loot).
    /// They are logged; `snapshot` sends each one to the viewers allowed to
    /// know it, and resyncs whole chunks only once they are quiet.
    fn ev_item(&mut self, id: u16) {
        let it = self.items[id as usize];
        self.log_item(id, it);
    }

    fn del_item(&mut self, id: u16) {
        if let Some(it) = self.items.get_mut(id as usize) {
            if it.alive {
                it.alive = false;
                let it = *it;
                self.free_items.push(id);
                self.log_item(id, it);
            }
        }
    }

    fn log_item(&mut self, id: u16, it: Item) {
        let c = item_chunk(it.pos.x, it.pos.z);
        self.chunk_ver[c] = self.chunk_ver[c].wrapping_add(1);
        self.chunk_tick[c] = self.tick;
        let k = self.ilog_n as usize % ILOG;
        self.ilog[k] = ILogE { tick: self.tick, id, actor: self.item_actor, chunk: c as u8, ver: self.chunk_ver[c], it };
        self.ilog_n = self.ilog_n.wrapping_add(1);
        for p in self.players.iter_mut() {
            p.ilog_sent[k / 64] &= !(1u64 << (k % 64));
        }
    }

    fn drop_item_near(&mut self, kind: u8, amount: u16, base: V3, k: usize) {
        // Spiral outwards, skipping spots that are not reachable from the
        // drop point (behind a wall, inside a crate).
        let eye = base + V3::new(0.0, 0.9, 0.0);
        let rr = 0.6 + 0.28 * (k as f32).sqrt();
        let a0 = k as f32 * 2.399 + self.rng.range(0.0, 0.4);
        let mut spot = V3::new(base.x, self.w.ground_below(base.x, base.z, base.y + 0.3, 0.2).max(WATER - 1.2), base.z);
        for tryk in 0..8 {
            let a = a0 + tryk as f32 * 0.785;
            let x = clampf(base.x + a.cos() * rr, 2.0, MAP - 2.0);
            let z = clampf(base.z + a.sin() * rr, 2.0, MAP - 2.0);
            let y = self.w.ground_below(x, z, base.y + 0.3, 0.2).max(WATER - 1.2);
            let p = V3::new(x, y, z);
            if !self.w.blocked(eye, p + V3::new(0.0, 0.25, 0.0)) {
                spot = p;
                break;
            }
        }
        self.add_item(kind, amount, spot + V3::new(0.0, 0.05, 0.0));
    }

    // ------------------------------------------------------------------
    // Events
    // ------------------------------------------------------------------
    fn ev_begin(&self) -> u32 {
        self.ev.len() as u32
    }
    fn ev_end(&mut self, start: u32, to: u8, at: Option<V3>) {
        let len = (self.ev.len() as u32 - start) as u16;
        let (x, z, filt) = match at {
            Some(p) => (p.x, p.z, true),
            None => (0.0, 0.0, false),
        };
        self.evs.push(EvRec { to, start, len, x, z, filt, shooter: 255, victim: 255, alt: V3::ZERO, alt_kind: 0, dir: V3::ZERO });
    }
    fn ev_pinfo(&mut self, i: usize, left: bool) {
        let p = &self.players[i];
        let name = p.name.as_bytes().to_vec();
        let flags = (p.bot as u8) | if left { 2 } else { 0 };
        let color = p.color;
        let start = self.ev_begin();
        let mut w = Wr::new(&mut self.ev);
        w.u8(E_PINFO);
        w.u8(i as u8);
        w.u8(flags);
        w.u8(color);
        w.u8(name.len() as u8);
        w.bytes(&name);
        self.ev_end(start, 255, None);
    }

    // ------------------------------------------------------------------
    // Players
    // ------------------------------------------------------------------
    pub fn human_count(&self) -> usize {
        self.players.iter().filter(|p| p.used && p.human && p.connected).count()
    }
    pub fn alive_count(&self) -> usize {
        self.players.iter().filter(|p| p.used && p.alive).count()
    }

    pub fn add_human(&mut self, raw_name: &[u8]) -> Option<u8> {
        if self.phase != PH_LOBBY {
            return None;
        }
        let i = (0..self.slots).find(|&i| !self.players[i].used)?;
        let base = sanitize_name(raw_name, "Player");
        let mut name = base.clone();
        let mut k = 2;
        while self.players.iter().any(|p| p.used && name_key(&p.name) == name_key(&name)) && k < 40 {
            name = format!("{}#{}", base, k);
            k += 1;
        }
        let color = self.rng.below(12) as u8;
        let p = &mut self.players[i];
        *p = Player::default();
        p.used = true;
        p.human = true;
        p.connected = true;
        p.name = name;
        p.color = color;
        p.need_welcome = true;
        p.rew_floor = REWIND_MAX;
        p.reset_combat();
        p.ms.mode = MODE_PLANE;
        self.ev_pinfo(i, false);
        Some(i as u8)
    }

    pub fn remove_human(&mut self, pid: u8) {
        let i = pid as usize;
        if i >= MAX_PLAYERS || !self.players[i].used || !self.players[i].human {
            return;
        }
        if self.phase == PH_LOBBY {
            self.ev_pinfo(i, true);
            self.players[i] = Player::default();
        } else {
            // AI takes over the abandoned character (on foot, and not as one of the drivers).
            self.exit_car(i);
            self.leave_heli(i);
            let seed = self.rng.u32();
            let pl = self.plane;
            let p = &mut self.players[i];
            p.connected = false;
            p.human = false;
            p.bot = true;
            p.cmdq.clear();
            p.brain = crate::bot::Brain::new(seed, 0.45);
            p.brain.yaw = p.yaw;
            if p.ms.mode == MODE_PLANE && pl.active {
                let (lx, lz, jt) = crate::bot::pick_landing(&self.w, &pl, &mut p.brain, &[]);
                p.brain.land = (lx, lz);
                p.brain.jump_t = jt.max(pl.t);
            } else {
                let f = fwd(p.yaw);
                p.brain.land = (clampf(p.ms.pos.x + f.x * 60.0, 40.0, MAP - 40.0), clampf(p.ms.pos.z + f.z * 60.0, 40.0, MAP - 40.0));
                p.brain.jump_t = pl.t;
            }
            self.ev_pinfo(i, false);
        }
    }

    fn start_match(&mut self) {
        self.phase = PH_PLAY;
        self.match_t = 0.0;
        // fill with bots
        let mut names: Vec<usize> = (0..BOT_NAMES.len()).collect();
        for i in (1..names.len()).rev() {
            let j = self.rng.below(i as u32 + 1) as usize;
            names.swap(i, j);
        }
        let taken: Vec<String> = self.players.iter().filter(|p| p.used).map(|p| name_key(&p.name)).collect();
        let mut ni = 0;
        for i in 0..self.slots {
            if self.players[i].used {
                continue;
            }
            let mut name = String::new();
            while ni < names.len() {
                let cand = BOT_NAMES[names[ni]];
                ni += 1;
                if !taken.iter().any(|t| *t == name_key(cand)) {
                    name = cand.to_string();
                    break;
                }
            }
            if name.is_empty() {
                name = format!("Bot{}", i);
            }
            let skill = self.rng.range(0.1, 0.65);
            let seed = self.rng.u32();
            let color = self.rng.below(12) as u8;
            let p = &mut self.players[i];
            *p = Player::default();
            p.used = true;
            p.bot = true;
            p.name = name;
            p.color = color;
            p.brain = crate::bot::Brain::new(seed, skill);
            p.reset_combat();
            self.ev_pinfo(i, false);
        }
        crate::bot::pick_drivers(&mut self.players);
        // plane path
        let a = self.rng.range(0.0, TAU);
        let (dx, dz) = (a.cos(), a.sin());
        let cx = MAP * 0.5 + self.rng.range(-170.0, 170.0);
        let cz = MAP * 0.5 + self.rng.range(-170.0, 170.0);
        let half = 820.0;
        let (sx, sz) = (cx - dx * half, cz - dz * half);
        // compute entry/exit of the playable square [30, MAP-30]
        let mut enter = f32::MAX;
        let mut exit = 0.0f32;
        let mut t = 0.0;
        while t <= 2.0 * half {
            let x = sx + dx * t;
            let z = sz + dz * t;
            if x > 130.0 && z > 130.0 && x < MAP - 130.0 && z < MAP - 130.0 {
                enter = enter.min(t);
                exit = exit.max(t);
            }
            t += 5.0;
        }
        if enter > exit {
            enter = half - 100.0;
            exit = half + 100.0;
        }
        self.plane = Plane { sx, sz, dx, dz, t: (enter - 260.0).max(0.0), enter_t: enter, exit_t: exit, active: true };
        for k in 0..self.props.len() {
            let home = self.props[k].home;
            self.props[k].pos = home;
            self.props[k].vel = V3::ZERO;
            self.props[k].moving = false;
            self.props[k].moved = self.tick;
            self.w.move_box(self.props[k].bx, home);
        }
        self.w.cars = self.car0.clone();
        for k in 0..self.car_driver.len() {
            self.car_driver[k] = 255;
            self.car_moved[k] = self.tick;
            self.car_slide[k] = [0.0; 2];
            self.car_spin[k] = 0.0;
            self.car_pusher[k] = 255;
            self.car_hp[k] = self.car_max_hp(k);
            self.car_hit_by[k] = 255;
            self.car_fire[k] = 0;
        }
        self.w.helis = self.heli0.clone();
        for h in self.helis.iter_mut() {
            *h = Heli::parked();
        }
        self.missiles.clear();
        self.zone = Zone { cx: MAP * 0.5, cz: MAP * 0.5, r: 780.0, timer: ZONES[0].wait, ..Default::default() };
        self.pick_next_zone();
        let ppos = self.plane.pos();
        let mut landings: Vec<(f32, f32)> = Vec::new();
        for i in 0..self.slots {
            let pl = self.plane;
            let p = &mut self.players[i];
            if !p.used {
                continue;
            }
            p.reset_combat();
            p.ms = MoveState { pos: ppos, mode: MODE_PLANE, ..Default::default() };
            p.yaw = yaw_to(dx, dz);
            if p.bot {
                let (lx, lz, jt) = crate::bot::pick_landing(&self.w, &pl, &mut p.brain, &landings);
                p.brain.land = (lx, lz);
                p.brain.jump_t = jt;
                landings.push((lx, lz));
            }
        }
        self.record_hist();
    }

    // ------------------------------------------------------------------
    // Input
    // ------------------------------------------------------------------
    pub fn on_message(&mut self, pid: u8, data: &[u8]) {
        let i = pid as usize;
        if i >= MAX_PLAYERS || !self.players[i].used || !self.players[i].human || data.is_empty() {
            return;
        }
        let mut r = Rd::new(data);
        match r.u8() {
            C_INPUT => {
                // A message may carry several commands: u8 count, then 17 bytes each.
                let n = r.u8().min(8);
                for _ in 0..n {
                    let seq = r.u32();
                    let view = r.u32();
                    let mx = r.i8();
                    let mz = r.i8();
                    let yaw = r.u16();
                    let pitch = r.i16();
                    let buttons = r.u16();
                    if r.bad {
                        return;
                    }
                    let c = Cmd::from_quant(seq, view, mx, mz, yaw, pitch, buttons);
                    let phase = self.phase;
                    let p = &mut self.players[i];
                    if seq <= p.last_seq {
                        continue;
                    }
                    p.last_seq = seq;
                    if phase == PH_PLAY {
                        // Simulated one per tick in step(): arrival timing can never
                        // speed a player up, freeze gravity or hide positions between ticks.
                        if p.cmdq.len() >= CMDQ_CAP {
                            if let Some(old) = p.cmdq.pop_front() {
                                p.carry = Self::absorb(p.carry, old.buttons);
                                p.acked_seq = old.seq;
                                p.last_cmd = old;
                                // the oldest queued commands are the late ones that
                                // idle steps already covered: this drop pays for one
                                p.debt = p.debt.saturating_sub(1);
                            }
                        }
                        p.cmdq.push_back(c);
                    } else {
                        p.yaw = c.yaw;
                        p.pitch = c.pitch;
                        p.acked_seq = seq;
                        p.last_cmd = c;
                    }
                }
            }
            C_START => {
                if self.phase == PH_LOBBY && self.flags & FLAG_PRIVATE != 0 {
                    let host = (0..MAX_PLAYERS).find(|&k| self.players[k].used && self.players[k].human);
                    if host == Some(i) {
                        self.lobby_t = self.lobby_t.min(3.0);
                    }
                }
            }
            _ => {}
        }
    }

    fn apply_cmd(&mut self, i: usize, c: Cmd) {
        let tick = self.tick;
        let human = self.players[i].human;
        let prev = self.players[i].prev_buttons;
        // Humans get one action per press: holding a button must not repeat it.
        let pressed = |b: u16| c.has(b) && (!human || prev & b == 0);
        let (pick, spect, use_) = (pressed(B_PICKUP), pressed(B_SPECT), pressed(B_USE));
        {
            let p = &mut self.players[i];
            p.prev_buttons = c.buttons;
            p.yaw = wrap_angle(c.yaw);
            p.pitch = clampf(c.pitch, -PITCH_LIMIT, PITCH_LIMIT);
        }
        if !self.players[i].alive {
            if spect && self.players[i].spect_cd <= 0.0 {
                self.cycle_spect(i);
            }
            return;
        }
        let mode = self.players[i].ms.mode;
        if mode == MODE_PLANE {
            if c.has(B_JUMP) && self.plane.can_jump() {
                self.eject(i);
            }
            return;
        }
        if mode == MODE_DRIVE && use_ && car_speed(&self.players[i].ms).abs() < 9.0 {
            self.exit_car(i);
            return;
        }
        if mode == MODE_FLY {
            let ms = self.players[i].ms;
            // out only once it has landed and stands still
            if use_ && ms.on_ground && ms.vel.len() < 1.5 {
                self.exit_heli(i);
                return;
            }
            step_move(&self.w, &mut self.players[i].ms, &c, 1.0, DT);
            self.heli_moved_by(i, ms.vel, &c);
            return;
        }
        let mult = self.players[i].speed_mult();
        let w = &self.w;
        let p = &mut self.players[i];
        let v0 = if mode == MODE_DRIVE { car_speed(&p.ms).abs() } else { 0.0 };
        let vel0 = p.ms.vel;
        let mut hits = DriveHits::default();
        let ev = step_move_hits(w, &mut p.ms, &c, mult, DT, &mut hits);
        if mode == MODE_DRIVE {
            self.car_moved_by(i, v0, vel0, &hits);
            return;
        }
        if ev & EV_LANDED != 0 && mode != MODE_GROUND {
            let pos = p.ms.pos;
            let start = self.ev_begin();
            let mut wr = Wr::new(&mut self.ev);
            wr.u8(E_LAND);
            wr.u8(i as u8);
            self.ev_end(start, i as u8, Some(pos));
        }
        let p = &mut self.players[i];
        if p.ms.mode != MODE_GROUND {
            p.ads = false;
            return;
        }
        p.ads = c.has(B_ADS) && !p.ms.in_water;
        // weapon slot request (edge): 0 none, 1 slot A, 2 slot B, 3 fists
        let req = c.slot();
        if req > 0 {
            let want = req - 1;
            if want != p.cur {
                p.cur = want;
                p.reload_t = 0.0;
                p.heal_t = 0.0;
                p.switch_t = 0.35;
            }
        }
        let wk2 = p.weapon();
        if c.has(B_RELOAD) {
            Self::start_reload(p);
        }
        let hk = c.heal();
        if hk > 0 && p.heal_t <= 0.0 {
            let k = (hk - 1) as usize;
            let ok = p.heals[k] > 0 && match k {
                0 => p.health < 75.0,
                1 => p.health < 100.0,
                _ => true,
            };
            if ok {
                p.heal_t = HEAL_TIME[k];
                p.heal_kind = hk;
                p.reload_t = 0.0;
            }
        }
        if use_ && (self.enter_car(i) || self.enter_heli(i)) {
            return;
        }
        // leaning on a crate slides it along
        if c.mx != 0.0 || c.mz != 0.0 {
            let ms = self.players[i].ms;
            let wish = (right(c.yaw) * c.mx + fwd(c.yaw) * c.mz).norm();
            for j in 0..self.props.len() {
                let pr = self.props[j];
                if !pr.crate_ || pr.moving {
                    continue;
                }
                let b = self.w.boxes[pr.bx];
                if (b.min.x - ms.pos.x).abs() > 4.0 || (b.min.z - ms.pos.z).abs() > 4.0 || b.max.y <= ms.pos.y + STEP || b.min.y >= ms.pos.y + 1.5 {
                    continue;
                }
                let (dx, dz) = (clampf(ms.pos.x, b.min.x, b.max.x) - ms.pos.x, clampf(ms.pos.z, b.min.z, b.max.z) - ms.pos.z);
                if dx * dx + dz * dz < (P_RADIUS + 0.1) * (P_RADIUS + 0.1) && dx * wish.x + dz * wish.z > 0.02 {
                    self.kick_prop(j, wish * 2.6);
                }
            }
        }
        if pick && self.players[i].pick_cd <= 0.0 {
            self.manual_pickup(i);
        }
        if c.has(B_NADE) {
            self.throw_nade(i, &c);
        }
        let p = &mut self.players[i];
        if c.has(B_FIRE) && !p.ms.in_water && !(c.has(B_SPRINT) && c.mz > 0.3 && !c.has(B_ADS)) {
            p.heal_t = 0.0;
            if p.switch_t <= 0.0 && p.reload_t <= 0.0 {
                if wk2 == W_FISTS {
                    if p.fire_cd <= 0.0 {
                        p.fire_cd += WD[0].interval;
                        p.firing_t = 0.3;
                        self.melee(i, tick);
                    }
                } else {
                    let s = p.cur as usize;
                    let def = &WD[wk2 as usize];
                    let mut shots = 0;
                    while p.fire_cd <= 0.0 && p.slots[s].1 > 0 && shots < 3 {
                        p.fire_cd += def.interval;
                        p.slots[s].1 -= 1;
                        shots += 1;
                    }
                    if shots > 0 {
                        p.firing_t = 0.25;
                    }
                    let empty = p.slots[s].1 == 0;
                    for _ in 0..shots {
                        self.shoot(i, wk2, c.view);
                    }
                    if empty {
                        Self::start_reload(&mut self.players[i]);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Cars
    // ------------------------------------------------------------------
    /// Get into the nearest free car within reach.
    fn enter_car(&mut self, i: usize) -> bool {
        let pos = self.players[i].ms.pos;
        let mut best: Option<(usize, f32)> = None;
        for k in 0..self.w.cars.len() {
            if self.car_driver[k] != 255 || self.car_hp[k] <= 0.0 {
                continue;
            }
            let c = self.w.cars[k];
            if (pos.y - c[1]).abs() > 1.6 {
                continue;
            }
            // distance to the body: how far a 1.5 m circle overlaps it
            let mut q = pos;
            if !self.w.push_car(k, &mut q, 1.5) {
                continue;
            }
            // not through a wall
            if self.w.blocked_static(pos + V3::new(0.0, 1.0, 0.0), V3::new(c[0], c[1] + 1.0, c[2])) {
                continue;
            }
            let d = dist_xz(pos.x, pos.z, c[0], c[2]);
            if best.map_or(true, |b| d < b.1) {
                best = Some((k, d));
            }
        }
        let Some((k, _)) = best else { return false };
        let c = self.w.cars[k];
        self.car_driver[k] = i as u8;
        self.car_moved[k] = self.tick;
        let p = &mut self.players[i];
        p.ms = MoveState { pos: V3::new(c[0], c[1], c[2]), mode: MODE_DRIVE, on_ground: true, airtime: c[3], car: k as u8, ..Default::default() };
        p.ads = false;
        p.reload_t = 0.0;
        p.heal_t = 0.0;
        p.firing_t = 0.0;
        true
    }

    pub fn car_max_hp(&self, k: usize) -> f32 {
        if self.w.car_kind.get(k) == Some(&VK_TRUCK) { 1100.0 } else { 600.0 }
    }

    /// Damage the bodywork of car `k`; it blows up when nothing is left.
    pub fn hurt_car(&mut self, k: usize, dmg: f32, by: u8) {
        if k >= self.car_hp.len() || self.car_hp[k] <= 0.0 || dmg <= 0.0 || self.phase != PH_PLAY {
            return;
        }
        self.car_hp[k] -= dmg;
        self.car_hit_by[k] = by;
        self.car_moved[k] = self.tick;
        if self.car_hp[k] <= 0.0 {
            self.explode_car(k);
        } else if self.car_hp[k] < self.car_max_hp(k) * CAR_FIRE_AT && self.car_fire[k] == 0 {
            // the engine catches fire: nothing saves the car now
            self.car_fire[k] = self.tick.max(1);
        }
    }

    // ------------------------------------------------------------------
    // Loose props
    // ------------------------------------------------------------------
    /// Set a prop moving. Whatever stood on it falls.
    pub fn kick_prop(&mut self, k: usize, vel: V3) {
        let p = self.props[k];
        self.props[k].vel = if p.moving { p.vel + vel * 0.5 } else { vel };
        self.props[k].moving = true;
        self.props[k].moved = self.tick;
        let b = self.w.boxes[p.bx];
        for j in 0..self.props.len() {
            let q = self.props[j];
            let o = self.w.boxes[q.bx];
            if j != k && !q.moving && (o.min.y - b.max.y).abs() < 0.05 && o.min.x < b.max.x && o.max.x > b.min.x && o.min.z < b.max.z && o.max.z > b.min.z {
                self.props[j].moving = true;
                self.props[j].moved = self.tick;
            }
        }
    }

    /// Props near `pos` thrown outwards by a blast.
    fn blast_props(&mut self, pos: V3, radius: f32) {
        for k in 0..self.props.len() {
            let p = self.props[k];
            let b = self.w.boxes[p.bx];
            let c = (b.min + b.max) * 0.5;
            let d = (c - pos).len();
            if d >= radius {
                continue;
            }
            let f = 1.0 - d / radius;
            let dir = V3::new(c.x - pos.x, 0.0, c.z - pos.z).norm();
            let v = if p.crate_ { dir * (16.0 * f + 2.0) + V3::new(0.0, 4.0 + 9.0 * f, 0.0) } else { dir * (4.5 * f) + V3::new(0.0, 2.5 * f, 0.0) };
            self.kick_prop(k, v);
        }
    }

    /// Moving props fly, bounce, slide and come to rest.
    fn update_props(&mut self) {
        let tick = self.tick;
        for k in 0..self.props.len() {
            if !self.props[k].moving {
                continue;
            }
            let mut p = self.props[k];
            let b = self.w.boxes[p.bx];
            let (hx, hz, hy) = ((b.max.x - b.min.x) * 0.5, (b.max.z - b.min.z) * 0.5, b.max.y - b.min.y);
            let r = hx.min(hz);
            // out of the grid while it moves, so it does not collide with itself
            self.w.unlink_box(p.bx);
            p.vel.y -= 20.0 * DT;
            let mut np = p.pos + p.vel * DT;
            let mut q = V3::new(np.x, np.y + 0.05, np.z);
            if self.w.collide_body(&mut q, r, hy) | self.w.push_cars(&mut q, r, usize::MAX) | self.w.push_helis(&mut q, r, hy, usize::MAX) {
                // off a wall, a car or another prop: most of the speed is lost
                let (dx, dz) = (q.x - np.x, q.z - np.z);
                let l = (dx * dx + dz * dz).sqrt().max(1e-4);
                let into = (p.vel.x * dx + p.vel.z * dz) / l;
                if into < 0.0 {
                    p.vel.x -= dx / l * into * 1.3;
                    p.vel.z -= dz / l * into * 1.3;
                }
                p.vel.x *= 0.8;
                p.vel.z *= 0.8;
                np.x = q.x;
                np.z = q.z;
            }
            let g = self.w.ground_below(np.x, np.z, np.y.max(p.pos.y), r);
            let mut grounded = false;
            if np.y <= g {
                np.y = g;
                grounded = true;
                if p.vel.y < -4.0 {
                    p.vel.y *= -0.32;
                } else {
                    p.vel.y = 0.0;
                }
                let fr = if p.crate_ { 5.0 } else { 11.0 };
                let k2 = (1.0 - fr * DT).max(0.0);
                p.vel.x *= k2;
                p.vel.z *= k2;
            }
            np.x = clampf(np.x, 4.0, MAP - 4.0);
            np.z = clampf(np.z, 4.0, MAP - 4.0);
            if !np.finite() || !p.vel.finite() {
                np = p.home;
                p.vel = V3::ZERO;
            }
            p.pos = np;
            if grounded && p.vel.len() < 0.35 {
                p.vel = V3::ZERO;
                p.moving = false;
            }
            p.moved = tick;
            self.props[k] = p;
            // (move_box unlinks first: harmless for a box already out of the grid)
            self.w.move_box(p.bx, np);
            if grounded && !p.crate_ {
                self.box_clears_cars(p.bx);
            }
        }
    }

    /// A container that came down where an empty car stands (the stack it
    /// was on was knocked away): the car is moved out from under it. (A
    /// driven one is kept out of it by its own step.)
    fn box_clears_cars(&mut self, bx: usize) {
        let b = self.w.boxes[bx];
        for j in 0..self.w.cars.len() {
            let c = self.w.cars[j];
            if c[0] < b.min.x - 4.0 || c[0] > b.max.x + 4.0 || c[2] < b.min.z - 4.0 || c[2] > b.max.z + 4.0 || b.max.y < c[1] || b.min.y > c[1] + 1.6 {
                continue;
            }
            let dr = self.car_driver[j] as usize;
            if dr < MAX_PLAYERS && self.car_of(dr) == Some(j) {
                continue;
            }
            let mut np = V3::new(c[0], c[1], c[2]);
            car_out_of_walls(&self.w, self.w.car_kind[j], &mut np, c[3]);
            if !np.finite() || (np.x == c[0] && np.z == c[2]) {
                continue;
            }
            np.x = clampf(np.x, 4.0, MAP - 4.0);
            np.z = clampf(np.z, 4.0, MAP - 4.0);
            self.w.cars[j] = [np.x, self.w.height(np.x, np.z), np.z, c[3]];
            self.car_moved[j] = self.tick;
            self.cars_make_room(j);
        }
    }

    /// A car knocks crates flying and shoves containers.
    fn car_hits_props(&mut self, k: usize, v0: f32) {
        let f = fwd(self.w.cars[k][3]);
        let dir = if self.players[self.car_driver[k] as usize % MAX_PLAYERS].ms.vel.dot(f) >= 0.0 { f } else { f * -1.0 };
        self.car_knocks_props(k, v0, dir);
    }

    /// Car `k`, going `v0` m/s along `dir`, against the props it touches.
    fn car_knocks_props(&mut self, k: usize, v0: f32, dir: V3) {
        if v0 < 1.5 {
            return;
        }
        let c = self.w.cars[k];
        let f = fwd(c[3]);
        let (circles, cr, _) = car_shape(self.w.car_kind[k]);
        for j in 0..self.props.len() {
            let p = self.props[j];
            let b = self.w.boxes[p.bx];
            if (b.min.x - c[0]).abs() > 9.0 || (b.min.z - c[2]).abs() > 9.0 || b.min.y > c[1] + 1.6 || b.max.y < c[1] {
                continue;
            }
            let reach = cr + if p.crate_ { 0.1 } else { 0.3 };
            let touch = circles.iter().any(|&off| {
                let (x, z) = (c[0] + f.x * off, c[2] + f.z * off);
                let (dx, dz) = (x - clampf(x, b.min.x, b.max.x), z - clampf(z, b.min.z, b.max.z));
                dx * dx + dz * dz < reach * reach
            });
            if !touch {
                continue;
            }
            let away = V3::new((b.min.x + b.max.x) * 0.5 - c[0], 0.0, (b.min.z + b.max.z) * 0.5 - c[2]).norm();
            if p.crate_ {
                if p.moving && p.vel.len() > v0 * 0.6 {
                    continue; // already on its way
                }
                let jitter = self.rng.range(-0.25, 0.25);
                self.kick_prop(j, (dir * 0.75 + away * 0.45 + right(c[3]) * jitter) * (v0 * 1.25 + 2.0) + V3::new(0.0, 3.5 + v0 * 0.42, 0.0));
            } else if !p.moving && away.dot(dir) > 0.2 {
                self.kick_prop(j, dir * (v0 * 0.45).min(5.5));
            }
        }
    }

    /// Burning cars blow up when their fuse runs out.
    fn update_car_fires(&mut self) {
        for k in 0..self.car_fire.len() {
            let lit = self.car_fire[k];
            if lit != 0 && self.car_hp[k] > 0.0 && self.tick.wrapping_sub(lit) >= CAR_FUSE_TICKS {
                self.explode_car(k);
            }
        }
    }

    /// Car `k` blows up: whoever is in it dies, people and cars nearby are
    /// hurt, and a wreck is left that still blocks the way.
    fn explode_car(&mut self, k: usize) {
        let c = self.w.cars[k];
        let by = self.car_hit_by[k];
        self.car_hp[k] = 0.0;
        self.car_moved[k] = self.tick;
        let pos = V3::new(c[0], c[1] + 1.0, c[2]);
        let dr = self.car_driver[k] as usize;
        if dr < MAX_PLAYERS {
            let who = if (by as usize) < MAX_PLAYERS { by } else { dr as u8 };
            self.damage(dr, who, 1.0e6, false, WK_BOOM, pos);
            if self.players[dr].alive {
                self.exit_car(dr);
            }
        }
        self.car_driver[k] = 255;
        let start = self.ev_begin();
        let mut w = Wr::new(&mut self.ev);
        w.u8(E_BOOM);
        w.f32(pos.x);
        w.f32(pos.y);
        w.f32(pos.z);
        self.ev_end(start, 255, Some(pos));
        let radius = 8.5;
        for j in 0..MAX_PLAYERS {
            let q = &self.players[j];
            if !q.used || !q.alive || q.ms.mode == MODE_PLANE || q.ms.mode == MODE_FLY {
                continue;
            }
            let at = q.ms.pos + V3::new(0.0, 0.9, 0.0);
            let d = (at - pos).len();
            // the wreck itself is no cover from its own blast
            if d > radius || self.w.blocked_static(pos + V3::new(0.0, 0.8, 0.0), at) {
                continue;
            }
            let dmg = 125.0 * (1.0 - d / radius).powf(1.3);
            if dmg > 1.0 {
                let who = if (by as usize) < MAX_PLAYERS { by } else { j as u8 };
                self.damage(j, who, dmg, false, WK_BOOM, pos);
            }
        }
        self.blast_props(pos, radius + 1.5);
        for o in 0..self.w.cars.len() {
            let oc = self.w.cars[o];
            let d = dist_xz(oc[0], oc[2], c[0], c[2]);
            if o != k && d < radius {
                self.hurt_car(o, 520.0 * (1.0 - d / radius), by);
            }
        }
        self.blast_helis(pos, radius, 520.0, 1.0, by);
    }

    /// The car `i` drives, if any.
    fn car_of(&self, i: usize) -> Option<usize> {
        let p = &self.players[i];
        let k = p.ms.car as usize;
        (p.ms.mode == MODE_DRIVE && k < self.car_driver.len() && self.car_driver[k] == i as u8).then_some(k)
    }

    /// Step out beside the car (which stops where it is).
    fn exit_car(&mut self, i: usize) {
        let Some(k) = self.car_of(i) else {
            if self.players[i].ms.mode == MODE_DRIVE {
                self.players[i].ms.mode = MODE_GROUND;
            }
            return;
        };
        self.car_driver[k] = 255;
        self.car_moved[k] = self.tick;
        self.sync_car_vel(k);
        let c = self.w.cars[k];
        let (f, r) = (fwd(c[3]), right(c[3]));
        let (hw, hl, _) = self.w.car_dims(k);
        let (ds, dl) = (hw + 0.65, hl + 0.65);
        // driver's side first, then the other side, the back and the front
        // (never through a wall the car is parked against; with every side
        // walled in, out of the middle of the car)
        let mid = V3::new(c[0], c[1] + 1.0, c[2]);
        let mut out = V3::new(c[0], c[1], c[2]);
        let mut least = f32::MAX;
        for (sx, sz) in [(-ds, 0.0), (ds, 0.0), (0.0, -dl), (0.0, dl)] {
            let (x, z) = (c[0] + r.x * sx + f.x * sz, c[2] + r.z * sx + f.z * sz);
            let y = self.w.ground_below(x, z, c[1] + 1.0, P_RADIUS);
            if (y - c[1]).abs() > 1.2 {
                continue;
            }
            let at = V3::new(x, y, z);
            if self.w.blocked_static(mid, at + V3::new(0.0, 1.0, 0.0)) {
                continue;
            }
            let mut q = at;
            self.w.collide_body(&mut q, P_RADIUS, 1.8);
            self.w.push_cars(&mut q, P_RADIUS, usize::MAX);
            self.w.push_helis(&mut q, P_RADIUS, 1.8, usize::MAX);
            let moved = (q - at).len();
            if moved < least {
                least = moved;
                out = q;
            }
            if moved < 0.01 {
                break;
            }
        }
        let p = &mut self.players[i];
        p.ms = MoveState { pos: out, mode: MODE_GROUND, on_ground: true, ..Default::default() };
        p.switch_t = p.switch_t.max(0.35);
    }

    /// How fast car `k` really moves: its driver's speed and its slide.
    pub fn car_velocity(&self, k: usize) -> (f32, f32) {
        let s = self.car_slide[k];
        let dr = self.car_driver[k] as usize;
        if dr < MAX_PLAYERS && self.car_of(dr) == Some(k) {
            let v = self.players[dr].ms.vel;
            (s[0] + v.x, s[1] + v.z)
        } else {
            (s[0], s[1])
        }
    }

    fn sync_car_vel(&mut self, k: usize) {
        let (x, z) = self.car_velocity(k);
        if let Some(v) = self.w.car_vel.get_mut(k) {
            *v = [x, z];
        }
    }

    /// Change the velocity of car `k` by (`dvx`, `dvz`), pushed at the point
    /// (`cx`, `cz`): off-centre it turns the car as well. With a driver
    /// aboard, what goes along the car is added to their own speed.
    pub fn shove_car(&mut self, k: usize, dvx: f32, dvz: f32, cx: f32, cz: f32, by: u8) {
        if k >= self.car_slide.len() || !(dvx.is_finite() && dvz.is_finite()) {
            return;
        }
        let c = self.w.cars[k];
        let kind = self.w.car_kind[k];
        let f = fwd(c[3]);
        let (mut sx, mut sz) = (dvx, dvz);
        let dr = self.car_driver[k] as usize;
        if dr < MAX_PLAYERS && self.car_of(dr) == Some(k) {
            let along = dvx * f.x + dvz * f.z;
            let lim = car_shape(kind).2 + 8.0;
            let ms = &mut self.players[dr].ms;
            let v = clampf(car_speed(ms) + along, -lim, lim);
            ms.vel = V3::new(f.x * v, 0.0, f.z * v);
            sx -= f.x * along;
            sz -= f.z * along;
        }
        let s = &mut self.car_slide[k];
        s[0] += sx;
        s[1] += sz;
        let l = (s[0] * s[0] + s[1] * s[1]).sqrt();
        if l > CAR_SLIDE_MAX {
            s[0] *= CAR_SLIDE_MAX / l;
            s[1] *= CAR_SLIDE_MAX / l;
        }
        self.car_spin[k] = clampf(self.car_spin[k] + shove_spin(kind, dvx, dvz, cx - c[0], cz - c[2]), -CAR_SPIN_MAX, CAR_SPIN_MAX);
        self.car_pusher[k] = by;
        self.car_moved[k] = self.tick;
        self.sync_car_vel(k);
    }

    /// Car `k`, moving at (`vx`, `vz`), touches car `j`; (`nx`, `nz`) points
    /// from `j` to `k`. `j` is shoved along and dented by a hard knock.
    /// Returns the closing speed (0 when they are not closing).
    fn car_hit_car(&mut self, k: usize, j: usize, nx: f32, nz: f32, cx: f32, cz: f32, vx: f32, vz: f32, by: u8) -> f32 {
        let (ox, oz) = self.car_velocity(j);
        let closing = -((vx - ox) * nx + (vz - oz) * nz);
        self.car_struck(k, j, nx, nz, cx, cz, closing, by)
    }

    /// Car `j` is run into by car `k` at `closing` m/s along (`nx`, `nz`)
    /// (from `j` to `k`): shoved along, and dented by a hard knock.
    fn car_struck(&mut self, k: usize, j: usize, nx: f32, nz: f32, cx: f32, cz: f32, closing: f32, by: u8) -> f32 {
        if !(closing > 0.05) || j >= self.w.cars.len() || j == k {
            return 0.0;
        }
        let (mk, mj) = (car_mass(self.w.car_kind[k]), car_mass(self.w.car_kind[j]));
        let dv = closing * hit_share(mj, mk);
        self.shove_car(j, -nx * dv, -nz * dv, cx, cz, by);
        if closing > CAR_CRASH_V {
            self.hurt_car(j, (closing - CAR_CRASH_V) * CAR_CRASH_DMG * 2.0 * mk / (mk + mj), by);
        }
        closing
    }

    /// Car `k`, moving at (`vx`, `vz`), against the helicopters it touches:
    /// they are far heavier, so a hard knock only nudges them.
    fn car_hits_helis(&mut self, k: usize, vx: f32, vz: f32, by: u8) {
        if vx * vx + vz * vz < HELI_NUDGE_V * HELI_NUDGE_V {
            return;
        }
        let c = self.w.cars[k];
        let kind = self.w.car_kind[k];
        let f = fwd(c[3]);
        let (circles, cr, _) = car_shape(kind);
        for h in 0..self.helis.len() {
            let o = self.w.helis[h];
            if (o[0] - c[0]).abs() > 14.0 || (o[2] - c[2]).abs() > 14.0 {
                continue;
            }
            let mut best: Option<(f32, f32, f32)> = None;
            for off in circles {
                let base = V3::new(c[0] + f.x * off, c[1] + 0.45, c[2] + f.z * off);
                let mut q = base;
                if self.w.push_heli(h, &mut q, cr + CAR_TOUCH, 1.0) {
                    let (dx, dz) = (q.x - base.x, q.z - base.z);
                    let d = (dx * dx + dz * dz).sqrt();
                    if d > 1e-5 && best.map_or(true, |b| d > b.2) {
                        best = Some((dx / d, dz / d, d));
                    }
                }
            }
            let Some((nx, nz, _)) = best else { continue };
            let pilot = self.helis[h].pilot as usize;
            let piloted = pilot < MAX_PLAYERS && self.heli_of(pilot) == Some(h);
            let hv = if piloted { self.players[pilot].ms.vel } else { self.helis[h].vel };
            let closing = -((vx - hv.x) * nx + (vz - hv.z) * nz);
            if closing < HELI_NUDGE_V {
                continue;
            }
            // (never enough to set a standing one skidding)
            let dv = (closing * hit_share(HELI_MASS, car_mass(kind))).min(HELI_STAND_V * 0.9);
            let had = (hv.x * hv.x + hv.z * hv.z).sqrt();
            if had + dv >= HELI_STAND_V && !self.helis[h].air {
                continue;
            }
            if piloted {
                let v = &mut self.players[pilot].ms.vel;
                v.x -= nx * dv;
                v.z -= nz * dv;
            } else {
                let v = &mut self.helis[h].vel;
                v.x -= nx * dv;
                v.z -= nz * dv;
            }
            if closing > CAR_CRASH_V {
                self.hurt_heli(h, (closing - CAR_CRASH_V) * 6.0 * car_mass(kind), by);
            }
        }
    }

    /// A helicopter in the air (or skidding fast) that touches a car shoves
    /// it out of its way. `v0` is how fast the helicopter came in.
    fn heli_shoves_cars(&mut self, h: usize, v0: V3, by: u8) {
        let o = self.w.helis[h];
        let f = fwd(o[3]);
        for k in 0..self.w.cars.len() {
            let c = self.w.cars[k];
            if (c[0] - o[0]).abs() > 14.0 || (c[2] - o[2]).abs() > 14.0 {
                continue;
            }
            let mut best: Option<(f32, f32, f32, V3)> = None;
            for off in HELI_CIRCLES {
                let base = V3::new(o[0] + f.x * off, o[1] + 0.2, o[2] + f.z * off);
                let mut q = base;
                if self.w.push_car(k, &mut q, HELI_CR + CAR_TOUCH) {
                    let (dx, dz) = (q.x - base.x, q.z - base.z);
                    let d = (dx * dx + dz * dz).sqrt();
                    if d > 1e-5 && best.map_or(true, |b| d > b.2) {
                        best = Some((dx / d, dz / d, d, base));
                    }
                }
            }
            // (nx, nz): from the car to the helicopter
            let Some((nx, nz, _, base)) = best else { continue };
            let (cvx, cvz) = self.car_velocity(k);
            let closing = -((v0.x - cvx) * nx + (v0.z - cvz) * nz);
            // out of the way at a walking pace at least (it is coming down on the car)
            let dv = (closing.max(0.0) * hit_share(car_mass(self.w.car_kind[k]), HELI_MASS)).max(HELI_SHOVE_MIN + closing.min(0.0));
            if dv <= 0.0 {
                continue;
            }
            self.shove_car(k, -nx * dv, -nz * dv, base.x - nx * HELI_CR, base.z - nz * HELI_CR, by);
            if closing > CAR_CRASH_V {
                self.hurt_car(k, (closing - CAR_CRASH_V) * CAR_CRASH_DMG * 1.8, by);
            }
        }
    }

    /// Car `k` was pushed back out of a wall, maybe into the vehicles that
    /// pinned it there: those are moved out of its way (and out of walls
    /// themselves), so that nothing comes to rest inside something else.
    fn cars_make_room(&mut self, k: usize) {
        let tick = self.tick;
        for j in 0..self.w.cars.len() {
            let (a, b) = (self.w.cars[k], self.w.cars[j]);
            if j == k || (a[0] - b[0]).abs() > 8.0 || (a[2] - b[2]).abs() > 8.0 || (a[1] - b[1]).abs() > 3.0 {
                continue;
            }
            let jk = self.w.car_kind[j];
            let mut moved = false;
            for _ in 0..2 {
                let b = self.w.cars[j];
                let Some((nx, nz, depth)) = car_overlap(a, self.w.car_kind[k], b, jk) else { break };
                let mut np = V3::new(b[0] - nx * depth, b[1], b[2] - nz * depth);
                car_out_of_walls(&self.w, jk, &mut np, b[3]);
                np.x = clampf(np.x, 4.0, MAP - 4.0);
                np.z = clampf(np.z, 4.0, MAP - 4.0);
                np.y = self.w.height(np.x, np.z);
                if !np.finite() || (np.x == b[0] && np.z == b[2]) {
                    break;
                }
                self.w.cars[j] = [np.x, np.y, np.z, b[3]];
                moved = true;
            }
            if moved {
                self.car_moved[j] = tick;
                let dr = self.car_driver[j] as usize;
                if dr < MAX_PLAYERS && self.car_of(dr) == Some(j) {
                    let c = self.w.cars[j];
                    self.players[dr].ms.pos = V3::new(c[0], c[1], c[2]);
                }
            }
        }
    }

    /// Shoved cars slide on, slow down, turn, bump into the world and into
    /// each other, and come to rest.
    fn update_cars(&mut self) {
        let tick = self.tick;
        for k in 0..self.w.cars.len() {
            let [mut sx, mut sz] = self.car_slide[k];
            let mut spin = self.car_spin[k];
            if sx == 0.0 && sz == 0.0 && spin == 0.0 {
                continue;
            }
            let c = self.w.cars[k];
            let kind = self.w.car_kind[k];
            let by = self.car_pusher[k];
            let dr = self.car_driver[k] as usize;
            let driver = (dr < MAX_PLAYERS && self.car_of(dr) == Some(k)).then_some(dr);
            let (vx, vz) = self.car_velocity(k);
            let sp = (sx * sx + sz * sz).sqrt();
            let dir = if sp > 1e-4 { V3::new(sx / sp, 0.0, sz / sp) } else { V3::ZERO };
            let yaw = wrap_angle(c[3] + spin * DT);
            let mut np = V3::new(c[0] + sx * DT, c[1], c[2] + sz * DT);
            // a bank too steep to slide up
            let reach = car_shape(kind).0[2] + 0.7;
            let rise = self.w.height(np.x + dir.x * reach, np.z + dir.z * reach) - self.w.height(np.x, np.z);
            if rise > 1.9 {
                np.x = c[0];
                np.z = c[2];
                sx = 0.0;
                sz = 0.0;
            } else if rise > 0.5 {
                let k2 = (1.0 - (rise - 0.5) * 9.0 * DT / sp.max(0.5)).max(0.0);
                sx *= k2;
                sz *= k2;
            }
            // (a fast one in short steps, up to whatever it touches first: it
            // must not skip through a thin wall)
            let mut hits = [CarHit::default(); CAR_HITS];
            let (mut solid, mut n) = (V3::ZERO, 0);
            let steps = ((sp * DT / 0.5).ceil() as usize).clamp(1, 4);
            let (ex, ez) = (np.x - c[0], np.z - c[2]);
            np.x = c[0];
            np.z = c[2];
            for _ in 0..steps {
                np.x += ex / steps as f32;
                np.z += ez / steps as f32;
                let (_, s2, n2) = car_collide(&self.w, k, kind, &mut np, yaw, &mut hits);
                solid = s2;
                n = n2;
                if n2 > 0 || s2.x != 0.0 || s2.z != 0.0 {
                    break;
                }
            }
            // squeezed between a vehicle and a wall: the wall wins (and the
            // vehicle is moved out of the way below)
            let walled = {
                let w0 = np;
                car_out_of_walls(&self.w, kind, &mut np, yaw);
                solid.x += np.x - w0.x;
                solid.z += np.z - w0.z;
                np.x != w0.x || np.z != w0.z
            };
            let mut dent = 0.0f32;
            let sl = (solid.x * solid.x + solid.z * solid.z).sqrt();
            if sl > 1e-4 {
                // a wall, a tree or a helicopter takes the speed towards it
                let (nx, nz) = (solid.x / sl, solid.z / sl);
                let into = sx * nx + sz * nz;
                if into < 0.0 {
                    sx -= nx * into * 1.1;
                    sz -= nz * into * 1.1;
                    dent = dent.max(-into - CAR_CRASH_V);
                }
                spin *= 0.5;
            }
            let mk = car_mass(kind);
            for h in &hits[..n] {
                let Some((nx, nz, cx, cz)) = h.normal() else { continue };
                let closing = self.car_hit_car(k, h.k, nx, nz, cx, cz, vx, vz, by);
                if closing > 0.0 {
                    let mj = car_mass(self.w.car_kind[h.k]);
                    let dv = closing * hit_share(mk, mj);
                    sx += nx * dv;
                    sz += nz * dv;
                    spin = clampf(spin + shove_spin(kind, nx * dv, nz * dv, cx - np.x, cz - np.z), -CAR_SPIN_MAX, CAR_SPIN_MAX);
                    dent = dent.max((closing - CAR_CRASH_V) * 2.0 * mj / (mk + mj));
                }
            }
            // rolling along its wheels is easy, sideways the tyres bite
            let f = fwd(yaw);
            (sx, sz, spin) = slide_friction(sx, sz, spin, yaw, self.w.height(np.x, np.z) < WATER - 0.45, DT);
            let (bx, bz) = (clampf(np.x, 4.0, MAP - 4.0), clampf(np.z, 4.0, MAP - 4.0));
            if bx != np.x {
                sx = 0.0;
            }
            if bz != np.z {
                sz = 0.0;
            }
            np.x = bx;
            np.z = bz;
            np.y = self.w.height(np.x, np.z);
            if sx * sx + sz * sz < CAR_REST_V * CAR_REST_V && spin.abs() < CAR_REST_SPIN {
                sx = 0.0;
                sz = 0.0;
                spin = 0.0;
            }
            if !(np.finite() && sx.is_finite() && sz.is_finite() && spin.is_finite() && yaw.is_finite()) {
                self.car_slide[k] = [0.0; 2];
                self.car_spin[k] = 0.0;
                self.sync_car_vel(k);
                continue;
            }
            self.w.cars[k] = [np.x, np.y, np.z, yaw];
            self.car_moved[k] = tick;
            if walled {
                self.cars_make_room(k);
            }
            self.car_slide[k] = [sx, sz];
            self.car_spin[k] = spin;
            if let Some(d) = driver {
                // the driver goes with the car, at the speed they had along it
                let ms = &mut self.players[d].ms;
                let v = car_speed(ms);
                ms.pos = np;
                ms.airtime = yaw;
                ms.vel = V3::new(f.x * v, 0.0, f.z * v);
            }
            self.sync_car_vel(k);
            self.car_knocks_props(k, sp, dir);
            self.car_hits_helis(k, vx, vz, by);
            if driver.is_none() {
                // people in its way: pushed aside, and hurt by a fast one
                // (a driven car does this after its driver's step)
                for j in 0..MAX_PLAYERS {
                    let q = &self.players[j];
                    if !q.used || !q.alive || q.ms.mode != MODE_GROUND {
                        continue;
                    }
                    let mut at = q.ms.pos;
                    if !self.w.push_car(k, &mut at, P_RADIUS) {
                        continue;
                    }
                    let hgt = body_h(q.ms.crouch);
                    self.w.collide_body(&mut at, P_RADIUS, hgt);
                    let towards = (q.ms.pos.x - c[0]) * dir.x + (q.ms.pos.z - c[2]) * dir.z > 0.0;
                    let q = &mut self.players[j];
                    q.ms.pos.x = at.x;
                    q.ms.pos.z = at.z;
                    if sp > CAR_SLIDE_HURT_V && towards && tick.wrapping_sub(q.ram_tick) > 24 {
                        q.ram_tick = tick;
                        let who = if (by as usize) < MAX_PLAYERS && self.players[by as usize].used { by } else { j as u8 };
                        let wk = if kind == VK_TRUCK { WK_TRUCK } else { WK_CAR };
                        self.damage(j, who, 8.0 + (sp - CAR_SLIDE_HURT_V) * 4.0, false, wk, np);
                        let s = &mut self.car_slide[k];
                        s[0] *= 0.85;
                        s[1] *= 0.85;
                    }
                }
            }
            if dent > 0.0 {
                self.hurt_car(k, dent * CAR_CRASH_DMG, by);
            }
        }
    }

    /// After the driver's step: move the car with them and run people over.
    fn car_moved_by(&mut self, i: usize, v0: f32, vel0: V3, hits: &DriveHits) {
        let Some(k) = self.car_of(i) else {
            self.exit_car(i);
            return;
        };
        let tick = self.tick;
        let ms = self.players[i].ms;
        let now = [ms.pos.x, ms.pos.y, ms.pos.z, ms.airtime];
        if self.w.cars[k] != now {
            self.w.cars[k] = now;
            self.car_moved[k] = tick;
        }
        let v = car_speed(&ms);
        self.car_hits_props(k, v0.max(v.abs()));
        // the vehicles it ran into are shoved along: each gets exactly what
        // the driver paid for with its own speed in `step_drive` (the same
        // closing speed, so that pushing loses no engine power)
        {
            let s = self.car_slide[k];
            let (vx, vz) = (vel0.x + s[0], vel0.z + s[1]);
            for h in &hits.hits[..hits.n] {
                if let Some((nx, nz, cx, cz)) = h.normal() {
                    self.car_struck(k, h.k, nx, nz, cx, cz, h.closing, i as u8);
                }
            }
            self.car_hits_helis(k, vx, vz, i as u8);
            self.sync_car_vel(k);
            if self.car_of(i) != Some(k) {
                // (the crash blew something up, and that killed the driver)
                return;
            }
        }
        // a crash: the speed lost in one step dents the car
        let lost = v0 - v.abs();
        if lost > 5.0 {
            self.hurt_car(k, (lost - 5.0) * 14.0, i as u8);
            if self.car_hp[k] <= 0.0 {
                return;
            }
        }
        for j in 0..MAX_PLAYERS {
            let q = &self.players[j];
            if j == i || !q.used || !q.alive || q.ms.mode != MODE_GROUND {
                continue;
            }
            let mut at = q.ms.pos;
            if !self.w.push_car(k, &mut at, P_RADIUS) {
                continue;
            }
            let hgt = body_h(q.ms.crouch);
            self.w.collide_body(&mut at, P_RADIUS, hgt);
            let q = &mut self.players[j];
            q.ms.pos.x = at.x;
            q.ms.pos.z = at.z;
            if v.abs() > 4.0 && tick.wrapping_sub(q.ram_tick) > 24 {
                q.ram_tick = tick;
                let dmg = 12.0 + (v.abs() - 4.0) * 5.0;
                let wk = if self.w.car_kind[k] == VK_TRUCK { WK_TRUCK } else { WK_CAR };
                self.damage(j, i as u8, dmg, false, wk, ms.pos);
                // the hit costs the car some speed
                let p = &mut self.players[i];
                p.ms.vel = p.ms.vel * 0.85;
            }
        }
    }

    // ------------------------------------------------------------------
    // Helicopters
    // ------------------------------------------------------------------
    /// The helicopter `i` flies, if any.
    fn heli_of(&self, i: usize) -> Option<usize> {
        let p = &self.players[i];
        let k = p.ms.car as usize;
        (p.ms.mode == MODE_FLY && k < self.helis.len() && self.helis[k].pilot == i as u8).then_some(k)
    }

    /// Board the nearest landed, unpiloted, intact helicopter within reach.
    fn enter_heli(&mut self, i: usize) -> bool {
        let pos = self.players[i].ms.pos;
        let mut best: Option<(usize, f32)> = None;
        for k in 0..self.helis.len() {
            let h = self.helis[k];
            if h.pilot != 255 || h.hp <= 0.0 || h.air {
                continue;
            }
            let c = self.w.helis[k];
            if (pos.y - c[1]).abs() > 1.6 {
                continue;
            }
            // distance to the hull: how far a 1.8 m circle overlaps it
            let mut q = pos;
            if !self.w.push_heli(k, &mut q, 1.8, 1.8) {
                continue;
            }
            // not through a wall
            if self.w.blocked_static(pos + V3::new(0.0, 1.0, 0.0), q + V3::new(0.0, 1.0, 0.0)) {
                continue;
            }
            let d = (q - pos).len();
            if best.map_or(true, |b| d > b.1) {
                best = Some((k, d));
            }
        }
        let Some((k, _)) = best else { return false };
        let c = self.w.helis[k];
        self.helis[k].pilot = i as u8;
        self.helis[k].vel = V3::ZERO;
        let p = &mut self.players[i];
        p.ms = MoveState { pos: V3::new(c[0], c[1], c[2]), mode: MODE_FLY, on_ground: true, airtime: c[3], car: k as u8, ..Default::default() };
        p.ads = false;
        p.reload_t = 0.0;
        p.heal_t = 0.0;
        p.firing_t = 0.0;
        true
    }

    /// Step out beside the landed helicopter.
    fn exit_heli(&mut self, i: usize) {
        let Some(k) = self.heli_of(i) else {
            if self.players[i].ms.mode == MODE_FLY {
                self.players[i].ms.mode = MODE_GROUND;
            }
            return;
        };
        self.helis[k].pilot = 255;
        self.helis[k].vel = V3::ZERO;
        let c = self.w.helis[k];
        let (f, r) = (fwd(c[3]), right(c[3]));
        let (ds, dl) = (HELI_HW + 0.75, HELI_HL + 0.75);
        // by the cockpit on the right, then on the left, further back, and
        // off the nose or the tail (never through a wall it stands against;
        // with every side walled in, out of the middle)
        let mid = V3::new(c[0], c[1] + 1.0, c[2]);
        let mut out = V3::new(c[0], c[1], c[2]);
        let mut least = f32::MAX;
        for (sx, sz) in [(ds, 4.5), (-ds, 4.5), (ds, 0.0), (-ds, 0.0), (ds, -4.5), (-ds, -4.5), (0.0, dl), (0.0, -dl)] {
            let (x, z) = (c[0] + r.x * sx + f.x * sz, c[2] + r.z * sx + f.z * sz);
            let y = self.w.ground_below(x, z, c[1] + 1.0, P_RADIUS);
            if (y - c[1]).abs() > 1.6 {
                continue;
            }
            let at = V3::new(x, y, z);
            if self.w.blocked_static(mid, at + V3::new(0.0, 1.0, 0.0)) {
                continue;
            }
            let mut q = at;
            self.w.collide_body(&mut q, P_RADIUS, 1.8);
            self.w.push_cars(&mut q, P_RADIUS, usize::MAX);
            self.w.push_helis(&mut q, P_RADIUS, 1.8, usize::MAX);
            let moved = (q - at).len();
            if moved < least {
                least = moved;
                out = q;
            }
            if moved < 0.01 {
                break;
            }
        }
        let p = &mut self.players[i];
        p.ms = MoveState { pos: out, mode: MODE_GROUND, on_ground: true, ..Default::default() };
        p.switch_t = p.switch_t.max(0.35);
    }

    /// The pilot goes whatever the helicopter is doing (its player left the
    /// match): beside it when it has landed, under a parachute when it is
    /// in the air. The empty helicopter then comes down by itself
    /// (`update_helis`).
    fn leave_heli(&mut self, i: usize) {
        let Some(k) = self.heli_of(i) else {
            if self.players[i].ms.mode == MODE_FLY {
                self.players[i].ms.mode = MODE_GROUND;
            }
            return;
        };
        if !self.helis[k].air {
            self.exit_heli(i);
            return;
        }
        let ms = self.players[i].ms;
        self.helis[k].pilot = 255;
        self.helis[k].vel = ms.vel;
        let r = right(ms.airtime);
        let mut at = ms.pos + r * (HELI_HW + 2.5) + V3::new(0.0, 1.0, 0.0);
        self.w.collide_body(&mut at, P_RADIUS, 1.8);
        at.y = at.y.max(self.w.ground_below(at.x, at.z, at.y, P_RADIUS));
        let p = &mut self.players[i];
        p.ms = MoveState { pos: at, vel: V3::new(ms.vel.x * 0.5, -2.0, ms.vel.z * 0.5), mode: MODE_PARA, ..Default::default() };
    }

    /// Damage the hull of helicopter `k`; it blows up when nothing is left.
    pub fn hurt_heli(&mut self, k: usize, dmg: f32, by: u8) {
        if k >= self.helis.len() || self.helis[k].hp <= 0.0 || dmg <= 0.0 || self.phase != PH_PLAY {
            return;
        }
        self.helis[k].hp -= dmg;
        self.helis[k].hit_by = by;
        if self.helis[k].hp <= 0.0 {
            self.explode_heli(k);
        }
    }

    /// Helicopters within `radius` of a blast at `pos` take up to `dmg`.
    fn blast_helis(&mut self, pos: V3, radius: f32, dmg: f32, pw: f32, by: u8) {
        for k in 0..self.helis.len() {
            let c = self.w.helis[k];
            // distance to the hull: to its axis, less its girth
            let f = fwd(c[3]);
            let along = clampf((pos.x - c[0]) * f.x + (pos.z - c[2]) * f.z, -6.0, 6.0);
            let d = ((V3::new(c[0] + f.x * along, c[1] + 1.8, c[2] + f.z * along) - pos).len() - 1.7).max(0.0);
            if d < radius {
                self.hurt_heli(k, dmg * (1.0 - d / radius).powf(pw), by);
            }
        }
    }

    /// Helicopter `k` blows up: its pilot dies, people and vehicles nearby
    /// are hurt, and a wreck is left (it falls if it was flying).
    fn explode_heli(&mut self, k: usize) {
        let c = self.w.helis[k];
        let by = self.helis[k].hit_by;
        self.helis[k].hp = 0.0;
        self.helis[k].ammo = 0;
        let pos = V3::new(c[0], c[1] + 1.8, c[2]);
        let dr = self.helis[k].pilot as usize;
        if dr < MAX_PLAYERS {
            let who = if (by as usize) < MAX_PLAYERS { by } else { dr as u8 };
            self.damage(dr, who, 1.0e6, false, WK_BOOM, pos);
        }
        self.helis[k].pilot = 255;
        let start = self.ev_begin();
        let mut w = Wr::new(&mut self.ev);
        w.u8(E_BOOM);
        w.f32(pos.x);
        w.f32(pos.y);
        w.f32(pos.z);
        self.ev_end(start, 255, Some(pos));
        let radius = 11.0;
        for j in 0..MAX_PLAYERS {
            let q = &self.players[j];
            if !q.used || !q.alive || q.ms.mode == MODE_PLANE || q.ms.mode == MODE_FLY {
                continue;
            }
            let at = q.ms.pos + V3::new(0.0, 0.9, 0.0);
            let d = (at - pos).len();
            // the wreck itself is no cover from its own blast
            if d > radius || self.w.blocked_static(pos, at) {
                continue;
            }
            let dmg = 130.0 * (1.0 - d / radius).powf(1.3);
            if dmg > 1.0 {
                let who = if (by as usize) < MAX_PLAYERS { by } else { j as u8 };
                self.damage(j, who, dmg, false, WK_BOOM, pos);
            }
        }
        self.blast_props(pos, radius + 1.5);
        for o in 0..self.w.cars.len() {
            let oc = self.w.cars[o];
            let d = (V3::new(oc[0], oc[1] + 0.7, oc[2]) - pos).len();
            if d < radius {
                self.hurt_car(o, 600.0 * (1.0 - d / radius), by);
            }
        }
        // (the wreck itself has nothing left to lose)
        self.blast_helis(pos, radius, 600.0, 1.0, by);
    }

    /// After the pilot's step: move the helicopter with them, dent it when
    /// it flew into something, shove people aside and launch missiles.
    fn heli_moved_by(&mut self, i: usize, v0: V3, c: &Cmd) {
        let Some(k) = self.heli_of(i) else {
            self.exit_heli(i);
            return;
        };
        let tick = self.tick;
        let ms = self.players[i].ms;
        let old = self.w.helis[k];
        self.w.helis[k] = [ms.pos.x, ms.pos.y, ms.pos.z, ms.airtime];
        {
            // attitude, for show: nose down with forward speed and thrust,
            // banking into a turn and with a sideways slide
            let (f, r) = (fwd(ms.airtime), right(ms.airtime));
            let a = (ms.vel - v0) * (1.0 / DT);
            let (vf, vs) = (ms.vel.dot(f), ms.vel.dot(r));
            let turn = wrap_angle(ms.airtime - old[3]) / DT;
            let (tp, tr) = if ms.on_ground {
                (0.0, 0.0)
            } else {
                (clampf(vf / HELI_VMAX * 0.12 + a.dot(f) * 0.014, -0.3, 0.3), clampf(vs / HELI_VSIDE * 0.08 + a.dot(r) * 0.014 - turn * vf * 0.004, -0.35, 0.35))
            };
            let h = &mut self.helis[k];
            h.pitch = lerpf(h.pitch, tp, 0.12);
            h.roll = lerpf(h.roll, tr, 0.12);
            h.vel = ms.vel;
            h.air = !ms.on_ground;
        }
        if !ms.on_ground || v0.x * v0.x + v0.z * v0.z > HELI_SHOVE_V * HELI_SHOVE_V {
            self.heli_shoves_cars(k, v0, i as u8);
        }
        // a crash: the speed lost in one step (a wall, a tree, the ground
        // taken at speed) dents the hull
        let lost = v0.len() - ms.vel.len();
        if lost > 2.0 {
            self.hurt_heli(k, (lost - 2.0) * 40.0, i as u8);
            if self.helis[k].hp <= 0.0 {
                return;
            }
        }
        let v = ms.vel.len();
        for j in 0..MAX_PLAYERS {
            let q = &self.players[j];
            if j == i || !q.used || !q.alive || q.ms.mode != MODE_GROUND {
                continue;
            }
            let hgt = body_h(q.ms.crouch);
            let mut at = q.ms.pos;
            if !self.w.push_heli(k, &mut at, P_RADIUS, hgt) {
                continue;
            }
            self.w.collide_body(&mut at, P_RADIUS, hgt);
            let q = &mut self.players[j];
            q.ms.pos.x = at.x;
            q.ms.pos.z = at.z;
            if v > 4.0 && tick.wrapping_sub(q.ram_tick) > 24 {
                q.ram_tick = tick;
                self.damage(j, i as u8, 12.0 + (v - 4.0) * 5.0, false, WK_HELI, ms.pos);
            }
        }
        if c.has(B_FIRE) {
            self.fire_missile(i, k);
        }
    }

    /// Launch a missile from helicopter `k` along its pilot's aim, if one
    /// is ready. The pods fire in turn.
    fn fire_missile(&mut self, i: usize, k: usize) {
        let h = self.helis[k];
        if h.cd > 0.0 || h.ammo == 0 || h.hp <= 0.0 {
            return;
        }
        let c = self.w.helis[k];
        let p = &self.players[i];
        // only ahead: within 60 degrees of the nose, and not straight up
        let off = clampf(wrap_angle(p.yaw - c[3]), -1.05, 1.05);
        let d = dir_from(c[3] + off, clampf(p.pitch, -1.3, 0.6));
        let (f, r) = (fwd(c[3]), right(c[3]));
        let pos = V3::new(c[0], c[1], c[2]);
        let side = if h.side { 1.0 } else { -1.0 };
        let from = pos + r * (side * MISSILE_POD[0]) + V3::UP * MISSILE_POD[1] + f * MISSILE_POD[2];
        // Both pods converge on what the line from the middle of the hull
        // along the aim meets (the world only: vehicles move).
        let mid = pos + V3::new(0.0, 2.0, 0.0);
        let reach = MISSILE_SPEED * MISSILE_LIFE as f32 * DT;
        let t = self.w.raycast_static(mid, d, reach).map_or(reach, |x| x.0).max(30.0);
        let dir = (mid + d * t - from).norm();
        self.missile_seq = self.missile_seq.wrapping_add(1);
        let id = self.missile_seq;
        self.missiles.push(Missile { pos: from, dir, life: MISSILE_LIFE, owner: i as u8, heli: k as u8, id });
        let h = &mut self.helis[k];
        h.ammo -= 1;
        h.cd = MISSILE_CD;
        h.side = !h.side;
    }

    /// Missiles fly straight and go off on the world, a vehicle, close to
    /// a player, or at the end of their run.
    fn update_missiles(&mut self) {
        let mut k = 0;
        while k < self.missiles.len() {
            let mut m = self.missiles[k];
            let mut lim = MISSILE_SPEED * DT;
            let mut hit = false;
            if let Some((t, _, _)) = self.w.raycast_static(m.pos, m.dir, lim) {
                lim = t;
                hit = true;
            }
            if let Some((t, _, _)) = self.w.ray_cars(m.pos, m.dir, lim) {
                lim = t;
                hit = true;
            }
            if let Some((t, _, _)) = self.w.ray_helis(m.pos, m.dir, lim, m.heli as usize) {
                lim = t;
                hit = true;
            }
            // the sea
            if m.dir.y < -1e-4 && m.pos.y > WATER && (m.pos.y - WATER) / -m.dir.y < lim {
                lim = (m.pos.y - WATER) / -m.dir.y;
                hit = true;
            }
            // passing within a metre of somebody on foot or in the air
            for j in 0..MAX_PLAYERS {
                let q = &self.players[j];
                if j == m.owner as usize || !q.used || !q.alive || !matches!(q.ms.mode, MODE_GROUND | MODE_FALL | MODE_PARA) {
                    continue;
                }
                let c = q.ms.pos + V3::new(0.0, 0.9, 0.0);
                let along = clampf((c - m.pos).dot(m.dir), 0.0, lim);
                if (m.pos + m.dir * along - c).len() < 1.1 {
                    lim = (along - 0.4).max(0.0).min(lim);
                    hit = true;
                }
            }
            m.pos += m.dir * lim;
            m.life = m.life.saturating_sub(1);
            let out = m.pos.x < 1.0 || m.pos.z < 1.0 || m.pos.x > MAP - 1.0 || m.pos.z > MAP - 1.0;
            if hit || out || m.life == 0 || !m.pos.finite() {
                self.missiles.swap_remove(k);
                // one that leaves the map is simply gone: a blast at the edge
                // would wreck the helicopter that fired it from there
                if m.pos.finite() && (hit || !out) {
                    // (just short of what it hit, so the blast is not inside it)
                    self.blast(m.pos - m.dir * 0.25, m.owner, MISSILE_R, MISSILE_DMG, 1.4, MISSILE_CAR_DMG, WK_MISSILE);
                }
                continue;
            }
            self.missiles[k] = m;
            k += 1;
        }
    }

    /// Per tick: cooldowns, rotors, and helicopters nobody flies coming
    /// down. An intact one left in the air autorotates: it sinks, slows
    /// and settles gently where it is; a wreck drops.
    fn update_helis(&mut self) {
        for k in 0..self.helis.len() {
            let pi = self.helis[k].pilot as usize;
            if pi < MAX_PLAYERS && self.heli_of(pi).is_none() {
                // the pilot is gone (dead, or put out of it)
                self.helis[k].pilot = 255;
            }
            let c = self.w.helis[k];
            let mut h = self.helis[k];
            h.cd = (h.cd - DT).max(0.0);
            h.rotor = if h.hp <= 0.0 {
                (h.rotor - DT / 2.0).max(0.0)
            } else if h.pilot != 255 {
                (h.rotor + DT / 3.0).min(1.0)
            } else if h.air {
                approach(h.rotor, 0.7, DT / 4.0)
            } else {
                (h.rotor - DT / 8.0).max(0.0)
            };
            if h.pilot != 255 {
                self.helis[k] = h;
                continue;
            }
            h.pitch = lerpf(h.pitch, 0.0, 0.08);
            h.roll = lerpf(h.roll, 0.0, 0.08);
            let pos = V3::new(c[0], c[1], c[2]);
            if !h.air {
                if h.vel.x * h.vel.x + h.vel.z * h.vel.z < 0.01 {
                    h.vel = V3::ZERO;
                    self.helis[k] = h;
                    continue;
                }
                // knocked by a vehicle: it skids a little and stands again
                let mut ms = MoveState { pos, vel: V3::new(h.vel.x, 0.0, h.vel.z), mode: MODE_FLY, on_ground: true, airtime: c[3], car: k as u8, ..Default::default() };
                step_fly(&self.w, &mut ms, &Cmd { yaw: c[3], ..Default::default() }, DT);
                if ms.pos.finite() && ms.vel.finite() {
                    self.w.helis[k] = [ms.pos.x, ms.pos.y, ms.pos.z, c[3]];
                    h.vel = ms.vel;
                    h.air = !ms.on_ground;
                } else {
                    h.vel = V3::ZERO;
                }
                self.helis[k] = h;
                continue;
            }
            let mut impact = 0.0;
            if h.hp > 0.0 {
                let v0 = h.vel;
                let mut ms = MoveState { pos, vel: h.vel, mode: MODE_FLY, airtime: c[3], car: k as u8, ..Default::default() };
                step_fly(&self.w, &mut ms, &Cmd { yaw: c[3], buttons: B_CROUCH, ..Default::default() }, DT);
                self.w.helis[k] = [ms.pos.x, ms.pos.y, ms.pos.z, c[3]];
                h.vel = ms.vel;
                h.air = !ms.on_ground;
                self.helis[k] = h;
                // (coming down on a car, it shoves the car out from under it)
                self.heli_shoves_cars(k, v0, h.hit_by);
                h = self.helis[k];
            } else {
                h.vel.y = (h.vel.y - 14.0 * DT).max(-40.0);
                h.vel.x *= 0.99;
                h.vel.z *= 0.99;
                let mut np = pos + h.vel * DT;
                np.x = clampf(np.x, 10.0, MAP - 10.0);
                np.z = clampf(np.z, 10.0, MAP - 10.0);
                let push = heli_collide(&self.w, k, &mut np, c[3]);
                if push.len2() > 1e-8 {
                    h.vel.x = 0.0;
                    h.vel.z = 0.0;
                }
                let g = heli_floor(&self.w, np, c[3], pos.y.max(np.y));
                if np.y <= g {
                    np.y = g;
                    impact = h.vel.len();
                    h.vel = V3::ZERO;
                    h.air = false;
                }
                if !np.finite() {
                    np = pos;
                    h.air = false;
                }
                self.w.helis[k] = [np.x, np.y, np.z, c[3]];
            }
            self.helis[k] = h;
            if impact > 12.0 {
                // the wreck hits the ground
                let at = V3::new(self.w.helis[k][0], self.w.helis[k][1] + 1.5, self.w.helis[k][2]);
                self.blast(at, h.hit_by, 7.0, 90.0, 1.3, 300.0, WK_BOOM);
            }
        }
    }

    fn start_reload(p: &mut Player) {
        if p.cur >= 2 || p.reload_t > 0.0 || p.switch_t > 0.0 {
            return;
        }
        let (wk, mag) = p.slots[p.cur as usize];
        if wk == 0 {
            return;
        }
        let def = &WD[wk as usize];
        if mag >= def.mag || p.ammo[def.ammo as usize] == 0 {
            return;
        }
        p.reload_t = def.reload;
        p.heal_t = 0.0;
    }

    fn cycle_spect(&mut self, i: usize) {
        let start = self.players[i].spect as usize;
        for k in 1..=MAX_PLAYERS {
            let j = (start.wrapping_add(k)) % MAX_PLAYERS;
            if j != i && self.players[j].used && self.players[j].alive {
                self.set_spect(i, j as u8);
                // switching is rate limited (cycling through everyone would be a radar)
                self.players[i].spect_cd = 1.5;
                return;
            }
        }
    }

    /// Spectate another player. Visibility is recomputed from the new eye, so
    /// the union of several players' views is never sent.
    fn set_spect(&mut self, i: usize, j: u8) {
        let p = &mut self.players[i];
        if p.spect != j {
            p.spect = j;
            p.seen_tick = [0; MAX_PLAYERS];
            p.vis_check = [0; MAX_PLAYERS];
        }
    }

    fn eject(&mut self, i: usize) {
        let pp = self.plane.pos();
        let (dx, dz) = (self.plane.dx, self.plane.dz);
        let p = &mut self.players[i];
        p.ms.mode = MODE_FALL;
        p.ms.pos = pp - V3::new(dx, 0.0, dz) * 4.0 - V3::new(0.0, 3.0, 0.0);
        p.ms.vel = V3::new(dx * 20.0, -5.0, dz * 20.0);
        p.ms.airtime = 0.0;
    }

    // ------------------------------------------------------------------
    // Combat
    // ------------------------------------------------------------------
    /// Player `j` as viewer `i`'s client drew it at `view`.
    fn rewound(&self, i: usize, j: usize, view: u32) -> Option<HistE> {
        let now = self.tick as f32;
        // Clients render ~4.5 ticks behind + latency + command queueing;
        // 15 ticks (500 ms) covers RTTs up to ~250 ms. Commands run before
        // record_hist() writes this tick's slot, so tick - 1 is the newest
        // recorded state (hist[tick % HIST] still holds tick - HIST).
        let vt = clampf(view as f32 / 256.0, (now - REWIND_MAX).max(0.0), (now - 1.0).max(0.0));
        let t0u = vt.floor() as u32;
        if self.tick >= HIST as u32 && t0u + (HIST as u32) <= self.tick {
            return None;
        }
        let a = self.hist[t0u as usize % HIST][j];
        let b = if t0u + 1 < self.tick { self.hist[(t0u as usize + 1) % HIST][j] } else { a };
        if !a.alive && !b.alive {
            return None;
        }
        // The player as the client draws it: snapshots go out every
        // snap_every ticks, and the client interpolates position, yaw and
        // pitch between the two around its render time and takes the stance
        // (crouch, mode) from the newer one. A pose held only between
        // snapshots (crouching on odd ticks, say) is never drawn, so it must
        // not decide the hit either.
        let se = self.snap_every.max(1);
        let latest = (self.tick.saturating_sub(1) / se) * se;
        let sa = ((t0u / se) * se).min(latest);
        let sb = (sa + se).min(latest);
        let (pa, pb) = (self.hist[sa as usize % HIST][j], self.hist[sb as usize % HIST][j]);
        let mut f = if sb > sa { clampf((vt - sa as f32) / (sb - sa) as f32, 0.0, 1.0) } else { 0.0 };
        // A player missing from the older snapshot (it just came into view)
        // is drawn at the newer one, without interpolation.
        let from = self.players[i].sent_from[j];
        if from > sa && from <= sb {
            f = 1.0;
        }
        // Yaw and pitch exactly as the client computes them: from the
        // snapshot's quantised values, in f64 (src/protocol.ts, lerpAngle in
        // src/render/math.ts). A half turn between snapshots is a tie that
        // f32 and f64 rounding would break in opposite directions.
        let tau = std::f64::consts::PI * 2.0;
        let (ya, yb) = (q_yaw(pa.yaw) as f64 * tau / 65536.0, q_yaw(pb.yaw) as f64 * tau / 65536.0);
        let mut dy = (yb - ya) % tau;
        if dy > std::f64::consts::PI {
            dy -= tau;
        } else if dy <= -std::f64::consts::PI {
            dy += tau;
        }
        let (qa, qb) = (q_pitch(pa.pitch) as f64 * 1.5 / 32767.0, q_pitch(pb.pitch) as f64 * 1.5 / 32767.0);
        let f64f = f as f64;
        Some(HistE {
            pos: pa.pos.lerp(pb.pos, f),
            alive: true,
            yaw: (ya + dy * f64f) as f32,
            pitch: (qa + (qb - qa) * f64f) as f32,
            ..pb
        })
    }

    /// Returns (victim, distance, headshot).
    pub fn ray_players(&self, shooter: usize, o: V3, d: V3, maxt: f32, view: u32) -> Option<(usize, f32, bool)> {
        let mut best: Option<(usize, f32, bool)> = None;
        let mut lim = maxt;
        for j in 0..MAX_PLAYERS {
            if j == shooter {
                continue;
            }
            let q = &self.players[j];
            if !q.used || !q.alive || q.ms.mode == MODE_PLANE || q.ms.mode == MODE_DRIVE || q.ms.mode == MODE_FLY {
                continue;
            }
            let h = if self.players[shooter].bot { HistE::of(q) } else {
                match self.rewound(shooter, j, view) {
                    Some(h) => h,
                    None => continue,
                }
            };
            let p = h.pos;
            // cheap reject
            let to = p - o;
            let along = to.dot(d);
            if along < -2.0 || along > lim + 2.0 {
                continue;
            }
            let perp2 = (to - d * along).len2();
            if perp2 > 4.0 {
                continue;
            }
            // head: a sphere on the rendered head plus the drawn head boxes
            // (helmet included); the body capsule reaches the shoulders. A ray that reaches both is a body hit only if it
            // enters the capsule at the legs (below the hip along the body
            // axis, which is tilted in free fall) or passes a drawn part of
            // the upper body (torso, arms, weapon, medkit) before the head:
            // the capsule is fatter than the drawn body, and a ray that rises
            // past the chest to the chin is drawn hitting the head.
            let pose = body_pose(p, h.yaw, h.pitch, h.crouch, h.sprint, h.mode);
            let gear = h.gear();
            let th = pose.ray_head(o, d, lim, gear.helm);
            let upright = h.mode != MODE_FALL;
            let tb = if upright { ray_capsule(o, d, p.x, p.z, pose.a.y, pose.b.y, 0.33, lim) } else { ray_capsule_seg(o, d, pose.a, pose.b, 0.33, lim) };
            let legs = |tb: f32| {
                let ax = (pose.b - pose.a).norm();
                (o + d * tb - pose.a).dot(ax) < (pose.hip - pose.a).dot(ax)
            };
            let hit = match (th, tb) {
                (Some(th), Some(tb)) if tb < th && legs(tb) => Some((tb, false)),
                (Some(th), tb) if pose.ray_upper(o, d, th, gear) => Some((tb.map_or(th, |tb| tb.min(th)), false)),
                (Some(th), _) => Some((th, true)),
                (None, Some(tb)) => Some((tb, false)),
                (None, None) => None,
            };
            if let Some((t, head)) = hit {
                lim = t;
                best = Some((j, t, head));
            }
        }
        best
    }

    fn shoot(&mut self, i: usize, wk: u8, view: u32) {
        let p = &self.players[i];
        let def = WD[wk as usize];
        let scoped = p.ads && def.zoom >= SCOPE_ZOOM;
        let (o, d0) = aim_ray_safe(&self.w, p.ms.pos, p.yaw, p.pitch, p.ms.crouch, p.ads, scoped);
        let muz = muzzle(p.ms.pos, p.yaw, p.pitch, p.ms.crouch);
        // The eye is on the body axis, so it is never inside a wall.
        let eye = p.ms.pos + V3::new(0.0, eye_h(p.ms.crouch), 0.0);
        let speed = p.ms.vel.len_xz();
        let mut spread = if p.ads { def.ads } else { def.hip };
        if speed > 1.0 {
            spread *= 1.7;
        }
        if !p.ms.on_ground {
            spread *= 2.5;
        }
        if p.ms.crouch {
            spread *= 0.7;
        }
        let r = right(p.yaw);
        let up = r.cross(d0).norm();
        let view = if p.bot { self.tick * 256 } else { view };
        for _ in 0..def.pellets {
            let a = self.crng.range(0.0, TAU);
            let rad = spread * self.crng.f().sqrt();
            let d = (d0 + r * (a.cos() * rad) + up * (a.sin() * rad)).norm();
            let wh = self.w.raycast(o, d, def.range);
            let tw = wh.map_or(def.range, |h| h.0);
            let ph = self.ray_players(i, o, d, tw, view);
            let end_t = ph.map_or(tw, |h| h.1);
            let target = o + d * end_t;
            // obstruction between the shooter's eye and the aim point
            let md = target - eye;
            let ml = md.len();
            let mut end = target;
            let mut hit_kind = if ph.is_some() { 1u8 } else if wh.is_some() { 3 } else { 0 };
            let mut victim = ph;
            // the shot as a viewer that is not sent the victim sees it
            let alt = o + d * tw;
            let alt_kind = if wh.is_some() { 3u8 } else { 0 };
            if ml > 0.3 {
                let dd = md * (1.0 / ml);
                if let Some(h) = self.w.raycast(eye, dd, ml - 0.2) {
                    end = eye + dd * h.0;
                    victim = None;
                    hit_kind = 3;
                }
            }
            if victim.is_none() && hit_kind == 3 && wh.map_or(false, |h| h.2 == BK_CAR) && (end - alt).len2() < 0.01 {
                // the bodywork takes some of a bullet that hits an occupied car
                if let Some((_, _, k)) = self.w.ray_cars(o, d, tw + 0.05) {
                    let dr = self.car_driver[k] as usize;
                    let fall = if tw <= def.f0 { 1.0 } else if tw >= def.f1 { def.fmin } else { lerpf(1.0, def.fmin, (tw - def.f0) / (def.f1 - def.f0)) };
                    if dr < MAX_PLAYERS && dr != i {
                        self.damage(dr, i as u8, def.dmg * fall * 0.55, false, wk, self.players[i].ms.pos);
                    }
                    // and the car itself wears down until it blows up
                    self.hurt_car(k, def.dmg * fall, i as u8);
                }
            }
            if victim.is_none() && hit_kind == 3 && wh.map_or(false, |h| h.2 == BK_HELI) && (end - alt).len2() < 0.01 {
                // a helicopter's hull stops the bullet and wears down
                if let Some((_, _, k)) = self.w.ray_helis(o, d, tw + 0.05, usize::MAX) {
                    let fall = if tw <= def.f0 { 1.0 } else if tw >= def.f1 { def.fmin } else { lerpf(1.0, def.fmin, (tw - def.f0) / (def.f1 - def.f0)) };
                    self.hurt_heli(k, def.dmg * fall, i as u8);
                }
            }
            if let Some((j, dist, head)) = victim {
                let fall = if dist <= def.f0 {
                    1.0
                } else if dist >= def.f1 {
                    def.fmin
                } else {
                    lerpf(1.0, def.fmin, (dist - def.f0) / (def.f1 - def.f0))
                };
                let mut dmg = def.dmg * fall;
                if head {
                    dmg *= def.head;
                    hit_kind = 2;
                }
                self.damage(j, i as u8, dmg, head, wk, self.players[i].ms.pos);
            }
            let start = self.ev_begin();
            let mut w = Wr::new(&mut self.ev);
            w.u8(E_SHOT);
            w.u8(i as u8);
            w.u8(wk);
            w.f32(muz.x);
            w.f32(muz.y);
            w.f32(muz.z);
            w.f32(end.x);
            w.f32(end.y);
            w.f32(end.z);
            w.u8(hit_kind);
            self.ev_end(start, 255, Some(muz));
            if let Some(e) = self.evs.last_mut() {
                e.shooter = i as u8;
                e.dir = d;
                if let (Some((j, _, _)), 1 | 2) = (victim, hit_kind) {
                    e.victim = j as u8;
                    e.alt = alt;
                    e.alt_kind = alt_kind;
                }
            }
        }
    }

    fn melee(&mut self, i: usize, _tick: u32) {
        let p = &self.players[i];
        let head = p.ms.pos + V3::new(0.0, eye_h(p.ms.crouch) - 0.3, 0.0);
        let d = dir_from(p.yaw, p.pitch * 0.5);
        let mut best: Option<(usize, f32)> = None;
        for j in 0..MAX_PLAYERS {
            if j == i {
                continue;
            }
            let q = &self.players[j];
            if !q.used || !q.alive || q.ms.mode != MODE_GROUND {
                continue;
            }
            let to = q.ms.pos + V3::new(0.0, 1.0, 0.0) - head;
            let dist = to.len();
            if dist > 2.0 {
                continue;
            }
            if to.norm().dot(d) < 0.6 {
                continue;
            }
            if self.w.blocked(head, q.ms.pos + V3::new(0.0, 1.0, 0.0)) {
                continue;
            }
            if best.map_or(true, |b| dist < b.1) {
                best = Some((j, dist));
            }
        }
        if let Some((j, _)) = best {
            let pos = self.players[i].ms.pos;
            self.damage(j, i as u8, WD[0].dmg, false, W_FISTS, pos);
        }
    }

    fn throw_nade(&mut self, i: usize, c: &Cmd) {
        let p = &mut self.players[i];
        if p.nades == 0 || p.nade_cd > 0.0 {
            return;
        }
        p.nades -= 1;
        p.nade_cd = 1.0;
        p.heal_t = 0.0;
        let d = dir_from(c.yaw, clampf(c.pitch + 0.25, -1.2, 1.3));
        let eye = p.ms.pos + V3::new(0.0, eye_h(p.ms.crouch), 0.0);
        let want = eye + right(c.yaw) * 0.25 + d * 0.5;
        let off = want - eye;
        let ol = off.len();
        let pos = match self.w.raycast(eye, off * (1.0 / ol), ol + 0.15) {
            Some((t, _, _)) => eye + off * ((t - 0.15).max(0.0) / ol),
            None => want,
        };
        let p = &self.players[i];
        let vel = d * 19.0 + V3::new(p.ms.vel.x * 0.5, 0.0, p.ms.vel.z * 0.5);
        self.nade_seq = self.nade_seq.wrapping_add(1);
        let id = self.nade_seq;
        self.nades.push(Nade { pos, vel, fuse: 3.2, owner: i as u8, id });
        let start = self.ev_begin();
        let mut w = Wr::new(&mut self.ev);
        w.u8(E_THROW);
        w.u8(i as u8);
        self.ev_end(start, i as u8, None);
    }

    fn update_nades(&mut self) {
        let mut k = 0;
        while k < self.nades.len() {
            let mut n = self.nades[k];
            n.fuse -= DT;
            n.vel.y -= 20.0 * DT;
            let step = n.vel * DT;
            let l = step.len();
            if l > 1e-5 {
                let d = step * (1.0 / l);
                if let Some((t, nrm, _)) = self.w.raycast(n.pos, d, l + 0.12) {
                    n.pos += d * (t - 0.1).max(0.0);
                    let vn = nrm * n.vel.dot(nrm);
                    n.vel = (n.vel - vn * 1.0) * 0.6 - vn * 0.35;
                    if n.vel.len() < 0.8 {
                        n.vel = V3::ZERO;
                    }
                } else {
                    n.pos += step;
                }
            }
            let g = self.w.height(n.pos.x, n.pos.z);
            if n.pos.y < g + 0.08 {
                n.pos.y = g + 0.08;
                if n.vel.y < 0.0 {
                    n.vel.y = -n.vel.y * 0.3;
                }
                n.vel.x *= 0.8;
                n.vel.z *= 0.8;
            }
            n.pos.x = clampf(n.pos.x, 1.0, MAP - 1.0);
            n.pos.z = clampf(n.pos.z, 1.0, MAP - 1.0);
            if n.fuse <= 0.0 {
                self.nades.swap_remove(k);
                self.explode(n);
                continue;
            }
            self.nades[k] = n;
            k += 1;
        }
    }

    fn explode(&mut self, n: Nade) {
        self.blast(n.pos, n.owner, 7.5, 115.0, 1.4, 460.0, WK_NADE);
    }

    /// An explosion at `pos`: players in sight of it within `radius` are
    /// hurt (`pdmg` at the centre, falling off with the power `pw`), props
    /// fly, vehicles take up to `cdmg`.
    fn blast(&mut self, pos: V3, owner: u8, radius: f32, pdmg: f32, pw: f32, cdmg: f32, wk: u8) {
        let start = self.ev_begin();
        let mut w = Wr::new(&mut self.ev);
        w.u8(E_BOOM);
        w.f32(pos.x);
        w.f32(pos.y);
        w.f32(pos.z);
        self.ev_end(start, 255, Some(pos));
        for j in 0..MAX_PLAYERS {
            let q = &self.players[j];
            // (a pilot is behind the hull: the helicopter takes it)
            if !q.used || !q.alive || q.ms.mode == MODE_PLANE || q.ms.mode == MODE_FLY {
                continue;
            }
            // a driver is reached over the roof, and shielded a little
            let driving = q.ms.mode == MODE_DRIVE;
            let roof = if driving { self.w.car_dims(q.ms.car as usize).2 + 0.3 } else { 0.9 };
            let c = q.ms.pos + V3::new(0.0, roof, 0.0);
            let d = (c - pos).len();
            if d > radius {
                continue;
            }
            if self.w.blocked(pos + V3::new(0.0, 0.3, 0.0), c) {
                continue;
            }
            let k = 1.0 - d / radius;
            let dmg = pdmg * k.powf(pw) * if driving { 0.7 } else { 1.0 };
            if dmg > 1.0 {
                self.damage(j, owner, dmg, false, wk, pos);
            }
        }
        self.blast_props(pos, radius);
        for k in 0..self.w.cars.len() {
            let c = self.w.cars[k];
            let d = (V3::new(c[0], c[1] + 0.7, c[2]) - pos).len();
            if d < radius {
                self.hurt_car(k, cdmg * (1.0 - d / radius).powf(1.2), owner);
            }
        }
        self.blast_helis(pos, radius, cdmg, 1.2, owner);
    }

    /// Apply damage. attacker 255 = zone. weapon 200 = grenade.
    pub fn damage(&mut self, j: usize, attacker: u8, raw: f32, head: bool, weapon: u8, src: V3) {
        if !self.players[j].alive || raw <= 0.0 || self.phase != PH_PLAY {
            return;
        }
        let mut dmg = raw;
        let mut armor_hit = false;
        {
            let q = &mut self.players[j];
            if attacker != KILLER_ZONE && weapon != WK_NADE {
                if head && q.helm > 0 {
                    let red = ARMOR_RED[q.helm as usize];
                    dmg *= 1.0 - red;
                    q.helm_dur -= raw * 0.6;
                    armor_hit = true;
                    if q.helm_dur <= 0.0 {
                        q.helm = 0;
                        q.helm_dur = 0.0;
                    }
                } else if !head && q.vest > 0 {
                    let red = ARMOR_RED[q.vest as usize];
                    dmg *= 1.0 - red;
                    q.vest_dur -= raw * 0.6;
                    armor_hit = true;
                    if q.vest_dur <= 0.0 {
                        q.vest = 0;
                        q.vest_dur = 0.0;
                    }
                }
            }
            q.health -= dmg;
            if attacker != KILLER_ZONE {
                q.hurt_by = attacker;
                q.hurt_t = 4.0;
                // being shot interrupts healing; the zone does not (or nobody
                // could ever heal outside the circle)
                q.heal_t = 0.0;
            }
        }
        let killed = self.players[j].health <= 0.0;
        if (attacker as usize) < MAX_PLAYERS && attacker as usize != j {
            let a = &mut self.players[attacker as usize];
            a.dmg_dealt += dmg;
            let start = self.ev_begin();
            let mut w = Wr::new(&mut self.ev);
            w.u8(E_HIT);
            w.u8(j as u8);
            w.u8(dmg.round().clamp(1.0, 255.0) as u8);
            w.u8((head as u8) | if killed { 2 } else { 0 } | if armor_hit { 4 } else { 0 });
            self.ev_end(start, attacker, None);
        }
        {
            let start = self.ev_begin();
            let mut w = Wr::new(&mut self.ev);
            w.u8(E_HURT);
            w.u8(attacker);
            w.f32(src.x);
            w.f32(src.z);
            w.u8(dmg.round().clamp(1.0, 255.0) as u8);
            self.ev_end(start, j as u8, None);
        }
        if killed {
            self.kill(j, attacker, weapon, head);
        }
    }

    fn kill(&mut self, j: usize, attacker: u8, weapon: u8, head: bool) {
        // players who die in the same tick share the placement
        let place = self.alive_start.max(1) as u8;
        let tick = self.tick;
        let pos;
        {
            let q = &mut self.players[j];
            q.alive = false;
            q.health = 0.0;
            q.place = place;
            q.killer = attacker;
            q.death_tick = tick;
            if q.ms.mode == MODE_DRIVE {
                let k = q.ms.car as usize;
                if k < self.car_driver.len() && self.car_driver[k] == j as u8 {
                    self.car_driver[k] = 255;
                    self.car_moved[k] = tick;
                    // (it stops, but for what it still slides at)
                    self.w.car_vel[k] = self.car_slide[k];
                }
            }
            let mut at = q.ms.pos;
            if q.ms.mode == MODE_FLY {
                let k = q.ms.car as usize;
                if k < self.helis.len() && self.helis[k].pilot == j as u8 {
                    self.helis[k].pilot = 255;
                    // what the pilot carried lands beside the hull, not in it
                    let r = right(self.w.helis[k][3]);
                    at = at + r * -(HELI_HW + 1.3);
                }
            }
            q.ms.mode = MODE_DEAD;
            q.ms.vel = V3::ZERO;
            q.reload_t = 0.0;
            q.heal_t = 0.0;
            pos = at;
        }
        let follow = if (attacker as usize) < MAX_PLAYERS && attacker as usize != j { attacker } else { 255 };
        self.set_spect(j, follow);
        if (attacker as usize) < MAX_PLAYERS && attacker as usize != j {
            self.players[attacker as usize].kills += 1;
        }
        {
            let start = self.ev_begin();
            let mut w = Wr::new(&mut self.ev);
            w.u8(E_KILL);
            w.u8(attacker);
            w.u8(j as u8);
            w.u8(weapon);
            w.u8(head as u8);
            self.ev_end(start, 255, None);
        }
        // drop loot (visible to whoever could see the victim)
        self.item_actor = j as u8;
        let q = self.players[j].clone();
        let mut k = 0;
        for s in 0..2 {
            if q.slots[s].0 != 0 {
                self.drop_item_near(q.slots[s].0, q.slots[s].1, pos, k);
                k += 1;
            }
        }
        for a in 0..NUM_AMMO {
            let mut left = q.ammo[a];
            while left > 0 && k < 24 {
                let take = left.min(AMMO_STACK[a] * 2);
                self.drop_item_near(IK_AMMO + a as u8, take, pos, k);
                left -= take;
                k += 1;
            }
        }
        for h in 0..3 {
            if q.heals[h] > 0 {
                self.drop_item_near(IK_BANDAGE + h as u8, q.heals[h] as u16, pos, k);
                k += 1;
            }
        }
        if q.vest > 0 {
            self.drop_item_near(IK_VEST + q.vest - 1, q.vest_dur.max(1.0) as u16, pos, k);
            k += 1;
        }
        if q.helm > 0 {
            self.drop_item_near(IK_HELM + q.helm - 1, q.helm_dur.max(1.0) as u16, pos, k);
            k += 1;
        }
        if q.nades > 0 {
            self.drop_item_near(IK_NADE, q.nades as u16, pos, k);
        }
        self.item_actor = 255;
        let p = &mut self.players[j];
        p.slots = [(0, 0); 2];
        p.ammo = [0; NUM_AMMO];
        p.heals = [0; 3];
        p.nades = 0;
        p.vest = 0;
        p.helm = 0;
        // spectators of the victim follow the killer
        for s in 0..MAX_PLAYERS {
            if s != j && self.players[s].used && !self.players[s].alive && self.players[s].spect as usize == j {
                self.set_spect(s, attacker);
            }
        }
    }

    // ------------------------------------------------------------------
    // Pickups
    // ------------------------------------------------------------------
    /// Try to take an item. Returns true if the item was consumed (fully or partially).
    fn try_take(&mut self, i: usize, id: u16, manual: bool) -> bool {
        let it = self.items[id as usize];
        if !it.alive {
            return false;
        }
        let p = &mut self.players[i];
        let kind = it.kind;
        let mut remaining = it.amount;
        let mut took = 0u16;
        let mut drop: Option<(u8, u16)> = None;
        if is_weapon(kind) {
            let empty = (0..2).find(|&s| p.slots[s].0 == 0);
            if let Some(s) = empty {
                p.slots[s] = (kind, it.amount);
                if p.cur == 2 {
                    p.cur = s as u8;
                    p.switch_t = 0.35;
                }
                took = 1;
                remaining = 0;
            } else if manual {
                let s = if p.cur < 2 { p.cur as usize } else { 0 };
                drop = Some(p.slots[s]);
                p.slots[s] = (kind, it.amount);
                p.cur = s as u8;
                p.reload_t = 0.0;
                p.switch_t = 0.35;
                took = 1;
                remaining = 0;
            }
        } else if (IK_AMMO..IK_AMMO + NUM_AMMO as u8).contains(&kind) {
            let a = (kind - IK_AMMO) as usize;
            let space = AMMO_CAP[a].saturating_sub(p.ammo[a]);
            let t = space.min(remaining);
            p.ammo[a] += t;
            remaining -= t;
            took = t;
        } else if (IK_BANDAGE..=IK_DRINK).contains(&kind) {
            let h = (kind - IK_BANDAGE) as usize;
            let space = HEAL_CAP[h].saturating_sub(p.heals[h]) as u16;
            let t = space.min(remaining);
            p.heals[h] += t as u8;
            remaining -= t;
            took = t;
        } else if (IK_VEST..IK_VEST + 3).contains(&kind) {
            let lv = kind - IK_VEST + 1;
            if lv > p.vest || (manual && lv == p.vest && it.amount as f32 > p.vest_dur + 1.0) {
                if p.vest > 0 {
                    drop = Some((IK_VEST + p.vest - 1, p.vest_dur.max(1.0) as u16));
                }
                p.vest = lv;
                p.vest_dur = it.amount.min(100) as f32;
                took = 1;
                remaining = 0;
            }
        } else if (IK_HELM..IK_HELM + 3).contains(&kind) {
            let lv = kind - IK_HELM + 1;
            if lv > p.helm || (manual && lv == p.helm && it.amount as f32 > p.helm_dur + 1.0) {
                if p.helm > 0 {
                    drop = Some((IK_HELM + p.helm - 1, p.helm_dur.max(1.0) as u16));
                }
                p.helm = lv;
                p.helm_dur = it.amount.min(100) as f32;
                took = 1;
                remaining = 0;
            }
        } else if kind == IK_NADE {
            let space = NADE_CAP.saturating_sub(p.nades) as u16;
            let t = space.min(remaining);
            p.nades += t as u8;
            remaining -= t;
            took = t;
        }
        if took == 0 {
            return false;
        }
        let pos = p.ms.pos;
        {
            let start = self.ev_begin();
            let mut w = Wr::new(&mut self.ev);
            w.u8(E_PICK);
            w.u8(kind);
            w.u16(took);
            self.ev_end(start, i as u8, None);
        }
        self.item_actor = i as u8;
        if remaining == 0 {
            self.del_item(id);
        } else {
            self.items[id as usize].amount = remaining;
            self.ev_item(id);
        }
        if let Some((k, a)) = drop {
            if k != 0 {
                let pp = V3::new(pos.x, pos.y, pos.z);
                self.drop_item_near(k, a, pp, 3 + (self.tick % 5) as usize);
            }
        }
        self.item_actor = 255;
        true
    }

    fn manual_pickup(&mut self, i: usize) {
        let p = &self.players[i];
        let pos = p.ms.pos;
        let f = fwd(p.yaw);
        let mut best: Option<(u16, f32)> = None;
        for (id, it) in self.items.iter().enumerate() {
            if !it.alive {
                continue;
            }
            let dx = it.pos.x - pos.x;
            let dz = it.pos.z - pos.z;
            let d = (dx * dx + dz * dz).sqrt();
            if d > 2.6 || (it.pos.y - pos.y).abs() > 1.8 {
                continue;
            }
            if self.w.blocked(pos + V3::new(0.0, 0.9, 0.0), it.pos + V3::new(0.0, 0.25, 0.0)) {
                continue;
            }
            // prefer items in front and weapons
            let facing = if d > 0.01 { (dx * f.x + dz * f.z) / d } else { 1.0 };
            let score = d - facing * 0.6 - if is_weapon(it.kind) { 0.5 } else { 0.0 };
            if best.map_or(true, |b| score < b.1) {
                best = Some((id as u16, score));
            }
        }
        if let Some((id, _)) = best {
            if self.try_take(i, id, true) {
                // a weapon swap changes two items: don't let it repeat every tick
                self.players[i].pick_cd = 0.3;
            }
        }
    }

    fn auto_pickups(&mut self) {
        for i in 0..MAX_PLAYERS {
            let p = &self.players[i];
            if !p.used || !p.alive || p.ms.mode != MODE_GROUND {
                continue;
            }
            if (self.tick + i as u32) % 3 != 0 {
                continue;
            }
            let pos = p.ms.pos;
            let mut cands: [u16; 8] = [0; 8];
            let mut n = 0;
            for (id, it) in self.items.iter().enumerate() {
                if !it.alive || n >= 8 {
                    continue;
                }
                let dx = it.pos.x - pos.x;
                let dz = it.pos.z - pos.z;
                if dx * dx + dz * dz < 1.6 * 1.6
                    && (it.pos.y - pos.y).abs() < 1.6
                    && !self.w.blocked(pos + V3::new(0.0, 0.9, 0.0), it.pos + V3::new(0.0, 0.25, 0.0))
                {
                    cands[n] = id as u16;
                    n += 1;
                }
            }
            for &id in cands.iter().take(n) {
                self.try_take(i, id, false);
            }
        }
    }

    // ------------------------------------------------------------------
    // Tick
    // ------------------------------------------------------------------
    fn record_hist(&mut self) {
        let slot = self.tick as usize % HIST;
        for j in 0..MAX_PLAYERS {
            let p = &mut self.players[j];
            // the client's animation speed: |change of position| / dt, capped
            // at 12 m/s, smoothed with a 0.1 s time constant
            let moved = dist_xz(p.ms.pos.x, p.ms.pos.z, p.draw_at.0, p.draw_at.1);
            p.draw_speed = if p.ms.mode == MODE_PLANE { 0.0 } else { lerpf(p.draw_speed, (moved / DT).min(12.0), 0.3) };
            p.draw_at = (p.ms.pos.x, p.ms.pos.z);
            self.hist[slot][j] = HistE::of(p);
        }
    }

    fn update_zone(&mut self) {
        let z = &mut self.zone;
        if z.done {
            return;
        }
        z.timer -= DT;
        let st = &ZONES[(z.stage as usize).min(ZONES.len() - 1)];
        if !z.shrinking {
            if z.timer <= 0.0 {
                z.shrinking = true;
                z.timer = st.shrink;
                z.sx = z.cx;
                z.sz = z.cz;
                z.sr = z.r;
            }
        } else {
            let k = clampf(1.0 - z.timer / st.shrink, 0.0, 1.0);
            z.cx = lerpf(z.sx, z.nx, k);
            z.cz = lerpf(z.sz, z.nz, k);
            z.r = lerpf(z.sr, z.nr, k);
            if z.timer <= 0.0 {
                z.cx = z.nx;
                z.cz = z.nz;
                z.r = z.nr;
                z.stage += 1;
                z.shrinking = false;
                if (z.stage as usize) >= ZONES.len() {
                    z.done = true;
                    z.stage = (ZONES.len() - 1) as u8;
                } else {
                    z.timer = ZONES[z.stage as usize].wait;
                    self.pick_next_zone();
                }
            }
        }
    }

    fn pick_next_zone(&mut self) {
        let stage = (self.zone.stage as usize).min(ZONES.len() - 1);
        let nr = ZONES[stage].r;
        let (cx, cz, r) = (self.zone.cx, self.zone.cz, self.zone.r);
        let max_off = if stage == 0 { 260.0 } else { (r - nr).max(0.0) * 0.92 };
        let mut best = (cx, cz);
        for attempt in 0..40 {
            let a = self.rng.range(0.0, TAU);
            let d = self.rng.f().sqrt() * max_off;
            let (x, z) = (cx + a.cos() * d, cz + a.sin() * d);
            if stage == 0 && dist_xz(x, z, MAP * 0.5, MAP * 0.5) > 300.0 {
                continue;
            }
            let land = self.w.height(x, z) > 1.5;
            best = (clampf(x, 60.0, MAP - 60.0), clampf(z, 60.0, MAP - 60.0));
            if land || attempt > 30 {
                break;
            }
        }
        self.zone.nx = best.0;
        self.zone.nz = best.1;
        self.zone.nr = nr;
    }

    fn zone_damage(&mut self) {
        let (cx, cz, r) = (self.zone.cx, self.zone.cz, self.zone.r);
        let dps = self.zone.dps();
        for j in 0..MAX_PLAYERS {
            let p = &mut self.players[j];
            if !p.used || !p.alive || p.ms.mode == MODE_PLANE {
                continue;
            }
            if dist_xz(p.ms.pos.x, p.ms.pos.z, cx, cz) > r {
                p.zone_acc += dps * DT;
            }
            if p.zone_acc >= 1.0 {
                let d = p.zone_acc.floor();
                p.zone_acc -= d;
                let pos = p.ms.pos;
                self.damage(j, KILLER_ZONE, d, false, 255, pos);
            }
        }
    }

    fn update_timers(&mut self) {
        for i in 0..MAX_PLAYERS {
            let p = &mut self.players[i];
            if !p.used || !p.alive {
                continue;
            }
            p.fire_cd = (p.fire_cd - DT).max(-DT);
            p.switch_t = (p.switch_t - DT).max(0.0);
            p.nade_cd = (p.nade_cd - DT).max(0.0);
            p.firing_t = (p.firing_t - DT).max(0.0);
            p.hurt_t = (p.hurt_t - DT).max(0.0);
            if p.reload_t > 0.0 {
                p.reload_t -= DT;
                if p.reload_t <= 0.0 {
                    p.reload_t = 0.0;
                    if p.cur < 2 {
                        let s = p.cur as usize;
                        let wk = p.slots[s].0;
                        if wk != 0 {
                            let def = &WD[wk as usize];
                            let a = def.ammo as usize;
                            let need = def.mag.saturating_sub(p.slots[s].1);
                            let take = need.min(p.ammo[a]);
                            p.slots[s].1 += take;
                            p.ammo[a] -= take;
                        }
                    }
                }
            }
            if p.heal_t > 0.0 {
                p.heal_t -= DT;
                if p.heal_t <= 0.0 {
                    p.heal_t = 0.0;
                    let k = (p.heal_kind.max(1) - 1) as usize;
                    if p.heals[k] > 0 {
                        p.heals[k] -= 1;
                        match k {
                            0 => p.health = (p.health + 15.0).min(75.0f32.max(p.health)),
                            1 => p.health = 100.0,
                            _ => p.boost = (p.boost + 40.0).min(100.0),
                        }
                    }
                    p.heal_kind = 0;
                }
            }
            if p.boost > 0.0 {
                let rate = 40.0 / 30.0;
                let b = (rate * DT).min(p.boost);
                p.boost -= b;
                p.health = (p.health + b * 0.5).min(100.0);
            }
        }
    }

    /// Merge the edge buttons of a dropped command into `carry` (the latest
    /// slot / heal request wins).
    fn absorb(carry: u16, b: u16) -> u16 {
        let mut out = carry | (b & EDGE_BUTTONS);
        for sh in [8u16, 10] {
            if b & (3 << sh) != 0 {
                out = (out & !(3 << sh)) | (b & (3 << sh));
            }
        }
        out
    }

    /// Buttons of a simulated command with the taps of dropped ones merged in
    /// (the command's own slot / heal request wins).
    fn with_carry(b: u16, carry: u16) -> u16 {
        let mut out = b | (carry & EDGE_BUTTONS);
        for sh in [8u16, 10] {
            if b & (3 << sh) == 0 {
                out |= carry & (3 << sh);
            }
        }
        out
    }

    /// Humans: queued commands are simulated with per-player lag accounting.
    /// Each tick adds one unit of lag and every simulated step (a command, or
    /// an idle step when input is late) removes one, so a player is always
    /// simulated for between (ticks - LAG_GRACE - 1) and (ticks) steps. No
    /// arrival pattern can speed a player up (bursts, floods, late pairs) or
    /// slow one down (starving the server to hover), and late commands that
    /// idle steps already covered are dropped, so no backlog (input latency)
    /// builds up after a stall.
    fn step_humans(&mut self) {
        let now = self.tick as f32;
        for i in 0..MAX_PLAYERS {
            if !(self.players[i].used && self.players[i].human) {
                continue;
            }
            let p = &mut self.players[i];
            p.pick_cd = (p.pick_cd - DT).max(0.0);
            p.spect_cd = (p.spect_cd - DT).max(0.0);
            p.lag = (p.lag + 1).min(LAG_GRACE + 1);
            // A burst after a stall: its oldest commands are for ticks that
            // idle steps already simulated. Drop them, keeping their taps.
            while p.debt > 0 && p.cmdq.len() > p.lag as usize {
                let Some(c) = p.cmdq.pop_front() else { break };
                p.carry = Self::absorb(p.carry, c.buttons);
                p.acked_seq = c.seq;
                p.last_cmd = c;
                p.debt -= 1;
            }
            let mut applied = 0;
            while applied < 2 && self.players[i].lag > 0 {
                let p = &mut self.players[i];
                let Some(mut c) = p.cmdq.pop_front() else { break };
                p.lag -= 1;
                c.buttons = Self::with_carry(c.buttons, p.carry);
                p.carry = 0;
                // The view tick (lag compensation) is client-claimed. The
                // applied rewind always lies within [floor, floor + 4], where
                // the floor follows the client's lowest recent claim: down at
                // most one tick per command, up 0.08. So a low-latency client
                // cannot rewind one shot much further than the others, nor a
                // high-latency one a shot much less (15 ticks, then 1).
                let raw = now - c.view as f32 / 256.0;
                let fl = p.rew_floor;
                let r = if raw < 1.0 || self.catch_up {
                    // A view at or ahead of the server tick (a forged view), or
                    // one made in real time while the server catches up on a
                    // hitch, says nothing about this client's latency: keep
                    // the floor, and never rewind less than it.
                    clampf(raw, fl.max(1.0), (fl + REWIND_MARGIN).min(REWIND_MAX))
                } else {
                    let r = clampf(raw, 1.0, REWIND_MAX);
                    p.rew_floor = if r < fl { (fl - 1.0).max(r) } else { (fl + 0.08).min(r) };
                    clampf(r, p.rew_floor, (p.rew_floor + REWIND_MARGIN).min(REWIND_MAX))
                };
                c.view = ((now - r) * 256.0).max(0.0) as u32;
                p.acked_seq = c.seq;
                p.last_cmd = c;
                p.idle_run = 0;
                p.cmds_applied = p.cmds_applied.wrapping_add(1);
                self.apply_cmd(i, c);
                applied += 1;
            }
            let p = &mut self.players[i];
            let mut idle = false;
            let mut drift = false;
            if p.lag > LAG_GRACE {
                // Input is more than LAG_GRACE ticks late: simulate the tick
                // anyway and drop the late command when it shows up.
                idle = true;
                p.debt = (p.debt + 1).min(DEBT_MAX);
                p.debt_ttl = DEBT_TTL;
            } else {
                if p.debt_ttl > 0 {
                    p.debt_ttl -= 1;
                    if p.debt_ttl == 0 {
                        // no burst followed: the client skipped that time
                        p.debt = 0;
                    }
                }
                // The lag can get stuck above zero (a client that skipped
                // time, a route that got slower): win the grace back slowly.
                if p.lag == 0 {
                    p.lag_hi_t = 0;
                } else {
                    p.lag_hi_t += 1;
                    if p.lag_hi_t >= DRIFT_T && applied < 2 {
                        p.lag_hi_t = 0;
                        idle = true;
                        drift = true;
                    }
                }
            }
            if idle {
                let p = &mut self.players[i];
                p.lag -= 1;
                p.idle_steps = p.idle_steps.wrapping_add(1);
                if p.alive && p.ms.mode != MODE_PLANE {
                    // Held stance, briefly the last movement; never edge
                    // actions or firing, so starving the server never helps.
                    // A drift step replaces no command, so it must not move
                    // the player; a late-input step stands in for the late
                    // command and briefly repeats its movement.
                    let mut c = p.last_cmd;
                    if drift || p.idle_run >= IDLE_REPEAT {
                        c.mx = 0.0;
                        c.mz = 0.0;
                    }
                    // (a pilot's climb is a held control, like its descent)
                    c.buttons &= B_CROUCH | B_ADS | B_SPRINT | if p.ms.mode == MODE_FLY { B_JUMP } else { 0 };
                    p.idle_run = p.idle_run.saturating_add(1);
                    self.apply_cmd(i, c);
                }
            }
            // A queue that never empties (the route got faster, a client clock
            // running fast, a server time skip) is standing input latency: drop
            // the oldest command now and then, keeping its taps.
            let p = &mut self.players[i];
            if p.cmdq.len() > BACKLOG_MAX {
                p.backlog_t += 1;
                if p.backlog_t >= BACKLOG_T {
                    p.backlog_t = 0;
                    if let Some(c) = p.cmdq.pop_front() {
                        p.carry = Self::absorb(p.carry, c.buttons);
                        p.acked_seq = c.seq;
                        p.last_cmd = c;
                    }
                }
            } else {
                p.backlog_t = 0;
            }
        }
    }

    /// Advance one fixed tick. Returns bit 1 = snapshot due, bit 2 = match over.
    pub fn step(&mut self) -> u32 {
        self.tick = self.tick.wrapping_add(1);
        match self.phase {
            PH_LOBBY => {
                if self.human_count() > 0 {
                    self.lobby_t -= DT;
                    let humans = self.players.iter().filter(|p| p.used).count();
                    if self.lobby_t <= 0.0 || humans >= self.slots {
                        self.start_match();
                    }
                }
            }
            PH_PLAY => {
                self.match_t += DT;
                self.alive_start = self.alive_count();
                // plane
                if self.plane.active {
                    self.plane.t += PLANE_SPEED * DT;
                    let pp = self.plane.pos();
                    let force = self.plane.t > self.plane.exit_t;
                    for i in 0..MAX_PLAYERS {
                        if self.players[i].used && self.players[i].ms.mode == MODE_PLANE {
                            self.players[i].ms.pos = pp;
                            if force {
                                self.eject(i);
                            }
                        }
                    }
                    if force {
                        self.plane.active = false;
                    }
                }
                // (what the cars are doing, for whoever drives into one)
                for k in 0..self.w.cars.len() {
                    self.sync_car_vel(k);
                }
                // bots
                for i in 0..MAX_PLAYERS {
                    if self.players[i].used && self.players[i].bot && self.players[i].alive {
                        let c = crate::bot::think(self, i);
                        self.apply_cmd(i, c);
                    }
                }
                self.step_humans();
                self.update_timers();
                self.auto_pickups();
                self.update_nades();
                self.update_car_fires();
                self.update_cars();
                self.update_helis();
                self.update_missiles();
                self.update_props();
                self.update_zone();
                self.zone_damage();
                self.record_hist();
                if self.alive_count() <= 1 {
                    self.phase = PH_END;
                    self.end_t = 12.0;
                    let survivor = (0..MAX_PLAYERS).find(|&k| self.players[k].used && self.players[k].alive);
                    // Nobody left (last players died in the same tick): the one
                    // with the most kills, then damage, among them wins.
                    let tick = self.tick;
                    let last_group = || {
                        (0..MAX_PLAYERS)
                            .filter(|&k| self.players[k].used && !self.players[k].alive && self.players[k].death_tick == tick)
                            .max_by(|&a, &b| {
                                let (pa, pb) = (&self.players[a], &self.players[b]);
                                pa.kills.cmp(&pb.kills).then(pa.dmg_dealt.total_cmp(&pb.dmg_dealt)).then(b.cmp(&a))
                            })
                    };
                    self.winner = survivor.or_else(last_group).map_or(255, |k| k as u8);
                    if (self.winner as usize) < MAX_PLAYERS {
                        self.players[self.winner as usize].place = 1;
                    }
                    let start = self.ev_begin();
                    let mut w = Wr::new(&mut self.ev);
                    w.u8(E_END);
                    w.u8(self.winner);
                    self.ev_end(start, 255, None);
                }
            }
            _ => {
                self.end_t -= DT;
                if self.end_t <= 0.0 {
                    self.ended_flag = true;
                }
            }
        }
        let mut r = 0;
        if self.tick % self.snap_every == 0 {
            r |= 1;
        }
        if self.ended_flag {
            r |= 2;
        }
        r
    }

    // ------------------------------------------------------------------
    // Output
    // ------------------------------------------------------------------
    pub fn welcome(&mut self, pid: u8) -> &[u8] {
        self.out.clear();
        let mut w = Wr::new(&mut self.out);
        w.u8(S_WELCOME);
        w.u16(PROTO);
        w.u8(pid);
        w.u32(self.w.seed);
        w.u32(self.w.checksum);
        w.u8(TICK_HZ as u8);
        w.u8(self.snap_every as u8);
        w.u8(self.slots as u8);
        w.u8(self.flags as u8);
        w.f32(self.lobby_t);
        let used: Vec<usize> = (0..MAX_PLAYERS).filter(|&k| self.players[k].used).collect();
        w.u8(used.len() as u8);
        for &k in &used {
            let p = &self.players[k];
            w.u8(k as u8);
            w.u8(p.bot as u8);
            w.u8(p.color);
            let nb = p.name.as_bytes();
            w.u8(nb.len() as u8);
            w.bytes(nb);
        }
        // listed by wire id: in internal (creation) order, the position of
        // an item in the list would date it
        let mut alive: Vec<usize> = (0..self.items.len()).filter(|&k| self.items[k].alive).collect();
        alive.sort_by_key(|&k| item_wire(self.salt, k as u16));
        w.u16(alive.len() as u16);
        for &k in &alive {
            let it = &self.items[k];
            w.u16(item_wire(self.salt, k as u16));
            w.u8(it.kind);
            w.u16(it.amount);
            w.f32(it.pos.x);
            w.f32(it.pos.y);
            w.f32(it.pos.z);
        }
        if (pid as usize) < MAX_PLAYERS {
            let v = self.chunk_ver.clone();
            let n = self.ilog_n;
            let p = &mut self.players[pid as usize];
            p.need_welcome = false;
            p.item_ver = v;
            p.ilog_from = n;
            p.ilog_sent = [0; ILOG / 64];
        }
        &self.out
    }

    /// The id an item is sent with (see `item_wire`), and back.
    pub fn item_wire_id(&self, k: u16) -> u16 {
        item_wire(self.salt, k)
    }
    pub fn item_from_wire(&self, w: u16) -> u16 {
        item_unwire(self.salt, w)
    }

    /// Quiet period before a chunk with withheld changes is resynced to
    /// viewer `i`: 10-15 s, secret per viewer, chunk and change (its chunk
    /// version), so the resync's arrival does not date the hidden change.
    fn quiet_ticks(&self, i: usize, c: usize, ver: u32) -> u32 {
        let x = mix64(self.salt ^ ((i as u64) << 40) ^ ((c as u64) << 32) ^ ver as u64);
        ITEM_QUIET + (x % 151) as u32
    }

    pub fn snapshot(&mut self, pid: u8) -> &[u8] {
        self.out.clear();
        let i = pid as usize;
        if i >= MAX_PLAYERS || !self.players[i].used {
            return &self.out;
        }
        let me = self.players[i].clone();
        let salt = self.salt;
        // interest center: me, or who I spectate
        let mut center = me.ms.pos;
        if !me.alive && (me.spect as usize) < MAX_PLAYERS && self.players[me.spect as usize].used {
            center = self.players[me.spect as usize].ms.pos;
        }
        let alive = self.alive_count() as u8;
        let total = self.players.iter().filter(|p| p.used).count() as u8;
        let mut w = Wr::new(&mut self.out);
        w.u8(S_SNAP);
        w.u32(self.tick);
        w.u32(me.acked_seq);
        w.u8(self.phase);
        w.f32(match self.phase {
            PH_LOBBY => self.lobby_t,
            PH_PLAY => self.match_t,
            _ => self.end_t,
        });
        w.u8(alive);
        w.u8(total);
        w.u8(self.winner);
        // ---- self
        w.u8(me.ms.mode);
        w.u8((me.alive as u8) | ((me.ms.on_ground as u8) << 1) | ((me.ms.crouch as u8) << 2) | ((me.ms.in_water as u8) << 3) | ((me.ads as u8) << 4));
        w.f32(me.ms.pos.x);
        w.f32(me.ms.pos.y);
        w.f32(me.ms.pos.z);
        w.f32(me.ms.vel.x);
        w.f32(me.ms.vel.y);
        w.f32(me.ms.vel.z);
        w.f32(me.ms.airtime);
        w.f32(me.health);
        w.u8(me.vest);
        w.u8(me.vest_dur.clamp(0.0, 255.0) as u8);
        w.u8(me.helm);
        w.u8(me.helm_dur.clamp(0.0, 255.0) as u8);
        w.u8(me.cur);
        for s in 0..2 {
            w.u8(me.slots[s].0);
            w.u16(me.slots[s].1);
        }
        for a in 0..NUM_AMMO {
            w.u16(me.ammo[a]);
        }
        for h in 0..3 {
            w.u8(me.heals[h]);
        }
        w.u8(me.nades);
        w.f32(me.reload_t);
        w.f32(me.heal_t);
        w.u8(me.heal_kind);
        w.f32(me.boost);
        w.f32(me.switch_t);
        w.u16(me.kills);
        w.u16(me.dmg_dealt.min(65535.0) as u16);
        w.u8(me.spect);
        w.u8(me.place);
        w.u8(me.killer);
        // Late commands that idle steps already stood in for: the server drops
        // that many of the oldest when they arrive, so the client skips them
        // when it replays its pending input.
        w.u8(me.debt);
        // ---- zone
        let z = self.zone;
        w.u8(z.stage);
        w.u8((z.shrinking as u8) | ((z.done as u8) << 1));
        w.f32(z.timer);
        w.f32(z.cx);
        w.f32(z.cz);
        w.f32(z.r);
        w.f32(z.nx);
        w.f32(z.nz);
        w.f32(z.nr);
        w.f32(z.dps());
        // ---- plane
        let pl = self.plane;
        w.u8(pl.active as u8 | ((pl.can_jump() as u8) << 1));
        w.f32(pl.sx);
        w.f32(pl.sz);
        w.f32(pl.dx);
        w.f32(pl.dz);
        w.f32(pl.t);
        w.f32(pl.enter_t);
        w.f32(pl.exit_t);
        // ---- players (line-of-sight culled to limit wallhack/radar leaks)
        let tick = self.tick;
        let spect = if me.alive { 255usize } else { me.spect as usize };
        let (ceye, cskip) = if spect < MAX_PLAYERS && self.players[spect].used {
            let t = &self.players[spect];
            (t.ms.pos + V3::new(0.0, eye_h(t.ms.crouch), 0.0), spect)
        } else {
            (me.ms.pos + V3::new(0.0, eye_h(me.ms.crouch), 0.0), i)
        };
        // looking out of a helicopter (flying it, or watching its pilot)
        let fly_view = self.players[cskip].ms.mode == MODE_FLY;
        let mut send = [false; MAX_PLAYERS];
        for j in 0..MAX_PLAYERS {
            if j == i {
                continue;
            }
            let q = &self.players[j];
            if !q.used || q.ms.mode == MODE_PLANE {
                continue;
            }
            if !q.alive && tick.wrapping_sub(q.death_tick) > 30 * 12 {
                continue;
            }
            if j == spect {
                send[j] = true;
                continue;
            }
            let d = dist_xz(q.ms.pos.x, q.ms.pos.z, center.x, center.z);
            if d > INTEREST {
                continue;
            }
            if d < ALWAYS_SEND_R || j == cskip {
                send[j] = true;
                continue;
            }
            let pv = &self.players[i];
            // re-check each pair every other snapshot round, staggered by target
            if pv.vis_check[j] == 0 || (pv.snap_round.wrapping_add(j as u32)) % 2 == 0 {
                // a driver is seen through the roof line of the car
                let lift = if q.ms.mode == MODE_DRIVE { self.w.car_dims(q.ms.car as usize).2 - 0.7 } else { 0.0 };
                let head = q.ms.pos + V3::new(0.0, eye_h(q.ms.crouch) + 0.1 + lift, 0.0);
                let chest = q.ms.pos + V3::new(0.0, 0.9 + lift, 0.0);
                let visible = if q.ms.mode == MODE_FLY || fly_view {
                    // A pilot is seen as its helicopter is (the top of the
                    // hull, or its middle), and sees from inside one: only
                    // the world itself hides anything, never a vehicle.
                    let (top, mid) = if q.ms.mode == MODE_FLY { (q.ms.pos + V3::new(0.0, HELI_TOP, 0.0), q.ms.pos + V3::new(0.0, 2.0, 0.0)) } else { (head, chest) };
                    !self.w.blocked_static(ceye, top) || !self.w.blocked_static(ceye, mid)
                } else {
                    !self.w.blocked(ceye, head) || !self.w.blocked(ceye, chest)
                };
                let pv = &mut self.players[i];
                pv.vis_check[j] = tick;
                if visible {
                    pv.seen_tick[j] = tick;
                }
            }
            let seen = self.players[i].seen_tick[j];
            send[j] = seen != 0 && tick.wrapping_sub(seen) <= 30;
        }
        {
            let pv = &mut self.players[i];
            for j in 0..MAX_PLAYERS {
                pv.sent_from[j] = if !send[j] { 0 } else if pv.sent_from[j] == 0 { tick.max(1) } else { pv.sent_from[j] };
            }
        }
        let mut w = Wr::new(&mut self.out);
        let count_pos = w.len();
        w.u8(0);
        let mut count = 0u8;
        for j in 0..MAX_PLAYERS {
            if !send[j] {
                continue;
            }
            let q = &self.players[j];
            w.u8(j as u8);
            w.u8(q.ms.mode);
            let fl = (q.alive as u8)
                | ((q.ms.crouch as u8) << 1)
                | ((q.ads as u8) << 2)
                | (((q.firing_t > 0.0) as u8) << 3)
                | (((q.heal_t > 0.0) as u8) << 4)
                | (((q.reload_t > 0.0) as u8) << 5)
                | ((q.ms.in_water as u8) << 6)
                | ((q.ms.on_ground as u8) << 7);
            w.u8(fl);
            w.f32(q.ms.pos.x);
            w.f32(q.ms.pos.y);
            w.f32(q.ms.pos.z);
            w.u16(q_yaw(q.yaw));
            w.i16(q_pitch(q.pitch));
            w.u8(q.weapon());
            w.u8(q.vest | (q.helm << 2));
            // exact health only for the player being spectated
            w.u8(if j == spect { q.health.clamp(0.0, 100.0) as u8 } else { 0 });
            count += 1;
        }
        self.out[count_pos] = count;
        // ---- grenades (near, thrown by someone visible, or in line of sight)
        let mut nlist: Vec<(u16, V3)> = Vec::new();
        for n in &self.nades {
            if nlist.len() >= 40 {
                break;
            }
            let d = dist_xz(n.pos.x, n.pos.z, center.x, center.z);
            if d > INTEREST {
                continue;
            }
            let o = n.owner as usize;
            let ok = d < ALWAYS_SEND_R || o == i || (o < MAX_PLAYERS && send[o]) || !self.w.blocked(ceye, n.pos + V3::new(0.0, 0.1, 0.0));
            if ok {
                nlist.push((n.id, n.pos));
            }
        }
        let mut w = Wr::new(&mut self.out);
        w.u8(nlist.len() as u8);
        for &(id, pos) in &nlist {
            w.u16(id);
            w.f32(pos.x);
            w.f32(pos.y);
            w.f32(pos.z);
        }
        // ---- cars: the ones that moved lately (or are driven), and the
        // rest a few at a time so every client converges on the same poses
        let ncar = self.w.cars.len().min(255);
        let round = (tick / self.snap_every.max(1)) as usize;
        let cpos = w.len();
        w.u8(0);
        let mut ccount = 0u8;
        for k in 0..ncar {
            let c = self.w.cars[k];
            let dr = self.car_driver[k];
            let fresh = dr != 255 || tick.wrapping_sub(self.car_moved[k]) < 45;
            if !(fresh || (k + round) % 16 == 0) {
                continue;
            }
            if dr as usize != i && dist_xz(c[0], c[2], center.x, center.z) > INTEREST {
                continue;
            }
            w.u8(k as u8);
            w.f32(c[0]);
            w.f32(c[1]);
            w.f32(c[2]);
            w.u16(q_yaw(c[3]));
            // a driver this viewer is not sent stays anonymous
            w.u8(if dr as usize == i { dr } else if (dr as usize) < MAX_PLAYERS && send[dr as usize] { dr } else if dr != 255 { 254 } else { 255 });
            // bodywork left in 255ths (0 only for a wreck)
            let hp = self.car_hp[k];
            let full = if self.w.car_kind.get(k) == Some(&VK_TRUCK) { 1100.0 } else { 600.0 };
            w.u8(if hp <= 0.0 { 0 } else { ((hp / full) * 255.0).ceil().clamp(1.0, 255.0) as u8 });
            // how fast it moves (cm/s) and turns (rad/s in 50ths): a client
            // that runs into it predicts where the knock sends it
            let [vx, vz] = self.w.car_vel[k];
            w.u16((vx * 100.0).round().clamp(-32767.0, 32767.0) as i16 as u16);
            w.u16((vz * 100.0).round().clamp(-32767.0, 32767.0) as i16 as u16);
            w.u8((self.car_spin[k] * 50.0).round().clamp(-127.0, 127.0) as i8 as u8);
            ccount += 1;
        }
        self.out[cpos] = ccount;
        let mut w = Wr::new(&mut self.out);
        // ---- loose props: the ones that moved lately, the rest a few at a time
        let ppos = w.len();
        w.u8(0);
        let mut pcount = 0u8;
        for k in 0..self.props.len() {
            let pr = self.props[k];
            let fresh = pr.moving || tick.wrapping_sub(pr.moved) < 45;
            if !(fresh || (k + round) % 32 == 0) || pcount == 255 {
                continue;
            }
            if dist_xz(pr.pos.x, pr.pos.z, center.x, center.z) > INTEREST {
                continue;
            }
            w.u16(k as u16);
            w.f32(pr.pos.x);
            w.f32(pr.pos.y);
            w.f32(pr.pos.z);
            w.u8(pr.moving as u8);
            pcount += 1;
        }
        self.out[ppos] = pcount;
        let mut w = Wr::new(&mut self.out);
        // ---- helicopters: loud and plain to see, so every one within
        // range is sent; who flies it only as for the driver of a car
        let hpos = w.len();
        w.u8(0);
        let mut hcount = 0u8;
        for k in 0..self.helis.len().min(255) {
            let c = self.w.helis[k];
            let h = self.helis[k];
            if h.pilot as usize != i && dist_xz(c[0], c[2], center.x, center.z) > INTEREST {
                continue;
            }
            w.u8(k as u8);
            w.f32(c[0]);
            w.f32(c[1]);
            w.f32(c[2]);
            w.u16(q_yaw(c[3]));
            w.u8(((h.pitch * 100.0).round().clamp(-127.0, 127.0) as i8) as u8);
            w.u8(((h.roll * 100.0).round().clamp(-127.0, 127.0) as i8) as u8);
            w.u8((h.rotor * 255.0).round().clamp(0.0, 255.0) as u8);
            let dr = h.pilot;
            w.u8(if dr as usize == i { dr } else if (dr as usize) < MAX_PLAYERS && send[dr as usize] { dr } else if dr != 255 { 254 } else { 255 });
            // hull left in 255ths (0 only for a wreck)
            w.u8(if h.hp <= 0.0 { 0 } else { ((h.hp / HELI_HP) * 255.0).ceil().clamp(1.0, 255.0) as u8 });
            w.u8(h.ammo);
            w.u8(h.air as u8);
            hcount += 1;
        }
        self.out[hpos] = hcount;
        let mut w = Wr::new(&mut self.out);
        // ---- missiles in flight
        let mpos = w.len();
        w.u8(0);
        let mut mcount = 0u8;
        for m in &self.missiles {
            if mcount >= 60 {
                break;
            }
            if dist_xz(m.pos.x, m.pos.z, center.x, center.z) > INTEREST {
                continue;
            }
            w.u16(m.id);
            w.f32(m.pos.x);
            w.f32(m.pos.y);
            w.f32(m.pos.z);
            w.u16(q_yaw(yaw_to(m.dir.x, m.dir.z)));
            w.i16(q_pitch(m.dir.y.clamp(-1.0, 1.0).asin()));
            mcount += 1;
        }
        self.out[mpos] = mcount;
        let mut w = Wr::new(&mut self.out);
        // ---- events
        let cpos = w.len();
        w.u16(0);
        let mut ecount = 0u16;
        for e in &self.evs {
            if e.to != 255 && e.to as usize != i {
                continue;
            }
            let d = dist_xz(e.x, e.z, center.x, center.z);
            if e.filt && d > INTEREST {
                continue;
            }
            let s = e.start as usize;
            let l = e.len as usize;
            let sh = e.shooter as usize;
            let vi = e.victim as usize;
            let hidden_sh = sh < MAX_PLAYERS && sh != i && !send[sh];
            let hidden_vi = vi < MAX_PLAYERS && vi != i && !send[vi];
            let kind = self.ev[s];
            if kind == E_SHOT && l == 28 && (hidden_sh || hidden_vi) {
                let mut b = [0u8; 28];
                b.copy_from_slice(&self.ev[s..s + l]);
                if hidden_vi {
                    // A hit on a player this viewer is not sent: the impact
                    // would locate them, so show the shot as if it missed.
                    put_v3(&mut b, 15, e.alt);
                    b[27] = e.alt_kind;
                }
                if hidden_sh {
                    // A shooter this viewer cannot see: audible, but no id and
                    // only the centre of a cell of a nested world-aligned grid
                    // (32 / 16 / 8 m). The range cut-off and each refinement
                    // depend only on the centre of the enclosing coarser cell,
                    // so the cell is all a viewer learns (never a distance).
                    let m = get_v3(&b, 3);
                    let mut g = 32.0f32;
                    if dist_xz(cell_c(m.x, g), cell_c(m.z, g), center.x, center.z) > HIDDEN_SHOT_R {
                        continue;
                    }
                    while g > 8.0 && dist_xz(cell_c(m.x, g), cell_c(m.z, g), center.x, center.z) < g * 6.0 {
                        g *= 0.5;
                    }
                    let from = V3::new(cell_c(m.x, g), cell_c(m.y, 2.0), cell_c(m.z, g));
                    b[1] = 255;
                    put_v3(&mut b, 3, from);
                    // A miss ends exactly `range` from the aim origin, and a
                    // wall hit near the cell is the cover the shooter stands
                    // at (or where the eye-obstruction rule stopped the
                    // shot): either would locate the shooter. So would a wall
                    // hit the viewer cannot see, intersected with the map's
                    // lines of sight (the shooter's eye sees every impact).
                    // Send only a coarse direction, drawn from the coarse
                    // origin. The choice depends on values the viewer would
                    // be sent (the end, the cell centre) and its own eye, so
                    // it tells nothing about the shooter.
                    let endp = get_v3(&b, 15);
                    let near_wall = b[27] == 3 && (endp - from).len() < g + 8.0;
                    let unseen = b[27] == 3 && {
                        let dv = endp - ceye;
                        let l = dv.len();
                        // (stopping just short of the end: walls are 0.3 m
                        // thick, and an impact on the far face is unseen)
                        l > 0.5 && self.w.raycast(ceye, dv * (1.0 / l), (l - 0.03).max(0.0)).is_some()
                    };
                    if b[27] == 0 || near_wall || unseen {
                        let dir = if e.dir.len() > 0.5 { e.dir } else { (get_v3(&b, 15) - m).norm() };
                        let q = |v: f32| (v * 16.0).round() / 16.0;
                        let dq = V3::new(q(dir.x), q(dir.y), q(dir.z)).norm();
                        let range = WD.get(b[2] as usize).map_or(100.0, |w| w.range);
                        put_v3(&mut b, 15, from + dq * range);
                        b[27] = 0;
                    }
                }
                self.out.extend_from_slice(&b);
            } else if kind == E_HURT && l == 11 && hidden_sh_id(&self.ev[s..s + l], i, &send) {
                // Hurt by a player this viewer is not sent: the HUD only needs
                // a direction, so no id and no position.
                let mut b = [0u8; 11];
                b.copy_from_slice(&self.ev[s..s + l]);
                let sx = f32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                let sz = f32::from_le_bytes([b[6], b[7], b[8], b[9]]);
                let step = TAU / 32.0;
                let a = ((sx - me.ms.pos.x).atan2(sz - me.ms.pos.z) / step).round() * step;
                b[1] = HURT_HIDDEN;
                b[2..6].copy_from_slice(&(me.ms.pos.x + a.sin() * 20.0).to_le_bytes());
                b[6..10].copy_from_slice(&(me.ms.pos.z + a.cos() * 20.0).to_le_bytes());
                self.out.extend_from_slice(&b);
            } else {
                self.out.extend_from_slice(&self.ev[s..s + l]);
            }
            ecount += 1;
        }
        // item changes this viewer may know: near it, its own, or made by a
        // player it can see. Others wait (stale ghosts) until it comes near,
        // or until the chunk has been quiet long enough to resync.
        let mut fresh_hidden = [false; ICH * ICH];
        if self.players[i].item_ver.len() == ICH * ICH {
            let n = self.ilog_n;
            let from = self.players[i].ilog_from.max(n.saturating_sub(ILOG as u32));
            // Newest first: a change is only sent while the viewer holds no
            // newer state of that item (sent earlier, by a resync, or in this
            // snapshot), so a withheld older change (say, the creation of a
            // death pile far away) can never be replayed over a newer one
            // (its looting) and bring looted items back as ghosts.
            let mut covered = [0u64; MAX_ITEMS / 64 + 1];
            let mut now: Vec<usize> = Vec::new();
            // per chunk: ring entries newer than the viewer's version, and
            // whether one of them is still withheld
            let mut newer = [0u32; ICH * ICH];
            let mut withheld = [false; ICH * ICH];
            for idx in (from..n).rev() {
                let k = idx as usize % ILOG;
                let bit = 1u64 << (k % 64);
                let e = self.ilog[k];
                let c = e.chunk as usize;
                let (cw, cb) = (e.id as usize / 64, 1u64 << (e.id % 64));
                let is_newer = e.ver.wrapping_sub(self.players[i].item_ver[c]) as i32 > 0;
                if is_newer {
                    newer[c] += 1;
                }
                if self.players[i].ilog_sent[k / 64] & bit != 0 || covered[cw] & cb != 0 {
                    // delivered, or superseded by a newer state the viewer holds
                    self.players[i].ilog_sent[k / 64] |= bit;
                    covered[cw] |= cb;
                    continue;
                }
                let a = e.actor as usize;
                let near = dist_xz(e.it.pos.x, e.it.pos.z, center.x, center.z) < ALWAYS_SEND_R;
                // made by a player this viewer could see at the time (not
                // one that comes into view later: that would release its
                // hidden loot route, oldest first, the moment it is seen)
                let seen = a < MAX_PLAYERS && send[a] && {
                    let from = self.players[i].sent_from[a];
                    from != 0 && from <= e.tick
                };
                if a == i || near || seen {
                    now.push(k);
                    covered[cw] |= cb;
                    self.players[i].ilog_sent[k / 64] |= bit;
                } else {
                    withheld[c] |= is_newer;
                    // Each withheld change draws its own quiet period, so the
                    // viewer cannot re-draw it with changes of its own.
                    if tick.wrapping_sub(e.tick) < self.quiet_ticks(i, c, e.ver) {
                        fresh_hidden[c] = true;
                    }
                }
            }
            for &k in now.iter().rev() {
                let e = self.ilog[k];
                let mut w = Wr::new(&mut self.out);
                if e.it.alive {
                    w.u8(E_ITEM);
                    w.u16(item_wire(salt, e.id));
                    w.u8(e.it.kind);
                    w.u16(e.it.amount);
                    w.f32(e.it.pos.x);
                    w.f32(e.it.pos.y);
                    w.f32(e.it.pos.z);
                } else {
                    w.u8(E_IDEL);
                    w.u16(item_wire(salt, e.id));
                }
                ecount += 1;
            }
            // A chunk whose every change since the viewer's version reached it
            // as a delta is up to date. Resyncing it anyway (say, right after
            // the viewer's own pickup) would answer at once whether hidden
            // changes are pending there.
            for c in 0..ICH * ICH {
                let (v, cv) = (self.players[i].item_ver[c], self.chunk_ver[c]);
                if v != cv && !withheld[c] && newer[c] == cv.wrapping_sub(v) {
                    self.players[i].item_ver[c] = cv;
                }
            }
            // If the ring no longer holds every change of the quiet window, only
            // chunks quiet for the whole window may be resynced.
            let ring_full = n as usize > ILOG && tick.wrapping_sub(self.ilog[n as usize % ILOG].tick) < ITEM_QUIET + 150;
            if ring_full {
                for c in 0..ICH * ICH {
                    // timed from the newest change (a change of the viewer's
                    // own restarts the wait, so it cannot shorten it)
                    if tick.wrapping_sub(self.chunk_tick[c]) < self.quiet_ticks(i, c, self.chunk_ver[c]) {
                        fresh_hidden[c] = true;
                    }
                }
            }
        }
        // item chunk resyncs (nearest first, a few per snapshot)
        if self.players[i].item_ver.len() == ICH * ICH {
            let mut todo: Vec<(f32, usize)> = Vec::new();
            for c in 0..ICH * ICH {
                if self.players[i].item_ver[c] == self.chunk_ver[c] || fresh_hidden[c] {
                    continue;
                }
                let x0 = (c % ICH) as f32 * ICW;
                let z0 = (c / ICH) as f32 * ICW;
                let dx = (x0 - center.x).max(0.0).max(center.x - (x0 + ICW));
                let dz = (z0 - center.z).max(0.0).max(center.z - (z0 + ICW));
                let d = (dx * dx + dz * dz).sqrt();
                if d <= ITEM_SYNC_R {
                    todo.push((d, c));
                }
            }
            todo.sort_by(|a, b| a.0.total_cmp(&b.0));
            for &(_, c) in todo.iter().take(4) {
                let mut ids: Vec<usize> = (0..self.items.len())
                    .filter(|&k| self.items[k].alive && item_chunk(self.items[k].pos.x, self.items[k].pos.z) == c)
                    .collect();
                // by wire id, not in creation order (which would date them)
                ids.sort_by_key(|&k| item_wire(salt, k as u16));
                let mut w = Wr::new(&mut self.out);
                w.u8(E_ICHUNK);
                w.u8(c as u8);
                w.u16(ids.len() as u16);
                for &k in &ids {
                    let it = &self.items[k];
                    w.u16(item_wire(salt, k as u16));
                    w.u8(it.kind);
                    w.u16(it.amount);
                    w.f32(it.pos.x);
                    w.f32(it.pos.y);
                    w.f32(it.pos.z);
                }
                let v = self.chunk_ver[c];
                self.players[i].item_ver[c] = v;
                ecount += 1;
                // the resync carries the chunk's current state: older changes
                // in it must not be replayed later
                let n = self.ilog_n;
                let from = n.saturating_sub(ILOG as u32);
                for idx in from..n {
                    let k = idx as usize % ILOG;
                    if self.ilog[k].chunk as usize == c {
                        self.players[i].ilog_sent[k / 64] |= 1u64 << (k % 64);
                    }
                }
            }
        }
        self.out[cpos..cpos + 2].copy_from_slice(&ecount.to_le_bytes());
        self.players[i].snap_round = self.players[i].snap_round.wrapping_add(1);
        &self.out
    }

    /// Clear events after a snapshot round has been sent to everyone.
    pub fn flush(&mut self) {
        self.ev.clear();
        self.evs.clear();
    }
}
