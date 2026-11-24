//! Authoritative match simulation + wire protocol. One `Match` per room.

use crate::defs::*;
use crate::math::*;
use crate::net::*;
use crate::player::*;
use crate::rng::Rng;
use crate::world::*;

pub const PROTO: u16 = 5;
pub const HIST: usize = 32;
pub const MAX_ITEMS: usize = 4000;
pub const INTEREST: f32 = 520.0;
pub const CMDQ_CAP: usize = 8;
/// Ticks of late input absorbed before the server simulates a player without it.
pub const LAG_GRACE: u8 = 3;
/// Late commands covered by idle steps must arrive within this many ticks.
const DEBT_TTL: u8 = 10;
const DEBT_MAX: u8 = 16;
/// Idle steps keep the last movement input for this many ticks, then stop.
const IDLE_REPEAT: u8 = 6;
/// If the lag never returns to zero for this long, take one idle step.
const DRIFT_T: u16 = 60;
/// More than this many commands left queued for BACKLOG_T ticks: drop one.
const BACKLOG_MAX: usize = 3;
const BACKLOG_T: u8 = 30;
/// Lag compensation: the most a shot may be rewound, and how far above the
/// player's recent minimum rewind a single command may claim.
pub const REWIND_MAX: f32 = 15.0;
const REWIND_MARGIN: f32 = 4.0;
/// Buttons that act on a single command (carried over when one is dropped).
const EDGE_BUTTONS: u16 = B_JUMP | B_RELOAD | B_PICKUP | B_SPECT | B_NADE | B_FIRE;
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
const ILOG: usize = 1024;
/// A chunk is only resynced to a viewer when it holds no change younger than
/// this that the viewer was not allowed to see.
const ITEM_QUIET: u32 = 300;
/// Hidden shooters: shots farther than this are not sent, nearer ones only
/// with a coarse position and no shooter id.
const HIDDEN_SHOT_R: f32 = 330.0;

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
    /// Ticks the queue has held more than BACKLOG_MAX commands.
    pub backlog_t: u8,
    /// Recent minimum of the rewind (ticks) this client claims for its view.
    pub rew_floor: f32,
    pub cmds_applied: u32,
    pub idle_steps: u32,
    pub last_cmd: Cmd,
    /// Buttons of the previous simulated command (press-edge detection).
    pub prev_buttons: u16,
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
}

pub struct Match {
    pub w: World,
    pub rng: Rng,
    /// Shot spread only (see `with_secrets`).
    crng: Rng,
    pub tick: u32,
    pub phase: u8,
    pub lobby_t: f32,
    pub end_t: f32,
    pub match_t: f32,
    pub flags: u32,
    pub slots: usize,
    pub snap_every: u32,
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
}

/// Name characters: 0 = dropped, 1 = base character, 2 = combining mark,
/// 3 = space. A whitelist of letters, digits, punctuation and symbols of
/// common scripts: control, format, private-use, unassigned and invisible
/// filler characters are all dropped.
fn name_char_class(c: u32) -> u8 {
    match c {
        0x20 | 0xA0 | 0x2000..=0x200A | 0x205F | 0x3000 => 3,
        0x22 | 0x26 | 0x27 | 0x3C | 0x3E | 0x5C | 0x60 => 0, // " & ' < > \ `
        0x21..=0x7E => 1,
        0xA1..=0xAC | 0xAE..=0x2AF => 1,          // Latin-1 (no soft hyphen), Latin ext., IPA
        0x300..=0x36F => 2,                         // combining diacritics
        0x370..=0x377 | 0x37A..=0x37F | 0x384..=0x38A | 0x38C | 0x38E..=0x3A1 | 0x3A3..=0x3FF => 1, // Greek
        0x400..=0x482 | 0x48A..=0x52F => 1,         // Cyrillic
        0x483..=0x489 => 2,
        0x531..=0x556 | 0x561..=0x587 => 1,         // Armenian
        0x591..=0x5BD | 0x5BF | 0x5C1..=0x5C2 | 0x5C4..=0x5C5 | 0x5C7 => 2, // Hebrew points
        0x5D0..=0x5EA => 1,
        0x620..=0x64A | 0x660..=0x669 | 0x671..=0x6D3 => 1, // Arabic
        0x64B..=0x65F | 0x670 => 2,
        0x900..=0x903 | 0x93A..=0x93C | 0x93E..=0x94F | 0x951..=0x957 | 0x962..=0x963 => 2, // Devanagari signs
        0x904..=0x939 | 0x93D | 0x950 | 0x958..=0x961 | 0x964..=0x97F => 1,
        0xE31 | 0xE34..=0xE3A | 0xE47..=0xE4E => 2, // Thai marks
        0xE01..=0xE30 | 0xE32..=0xE33 | 0xE3F..=0xE46 | 0xE4F..=0xE5B => 1,
        0x1E00..=0x1EFF => 1,                       // Latin extended additional (Vietnamese)
        0x2010..=0x2027 | 0x2030..=0x205E => 1,     // punctuation (no separators / bidi / invisible)
        0x2070..=0x2071 | 0x2074..=0x208E | 0x2090..=0x209C => 1,
        0x20A0..=0x20C0 | 0x2100..=0x218B | 0x2190..=0x23FF | 0x2460..=0x27BF | 0x27C0..=0x2BFF => 1,
        0x2E80..=0x2FDF => 1,                       // CJK radicals
        0x302A..=0x302F | 0x3099..=0x309A => 2,
        0x3001..=0x3029 | 0x3030..=0x303F => 1,     // CJK punctuation
        0x3041..=0x3096 | 0x309B..=0x30FF => 1,     // kana
        0x3105..=0x312F | 0x3131..=0x3163 | 0x3165..=0x318E | 0x3190..=0x31FF => 1, // (no Hangul filler)
        0x3200..=0x4DBF | 0x4DC0..=0x9FFF | 0xAC00..=0xD7A3 | 0xF900..=0xFAFF | 0xFE30..=0xFE4F => 1,
        0xFF01..=0xFF5E | 0xFF61..=0xFF9F | 0xFFE0..=0xFFEE => 1, // full/halfwidth (no halfwidth filler)
        0x1F3FB..=0x1F3FF => 2,                     // emoji skin tones modify the previous emoji
        0x1F000..=0x1FAFF => 1,                     // emoji and symbols
        0x20000..=0x2FFFF => 1,                     // CJK extensions
        _ => 0,
    }
}

/// At most this many combining marks per base character (accents, Thai and
/// Indic vowel signs), so names cannot stack marks into the rows above.
const NAME_MARKS: u32 = 2;

fn sanitize_name(raw: &[u8], fallback: &str) -> String {
    let s = String::from_utf8_lossy(raw);
    let mut out = String::new();
    let mut count = 0;
    let mut last_space = true;
    let mut marks = NAME_MARKS; // no base character yet: no marks allowed
    for ch in s.chars() {
        match name_char_class(ch as u32) {
            1 => {
                out.push(ch);
                last_space = false;
                marks = 0;
            }
            2 => {
                if marks < NAME_MARKS {
                    out.push(ch);
                    marks += 1;
                }
                continue;
            }
            3 => {
                if last_space {
                    continue;
                }
                out.push(' ');
                last_space = true;
                marks = NAME_MARKS;
            }
            _ => continue,
        }
        count += 1;
        if count >= 14 {
            break;
        }
    }
    let t = out.trim().to_string();
    if t.is_empty() {
        fallback.to_string()
    } else {
        t
    }
}

/// Comparison key for duplicate names: case-folded, fullwidth folded to
/// ASCII, marks removed and common Cyrillic / Greek look-alikes mapped to
/// Latin, so 'Kaze', 'Ｋａｚｅ' and 'Kаze' (Cyrillic a) count as the same name.
pub fn name_key(name: &str) -> String {
    let mut k = String::new();
    for ch in name.chars() {
        let c = ch as u32;
        if name_char_class(c) == 2 {
            continue;
        }
        let ch = if (0xFF01..=0xFF5E).contains(&c) { char::from_u32(c - 0xFEE0).unwrap_or(ch) } else { ch };
        for lc in ch.to_lowercase() {
            k.push(match lc {
                'а' => 'a',
                'в' | 'β' => 'b',
                'е' | 'ё' | 'ε' => 'e',
                'к' | 'κ' => 'k',
                'м' => 'm',
                'н' | 'һ' => 'h',
                'о' | 'ο' => 'o',
                'р' | 'ρ' => 'p',
                'с' => 'c',
                'т' | 'τ' => 't',
                'у' => 'y',
                'х' | 'χ' => 'x',
                'і' | 'ι' => 'i',
                'ј' => 'j',
                'ѕ' => 's',
                'ԁ' => 'd',
                'α' => 'a',
                'η' => 'n',
                'ν' => 'v',
                'υ' => 'u',
                'ζ' => 'z',
                _ => lc,
            });
        }
    }
    k
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
        let mut m = Match {
            w,
            rng: Rng::new(secret ^ (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5EED),
            crng: Rng::new(combat ^ 0x6A09_E667_F3BC_C908),
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
        };
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
        self.ilog[k] = ILogE { tick: self.tick, id, actor: self.item_actor, chunk: c as u8, it };
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
        self.evs.push(EvRec { to, start, len, x, z, filt, shooter: 255 });
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
            // AI takes over the abandoned character.
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
            let skill = self.rng.range(0.15, 0.85);
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
        let (pick, spect) = (pressed(B_PICKUP), pressed(B_SPECT));
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
        let mult = self.players[i].speed_mult();
        let w = &self.w;
        let p = &mut self.players[i];
        let ev = step_move(w, &mut p.ms, &c, mult, DT);
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
    fn rewound(&self, j: usize, view: u32) -> Option<HistE> {
        let now = self.tick as f32;
        // Clients render ~4.5 ticks behind + latency + command queueing;
        // 15 ticks (500 ms) covers RTTs up to ~250 ms. Commands run before
        // record_hist() writes this tick's slot, so tick - 1 is the newest
        // recorded state (hist[tick % HIST] still holds tick - HIST).
        let vt = clampf(view as f32 / 256.0, (now - REWIND_MAX).max(0.0), (now - 1.0).max(0.0));
        let t0 = vt.floor();
        let frac = vt - t0;
        let t0u = t0 as u32;
        if self.tick >= HIST as u32 && t0u + (HIST as u32) <= self.tick {
            return None;
        }
        let a = self.hist[t0u as usize % HIST][j];
        let b = if t0u + 1 < self.tick { self.hist[(t0u as usize + 1) % HIST][j] } else { a };
        if !a.alive && !b.alive {
            return None;
        }
        Some(HistE { pos: a.pos.lerp(b.pos, frac), crouch: if frac < 0.5 { a.crouch } else { b.crouch }, alive: true })
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
            if !q.used || !q.alive || q.ms.mode == MODE_PLANE {
                continue;
            }
            let h = if self.players[shooter].bot { HistE { pos: q.ms.pos, crouch: q.ms.crouch, alive: true } } else {
                match self.rewound(j, view) {
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
            // head sphere around the rendered head; the body capsule reaches
            // the shoulders. Head vs body is decided by where the ray enters:
            // body below the neck (the rendered torso top), head above it.
            let head_c = p + V3::new(0.0, eye_h(h.crouch) + if h.crouch { 0.1 } else { 0.06 }, 0.0);
            let th = ray_sphere(o, d, head_c, 0.2, lim);
            let tb = ray_capsule(o, d, p.x, p.z, p.y + 0.3, head_c.y - 0.3, 0.33, lim);
            let neck = head_c.y - 0.1;
            let hit = match (th, tb) {
                (Some(th), Some(tb)) if tb < th && o.y + d.y * tb < neck => Some((tb, false)),
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
            if ml > 0.3 {
                let dd = md * (1.0 / ml);
                if let Some(h) = self.w.raycast(eye, dd, ml - 0.2) {
                    end = eye + dd * h.0;
                    victim = None;
                    hit_kind = 3;
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
        let start = self.ev_begin();
        let mut w = Wr::new(&mut self.ev);
        w.u8(E_BOOM);
        w.f32(n.pos.x);
        w.f32(n.pos.y);
        w.f32(n.pos.z);
        self.ev_end(start, 255, Some(n.pos));
        let radius = 7.5;
        for j in 0..MAX_PLAYERS {
            let q = &self.players[j];
            if !q.used || !q.alive || q.ms.mode == MODE_PLANE {
                continue;
            }
            let c = q.ms.pos + V3::new(0.0, 0.9, 0.0);
            let d = (c - n.pos).len();
            if d > radius {
                continue;
            }
            if self.w.blocked(n.pos + V3::new(0.0, 0.3, 0.0), c) {
                continue;
            }
            let k = 1.0 - d / radius;
            let dmg = 115.0 * k.powf(1.4);
            if dmg > 1.0 {
                self.damage(j, n.owner, dmg, false, 200, n.pos);
            }
        }
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
            if attacker != KILLER_ZONE && weapon != 200 {
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
            q.ms.mode = MODE_DEAD;
            q.ms.vel = V3::ZERO;
            q.reload_t = 0.0;
            q.heal_t = 0.0;
            pos = q.ms.pos;
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
            let p = &self.players[j];
            self.hist[slot][j] = HistE { pos: p.ms.pos, crouch: p.ms.crouch, alive: p.used && p.alive };
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
                // The view tick (lag compensation) is client-claimed: allow at
                // most REWIND_MARGIN ticks above this player's recent minimum,
                // so a low-latency client cannot rewind selectively.
                let r = clampf(now - c.view as f32 / 256.0, 1.0, REWIND_MAX);
                p.rew_floor = if r < p.rew_floor { r } else { (p.rew_floor + 0.08).min(r) };
                let r = r.min(p.rew_floor + REWIND_MARGIN);
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
                    c.buttons &= B_CROUCH | B_ADS | B_SPRINT;
                    p.idle_run = p.idle_run.saturating_add(1);
                    self.apply_cmd(i, c);
                }
            }
            // A queue that stays long means a client clock running fast: drop
            // the oldest command now and then so input latency stays low.
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
        let alive: Vec<usize> = (0..self.items.len()).filter(|&k| self.items[k].alive).collect();
        w.u16(alive.len() as u16);
        for &k in &alive {
            let it = &self.items[k];
            w.u16(k as u16);
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

    pub fn snapshot(&mut self, pid: u8) -> &[u8] {
        self.out.clear();
        let i = pid as usize;
        if i >= MAX_PLAYERS || !self.players[i].used {
            return &self.out;
        }
        let me = self.players[i].clone();
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
                let head = q.ms.pos + V3::new(0.0, eye_h(q.ms.crouch) + 0.1, 0.0);
                let chest = q.ms.pos + V3::new(0.0, 0.9, 0.0);
                let visible = !self.w.blocked(ceye, head) || !self.w.blocked(ceye, chest);
                let pv = &mut self.players[i];
                pv.vis_check[j] = tick;
                if visible {
                    pv.seen_tick[j] = tick;
                }
            }
            let seen = self.players[i].seen_tick[j];
            send[j] = seen != 0 && tick.wrapping_sub(seen) <= 30;
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
            if sh < MAX_PLAYERS && sh != i && !send[sh] && l == 28 {
                // A shooter this viewer cannot see: audible, but no id and
                // only a coarse position (a grid that grows with distance).
                if d > HIDDEN_SHOT_R {
                    continue;
                }
                let mut b = [0u8; 28];
                b.copy_from_slice(&self.ev[s..s + l]);
                b[1] = 255;
                let g = (d / 8.0).max(4.0);
                for (k, step) in [(3usize, g), (7, 2.0), (11, g)] {
                    let v = f32::from_le_bytes([b[k], b[k + 1], b[k + 2], b[k + 3]]);
                    let qv = ((v / step).floor() + 0.5) * step;
                    b[k..k + 4].copy_from_slice(&qv.to_le_bytes());
                }
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
            for idx in from..n {
                let k = idx as usize % ILOG;
                if self.players[i].ilog_sent[k / 64] & (1u64 << (k % 64)) != 0 {
                    continue;
                }
                let e = self.ilog[k];
                let a = e.actor as usize;
                let near = dist_xz(e.it.pos.x, e.it.pos.z, center.x, center.z) < ALWAYS_SEND_R;
                if a == i || near || (a < MAX_PLAYERS && send[a]) {
                    let mut w = Wr::new(&mut self.out);
                    if e.it.alive {
                        w.u8(E_ITEM);
                        w.u16(e.id);
                        w.u8(e.it.kind);
                        w.u16(e.it.amount);
                        w.f32(e.it.pos.x);
                        w.f32(e.it.pos.y);
                        w.f32(e.it.pos.z);
                    } else {
                        w.u8(E_IDEL);
                        w.u16(e.id);
                    }
                    ecount += 1;
                    self.players[i].ilog_sent[k / 64] |= 1u64 << (k % 64);
                } else if tick.wrapping_sub(e.tick) < ITEM_QUIET {
                    fresh_hidden[e.chunk as usize] = true;
                }
            }
            // If the ring no longer holds every change of the quiet window, only
            // chunks quiet for the whole window may be resynced.
            let ring_full = n as usize > ILOG && tick.wrapping_sub(self.ilog[n as usize % ILOG].tick) < ITEM_QUIET;
            if ring_full {
                for c in 0..ICH * ICH {
                    if tick.wrapping_sub(self.chunk_tick[c]) < ITEM_QUIET {
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
                let ids: Vec<usize> = (0..self.items.len())
                    .filter(|&k| self.items[k].alive && item_chunk(self.items[k].pos.x, self.items[k].pos.z) == c)
                    .collect();
                let mut w = Wr::new(&mut self.out);
                w.u8(E_ICHUNK);
                w.u8(c as u8);
                w.u16(ids.len() as u16);
                for &k in &ids {
                    let it = &self.items[k];
                    w.u16(k as u16);
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
