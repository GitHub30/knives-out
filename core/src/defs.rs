//! Game constants shared by client and server.

pub const TICK_HZ: u32 = 30;
pub const DT: f32 = 1.0 / 30.0;
pub const MAX_PLAYERS: usize = 32;

pub const MODE_PLANE: u8 = 0;
pub const MODE_FALL: u8 = 1;
pub const MODE_PARA: u8 = 2;
pub const MODE_GROUND: u8 = 3;
pub const MODE_DEAD: u8 = 4;

// Command buttons (held state unless noted as edge).
pub const B_FIRE: u16 = 1 << 0;
pub const B_ADS: u16 = 1 << 1;
pub const B_CROUCH: u16 = 1 << 2;
pub const B_SPRINT: u16 = 1 << 3;
pub const B_JUMP: u16 = 1 << 4; // edge
pub const B_RELOAD: u16 = 1 << 5; // edge
pub const B_PICKUP: u16 = 1 << 6; // edge
pub const B_SPECT: u16 = 1 << 7; // edge
// bits 8-9: slot request edge (0 none, 1 = slot A, 2 = slot B, 3 = fists)
// bits 10-11: heal request edge (0 none, 1 bandage, 2 medkit, 3 drink)
pub const B_NADE: u16 = 1 << 12; // edge
pub const B_MASK: u16 = 0x1FFF;

pub const W_FISTS: u8 = 0;
pub const W_PISTOL: u8 = 1;
pub const W_SMG: u8 = 2;
pub const W_AR: u8 = 3;
pub const W_SHOTGUN: u8 = 4;
pub const W_DMR: u8 = 5;
pub const W_SNIPER: u8 = 6;
pub const NUM_WEAPONS: usize = 7;

#[derive(Clone, Copy, Debug)]
pub struct WDef {
    /// Ammo type index (255 = none).
    pub ammo: u8,
    pub dmg: f32,
    /// Seconds between shots.
    pub interval: f32,
    pub mag: u16,
    pub reload: f32,
    /// Hip-fire cone half-angle (radians).
    pub hip: f32,
    /// ADS cone half-angle (radians).
    pub ads: f32,
    /// Damage falloff start / end distance and multiplier at the end.
    pub f0: f32,
    pub f1: f32,
    pub fmin: f32,
    pub pellets: u8,
    pub head: f32,
    pub range: f32,
    /// Movement speed multiplier while held.
    pub speed: f32,
    /// ADS zoom factor.
    pub zoom: f32,
    /// Desirability for bots.
    pub rank: u8,
    /// Client camera kick in radians.
    pub recoil: f32,
}

const fn wd(
    ammo: u8, dmg: f32, interval: f32, mag: u16, reload: f32, hip: f32, ads: f32, f0: f32, f1: f32,
    fmin: f32, pellets: u8, head: f32, range: f32, speed: f32, zoom: f32, rank: u8, recoil: f32,
) -> WDef {
    WDef { ammo, dmg, interval, mag, reload, hip, ads, f0, f1, fmin, pellets, head, range, speed, zoom, rank, recoil }
}

pub const WD: [WDef; NUM_WEAPONS] = [
    //  ammo dmg   intv  mag reload hip    ads    f0    f1    fmin pel head range speed zoom rank recoil
    wd(255, 20.0, 0.50, 0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1, 1.5, 1.9, 1.05, 1.0, 0, 0.0), // fists
    wd(0, 24.0, 0.17, 15, 1.4, 0.040, 0.016, 25.0, 70.0, 0.6, 1, 2.0, 160.0, 1.0, 1.3, 1, 0.018), // pistol
    wd(0, 17.0, 0.075, 30, 1.9, 0.050, 0.025, 20.0, 70.0, 0.55, 1, 1.8, 160.0, 1.0, 1.4, 3, 0.010), // smg
    wd(1, 23.0, 0.10, 30, 2.2, 0.045, 0.012, 50.0, 160.0, 0.7, 1, 2.0, 420.0, 0.95, 2.0, 5, 0.014), // ar
    wd(2, 14.0, 0.90, 5, 2.8, 0.095, 0.080, 8.0, 28.0, 0.25, 9, 1.5, 60.0, 0.95, 1.3, 3, 0.060), // shotgun
    wd(1, 45.0, 0.33, 10, 2.4, 0.060, 0.004, 100.0, 320.0, 0.8, 1, 2.2, 650.0, 0.92, 4.0, 4, 0.030), // dmr
    wd(3, 90.0, 1.40, 5, 3.0, 0.090, 0.001, 200.0, 650.0, 0.85, 1, 2.6, 900.0, 0.90, 7.0, 6, 0.070), // sniper
];

pub const NUM_AMMO: usize = 4;
pub const AMMO_CAP: [u16; NUM_AMMO] = [240, 240, 60, 40];
pub const AMMO_STACK: [u16; NUM_AMMO] = [30, 30, 10, 10];

// Item kinds. 1..=6 are weapons (kind == weapon id).
pub const IK_AMMO: u8 = 10; // 10..=13
pub const IK_BANDAGE: u8 = 20;
pub const IK_MEDKIT: u8 = 21;
pub const IK_DRINK: u8 = 22;
pub const IK_VEST: u8 = 30; // 30..=32 -> level 1..3
pub const IK_HELM: u8 = 33; // 33..=35 -> level 1..3
pub const IK_NADE: u8 = 40;

pub const HEAL_CAP: [u8; 3] = [15, 5, 5];
pub const HEAL_TIME: [f32; 3] = [3.0, 6.0, 2.0];
pub const NADE_CAP: u8 = 4;
pub const ARMOR_RED: [f32; 4] = [0.0, 0.25, 0.40, 0.55];

pub fn is_weapon(kind: u8) -> bool {
    (1..=6).contains(&kind)
}

pub struct ZStage {
    pub wait: f32,
    pub shrink: f32,
    pub r: f32,
    pub dps: f32,
}

pub const ZONES: [ZStage; 6] = [
    ZStage { wait: 100.0, shrink: 50.0, r: 360.0, dps: 1.0 },
    ZStage { wait: 55.0, shrink: 40.0, r: 210.0, dps: 2.0 },
    ZStage { wait: 45.0, shrink: 35.0, r: 120.0, dps: 4.0 },
    ZStage { wait: 35.0, shrink: 30.0, r: 60.0, dps: 6.0 },
    ZStage { wait: 25.0, shrink: 25.0, r: 25.0, dps: 9.0 },
    ZStage { wait: 20.0, shrink: 25.0, r: 0.0, dps: 14.0 },
];

pub const PLANE_SPEED: f32 = 75.0;
pub const PLANE_ALT: f32 = 330.0;

pub const BOT_NAMES: [&str; 48] = [
    "Kaze", "Sora_77", "Hayate", "Ren", "Mizuki", "Takumi", "Yuzu", "Kuro", "Akane", "Hikari",
    "Raiden", "Shion", "Nagi", "Kaito", "Rin_x", "Haru", "Aoi", "Kotetsu", "Tsubasa", "Natsu",
    "Ghost", "Viper", "Falcon", "Nomad", "Rook", "Echo", "Blaze", "Frost", "Onyx", "Specter",
    "Juno", "Atlas", "Kestrel", "Drift", "Cinder", "Moss", "Pike", "Quill", "Rune", "Saber",
    "Taro99", "Momo", "Genji", "Botan", "Daichi", "Ema", "Fuyu", "Goro",
];
