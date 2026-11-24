//! Minimal 3D math used by the simulation. Everything is f32 and pure so the
//! same wasm binary produces bit-identical results on client and server.

use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl V3 {
    pub const ZERO: V3 = V3 { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: V3 = V3 { x: 0.0, y: 1.0, z: 0.0 };
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> V3 {
        V3 { x, y, z }
    }
    #[inline]
    pub fn dot(self, o: V3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    #[inline]
    pub fn cross(self, o: V3) -> V3 {
        V3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    #[inline]
    pub fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
    #[inline]
    pub fn len2(self) -> f32 {
        self.dot(self)
    }
    #[inline]
    pub fn len_xz(self) -> f32 {
        (self.x * self.x + self.z * self.z).sqrt()
    }
    #[inline]
    pub fn norm(self) -> V3 {
        let l = self.len();
        if l > 1e-9 {
            self * (1.0 / l)
        } else {
            V3::ZERO
        }
    }
    #[inline]
    pub fn lerp(self, o: V3, t: f32) -> V3 {
        self + (o - self) * t
    }
    #[inline]
    pub fn finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl Add for V3 {
    type Output = V3;
    #[inline]
    fn add(self, o: V3) -> V3 {
        V3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for V3 {
    type Output = V3;
    #[inline]
    fn sub(self, o: V3) -> V3 {
        V3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Mul<f32> for V3 {
    type Output = V3;
    #[inline]
    fn mul(self, s: f32) -> V3 {
        V3::new(self.x * s, self.y * s, self.z * s)
    }
}
impl Neg for V3 {
    type Output = V3;
    #[inline]
    fn neg(self) -> V3 {
        V3::new(-self.x, -self.y, -self.z)
    }
}
impl AddAssign for V3 {
    #[inline]
    fn add_assign(&mut self, o: V3) {
        self.x += o.x;
        self.y += o.y;
        self.z += o.z;
    }
}
impl SubAssign for V3 {
    #[inline]
    fn sub_assign(&mut self, o: V3) {
        self.x -= o.x;
        self.y -= o.y;
        self.z -= o.z;
    }
}

pub const PI: f32 = std::f32::consts::PI;
pub const TAU: f32 = PI * 2.0;

#[inline]
pub fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}
#[inline]
pub fn lerpf(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
#[inline]
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clampf((x - e0) / (e1 - e0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
/// Wrap to (-PI, PI].
#[inline]
pub fn wrap_angle(a: f32) -> f32 {
    let mut a = a % TAU;
    if a > PI {
        a -= TAU
    } else if a <= -PI {
        a += TAU
    }
    a
}
/// Move `v` toward `target` by at most `d`.
#[inline]
pub fn approach(v: f32, target: f32, d: f32) -> f32 {
    if v < target {
        (v + d).min(target)
    } else {
        (v - d).max(target)
    }
}
/// Look direction. yaw=0 looks along +Z, yaw=PI/2 along +X, pitch>0 looks up.
#[inline]
pub fn dir_from(yaw: f32, pitch: f32) -> V3 {
    let cp = pitch.cos();
    V3::new(yaw.sin() * cp, pitch.sin(), yaw.cos() * cp)
}
#[inline]
pub fn fwd(yaw: f32) -> V3 {
    V3::new(yaw.sin(), 0.0, yaw.cos())
}
/// Screen-right for a camera looking along fwd(yaw) with +Y up (right-handed).
#[inline]
pub fn right(yaw: f32) -> V3 {
    V3::new(-yaw.cos(), 0.0, yaw.sin())
}
#[inline]
pub fn yaw_to(dx: f32, dz: f32) -> f32 {
    dx.atan2(dz)
}
#[inline]
pub fn dist_xz(ax: f32, az: f32, bx: f32, bz: f32) -> f32 {
    let dx = ax - bx;
    let dz = az - bz;
    (dx * dx + dz * dz).sqrt()
}

/// Slab test. Returns entry distance and hit axis (0=x,1=y,2=z). Rays that
/// start inside the box are ignored (returns None) which lets a shot fired from
/// a camera that clipped into a wall still pass through it.
pub fn ray_aabb(o: V3, d: V3, mn: V3, mx: V3, maxt: f32) -> Option<(f32, u8)> {
    let mut t0 = 0.0f32;
    let mut t1 = maxt;
    let mut axis = 255u8;
    let oa = [o.x, o.y, o.z];
    let da = [d.x, d.y, d.z];
    let mna = [mn.x, mn.y, mn.z];
    let mxa = [mx.x, mx.y, mx.z];
    for a in 0..3 {
        if da[a].abs() < 1e-9 {
            if oa[a] < mna[a] || oa[a] > mxa[a] {
                return None;
            }
        } else {
            let inv = 1.0 / da[a];
            let mut ta = (mna[a] - oa[a]) * inv;
            let mut tb = (mxa[a] - oa[a]) * inv;
            if ta > tb {
                std::mem::swap(&mut ta, &mut tb);
            }
            if ta > t0 {
                t0 = ta;
                axis = a as u8;
            }
            if tb < t1 {
                t1 = tb;
            }
            if t0 > t1 {
                return None;
            }
        }
    }
    if axis == 255 {
        return None;
    }
    Some((t0, axis))
}

/// Ray vs vertical cylinder side (no caps). `d` need not be normalized in xz.
pub fn ray_cyl(o: V3, d: V3, cx: f32, cz: f32, r: f32, y0: f32, y1: f32, maxt: f32) -> Option<f32> {
    let ox = o.x - cx;
    let oz = o.z - cz;
    let a = d.x * d.x + d.z * d.z;
    if a < 1e-9 {
        return None;
    }
    let b = ox * d.x + oz * d.z;
    let c = ox * ox + oz * oz - r * r;
    if c < 0.0 {
        return None;
    }
    let disc = b * b - a * c;
    if disc < 0.0 {
        return None;
    }
    let t = (-b - disc.sqrt()) / a;
    if t < 0.0 || t > maxt {
        return None;
    }
    let y = o.y + d.y * t;
    if y < y0 || y > y1 {
        return None;
    }
    Some(t)
}

/// Ray vs sphere, `d` normalized.
pub fn ray_sphere(o: V3, d: V3, c: V3, r: f32, maxt: f32) -> Option<f32> {
    let oc = o - c;
    let b = oc.dot(d);
    let cc = oc.len2() - r * r;
    if cc < 0.0 {
        return Some(0.0);
    }
    let disc = b * b - cc;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    if t >= 0.0 && t <= maxt {
        Some(t)
    } else {
        None
    }
}

/// Ray vs vertical capsule (segment from y0 to y1 at (cx,cz), radius r).
pub fn ray_capsule(o: V3, d: V3, cx: f32, cz: f32, y0: f32, y1: f32, r: f32, maxt: f32) -> Option<f32> {
    let mut best: Option<f32> = None;
    let mut take = |t: Option<f32>| {
        if let Some(t) = t {
            if best.map_or(true, |b| t < b) {
                best = Some(t);
            }
        }
    };
    take(ray_cyl(o, d, cx, cz, r, y0, y1, maxt));
    take(ray_sphere(o, d, V3::new(cx, y0, cz), r, maxt));
    take(ray_sphere(o, d, V3::new(cx, y1, cz), r, maxt));
    best
}
