//! Deterministic RNG (SplitMix64) and 2D Perlin noise.

use crate::math::lerpf;

#[derive(Clone, Copy, Debug, Default)]
pub struct Rng {
    s: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut r = Rng { s: seed ^ 0x9E37_79B9_7F4A_7C15 };
        r.next();
        r
    }
    #[inline]
    pub fn next(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    #[inline]
    pub fn u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }
    /// Uniform in [0, 1).
    #[inline]
    pub fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    #[inline]
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
    /// Uniform integer in [0, n).
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            (((self.next() >> 32) * n as u64) >> 32) as u32
        }
    }
    #[inline]
    pub fn chance(&mut self, p: f32) -> bool {
        self.f() < p
    }
    /// Pick an index from a weight table.
    pub fn weighted(&mut self, w: &[u32]) -> usize {
        let total: u32 = w.iter().sum();
        if total == 0 {
            return 0;
        }
        let mut r = self.below(total);
        for (i, &x) in w.iter().enumerate() {
            if r < x {
                return i;
            }
            r -= x;
        }
        w.len() - 1
    }
}

pub struct Noise {
    p: [u8; 512],
}

#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

impl Noise {
    pub fn new(seed: u64) -> Noise {
        let mut r = Rng::new(seed);
        let mut base = [0u8; 256];
        for (i, b) in base.iter_mut().enumerate() {
            *b = i as u8;
        }
        for i in (1..256usize).rev() {
            let j = r.below(i as u32 + 1) as usize;
            base.swap(i, j);
        }
        let mut p = [0u8; 512];
        for i in 0..512 {
            p[i] = base[i & 255];
        }
        Noise { p }
    }
    #[inline]
    fn grad(h: u8, x: f32, y: f32) -> f32 {
        match h & 7 {
            0 => x + y,
            1 => -x + y,
            2 => x - y,
            3 => -x - y,
            4 => x,
            5 => -x,
            6 => y,
            _ => -y,
        }
    }
    /// Classic 2D Perlin noise, roughly in [-1, 1].
    pub fn perlin(&self, x: f32, y: f32) -> f32 {
        let xf = x.floor();
        let yf = y.floor();
        let xi = ((xf as i32) & 255) as usize;
        let yi = ((yf as i32) & 255) as usize;
        let x = x - xf;
        let y = y - yf;
        let u = fade(x);
        let v = fade(y);
        let p = &self.p;
        let a = p[xi] as usize + yi;
        let b = p[xi + 1] as usize + yi;
        let aa = p[a];
        let ab = p[a + 1];
        let ba = p[b];
        let bb = p[b + 1];
        let x1 = lerpf(Self::grad(aa, x, y), Self::grad(ba, x - 1.0, y), u);
        let x2 = lerpf(Self::grad(ab, x, y - 1.0), Self::grad(bb, x - 1.0, y - 1.0), u);
        lerpf(x1, x2, v)
    }
    pub fn fbm(&self, x: f32, y: f32, oct: u32) -> f32 {
        let mut amp = 0.5;
        let mut freq = 1.0;
        let mut sum = 0.0;
        let mut norm = 0.0;
        for _ in 0..oct {
            sum += amp * self.perlin(x * freq, y * freq);
            norm += amp;
            amp *= 0.5;
            freq *= 2.03;
        }
        sum / norm
    }
}
