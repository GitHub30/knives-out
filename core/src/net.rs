//! Little-endian binary writer / bounds-checked reader for the wire protocol.

pub struct Wr<'a> {
    pub b: &'a mut Vec<u8>,
}

impl<'a> Wr<'a> {
    pub fn new(b: &'a mut Vec<u8>) -> Wr<'a> {
        Wr { b }
    }
    #[inline]
    pub fn u8(&mut self, v: u8) {
        self.b.push(v);
    }
    #[inline]
    pub fn u16(&mut self, v: u16) {
        self.b.extend_from_slice(&v.to_le_bytes());
    }
    #[inline]
    pub fn i16(&mut self, v: i16) {
        self.b.extend_from_slice(&v.to_le_bytes());
    }
    #[inline]
    pub fn u32(&mut self, v: u32) {
        self.b.extend_from_slice(&v.to_le_bytes());
    }
    #[inline]
    pub fn f32(&mut self, v: f32) {
        let v = if v.is_finite() { v } else { 0.0 };
        self.b.extend_from_slice(&v.to_le_bytes());
    }
    pub fn bytes(&mut self, s: &[u8]) {
        self.b.extend_from_slice(s);
    }
    pub fn len(&self) -> usize {
        self.b.len()
    }
}

pub struct Rd<'a> {
    b: &'a [u8],
    i: usize,
    pub bad: bool,
}

impl<'a> Rd<'a> {
    pub fn new(b: &'a [u8]) -> Rd<'a> {
        Rd { b, i: 0, bad: false }
    }
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        if self.i + N <= self.b.len() {
            out.copy_from_slice(&self.b[self.i..self.i + N]);
            self.i += N;
        } else {
            self.bad = true;
            self.i = self.b.len();
        }
        out
    }
    pub fn u8(&mut self) -> u8 {
        self.take::<1>()[0]
    }
    pub fn i8(&mut self) -> i8 {
        self.take::<1>()[0] as i8
    }
    pub fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.take::<2>())
    }
    pub fn i16(&mut self) -> i16 {
        i16::from_le_bytes(self.take::<2>())
    }
    pub fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take::<4>())
    }
    pub fn remaining(&self) -> usize {
        self.b.len() - self.i
    }
}

/// Encode yaw (radians) to u16 turns.
pub fn q_yaw(yaw: f32) -> u16 {
    let t = yaw / crate::math::TAU;
    let t = t - t.floor();
    ((t * 65536.0) as u32 & 0xFFFF) as u16
}
pub fn dq_yaw(q: u16) -> f32 {
    q as f32 * (crate::math::TAU / 65536.0)
}
pub fn q_pitch(p: f32) -> i16 {
    (crate::math::clampf(p, -1.5, 1.5) * (32767.0 / 1.5)) as i16
}
pub fn dq_pitch(q: i16) -> f32 {
    crate::math::clampf(q as f32 * (1.5 / 32767.0), -1.5, 1.5)
}
