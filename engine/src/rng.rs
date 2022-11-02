use std::cell::Cell;

const MULT: u64 = 0x243f6a8885a308d3;

pub struct Rng {
  state: Cell<u64>,
}

impl Rng {
  pub fn new(seed: u64) -> Rng {
    Rng {
      state: Cell::new(seed),
    }
  }

  /// Generate a uniformly random u64.
  #[inline]
  pub fn next_random(&self) -> u64 {
    let state = self.state.get().wrapping_add(1);
    self.state.set(state);
    let mut x = state.wrapping_mul(MULT);
    for _ in 0..4 {
      x ^= x >> 37;
      x = x.wrapping_mul(MULT);
    }
    x
  }

  /// Generate a uniformly random u32 in the range [0, max).
  #[inline]
  pub fn generate_range(&self, max: u32) -> u32 {
    // I don't care about the at most part per billion bias here.
    (self.next_random() % max as u64) as u32
  }

  /// Generate a uniformly random pair of f32s in the range [0, 1).
  #[inline]
  pub fn generate_float_pair(&self) -> (f32, f32) {
    let bits64 = self.next_random();
    let bits_lo = bits64 as u32;
    let bits_hi = (bits64 >> 32) as u32;
    let one: u32 = 0b0_01111111_00000000000000000000000;
    let mask: u32 = 0b0_00000000_11111111111111111111111;
    (
      unsafe { std::mem::transmute::<u32, f32>(one | (mask & bits_lo)) } - 1.0,
      unsafe { std::mem::transmute::<u32, f32>(one | (mask & bits_hi)) } - 1.0,
    )
  }

  /// Generate a uniformly random f32 in the range [0, 1).
  #[inline]
  pub fn generate_float(&self) -> f32 {
    self.generate_float_pair().0
  }

  /// Generate a gamma distributed f32 with the given shape and scale = 1.
  #[inline]
  pub fn generate_gamma_variate(&self, shape: f32) -> f32 {
    let e = std::f32::consts::E;
    loop {
      let (x, y) = self.generate_float_pair();
      let (z, _) = self.generate_float_pair();
      let (u, v, w) = (1.0 - x, 1.0 - y, 1.0 - z);
      let (xi, eta) = match u * (1.0 + shape / e) <= 1.0 {
        true => {
          let xi = v.powf(1.0 / shape);
          let eta = w * xi.powf(shape - 1.0);
          (xi, eta)
        }
        false => {
          let xi = 1.0 - v.ln();
          let eta = w * (-xi).exp();
          (xi, eta)
        }
      };
      if eta > xi.powf(shape - 1.0) * (-xi).exp() {
        continue;
      }
      return xi;
    }
  }
}
