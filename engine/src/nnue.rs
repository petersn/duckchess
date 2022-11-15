/// We have three implementations:
/// 1. A SIMD implementation for x86_64.
/// 2. A SIMD implementation for aarch64.
/// 3. A SIMD implementation for wasm32.

use std::collections::hash_map::DefaultHasher;
use crate::{
  inference::Evaluation,
  nnue_data::{
    INTEGER_SCALE, PARAMS_BLACK_DUCK_BIAS, PARAMS_BLACK_DUCK_WEIGHT, PARAMS_BLACK_MAIN_BIAS,
    PARAMS_BLACK_MAIN_WEIGHT, PARAMS_MAIN_EMBED_WEIGHT, PARAMS_WHITE_DUCK_BIAS,
    PARAMS_WHITE_DUCK_WEIGHT, PARAMS_WHITE_MAIN_BIAS, PARAMS_WHITE_MAIN_WEIGHT,
  },
  rules::{GameOutcome, Move, Player, State},
};

// Switch based on architecture.
cfg_if::cfg_if! {
  if #[cfg(target_arch = "x86_64")] {
    use std::arch::x86_64::*;
    type VecI8 = __m128i;
    type VecI16 = __m128i;
    const VECTOR_LENGTH_I8: usize = 32;
    const VECTOR_LENGTH_I16: usize = 16;

    fn veci16_zeros() -> VecI16 {
      unsafe { _mm_setzero_si128() }
    }

    fn veci8_zeros() -> VecI8 {
      unsafe { _mm_setzero_si128() }
    }
  } else if #[cfg(target_arch = "aarch64")] {
    use std::arch::aarch64::*;
    type VecI8 = int8x16_t;
    type VecI16 = int16x8_t;
    const VECTOR_LENGTH_I8: usize = 16;
    const VECTOR_LENGTH_I16: usize = 8;

    fn veci16_zeros() -> VecI16 {
      unsafe { vdupq_n_s16(0) }
    }

    fn veci8_zeros() -> VecI8 {
      unsafe { vdupq_n_s8(0) }
    }
  } else if #[cfg(target_arch = "wasm32")] {
    use std::arch::wasm32::*;
    type VecI8 = v128;
    type VecI16 = v128;
    const VECTOR_LENGTH_I8: usize = 16;
    const VECTOR_LENGTH_I16: usize = 8;

    fn veci16_zeros() -> VecI16 {
      unsafe { wasm_i8x16_splat(0) }
    }

    fn veci8_zeros() -> VecI8 {
      unsafe { wasm_i8x16_splat(0) }
    }
  } else {
    compile_error!("Unsupported architecture.");
  }
}

// We cast weights from &[[i8 or i16; inner]; outer] to a &[[Vec?; inner / VECTOR_LENGTH_?]; outer].
macro_rules! get_weight2d {
  (i8, $weights:ident, $inner:literal, $outer:literal) => {
    unsafe {
      std::slice::from_raw_parts(
        $weights.as_ptr() as *const [VecI8; $inner / VECTOR_LENGTH_I8],
        $outer,
      )
    }
  };
  (i16, $weights:ident, $inner:literal, $outer:literal) => {
    unsafe {
      std::slice::from_raw_parts(
        $weights.as_ptr() as *const [VecI16; $inner / VECTOR_LENGTH_I16],
        $outer,
      )
    }
  };
}

// We cast biases from &[i8 or i16; inner] to a &[Vec?; inner / VECTOR_LENGTH_?].
macro_rules! get_bias1d {
  (i8, $biases:ident, $inner:literal) => {
    unsafe {
      std::slice::from_raw_parts(
        $biases.as_ptr() as *const VecI8,
        $inner / VECTOR_LENGTH_I8,
      )
    }
  };
  (i16, $biases:ident, $inner:literal) => {
    unsafe {
      std::slice::from_raw_parts(
        $biases.as_ptr() as *const VecI16,
        $inner / VECTOR_LENGTH_I16,
      )
    }
  };
}

const LINEAR_STATE_SIZE: usize = 64;
const LINEAR_STATE_VECTOR_COUNT: usize = LINEAR_STATE_SIZE / VECTOR_LENGTH_I16;

// 6 layers for our pieces, 6 for theirs, 1 for the duck.
pub const DUCK_LAYER: usize = 12;

pub type LayerIndex = u16;
pub const NO_LAYER: LayerIndex = LayerIndex::MAX;

#[derive(Clone, Copy, Debug)]
pub struct UndoCookie {
  pub sub_layers: [LayerIndex; 2],
  pub add_layers: [LayerIndex; 2],
}

impl UndoCookie {
  pub fn new() -> UndoCookie {
    UndoCookie {
      sub_layers: [NO_LAYER; 2],
      add_layers: [NO_LAYER; 2],
    }
  }
}

pub struct Nnue {
  pub linear_state: [VecI16; LINEAR_STATE_VECTOR_COUNT],
  pub outputs:      [i16; 64 + 1],
  pub value:        i16,
}

impl Nnue {
  pub fn new(state: &State) -> Self {
    let linear_state = get_bias1d!(i16, PARAMS_MAIN_EMBED_WEIGHT, LINEAR_STATE_SIZE);
    let mut this = Self {
      linear_state,
      outputs:      [0; 64 + 1],
      value:        0,
    };
    for turn in [Player::Black, Player::White] {
      for (piece_layer_number, piece_array) in [
        &state.pawns,
        &state.knights,
        &state.bishops,
        &state.rooks,
        &state.queens,
        &state.kings,
      ]
      .into_iter()
      .enumerate()
      {
        let layer_number = piece_layer_number + 6 * turn as usize;
        let mut bitboard = piece_array[turn as usize].0;
        // Get all of the pieces.
        while let Some(pos) = crate::rules::iter_bits(&mut bitboard) {
          this.add_layer((64 * layer_number + pos as usize) as u16);
        }
      }
    }
    this
  }

  pub fn get_debugging_hash(&self) -> u64 {
    use std::hash::{Hash, Hasher};
    // We cast our linear state to a u8 array.
    let linear_state_u8 = unsafe {
      std::slice::from_raw_parts(
        self.linear_state.as_ptr() as *const u8,
        self.linear_state.len() * std::mem::size_of::<VecI16>(),
      )
    };
    let mut hasher = DefaultHasher::new();
    linear_state_u8.hash(&mut hasher);
    hasher.finish()
  }

  pub fn add_layer(&mut self, layer: u16) {
    
    let weights = unsafe {
      &*(PARAMS_MAIN_EMBED_WEIGHT as *const [[i16; LINEAR_STATE_SIZE]; 832] as *const [
        [VecI16; LINEAR_STATE_VECTOR_COUNT];
        832
      ])
    };
    for (i, weight) in weights[layer as usize].iter().enumerate() {
      self.linear_state[i] = veci16_add(self.linear_state[i], *weight);
    }
  }

  pub fn sub_layer(&mut self, layer: u16) {
    // We cast PARAMS_MAIN_EMBED_WEIGHT to a &[[VecI16; LINEAR_STATE_VECTOR_COUNT]; 832].
    let weights = unsafe {
      &*(PARAMS_MAIN_EMBED_WEIGHT as *const [[i16; LINEAR_STATE_SIZE]; 832] as *const [
        [VecI16; LINEAR_STATE_VECTOR_COUNT];
        832
      ])
    };
    for (i, weight) in weights[layer as usize].iter().enumerate() {
      self.linear_state[i] = veci16_sub(self.linear_state[i], *weight);
    }
  }

  pub fn sub_add_layers(&mut self, from: u16, to: u16) {
    for (i, weight) in PARAMS_MAIN_EMBED_WEIGHT[from as usize].iter().enumerate() {
      self.linear_state[i] -= weight;
    }
    for (i, weight) in PARAMS_MAIN_EMBED_WEIGHT[to as usize].iter().enumerate() {
      self.linear_state[i] += weight;
    }
  }

  pub fn undo(&mut self, cookie: UndoCookie) {
    for sub in cookie.sub_layers {
      if sub != u16::MAX {
        // Add because we're undoing a past sub.
        self.add_layer(sub);
      }
    }
    for add in cookie.add_layers {
      if add != u16::MAX {
        // Sub because we're undoing a past add.
        self.sub_layer(add);
      }
    }
  }

  pub fn evaluate(&mut self, state: &State) {
    // First, check if the state is terminal.
    match state.get_outcome() {
      Some(GameOutcome::Draw) => {
        self.value = 0;
        return;
      }
      Some(GameOutcome::Win(winner)) => {
        self.value = match winner == state.turn {
          true => 1_000_000_000,
          false => -1_000_000_000,
        };
        return;
      }
      None => {}
    }
    let (mat, bias) = match (state.turn, state.is_duck_move) {
      (Player::White, false) => (PARAMS_WHITE_MAIN_WEIGHT, PARAMS_WHITE_MAIN_BIAS),
      (Player::White, true) => (PARAMS_WHITE_DUCK_WEIGHT, PARAMS_WHITE_DUCK_BIAS),
      (Player::Black, false) => (PARAMS_BLACK_MAIN_WEIGHT, PARAMS_BLACK_MAIN_BIAS),
      (Player::Black, true) => (PARAMS_BLACK_DUCK_WEIGHT, PARAMS_BLACK_DUCK_BIAS),
    };
    // Rescale and ReLU the linear state.
    let mut scratch = [0.0; LINEAR_STATE_SIZE];
    for i in 0..LINEAR_STATE_SIZE {
      scratch[i] = self.linear_state[i].max(0) as f32 / INTEGER_SCALE;
    }
    // Compute the final output.
    for i in 0..65 {
      let mut sum = bias[i];
      for j in 0..LINEAR_STATE_SIZE {
        sum += mat[j][i] * scratch[j];
      }
      self.outputs[i] = sum;
    }
    // Rescale the value output as an integer.
    self.value = (self.outputs[64].tanh() * 1000.0) as i32;
  }
}
