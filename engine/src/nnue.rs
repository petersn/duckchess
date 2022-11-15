/// We have three implementations:
/// 1. A SIMD implementation for x86_64.
/// 2. A SIMD implementation for aarch64.
/// 3. A SIMD implementation for wasm32.
use std::collections::hash_map::DefaultHasher;

use crate::inference::Evaluation;
#[rustfmt::skip]
use crate::{
  nnue_data::{
    PARAMS_MAIN_EMBED_WEIGHT, PARAMS_MAIN_EMBED_BIAS,
    PARAMS_WHITE_MAIN_0_WEIGHT, PARAMS_WHITE_MAIN_0_BIAS,
    PARAMS_WHITE_MAIN_1_WEIGHT, PARAMS_WHITE_MAIN_1_BIAS,
    PARAMS_WHITE_MAIN_2_WEIGHT, PARAMS_WHITE_MAIN_2_BIAS,
    PARAMS_BLACK_MAIN_0_WEIGHT, PARAMS_BLACK_MAIN_0_BIAS,
    PARAMS_BLACK_MAIN_1_WEIGHT, PARAMS_BLACK_MAIN_1_BIAS,
    PARAMS_BLACK_MAIN_2_WEIGHT, PARAMS_BLACK_MAIN_2_BIAS,
    PARAMS_WHITE_DUCK_0_WEIGHT, PARAMS_WHITE_DUCK_0_BIAS,
    PARAMS_WHITE_DUCK_1_WEIGHT, PARAMS_WHITE_DUCK_1_BIAS,
    PARAMS_WHITE_DUCK_2_WEIGHT, PARAMS_WHITE_DUCK_2_BIAS,
    PARAMS_BLACK_DUCK_0_WEIGHT, PARAMS_BLACK_DUCK_0_BIAS,
    PARAMS_BLACK_DUCK_1_WEIGHT, PARAMS_BLACK_DUCK_1_BIAS,
    PARAMS_BLACK_DUCK_2_WEIGHT, PARAMS_BLACK_DUCK_2_BIAS,
  },
  rules::{Player, State},
};

const VECTOR_LENGTH_I8: usize = 16;
const VECTOR_LENGTH_I16: usize = 8;

// Our SIMD interface:
//
//   fn veci16_zeros() -> VecI16;
//     Return the vector of eight zeros.
//
//   fn veci16_{add,sub}(a: VecI16, b: VecI16) -> VecI16;
//     Perform elementwise *non-saturating* addition/subtraction.
//     This function is only used for maintaining the accumulator, and thus
//     we use non-saturating arithmetic, to guarantee invertibility.
//
//   fn vec_matmul_sat_fma(accum: VecI16, a: VecI8, b: VecI8) -> VecI16;
//     This function is used for implementing fast matrix multiplication.
//     The only guarantee is that the saturating horizontal sum of the result
//     will increase by the saturating dot product of a and b.
//
//   fn veci16_horizontal_sat_sum(a: VecI16) -> i16;
//     Returns the saturated sum of all eight elements of the vector.
//
// We now switch on the architecture to define these functions.
#[rustfmt::skip]
cfg_if::cfg_if! {
  if #[cfg(target_arch = "x86_64")] {
    use std::arch::x86_64::*;
    type VecI8 = __m128i;
    type VecI16 = __m128i;

    #[inline(always)]
    fn veci16_zeros() -> VecI16 { unsafe { _mm_setzero_si128() } }

    #[inline(always)]
    fn veci16_add(a: VecI16, b: VecI16) -> VecI16 { unsafe { _mm_add_epi16(a, b) } }

    #[inline(always)]
    fn veci16_sub(a: VecI16, b: VecI16) -> VecI16 { unsafe { _mm_sub_epi16(a, b) } }

    #[inline(always)]
    fn vec_matmul_sat_fma(accum: VecI16, a: VecI8, b: VecI8) -> VecI16 {
      unsafe { _mm_adds_epi16(accum, _mm_maddubs_epi16(a, b)) }
    }

    #[inline(always)]
    fn veci16_horizontal_sat_sum(a: VecI16) -> i16 {
      unsafe {
        let zeros = _mm_setzero_si128();
        let a = _mm_hadds_epi16(a, zeros);
        let a = _mm_hadds_epi16(a, zeros);
        let a = _mm_hadds_epi16(a, zeros);
        _mm_cvtsi128_si32(a) as i16
      }
    }
  } else if #[cfg(target_arch = "aarch64")] {
    use std::arch::aarch64::*;
    type VecI8 = int8x16_t;
    type VecI16 = int16x8_t;

    #[inline(always)]
    fn veci16_zeros() -> VecI16 { unsafe { vdupq_n_s16(0) } }

    #[inline(always)]
    fn veci16_add(a: VecI16, b: VecI16) -> VecI16 { unsafe { vaddq_s16(a, b) } }

    #[inline(always)]
    fn veci16_sub(a: VecI16, b: VecI16) -> VecI16 { unsafe { vsubq_s16(a, b) } }

    #[inline(always)]
    fn vec_matmul_sat_fma(accum: VecI16, a: VecI8, b: VecI8) -> VecI16 {
      compile_error!("TODO: Implement vec_matmul_sat_fma for aarch64");
    }

    #[inline(always)]
    fn veci16_horizontal_sat_sum(a: VecI16) -> i16 {
      compile_error!("TODO: Implement veci16_horizontal_sat_sum for aarch64");
    }
  } else if #[cfg(target_arch = "wasm32")] {
    use std::arch::wasm32::*;
    type VecI8 = v128;
    type VecI16 = v128;

    #[inline(always)]
    fn veci16_zeros() -> VecI16 { unsafe { i16x8_splat(0) } }

    #[inline(always)]
    fn veci16_add(a: VecI16, b: VecI16) -> VecI16 { unsafe { i16x8_add(a, b) } }

    #[inline(always)]
    fn veci16_sub(a: VecI16, b: VecI16) -> VecI16 { unsafe { i16x8_sub(a, b) } }

    #[inline(always)]
    fn vec_matmul_sat_fma(accum: VecI16, a: VecI8, b: VecI8) -> VecI16 {
      unsafe {
        let a = i16x8_extmul_low_i8x16(a, b);
        let b = i16x8_extmul_high_i8x16(a, b);
        i16x8_add(accum, i16x8_add(a, b))
      }
    }

    #[inline(always)]
    fn veci16_horizontal_sat_sum(a: VecI16) -> i16 {
      unsafe {
        let zeros = i16x8_splat(0);
        let a = i16x8_add_sat(a, i16x8_shuffle::<4, 5, 6, 7, 0, 1, 2, 3>(a, zeros));
        let a = i16x8_add_sat(a, i16x8_shuffle::<2, 3, 4, 5, 6, 7, 0, 1>(a, zeros));
        let a = i16x8_add_sat(a, i16x8_shuffle::<1, 2, 3, 4, 5, 6, 7, 0>(a, zeros));
        i16x8_extract_lane::<0>(a)
      }
    }
  } else {
    compile_error!("Unsupported architecture.");
  }
}

// We cast weights from &[[i8; inner]; outer] to a &[[VecI8; inner / VECTOR_LENGTH_I8]; outer].
macro_rules! get_weights2d_i8 {
  ($weights:expr, $inner:expr, $outer:expr) => {{
    static_assertions::const_assert_eq!($inner % VECTOR_LENGTH_I8, 0);
    unsafe {
      &*($weights as *const [i8; $inner] as *const [[VecI8; $inner / VECTOR_LENGTH_I8]; $outer])
    }
  }};
}

// We cast biases from &[i16; inner] to a &[VecI16; inner / VECTOR_LENGTH_I16].
macro_rules! get_bias1d_i16 {
  ($biases:expr, $inner:expr) => {{
    static_assertions::const_assert_eq!($inner % VECTOR_LENGTH_I16, 0);
    unsafe { &*($biases as *const i16 as *const [VecI16; $inner / VECTOR_LENGTH_I16]) }
  }};
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
  //pub outputs:      [i16; 64 + 1],
}

impl Nnue {
  pub fn new(state: &State) -> Self {
    let linear_state = get_bias1d_i16!(PARAMS_MAIN_EMBED_BIAS, LINEAR_STATE_SIZE);
    let mut this = Self {
      linear_state: *linear_state,
      //outputs:      [0; 64 + 1],
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
    let weights = get_bias1d_i16!(&PARAMS_MAIN_EMBED_WEIGHT[layer as usize], LINEAR_STATE_SIZE);
    for (i, weight) in weights.iter().enumerate() {
      self.linear_state[i] = veci16_add(self.linear_state[i], *weight);
    }
  }

  pub fn sub_layer(&mut self, layer: u16) {
    let weights = get_bias1d_i16!(&PARAMS_MAIN_EMBED_WEIGHT[layer as usize], LINEAR_STATE_SIZE);
    for (i, weight) in weights.iter().enumerate() {
      self.linear_state[i] = veci16_sub(self.linear_state[i], *weight);
    }
  }

  pub fn sub_add_layers(&mut self, from: u16, to: u16) {
    let add_weights = get_bias1d_i16!(&PARAMS_MAIN_EMBED_WEIGHT[to as usize], LINEAR_STATE_SIZE);
    let sub_weights = get_bias1d_i16!(&PARAMS_MAIN_EMBED_WEIGHT[from as usize], LINEAR_STATE_SIZE);
    for (i, (add_weight, sub_weight)) in add_weights.iter().zip(sub_weights.iter()).enumerate() {
      self.linear_state[i] = veci16_add(self.linear_state[i], *add_weight);
      self.linear_state[i] = veci16_sub(self.linear_state[i], *sub_weight);
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

  pub fn evaluate(&mut self, state: &State) -> Evaluation {
    if let Some(terminal_eval) = Evaluation::from_terminal_state(state) {
      return terminal_eval;
    }

    let (mat0, bias0, mat1, bias1, mat2, bias2) = match (state.turn, state.is_duck_move) {
      (Player::White, false) => (
        PARAMS_WHITE_MAIN_0_WEIGHT,
        PARAMS_WHITE_MAIN_0_BIAS,
        PARAMS_WHITE_MAIN_1_WEIGHT,
        PARAMS_WHITE_MAIN_1_BIAS,
        PARAMS_WHITE_MAIN_2_WEIGHT,
        PARAMS_WHITE_MAIN_2_BIAS,
      ),
      (Player::White, true) => (
        PARAMS_WHITE_DUCK_0_WEIGHT,
        PARAMS_WHITE_DUCK_0_BIAS,
        PARAMS_WHITE_DUCK_1_WEIGHT,
        PARAMS_WHITE_DUCK_1_BIAS,
        PARAMS_WHITE_DUCK_2_WEIGHT,
        PARAMS_WHITE_DUCK_2_BIAS,
      ),
      (Player::Black, false) => (
        PARAMS_BLACK_MAIN_0_WEIGHT,
        PARAMS_BLACK_MAIN_0_BIAS,
        PARAMS_BLACK_MAIN_1_WEIGHT,
        PARAMS_BLACK_MAIN_1_BIAS,
        PARAMS_BLACK_MAIN_2_WEIGHT,
        PARAMS_BLACK_MAIN_2_BIAS,
      ),
      (Player::Black, true) => (
        PARAMS_BLACK_DUCK_0_WEIGHT,
        PARAMS_BLACK_DUCK_0_BIAS,
        PARAMS_BLACK_DUCK_1_WEIGHT,
        PARAMS_BLACK_DUCK_1_BIAS,
        PARAMS_BLACK_DUCK_2_WEIGHT,
        PARAMS_BLACK_DUCK_2_BIAS,
      ),
    };
    // The horizontal sum of all of the entries in accum[i] represents the output of layer 0.
    let mut accums0 = [veci16_zeros(); 16];
    let weights0 = get_weights2d_i8!(mat0, 256, 16);
    for i in 0..LINEAR_STATE_VECTOR_COUNT {
      for j in 0..16 {
        let block = weights0[j][i];
        accums0[j] = vec_matmul_sat_fma(accums0[j], self.linear_state[i], block);
      }
    }
    // We now horizontally sum the accums0, add the bias, and apply the activation function.
    let mut layer0: [i8; 16] = [0; 16];
    for i in 0..16 {
      let intermediate = veci16_horizontal_sat_sum(accums0[i]) + bias0[i];
      layer0[i] = (intermediate / 128).clamp(-128, 127) as i8;
    }
    let layer0_simd: VecI8 = unsafe { std::mem::transmute(layer0) };

    // Second layer
    let mut accums1 = [veci16_zeros(); 32];
    let weights1 = get_weights2d_i8!(mat1, 16, 32);
    for j in 0..32 {
      let block = weights1[j][0];
      accums1[j] = vec_matmul_sat_fma(accums1[j], layer0_simd, block);
    }
    let mut layer1: [i8; 32] = [0; 32];
    for i in 0..32 {
      let intermediate = veci16_horizontal_sat_sum(accums1[i]) + bias1[i];
      layer1[i] = (intermediate / 128).clamp(-128, 127) as i8;
    }
    let layer1_simd: [VecI8; 2] = unsafe { std::mem::transmute(layer1) };

    // Final layer
    let weights2 = get_weights2d_i8!(mat2, 32, 1);
    let mut final_accum = veci16_zeros();
    final_accum = vec_matmul_sat_fma(final_accum, layer1_simd[0], weights2[0][0]);
    final_accum = vec_matmul_sat_fma(final_accum, layer1_simd[1], weights2[0][1]);
    let final_accum = veci16_horizontal_sat_sum(final_accum) + bias2[0];

    let network_output = (final_accum as f32 / 1024.0).tanh();
    //self.value = (self.outputs[64].tanh() * 1000.0) as i32;
    Evaluation {
      expected_score: (network_output + 1.0) / 2.0,
      perspective_player: state.turn,
    }
  }
}
