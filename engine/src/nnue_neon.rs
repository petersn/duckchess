//use std::arch::wasm32::*;
use std::arch::aarch64::*;
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

const LINEAR_STATE_SIZE: usize = 64;

// 6 layers for our pieces, 6 for theirs, 1 for the duck.
pub const DUCK_LAYER: usize = 12;

#[derive(Debug)]
pub struct UndoCookie {
  pub sub_layers: [u16; 2],
  pub add_layers: [u16; 2],
}

impl UndoCookie {
  pub fn new() -> UndoCookie {
    UndoCookie {
      sub_layers: [u16::MAX; 2],
      add_layers: [u16::MAX; 2],
    }
  }
}

pub struct Nnue {
  pub linear_state: [int32x4_t; LINEAR_STATE_SIZE / 4],
  pub outputs:      [f32; 64 + 1],
  pub value:        i32,
}

impl Nnue {
  pub fn new(state: &State) -> Self {
    // Generate a int32x4_t of all zeros.
    let zero = unsafe { vdupq_n_s32(0) };
    let mut this = Self {
      linear_state: [zero; LINEAR_STATE_SIZE / 4],
      outputs:      [0.0; 64 + 1],
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
    //use std::hash::{Hash, Hasher};
    //let mut hasher = DefaultHasher::new();
    //self.linear_state.hash(&mut hasher);
    //hasher.finish()
    0
  }

  pub fn add_layer(&mut self, layer: u16) {
    let row: &[int32x4_t] = unsafe {
      std::slice::from_raw_parts(
        PARAMS_MAIN_EMBED_WEIGHT.as_ptr().add(layer as usize) as *const int32x4_t,
        LINEAR_STATE_SIZE / 4,
      )
    };
    for i in 0..LINEAR_STATE_SIZE / 4 {
      self.linear_state[i] = unsafe { vaddq_s32(self.linear_state[i], row[i]) };
    }
    //for (i, weight) in PARAMS_MAIN_EMBED_WEIGHT[layer as usize].iter().enumerate() {
    //  self.linear_state[i] = i16x8_add(self.linear_state[i], i16x8_splat(*weight as i16));
    //}
  }

  pub fn sub_layer(&mut self, layer: u16) {
    let row: &[int32x4_t] = unsafe {
      std::slice::from_raw_parts(
        PARAMS_MAIN_EMBED_WEIGHT.as_ptr().add(layer as usize) as *const int32x4_t,
        LINEAR_STATE_SIZE / 4,
      )
    };
    for i in 0..LINEAR_STATE_SIZE / 4 {
      self.linear_state[i] = unsafe { vsubq_s32(self.linear_state[i], row[i]) };
    }
  }

  pub fn sub_add_layers(&mut self, from: u16, to: u16) {
    let (sub_row, add_row) = unsafe {
      (
        std::slice::from_raw_parts(
          PARAMS_MAIN_EMBED_WEIGHT.as_ptr().add(from as usize) as *const int32x4_t,
          LINEAR_STATE_SIZE / 4,
        ),
        std::slice::from_raw_parts(
          PARAMS_MAIN_EMBED_WEIGHT.as_ptr().add(to as usize) as *const int32x4_t,
          LINEAR_STATE_SIZE / 4,
        ),
      )
    };
    for i in 0..LINEAR_STATE_SIZE / 4 {
      self.linear_state[i] = unsafe { vsubq_s32(vaddq_s32(self.linear_state[i], add_row[i]), sub_row[i]) };
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
    // ===== OLD WASM CODE =====
    // // Rescale and ReLU the linear state.
    // let rescale = f32x4_splat(1.0 / INTEGER_SCALE);
    // let mut scratch = [f32x4_splat(0.0); LINEAR_STATE_SIZE / 4];
    // for i in 0..LINEAR_STATE_SIZE / 4 {
    //   scratch[i] = f32x4_mul(
    //     f32x4_abs(f32x4_convert_i32x4(self.linear_state[i])),
    //     rescale,
    //   );
    // }
    // // Compute the final output.
    // for i in 0..65 {
    //   let mut sum = f32x4_splat(0.0);
    //   let mat_row = unsafe {
    //     std::slice::from_raw_parts(mat.as_ptr().add(i) as *const float32x4_t, LINEAR_STATE_SIZE / 4)
    //   };
    //   for j in 0..LINEAR_STATE_SIZE / 4 {
    //     sum = f32x4_add(sum, f32x4_mul(scratch[j], mat_row[j]));
    //   }
    //   self.outputs[i] = bias[i]
    //     + f32x4_extract_lane::<0>(sum)
    //     + f32x4_extract_lane::<1>(sum)
    //     + f32x4_extract_lane::<2>(sum)
    //     + f32x4_extract_lane::<3>(sum);
    // }
    // // Rescale the value output as an integer.
    // self.value = (self.outputs[64].tanh() * 1000.0) as i32;
    // ===== NEW NEON CODE =====
    // Rescale and ReLU the linear state.
    unsafe {
      let rescale = vdupq_n_f32(1.0 / INTEGER_SCALE);
      let mut scratch = [vdupq_n_f32(0.0); LINEAR_STATE_SIZE / 4];
      for i in 0..LINEAR_STATE_SIZE / 4 {
        scratch[i] = vabsq_f32(vcvtq_f32_s32(self.linear_state[i]));
        scratch[i] = vmulq_f32(scratch[i], rescale);
      }
      // Compute the final output.
      for i in 0..65 {
        let mut sum = vdupq_n_f32(0.0);
        let mat_row = unsafe {
          std::slice::from_raw_parts(mat.as_ptr().add(i) as *const float32x4_t, LINEAR_STATE_SIZE / 4)
        };
        for j in 0..LINEAR_STATE_SIZE / 4 {
          sum = vaddq_f32(sum, vmulq_f32(scratch[j], mat_row[j]));
        }
        self.outputs[i] = bias[i]
          + vgetq_lane_f32(sum, 0)
          + vgetq_lane_f32(sum, 1)
          + vgetq_lane_f32(sum, 2)
          + vgetq_lane_f32(sum, 3);
      }
      // Rescale the value output as an integer.
      self.value = (self.outputs[64].tanh() * 1000.0) as i32;
    }
  }
}
