use std::collections::hash_map::DefaultHasher;
use std::arch::wasm32::*;

use crate::{
  inference::Evaluation,
  rules::{Move, State, Player, GameOutcome}, nnue_data::{
    INTEGER_SCALE,
    //PARAMS_MAIN_EMBED_WEIGHT,
    //PARAMS_WHITE_MAIN_WEIGHT,
    //PARAMS_WHITE_MAIN_BIAS,
    //PARAMS_WHITE_DUCK_WEIGHT,
    //PARAMS_WHITE_DUCK_BIAS,
    //PARAMS_BLACK_MAIN_WEIGHT,
    //PARAMS_BLACK_MAIN_BIAS,
    //PARAMS_BLACK_DUCK_WEIGHT,
    //PARAMS_BLACK_DUCK_BIAS,
  },
};

const LINEAR_STATE_SIZE: usize = 32;

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
  pub linear_state: [v128; LINEAR_STATE_SIZE],
  pub outputs:      [f32; 64 + 1],
  pub value:        i32,
  pub data:         Vec<v128>,
}

impl Nnue {
  pub fn new(state: &State) -> Self {
    let mut this = Self {
      linear_state: [i16x8_splat(0); LINEAR_STATE_SIZE],
      outputs:      [0.0; 64 + 1],
      value:        0,
      data:         vec![i16x8_splat(3); 10_000],
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
    for i in 0..LINEAR_STATE_SIZE {
      self.linear_state[i] = i16x8_add(self.linear_state[i], self.data[i + layer as usize]);
    }
    //for (i, weight) in PARAMS_MAIN_EMBED_WEIGHT[layer as usize].iter().enumerate() {
    //  self.linear_state[i] += weight;
    //}
  }

  pub fn sub_layer(&mut self, layer: u16) {
    for i in 0..LINEAR_STATE_SIZE {
      self.linear_state[i] = i16x8_sub(self.linear_state[i], self.data[i + layer as usize]);
    }
    //for (i, weight) in PARAMS_MAIN_EMBED_WEIGHT[layer as usize].iter().enumerate() {
    //  self.linear_state[i] -= weight;
    //}
  }

  pub fn sub_add_layers(&mut self, from: u16, to: u16) {
    for i in 0..LINEAR_STATE_SIZE {
      self.linear_state[i] = i16x8_sub(self.linear_state[i], self.data[i + from as usize]);
      self.linear_state[i] = i16x8_add(self.linear_state[i], self.data[i + to as usize]);
    }
    //for (i, weight) in PARAMS_MAIN_EMBED_WEIGHT[from as usize].iter().enumerate() {
    //  self.linear_state[i] -= weight;
    //}
    //for (i, weight) in PARAMS_MAIN_EMBED_WEIGHT[to as usize].iter().enumerate() {
    //  self.linear_state[i] += weight;
    //}
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
      },
      Some(GameOutcome::Win(winner)) => {
        self.value = match winner == state.turn {
          true => 1_000_000_000,
          false => -1_000_000_000,
        };
        return;
      },
      None => {},
    }
    // Relu, then affine.
    let mut relu = [i8x16_splat(0); LINEAR_STATE_SIZE / 2];
    for i in 0..LINEAR_STATE_SIZE / 2 {
      relu[i] = i8x16_narrow_i16x8(
        self.linear_state[2 * i + 0],
        self.linear_state[2 * i + 1],
      );
    }
    // Perform an affine layer.
    let mut accum0 = i16x8_splat(0);
    let mut accum1 = i16x8_splat(0);
    let mut accum2 = i16x8_splat(0);
    let mut accum3 = i16x8_splat(0);
    // Access the relu output as a slice of i8s.
    let relu = unsafe { std::slice::from_raw_parts(relu.as_ptr() as *const i8, relu.len() * 16) };
    for i in 0..relu.len() {
      let broadcast = i8x16_splat(relu[i]);
      accum0 = i16x8_add(accum0, i16x8_extmul_low_i8x16(broadcast, self.data[i]));
      accum1 = i16x8_add(accum1, i16x8_extmul_high_i8x16(broadcast, self.data[i]));
      accum2 = i16x8_add(accum2, i16x8_extmul_low_i8x16(broadcast, self.data[i + 1]));
      accum3 = i16x8_add(accum3, i16x8_extmul_high_i8x16(broadcast, self.data[i + 1]));
    }
    // Another relu.
    let accum0 = i8x16_narrow_i16x8(accum0, accum1);
    let accum1 = i8x16_narrow_i16x8(accum2, accum3);
    // Final inner product of this vector with the output weights.
    let prod_lo0 = i16x8_extmul_low_i8x16(accum0, self.data[relu.len()]);
    let prod_hi0 = i16x8_extmul_high_i8x16(accum0, self.data[relu.len()]);
    let prod_lo1 = i16x8_extmul_low_i8x16(accum1, self.data[relu.len() + 1]);
    let prod_hi1 = i16x8_extmul_high_i8x16(accum1, self.data[relu.len() + 1]);
    let mut accum = i16x8_add(i16x8_add(prod_lo0, prod_hi0), i16x8_add(prod_lo1, prod_hi1));
    // Sum up pairwise.
    accum = i32x4_extadd_pairwise_i16x8(accum);
    // Sum with shuffles.
    let zero = i32x4_splat(0);
    accum = i32x4_add(accum, i32x4_shuffle::<1, 0, 3, 2>(accum, zero));
    accum = i32x4_add(accum, i32x4_shuffle::<2, 3, 0, 1>(accum, zero));
    // Extract the value using i32x4_shuffle.
    self.value = i32x4_extract_lane::<0>(accum);

    //// Sum up the components using pairwise summation.
    //accum = i16x8_add(accum, i16x8_shuffle(accum, [1, 0, 3, 2, 5, 4, 7, 6]));
    //accum = i16x8_add(accum, i16x8_shuffle(accum, [2, 3, 0, 1, 6, 7, 4, 5]));
    //accum = i16x8_add(accum, i16x8_shuffle(accum, [4, 5, 6, 7, 0, 1, 2, 3]));
    //// Extract the value.
    //self.value = accum.extract(0);
    //let (mat, bias) = match (state.turn, state.is_duck_move) {
    //  (Player::White, false) => (PARAMS_WHITE_MAIN_WEIGHT, PARAMS_WHITE_MAIN_BIAS),
    //  (Player::White, true) => (PARAMS_WHITE_DUCK_WEIGHT, PARAMS_WHITE_DUCK_BIAS),
    //  (Player::Black, false) => (PARAMS_BLACK_MAIN_WEIGHT, PARAMS_BLACK_MAIN_BIAS),
    //  (Player::Black, true) => (PARAMS_BLACK_DUCK_WEIGHT, PARAMS_BLACK_DUCK_BIAS),
    //};
    // Rescale and ReLU the linear state.
    //let mut scratch = [0.0; LINEAR_STATE_SIZE];
    //for i in 0..LINEAR_STATE_SIZE {
    //  scratch[i] = self.linear_state[i].max(0) as f32 / INTEGER_SCALE;
    //}
    //// Compute the final output.
    //for i in 0..65 {
    //  let mut sum = bias[i];
    //  for j in 0..LINEAR_STATE_SIZE {
    //    sum += mat[j][i] * scratch[j];
    //  }
    //  self.outputs[i] = sum;
    //}
    // Rescale the value output as an integer.
    self.value = (self.outputs[64].tanh() * 1000.0) as i32;
  }
}
