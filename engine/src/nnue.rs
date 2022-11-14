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
  pub linear_state: [i32; LINEAR_STATE_SIZE],
  pub outputs:      [f32; 64 + 1],
  pub value:        i32,
}

impl Nnue {
  pub fn new(state: &State) -> Self {
    let mut this = Self {
      linear_state: [0; LINEAR_STATE_SIZE],
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
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    self.linear_state.hash(&mut hasher);
    hasher.finish()
  }

  pub fn add_layer(&mut self, layer: u16) {
    for (i, weight) in PARAMS_MAIN_EMBED_WEIGHT[layer as usize].iter().enumerate() {
      self.linear_state[i] += weight;
    }
  }

  pub fn sub_layer(&mut self, layer: u16) {
    for (i, weight) in PARAMS_MAIN_EMBED_WEIGHT[layer as usize].iter().enumerate() {
      self.linear_state[i] -= weight;
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
