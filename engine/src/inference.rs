use crate::rules::State;

// We have:
//   Six channels for white pieces: pawns, knights, bishops, rooks, queens, kings
//   Six channels for black pieces.
//   One channel for ducks.
//   One channel for whose turn it is.
//   One channel for if it's the duck subturn.
//   Four channels for castling rights.
//   One channel for an en passant square.
//   One channel for the last move.
//   One channel of all ones.
pub const CHANNEL_COUNT: usize = 6 + 6 + 1 + 1 + 1 + 4 + 1 + 1 + 1;

pub const POLICY_LEN: usize = 64 * 64;

/// Write out `state` into `array` in C, H, W order.
pub fn featurize_state<T: From<u8>>(state: &State, array: &mut [T; 64 * CHANNEL_COUNT]) {
  let mut layer_index = 0;
  let mut emit_bitboard = |bitboard: u64| {
    for i in 0..64 {
      array[64 * layer_index + i] = (((bitboard >> i) & 1) as u8).into();
    }
    layer_index += 1;
  };
  let bool_board = |b: bool| if b { u64::MAX } else { 0 };
  // Encode the pieces.
  for player in [1, 0] {
    for piece_array in [
      &state.pawns,
      &state.knights,
      &state.bishops,
      &state.rooks,
      &state.queens,
      &state.kings,
    ] {
      emit_bitboard(piece_array[player].0);
    }
  }
  // Encode the ducks.
  emit_bitboard(state.ducks.0);
  // Encode whose turn it is.
  emit_bitboard(bool_board(state.white_turn));
  // Encode if it's the duck subturn.
  emit_bitboard(bool_board(state.is_duck_move));
  // Encode castling rights.
  for player in [1, 0] {
    emit_bitboard(bool_board(state.castling_rights[player].king_side));
    emit_bitboard(bool_board(state.castling_rights[player].queen_side));
  }
  // Encode en passant square.
  emit_bitboard(state.en_passant.0);
  // Encode last move.
  emit_bitboard(state.highlight.0);
  // Encode all ones.
  emit_bitboard(u64::MAX);
  assert_eq!(layer_index, CHANNEL_COUNT);
}

#[derive(Clone)]
pub struct ModelOutputs {
  // policy[64 * from + to] is a probability 0 to 1.
  pub policy: [f32; POLICY_LEN],
  // value is a valuation for the current player from -1 to +1.
  pub value:  f32,
}

impl ModelOutputs {
  pub fn renormalize(&mut self, moves: &[crate::rules::Move]) {
    let mut temp = [0.0; POLICY_LEN];
    let mut sum = 0.0;
    for m in moves {
      let idx = m.to_index() as usize;
      let val = self.policy[idx];
      temp[idx] = val;
      sum += val;
    }
    let rescale = 1.0 / (1e-16 + sum);
    for i in 0..POLICY_LEN {
      self.policy[i] = temp[i] * rescale;
    }
  }
}

pub enum Fullness {
  NotFullYet,
  Full,
}

pub trait InferenceEngine {
  fn add_work(&mut self, state: &crate::rules::State) -> Fullness;
  fn predict(&self, outputs: &mut [&mut ModelOutputs]);
}
