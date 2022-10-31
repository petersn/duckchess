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

pub const FEATURES_SIZE: usize = 64 * CHANNEL_COUNT;

/// Write out `state` into `array` in C, H, W order.
pub fn featurize_state<T: From<u8>>(state: &State, array: &mut [T; FEATURES_SIZE]) {
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

slotmap::new_key_type! {
  pub struct PendingIndex;
}

pub struct InputBlock<const BATCH_SIZE: usize>
where
  [f32; BATCH_SIZE * FEATURES_SIZE]: Sized,
{
  pub cookies: Vec<PendingIndex>,
  pub data:    Box<[f32; BATCH_SIZE * FEATURES_SIZE]>,
}

pub fn add_to_input_blocks<const BATCH_SIZE: usize>(
  input_blocks: &mut Vec<InputBlock<BATCH_SIZE>>,
  state: &State,
  cookie: PendingIndex,
) -> usize
where
  [f32; BATCH_SIZE * FEATURES_SIZE]: Sized,
{
  // Create a new input block if we have none, or the last one is full.
  let do_create_new_block = match input_blocks.last() {
    Some(input_block) => input_block.cookies.len() >= BATCH_SIZE,
    None => true,
  };
  if do_create_new_block {
    input_blocks.push(InputBlock {
      cookies: vec![],
      data:    Box::new([0.0; BATCH_SIZE * FEATURES_SIZE]),
    })
  }
  // Add an entry to the last input block.
  let block = input_blocks.last_mut().unwrap();
  let range = block.cookies.len() * FEATURES_SIZE..(block.cookies.len() + 1) * FEATURES_SIZE;
  //let r = ;
  //let rr = r.try_into().unwrap();
  featurize_state(state, (&mut block.data[range]).try_into().unwrap());
  block.cookies.push(cookie);
  // Return the fullness of the input block.
  block.cookies.len()
}

pub struct InferenceResults<'a> {
  pub length:   usize,
  pub cookies:  &'a [PendingIndex],
  pub policies: &'a [&'a [f32; POLICY_LEN]],
  pub values:   &'a [f32],
}

impl<'a> InferenceResults<'a> {
  pub fn new(
    cookies: &'a [PendingIndex],
    policies: &'a [&'a [f32; POLICY_LEN]],
    values: &'a [f32],
  ) -> InferenceResults<'a> {
    assert_eq!(cookies.len(), policies.len());
    assert_eq!(policies.len(), values.len());
    InferenceResults {
      length: cookies.len(),
      cookies,
      policies,
      values,
    }
  }

  pub fn get(&self, index: usize) -> ModelOutputs {
    ModelOutputs {
      policy: *self.policies[index],
      value:  self.values[index],
    }
  }
}

pub trait InferenceEngine {
  const DESIRED_BATCH_SIZE: usize;

  /// Returns the nummber of entries queued up after adding `state`.
  fn add_work(&self, state: &crate::rules::State, cookie: PendingIndex) -> usize;
  /// Returns the number of entries processed.
  fn predict(&self, use_outputs: impl FnOnce(InferenceResults)) -> usize;
  fn clear(&self);
}
