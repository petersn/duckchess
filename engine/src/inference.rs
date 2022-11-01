use crate::rules::{State, Player, MOVE_HISTORY_LEN, GameOutcome};

#[derive(Debug, Clone, Copy)]
pub struct Evaluation {
  /// A value [0, 1] giving the expected score, loss = 0, draw = 0.5, win = 1.
  pub expected_score: f32,
  /// Whose perspective the score is from.
  pub perspective_player: Player,
}

impl Evaluation {
  pub const EVEN_EVAL: Self = Self {
    expected_score: 0.5,
    perspective_player: Player::White,
  };

  /// If a state is terminal, return the evaluation of the state.
  pub fn from_terminal_state(state: &State) -> Option<Self> {
    state.get_outcome().map(|outcome| match outcome {
      GameOutcome::Draw => Self::EVEN_EVAL,
      GameOutcome::Win(player) => Self {
        expected_score: 1.0,
        perspective_player: player,
      },
    })
  }

  pub fn expected_score_for_player(&self, player: Player) -> f32 {
    match player == self.perspective_player {
      true => self.expected_score,
      false => 1.0 - self.expected_score,
    }
  }
}

// We have:
//   Six channels for our pieces: pawns, knights, bishops, rooks, queens, kings
//   Six channels for their pieces.
//   One channel for ducks.
//   One channel that's all ones if it's white to move, zeros otherwise.
//   One channel for if it's the duck subturn.
//   Two channels for our castling rights: king side, queen side
//   Two channels for their castling rights.
//   One channel for an en passant square.
//   Pairs of (from, to) channels for the history of moves.
//   One channel of all ones.
pub const CHANNEL_COUNT: usize = 6 + 6 + 1 + 1 + 1 + 2 + 2 + 1 + (2 * MOVE_HISTORY_LEN) + 1;
// We have one policy plane for each possible from square.
pub const POLICY_PLANE_COUNT: usize = 64;
pub const POLICY_LEN: usize = POLICY_PLANE_COUNT * 64;
pub const FEATURES_SIZE: usize = CHANNEL_COUNT * 64;

/// Write out `state` into `array` in C, H, W order.
pub fn featurize_state<T: From<u8>>(state: &State, array: &mut [T; FEATURES_SIZE]) {
  let mut layer_index = 0;
  let mut emit_bitboard = |bitboard: u64| {
    let bitboard = match state.turn {
      Player::White => bitboard,
      Player::Black => bitboard.swap_bytes(),
    };
    for i in 0..64 {
      array[64 * layer_index + i] = (((bitboard >> i) & 1) as u8).into();
    }
    layer_index += 1;
  };
  let bool_board = |b: bool| if b { u64::MAX } else { 0 };
  // Encode the pieces.
  let player_order = match state.turn {
    Player::White => [Player::White, Player::Black],
    Player::Black => [Player::Black, Player::White],
  };
  for player in player_order {
    for piece_array in [
      &state.pawns,
      &state.knights,
      &state.bishops,
      &state.rooks,
      &state.queens,
      &state.kings,
    ] {
      emit_bitboard(piece_array[player as usize].0);
    }
  }
  // Encode the ducks.
  emit_bitboard(state.ducks.0);
  // Encode whose turn it is, just in case the model wants to care.
  emit_bitboard(bool_board(state.turn == Player::White));
  // Encode if it's the duck subturn.
  emit_bitboard(bool_board(state.is_duck_move));
  // Encode castling rights.
  for player in player_order {
    emit_bitboard(bool_board(state.castling_rights[player as usize].king_side));
    emit_bitboard(bool_board(state.castling_rights[player as usize].queen_side));
  }
  // Encode en passant square.
  emit_bitboard(state.en_passant.0);
  // Encode last four moves.
  for m in state.move_history.iter() {
    if let Some(m) = m {
      // TODO: Verify that these really are the right squares.
      emit_bitboard(1 << m.from);
      emit_bitboard(1 << m.to);
    } else {
      emit_bitboard(0);
      emit_bitboard(0);
    }
  }
  // Encode all ones.
  emit_bitboard(u64::MAX);
  assert_eq!(layer_index, CHANNEL_COUNT);
}

#[derive(Clone)]
pub struct ModelOutputs {
  // policy[64 * from + to] is a probability 0 to 1.
  pub policy: [f32; POLICY_LEN],
  // value is a valuation for the current player from -1 to +1.
  pub value:  Evaluation,
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
      value:  Evaluation {
        expected_score: (self.values[index] + 1.0) / 2.0,
        perspective_player: todo!(),
      },
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
