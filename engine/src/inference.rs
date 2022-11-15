use std::collections::VecDeque;

use crate::rules::{GameOutcome, Player, State, MOVE_HISTORY_LEN};

#[derive(Debug, Clone, Copy)]
pub struct Evaluation {
  /// A value [0, 1] giving the expected score, loss = 0, draw = 0.5, win = 1.
  pub expected_score:     f32,
  /// Whose perspective the score is from.
  pub perspective_player: Player,
}

impl Evaluation {
  pub const EVEN_EVAL: Self = Self {
    expected_score:     0.5,
    perspective_player: Player::White,
  };

  /// If a state is terminal, return the evaluation of the state.
  pub fn from_terminal_state(state: &State) -> Option<Self> {
    state.get_outcome().map(|outcome| match outcome {
      GameOutcome::Draw => Self::EVEN_EVAL,
      GameOutcome::Win(player) => Self {
        expected_score:     1.0,
        perspective_player: player,
      },
    })
  }

  pub fn expected_score_for_player(&self, player: Player) -> f32 {
    assert!(0.0 <= self.expected_score && self.expected_score <= 1.0);
    match player == self.perspective_player {
      true => self.expected_score,
      false => 1.0 - self.expected_score,
    }
  }
}

// Implement Display for Evaluation.
impl std::fmt::Display for Evaluation {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(
      f,
      "{}{:.3}{}\x1b[0m",
      match self.perspective_player {
        Player::White => "\x1b[91m",
        Player::Black => "\x1b[92m",
      },
      self.expected_score,
      match self.perspective_player {
        Player::White => "W",
        Player::Black => "B",
      }
    )
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
    // This swap_bytes is incredibly important, and is where vertically mirroring the board is implemented.
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
    emit_bitboard(bool_board(
      state.castling_rights[player as usize].queen_side,
    ));
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

pub type QuantizedProbability = u16;

fn quantize_probability(probability: f32) -> QuantizedProbability {
  assert!(0.0 <= probability && probability <= 1.0);
  (probability * (QuantizedProbability::MAX as f32)) as QuantizedProbability
}

fn dequantize_probability(probability: QuantizedProbability) -> f32 {
  (probability as f32) / (QuantizedProbability::MAX as f32)
}

#[derive(Clone)]
pub struct ModelOutputs {
  // policy[64 * from + to] is a probability 0 to 1.
  pub quantized_policy: Box<[QuantizedProbability; POLICY_LEN]>,
  // value is a valuation for the current player from -1 to +1.
  pub value:            Evaluation,
}

#[derive(Clone)]
pub struct FullPrecisionModelOutputs {
  pub policy: Box<[f32; POLICY_LEN]>,
  pub value:  Evaluation,
}

impl ModelOutputs {
  pub fn get_policy(&self, index: usize) -> f32 {
    dequantize_probability(self.quantized_policy[index])
  }

  pub fn set_policy(&mut self, index: usize, value: f32) {
    self.quantized_policy[index] = quantize_probability(value);
  }

  pub fn quantize_from(
    mut full_prec: FullPrecisionModelOutputs,
    moves: &[crate::rules::Move],
  ) -> ModelOutputs {
    let mut temp = Box::new([0.0; POLICY_LEN]);
    let mut sum: f32 = 0.0;
    // Copy over just the moves.
    for m in moves {
      let idx = m.to_index() as usize;
      let val = full_prec.policy[idx];
      temp[idx] = val;
      sum += val;
    }
    // Renormalize to sum to 1.
    let rescale = 1.0 / (1e-16 + sum);
    for i in 0..POLICY_LEN {
      full_prec.policy[i] = temp[i] * rescale;
    }
    // Check that the output policy is normalized.
    let sum = full_prec.policy.iter().sum::<f32>();
    debug_assert!(sum <= 1.001);
    let deficiency = (1.0 - sum).max(0.0);
    for m in moves {
      let idx = m.to_index() as usize;
      full_prec.policy[idx] += deficiency / moves.len() as f32;
    }
    let sum = full_prec.policy.iter().sum::<f32>();
    if !moves.is_empty() && (sum - 1.0).abs() > 1e-5 {
      println!("Renormalization failed: sum is {}", sum);
      for m in moves {
        let idx = m.to_index() as usize;
        println!("  {:?} -> {}", m, full_prec.policy[idx]);
      }
      panic!();
    }
    debug_assert!(moves.is_empty() || (sum - 1.0).abs() < 1e-5);
    // Finally, quantize and emit.
    let mut quantized_policy = Box::new([0; POLICY_LEN]);
    for i in 0..POLICY_LEN {
      quantized_policy[i] = quantize_probability(full_prec.policy[i]);
    }
    ModelOutputs {
      quantized_policy,
      value: full_prec.value,
    }

    //if (sum - 1.0).abs() > 1e-2 {
    //  panic!("\x1b[91m>>>\x1b0m Renormalized policy sum is {}, not 1.0", sum);
    //}
    //for m in moves {
    //  let idx = m.to_index() as usize;
    //  self.policy[idx] = temp[idx] * rescale;
    //}
  }
}

pub struct InputBlock<Cookie> {
  pub cookies: Vec<Cookie>,
  pub players: Vec<Player>,
  pub data:    Vec<f32>,
}

pub fn add_to_input_blocks<Cookie>(
  batch_size: usize,
  input_blocks: &mut VecDeque<InputBlock<Cookie>>,
  state: &State,
  cookie: Cookie,
) -> bool {
  // Create a new input block if we have none, or the last one is full.
  let do_create_new_block = match input_blocks.back() {
    Some(input_block) => input_block.cookies.len() >= batch_size,
    None => true,
  };
  if do_create_new_block {
    input_blocks.push_back(InputBlock {
      cookies: vec![],
      players: vec![],
      data:    vec![0.0; batch_size * FEATURES_SIZE],
    })
  }
  // Add an entry to the last input block.
  let block = input_blocks.back_mut().unwrap();
  let range = block.cookies.len() * FEATURES_SIZE..(block.cookies.len() + 1) * FEATURES_SIZE;
  //let r = ;
  //let rr = r.try_into().unwrap();
  featurize_state(state, (&mut block.data[range]).try_into().unwrap());
  block.cookies.push(cookie);
  block.players.push(state.turn);
  // Return if there's at least one full block
  input_blocks.front().map(|b| b.cookies.len() >= batch_size).unwrap_or(false)
}

fn make_perspective_policy(player: Player, policy: &[f32; POLICY_LEN]) -> Box<[f32; POLICY_LEN]> {
  //println!("Making perspective policy for player {:?}", player);
  // Check normalization.
  let sum: f32 = policy.iter().sum();
  debug_assert!((sum - 1.0).abs() < 1e-3);
  let mut result = Box::new([0.0; POLICY_LEN]);
  for from in 0..64 {
    for to in 0..64 {
      let remap_from = match player {
        Player::White => from,
        Player::Black => from ^ 56,
      };
      let remap_to = match player {
        Player::White => to,
        Player::Black => to ^ 56,
      };
      result[64 * remap_from + remap_to] = policy[64 * from + to];
    }
  }
  // Check that the result is also normalized.
  let sum: f32 = result.iter().sum();
  debug_assert!((sum - 1.0).abs() < 1e-3);
  result
}

#[derive(Debug)]
pub struct InferenceResults<'a, Cookie> {
  pub length:   usize,
  pub cookies:  &'a [Cookie],
  pub players:  &'a [Player],
  pub policies: &'a [&'a [f32; POLICY_LEN]],
  pub values:   &'a [f32],
}

impl<'a, Cookie> InferenceResults<'a, Cookie> {
  pub fn new(
    cookies: &'a [Cookie],
    players: &'a [Player],
    policies: &'a [&'a [f32; POLICY_LEN]],
    values: &'a [f32],
  ) -> InferenceResults<'a, Cookie> {
    assert_eq!(cookies.len(), policies.len());
    assert_eq!(cookies.len(), players.len());
    assert_eq!(cookies.len(), values.len());
    InferenceResults {
      length: cookies.len(),
      cookies,
      players,
      policies,
      values,
    }
  }

  pub fn get(&self, index: usize) -> FullPrecisionModelOutputs {
    //println!("Value: {}", self.values[index]);
    debug_assert!(-1.0 <= self.values[index] && self.values[index] <= 1.0);
    FullPrecisionModelOutputs {
      policy: make_perspective_policy(self.players[index], self.policies[index]),
      value:  Evaluation {
        expected_score:     (self.values[index] + 1.0) / 2.0,
        perspective_player: self.players[index],
      },
    }
  }
}

pub trait InferenceEngine<Cookie> {
  const DESIRED_BATCH_SIZE: usize;

  /// Returns if we have a full batch
  fn add_work(&self, state: &crate::rules::State, cookie: Cookie) -> bool;
  /// Returns the number of entries processed.
  fn predict(&self, use_outputs: impl FnOnce(InferenceResults<Cookie>)) -> usize;
  fn clear(&self);
}
