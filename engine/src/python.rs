use pyo3::prelude::*;

#[pyclass]
struct Engine {
  engine: crate::Engine,
}

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
const CHANNEL_COUNT: usize = 6 + 6 + 1 + 1 + 1 + 4 + 1 + 1 + 1;

#[pymethods]
impl Engine {
  #[new]
  fn new(seed: u64) -> Self {
    Self {
      engine: crate::new_engine(seed),
    }
  }

  fn get_state(&self) -> String {
    let state = crate::get_state_internal(&self.engine);
    serde_json::to_string(&state).unwrap()
  }

  fn get_state_into_array(&self, array_len: usize, array: usize) {
    assert_eq!(array_len, 64 * CHANNEL_COUNT);
    let array: &mut [[u8; 64]; CHANNEL_COUNT] = unsafe { &mut *(array as *mut _) };
    let state = crate::get_state_internal(&self.engine);
    let mut layer_index = 0;
    let mut emit_bitboard = |bitboard: u64| {
      for i in 0..64 {
        array[layer_index][i] = ((bitboard >> i) & 1) as u8;
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

  fn get_moves(&self) -> Vec<String> {
    let moves = crate::get_moves_internal(&self.engine);
    moves.into_iter().map(|m| serde_json::to_string(&m).unwrap()).collect()
  }

  fn run(&mut self, depth: u16) -> (i32, String) {
    let p = crate::run_internal(&mut self.engine, depth);
    let serialized = serde_json::to_string(&p.1).unwrap();
    (p.0, serialized)
  }

  fn apply_move(&mut self, m: &str) {
    let m: crate::Move = serde_json::from_str(m).unwrap();
    crate::apply_move_internal(&mut self.engine, m);
  }

  fn get_outcome(&self) -> Option<String> {
    let outcome = crate::get_outcome_internal(&self.engine);
    outcome.map(|s| s.to_owned())
  }
}

#[pyfunction]
fn encode_move(m: &str, array_len: usize, array: usize) {
  assert_eq!(array_len, 64 * 2);
  let array: &mut [[u8; 64]; 2] = unsafe { &mut *(array as *mut _) };
  let m: crate::Move = serde_json::from_str(m).unwrap();
  let mut layer_index = 0;
  let mut emit_bitboard = |bitboard: u64| {
    for i in 0..64 {
      array[layer_index][i] = ((bitboard >> i) & 1) as u8;
    }
    layer_index += 1;
  };
  emit_bitboard(if m.from == 64 { 0 } else { 1 << m.from });
  emit_bitboard(1 << m.to);
  assert_eq!(layer_index, 2);
}

#[pyfunction]
fn channel_count() -> usize {
  CHANNEL_COUNT
}

#[pyfunction]
fn version() -> i64 {
  1
}

#[pymodule]
fn engine(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
  m.add_function(wrap_pyfunction!(encode_move, m)?)?;
  m.add_function(wrap_pyfunction!(channel_count, m)?)?;
  m.add_function(wrap_pyfunction!(version, m)?)?;
  m.add_class::<Engine>()?;
  Ok(())
}
