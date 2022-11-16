use pyo3::prelude::*;

use crate::{inference, rules::{Move, Player}, search};

#[pyclass]
struct Engine {
  engine: search::Engine,
}

#[pymethods]
impl Engine {
  #[new]
  fn new(seed: u64) -> Self {
    Self {
      engine: search::Engine::new(seed, 1024),
    }
  }

  fn get_state(&self) -> String {
    serde_json::to_string(self.engine.get_state()).unwrap()
  }

  fn set_state(&mut self, state: String) {
    let new_state = serde_json::from_str(&state).unwrap();
    self.engine.set_state(new_state);
  }

  fn get_state_into_array(&self, array_len: usize, array: usize) {
    assert_eq!(array_len, inference::CHANNEL_COUNT * 8 * 8);
    let array: &mut [u8; inference::CHANNEL_COUNT * 8 * 8] = unsafe { &mut *(array as *mut _) };
    inference::featurize_state(self.engine.get_state(), array);
  }

  fn get_moves(&self) -> Vec<String> {
    let moves = self.engine.get_moves();
    moves.into_iter().map(|m| serde_json::to_string(&m).unwrap()).collect()
  }

  fn run(&mut self, depth: u16) -> (i32, String) {
    let (score, best_move) = self.engine.run(depth);
    let serialized = serde_json::to_string(&best_move).unwrap();
    (score, serialized)
  }

  fn apply_move(&mut self, m: &str) -> Option<&'static str> {
    let m: Move = serde_json::from_str(m).unwrap();
    self.engine.apply_move(m).err()
  }

  fn sanity_check(&self) -> Option<&'static str> {
    self.engine.get_state().sanity_check().err()
  }

  fn get_outcome(&self) -> Option<&'static str> {
    self.engine.get_outcome().map(|o| o.to_str())
  }

  fn get_nnue_feature_indices(&self) -> Option<Vec<usize>> {
    let state = self.engine.get_state();

    // Check if the state is terminal, and if so, refuse to featurize.
    if state.get_outcome().is_some() {
      return None;
    }

    // Figure out where each player's king is.
    let black_king_pos = crate::rules::get_square(state.kings[Player::Black as usize].0) as usize;
    let white_king_pos = crate::rules::get_square(state.kings[Player::White as usize].0) as usize;

    #[rustfmt::skip]
    let mut feature_indices = vec![
      128 * 11 * 64 +  0 + white_king_pos,
      128 * 11 * 64 + 64 + black_king_pos,
    ];

    // For each of 128 possible black or white king positions we have:
    //   * five white non-king pieces * 64 squares
    //   * five black non-king pieces * 64 squares
    //   * the duck * 64 squares
    // Finally, we have 128 additional biases.

    let king_pos_stride: usize = 11 * 64;
    let white_king_offset: usize = king_pos_stride * white_king_pos;
    let black_king_offset: usize = 64 * king_pos_stride + king_pos_stride * black_king_pos;

    for (turn, king_offset) in [(Player::Black, black_king_offset), (Player::White, white_king_offset)] {
      for (piece_layer_number, piece_array) in [
        &state.pawns,
        &state.knights,
        &state.bishops,
        &state.rooks,
        &state.queens,
      ]
      .into_iter()
      .enumerate()
      {
        let layer_number = piece_layer_number + 5 * turn as usize;
        let mut bitboard = piece_array[turn as usize].0;
        // Get all of the pieces.
        while let Some(pos) = crate::rules::iter_bits(&mut bitboard) {
          feature_indices.push(king_offset + layer_number * 64 + pos as usize);
        }
      }
      // Include the duck
      let mut bitboard = state.ducks.0;
      while let Some(pos) = crate::rules::iter_bits(&mut bitboard) {
        feature_indices.push(king_offset + 10 * 64 + pos as usize);
      }
    }

    Some(feature_indices)
  }
}

#[pyfunction]
fn index_to_move(index: u16) -> String {
  serde_json::to_string(&Move::from_index(index)).unwrap()
}

#[pyfunction]
fn move_to_index(m: &str) -> u16 {
  let m: Move = serde_json::from_str(m).unwrap();
  m.to_index()
}

#[pyfunction]
fn channel_count() -> usize {
  inference::CHANNEL_COUNT
}

#[pyfunction]
fn version() -> i64 {
  2
}

#[pyfunction]
fn is_duck_chess() -> bool {
  crate::rules::IS_DUCK_CHESS
}

#[pymodule]
fn engine(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
  m.add_function(wrap_pyfunction!(index_to_move, m)?)?;
  m.add_function(wrap_pyfunction!(move_to_index, m)?)?;
  m.add_function(wrap_pyfunction!(channel_count, m)?)?;
  m.add_function(wrap_pyfunction!(version, m)?)?;
  m.add_function(wrap_pyfunction!(is_duck_chess, m)?)?;
  m.add_class::<Engine>()?;
  Ok(())
}
