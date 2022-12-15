use pyo3::prelude::*;

use crate::{
  inference, nnue,
  rules::{Move, Player},
  search,
};

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

  fn have_quiescence_moves(&self) -> bool {
    // Figure out if we have any legal quiescence moves from this position.
    let mut quiescence_moves = Vec::new();
    self.engine.get_state().move_gen::<true>(&mut quiescence_moves);
    !quiescence_moves.is_empty()
  }

  fn run(&mut self, depth: u16) -> (i32, String) {
    let (score, best_move) = self.engine.run(depth, false);
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

  fn get_nnue_feature_indices(&self) -> Option<(Vec<nnue::LayerIndex>, usize)> {
    nnue::get_state_layers_and_net_index(self.engine.get_state())
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
