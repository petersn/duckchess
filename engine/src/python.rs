use pyo3::prelude::*;

use crate::{rules::Move, search, inference};

#[pyclass]
struct Engine {
  engine: search::Engine,
}

#[pymethods]
impl Engine {
  #[new]
  fn new(seed: u64) -> Self {
    Self {
      engine: search::Engine::new(seed),
    }
  }

  fn get_state(&self) -> String {
    serde_json::to_string(self.engine.get_state()).unwrap()
  }

  fn get_state_into_array(&self, array_len: usize, array: usize) {
    assert_eq!(array_len, 64 * inference::CHANNEL_COUNT);
    let array: &mut [u8; 64 * inference::CHANNEL_COUNT] = unsafe { &mut *(array as *mut _) };
    let state = self.engine.get_state();
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

  fn apply_move(&mut self, m: &str) {
    let m: Move = serde_json::from_str(m).unwrap();
    self.engine.apply_move(m);
  }

  fn get_outcome(&self) -> Option<&'static str> {
    self.engine.get_outcome().to_option_str()
  }
}

#[pyfunction]
fn encode_move(m: &str, array_len: usize, array: usize) {
  assert_eq!(array_len, 64 * 2);
  let array: &mut [[u8; 64]; 2] = unsafe { &mut *(array as *mut _) };
  let m: Move = serde_json::from_str(m).unwrap();
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
  inference::CHANNEL_COUNT
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
