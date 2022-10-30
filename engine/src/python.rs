use pyo3::prelude::*;

use crate::{inference, rules::Move, search};

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
    let array: &mut [u8; 64 * inference::CHANNEL_COUNT] =
      unsafe { &mut *(array as *mut _) };
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

  fn get_outcome(&self) -> Option<&'static str> {
    self.engine.get_outcome().to_option_str()
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
  1
}

#[pymodule]
fn engine(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
  m.add_function(wrap_pyfunction!(index_to_move, m)?)?;
  m.add_function(wrap_pyfunction!(move_to_index, m)?)?;
  m.add_function(wrap_pyfunction!(channel_count, m)?)?;
  m.add_function(wrap_pyfunction!(version, m)?)?;
  m.add_class::<Engine>()?;
  Ok(())
}
