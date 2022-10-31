use wasm_bindgen::prelude::*;

use crate::{inference, mcts, rules::State, web};
use crate::inference::{PendingIndex, InferenceResults};

pub struct TensorFlowJsEngine {
  array: [f32; 361],
}

impl TensorFlowJsEngine {
  pub fn new() -> Self {
    Self { array: [0.0; 361] }
  }

  pub async fn fetch_work(&self, array: js_sys::Float32Array) {
    // We wait for us to have a full array of data.

    array.copy_from(&self.array);
  }
}

impl inference::InferenceEngine for TensorFlowJsEngine {
  const DESIRED_BATCH_SIZE: usize = 32;

  fn add_work(&self, state: &crate::rules::State, cookie: PendingIndex) -> usize {
    todo!()
  }

  fn predict(&self, use_outputs: impl FnOnce(InferenceResults)) -> usize {
    todo!()
  }

  fn clear(&self) {
    todo!()
  }
}
