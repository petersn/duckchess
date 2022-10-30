use wasm_bindgen::prelude::*;

use crate::{inference, mcts, rules::State, web};

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
  fn add_work(&mut self, state: &crate::rules::State) -> inference::Fullness {
    //web::state_to_array(state, &mut self.array);
    inference::Fullness::Full
  }

  fn predict(&self, mut outputs: &mut [&mut inference::ModelOutputs]) {
    for output in outputs {
      output.value = 0.0;
      for i in 0..inference::POLICY_LEN {
        output.policy[i] = 0.0;
      }
    }
  }
}
