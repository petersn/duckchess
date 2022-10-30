use async_trait::async_trait;
use wasm_bindgen::prelude::*;
use crate::{mcts, rules::State, web};

pub struct TensorFlowJsEngine {
  array: [f32; 361],
}

impl TensorFlowJsEngine {
  pub fn new() -> Self {
    Self {
      array: [0.0; 361],
    }
  }

  pub async fn fetch_work(&self, array: js_sys::Float32Array) {
    // We wait for us to have a full array of data.
    
    array.copy_from(&self.array);
  }
}

#[async_trait]
impl mcts::InferenceEngine for TensorFlowJsEngine {
  async fn predict(&self, state: &State) -> mcts::ModelOutputs {
    //web::log(&format!("Predicting for state: {:?}", state));
    mcts::ModelOutputs {
      policy: [0.0; mcts::POLICY_LEN],
      value: 0.0,
    }
  }
}
