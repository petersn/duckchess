use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

use crate::rules::State;

pub struct InferenceEngine {}

impl InferenceEngine {
  pub async fn create() -> Self {
    Self {}
  }

  pub async fn predict(&self, state: &State, policy: &mut [f32; 64 * 64], value: &mut f32) {
    *value = 0.0;
    for i in 0..64 {
      for j in 0..64 {
        policy[i * 64 + j] = 0.0;
      }
    }
  }
}
