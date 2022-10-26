use engine::inference::InferenceEngine;

#[tokio::main]
async fn main() {
  let inference_engine = InferenceEngine::create().await;
  let state = engine::rules::State::starting_state();
  let mut policy = [0.0; 64 * 64];
  let mut value = 0.0;
  inference_engine.predict(&state, &mut policy, &mut value).await;
}
