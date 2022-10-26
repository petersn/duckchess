use engine::inference::InferenceEngine;

#[tokio::main]
async fn main() {
  let inference_engine = InferenceEngine::create("/tmp/keras").await;
  let state = engine::rules::State::starting_state();
  let model_outputs = inference_engine.predict(&state).await;
  println!("Value: {}", model_outputs.value);
}
