use engine::{inference::InferenceEngine, inference_desktop::TensorFlowEngine};

fn main() {
  let inference_engine = TensorFlowEngine::new("/tmp/keras");
  let state = engine::rules::State::starting_state();

  let mut sm = slotmap::SlotMap::with_key();
  let cookie = sm.insert(0);
  inference_engine.add_work(&state, cookie);
  inference_engine.predict(|outputs| {
    println!("cookies: {:?}", outputs.cookies);
  });
}
