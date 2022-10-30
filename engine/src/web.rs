use wasm_bindgen::prelude::*;

use crate::{inference_web, mcts, rules::Move, search};

#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = console)]
  pub fn log(s: &str);
}

#[wasm_bindgen]
pub struct Engine {
  engine: search::Engine,
  mcts:   mcts::Mcts,
}

#[wasm_bindgen]
impl Engine {
  pub fn get_state(&self) -> JsValue {
    serde_wasm_bindgen::to_value(self.engine.get_state()).unwrap_or_else(|e| {
      log(&format!("Failed to serialize state: {}", e));
      JsValue::NULL
    })
  }

  pub fn set_state(&mut self, state: JsValue) {
    let new_state = serde_wasm_bindgen::from_value(state).unwrap_or_else(|e| {
      log(&format!("Failed to deserialize state: {}", e));
      panic!("Failed to deserialize state: {}", e);
    });
    self.engine.set_state(new_state);
  }

  pub fn get_moves(&self) -> JsValue {
    let mut moves = Vec::new();
    self.engine.get_state().move_gen::<false>(&mut moves);
    serde_wasm_bindgen::to_value(&moves).unwrap_or_else(|e| {
      log(&format!("Failed to serialize moves: {}", e));
      JsValue::NULL
    })
  }

  pub fn apply_move(&mut self, m: JsValue) -> bool {
    let m: Move = serde_wasm_bindgen::from_value(m).unwrap_or_else(|e| {
      log(&format!("Failed to deserialize move: {}", e));
      panic!("Failed to deserialize move: {}", e);
    });
    match self.engine.get_state_mut().apply_move(m) {
      Ok(()) => true,
      Err(msg) => {
        log(&format!("Failed to apply move: {}", msg));
        false
      }
    }
  }

  //pub async fn step_until_inference_batch(&mut self, array: js_sys::Float32Array) {
  //  self.mcts.step_until_inference_batch().await;
  //}

  //pub async fn step(&mut self) {
  //  self.mcts.step().await;
  //  log(&format!("MCTS step done"));
  //}

  pub fn run(&mut self, depth: u16) -> JsValue {
    let p = self.engine.run(depth);
    serde_wasm_bindgen::to_value(&p).unwrap_or_else(|e| {
      log(&format!("Failed to serialize score: {}", e));
      JsValue::NULL
    })
  }
}

#[wasm_bindgen]
pub fn new_engine(seed: u64) -> Engine {
  let tfjs_inference_engine = Box::leak(Box::new(inference_web::TensorFlowJsEngine::new()));
  log(&format!("Created inference engine"));
  Engine {
    engine: search::Engine::new(seed),
    mcts:   mcts::Mcts::new(seed),
  }
}

#[wasm_bindgen]
pub fn modify_array(x: &mut [f32]) {
  for i in 0..x.len() {
    x[i] = i as f32;
  }
}
