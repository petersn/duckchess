use wasm_bindgen::prelude::*;

use crate::{rules::Move, search};

#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = console)]
  fn log(s: &str);
}

#[wasm_bindgen]
struct Engine {
  engine: search::Engine,
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
    self.engine.get_state_mut().apply_move(m)
  }

  pub fn run(&mut self, depth: u16) -> JsValue {
    let p = self.engine.run(depth);
    serde_wasm_bindgen::to_value(&p).unwrap_or_else(|e| {
      log(&format!("Failed to serialize score: {}", e));
      JsValue::NULL
    })
  }
}
