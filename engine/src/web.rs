use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = console)]
  fn log(s: &str);
}

#[wasm_bindgen]
struct Engine {
  engine: crate::Engine,
}

#[wasm_bindgen]
impl Engine {
  pub fn get_state(&self) -> JsValue {
    serde_wasm_bindgen::to_value(&self.state).unwrap_or_else(|e| {
      log(&format!("Failed to serialize state: {}", e));
      JsValue::NULL
    })
  }

  pub fn set_state(&mut self, state: JsValue) {
    self.state = serde_wasm_bindgen::from_value(state).unwrap_or_else(|e| {
      log(&format!("Failed to deserialize state: {}", e));
      panic!("Failed to deserialize state: {}", e);
    });
  }

  pub fn get_moves(&self) -> JsValue {
    let mut moves = Vec::new();
    self.state.move_gen::<false>(&mut moves);
    serde_wasm_bindgen::to_value(&moves).unwrap_or_else(|e| {
      log(&format!("Failed to serialize moves: {}", e));
      JsValue::NULL
    })
  }
}