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

  pub fn apply_move(&mut self, m: JsValue) -> bool {
    let m: Move = serde_wasm_bindgen::from_value(m).unwrap_or_else(|e| {
      log(&format!("Failed to deserialize move: {}", e));
      panic!("Failed to deserialize move: {}", e);
    });
    self.state.apply_move(&m)
  }

  pub fn run(&mut self, depth: u16) -> JsValue {
    self.nodes_searched = 0;
    let start_state = self.state.clone();
    // Apply iterative deepening.
    let mut p = (-1, (None, None));
    for d in 1..=depth {
      p = self.pvs::<false>(d, &start_state, VERY_NEGATIVE_EVAL, VERY_POSITIVE_EVAL);
      log(&format!(
        "Depth {}: {} (nodes={})",
        d, p.0, self.nodes_searched
      ));
    }
    serde_wasm_bindgen::to_value(&p).unwrap_or_else(|e| {
      log(&format!("Failed to serialize score: {}", e));
      JsValue::NULL
    })
  }
}