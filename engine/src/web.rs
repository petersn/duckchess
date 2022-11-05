use wasm_bindgen::prelude::*;

use crate::inference::{FEATURES_SIZE, POLICY_LEN};
use crate::inference_web::MAX_BATCH_SIZE;
use crate::mcts::PendingPath;
use crate::{inference, inference_web, mcts, rules::Move, search};

const MAX_STEPS_BEFORE_INFERENCE: usize = 40 * MAX_BATCH_SIZE;

#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = console)]
  pub fn log(s: &str);
}

#[wasm_bindgen]
pub struct Engine {
  inference_engine: &'static inference_web::TensorFlowJsEngine<(usize, PendingPath)>,
  engine:           search::Engine,
  mcts:             mcts::Mcts<'static, inference_web::TensorFlowJsEngine<(usize, PendingPath)>>,
  //input_array: Box<[f32; MAX_BATCH_SIZE * FEATURES_SIZE]>,
  //policy_array: Box<[f32; MAX_BATCH_SIZE * POLICY_LEN]>,
  //value_array: Box<[f32; MAX_BATCH_SIZE]>,
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
      Ok(()) => {}
      Err(msg) => {
        log(&format!("Failed to apply move: {}", msg));
        return false;
      }
    }
    self.mcts.apply_move(m);
    use crate::inference::InferenceEngine;
    self.inference_engine.clear();
    true
  }

  pub fn step_until_batch(&mut self, features_array: &mut [f32]) -> usize {
    for i in 0..MAX_STEPS_BEFORE_INFERENCE {
      self.mcts.step();
      if self.inference_engine.has_batch() {
        break;
      }
    }
    log(&format!(
      "Inference engine has batch: {}",
      self.inference_engine.has_batch()
    ));
    self.inference_engine.fetch_work(features_array)
  }

  pub fn give_answers(&mut self, policy_array: &[f32], value_array: &[f32]) {
    use crate::inference::InferenceEngine;
    self.inference_engine.give_answers(policy_array, value_array);
    self.mcts.get_state().sanity_check().unwrap();
    self.inference_engine.predict(|inference_results| {
      for (i, (cookie, pending_path)) in inference_results.cookies.into_iter().enumerate() {
        self.mcts.process_path(pending_path.clone(), inference_results.get(i));
      }
    });
    //self.mcts.predict_now();
    //self.inference_engine.give_answers(&self.policy_array, &self.value_array);
  }

  pub fn get_principal_variation(&self) -> JsValue {
    let pv = self.mcts.get_pv();
    serde_wasm_bindgen::to_value(&pv).unwrap_or_else(|e| {
      log(&format!("Failed to serialize pv: {}", e));
      JsValue::NULL
    })
  }

  //pub fn step(&mut self) {
  //  log(&format!("MCTS step done"));
  //  self.mcts.step().await;
  //
  //}

  //pub fn run(&mut self, depth: u16) -> JsValue {
  //  let p = self.engine.run(depth);
  //  serde_wasm_bindgen::to_value(&p).unwrap_or_else(|e| {
  //    log(&format!("Failed to serialize score: {}", e));
  //    JsValue::NULL
  //  })
  //}
}

#[wasm_bindgen]
pub fn new_engine(seed: u64) -> Engine {
  log(&format!("Creating new engine with seed {}", seed));
  let tfjs_inference_engine = Box::leak(Box::new(inference_web::TensorFlowJsEngine::new()));
  log(&format!("Created (1) inference engine"));
  Engine {
    inference_engine: tfjs_inference_engine,
    engine:           search::Engine::new(seed),
    mcts:             mcts::Mcts::new(0, seed, tfjs_inference_engine),
    //input_array: Box::new([0.0; MAX_BATCH_SIZE * FEATURES_SIZE]),
    //policy_array: Box::new([0.0; MAX_BATCH_SIZE * POLICY_LEN]),
    //value_array: Box::new([0.0; MAX_BATCH_SIZE]),
  }
}

#[wasm_bindgen]
pub fn max_batch_size() -> usize {
  MAX_BATCH_SIZE
}

#[wasm_bindgen]
pub fn channel_count() -> usize {
  inference::CHANNEL_COUNT
}

#[wasm_bindgen]
pub fn parse_pgn4(pgn4_str: &str) -> JsValue {
  let pgn4 = crate::pgn4_parse::parse(pgn4_str);
  serde_wasm_bindgen::to_value(&pgn4).unwrap_or_else(|e| {
    log(&format!("Failed to parse pgn4: {}", e));
    JsValue::NULL
  })
}

#[wasm_bindgen]
pub fn uci_to_move(uci: &str) -> JsValue {
  let m = crate::rules::Move::from_uci(uci);
  serde_wasm_bindgen::to_value(&m).unwrap_or_else(|e| {
    log(&format!("Failed to parse uci: {}", e));
    JsValue::NULL
  })
}

#[wasm_bindgen]
pub fn move_to_uci(m: JsValue) -> String {
  let m: Move = serde_wasm_bindgen::from_value(m).unwrap_or_else(|e| {
    log(&format!("Failed to deserialize move: {}", e));
    panic!("Failed to deserialize move: {}", e);
  });
  m.to_uci()
}

fn perft<const NNUE: bool>() {
  use engine::rules::{State, Move};
  use engine::nnue::UndoCookie;

  struct StackEntry {
    depth: usize,
    state: State,
    m: Move,
  }

  let start_time = std::time::Instant::now();
  let mut nodes = 0;
  let mut moves = vec![];
  let mut stack = vec![
    StackEntry {
      depth: 0,
      state: State::starting_state(),
      m: Move::from_uci("e2e4").unwrap(),
    }
  ];
  let mut nnue = engine::nnue::Nnue::new(&State::starting_state());
  while let Some(mut entry) = stack.pop() {
    nodes += 1;
    let undo_cookie = entry.state.apply_move::<NNUE>(entry.m, Some(&mut nnue)).unwrap();
    nnue.evaluate(&entry.state);
    if entry.depth == 4 {
      
    } else {
      entry.state.move_gen::<false>(&mut moves);
      for (i, m) in moves.iter().enumerate() {
        //let mut new_state = entry.state.clone();
        //let undo_cookie = new_state.apply_move::<false>(*m, None).unwrap();
        stack.push(StackEntry {
          depth: entry.depth + 1,
          state: entry.state.clone(),
          m: *m,
        });
      }
      moves.clear();
    }
    if NNUE {
      nnue.undo(undo_cookie);
    }
  }

  web::log(&format!("{} nodes", nodes));
  web::log(&format!("{} seconds", start_time.elapsed().as_secs_f64()));
  web::log(&format!("{} Mnodes/second", 1e-6 * nodes as f64 / start_time.elapsed().as_secs_f64()));
}
