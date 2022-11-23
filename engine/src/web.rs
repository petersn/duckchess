use std::arch::wasm32::*;

use js_sys::{Array, Atomics, Int32Array, SharedArrayBuffer, Uint8Array};
use wasm_bindgen::prelude::*;

use crate::inference::{FEATURES_SIZE, POLICY_LEN};
use crate::inference_web::MAX_BATCH_SIZE;
use crate::mcts::{PendingPath, SearchParams};
use crate::{inference, inference_web, mcts, rules::Move, search};

const MAX_STEPS_BEFORE_INFERENCE: usize = 40 * MAX_BATCH_SIZE;
const WEB_TRANSPOSITION_TABLE_SIZE: usize = 1 << 20;

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
    let mut optional_moves = Vec::new();
    let current_state = self.engine.get_state();
    current_state.move_gen::<false>(&mut moves);
    // If this is a duck move, also get the moves for the next state.
    if current_state.is_duck_move && current_state.get_outcome().is_none() {
      let mut next_state = current_state.clone();
      next_state.turn = next_state.turn.other_player();
      next_state.is_duck_move = false;
      // Remove the duck, because they could put it anywhere to not interfere.
      next_state.ducks.0 = 0;
      next_state.move_gen::<false>(&mut optional_moves);
    }

    serde_wasm_bindgen::to_value(&[moves, optional_moves]).unwrap_or_else(|e| {
      log(&format!("Failed to serialize moves: {}", e));
      JsValue::NULL
    })
  }

  pub fn apply_move(&mut self, m: JsValue, is_hidden: bool) -> bool {
    let m: Move = serde_wasm_bindgen::from_value(m).unwrap_or_else(|e| {
      log(&format!("Failed to deserialize move: {}", e));
      panic!("Failed to deserialize move: {}", e);
    });
    // Print all args.
    log(&format!("Applying move: {:?} -- {:?}", m, is_hidden));

    // First check if we need to apply a duck move first.
    if is_hidden {
      let current_state = self.engine.get_state();
      // If this is a duck move, also get the moves for the next state.
      if !current_state.is_duck_move || current_state.get_outcome().is_some() {
        log("BUG BUG BUG: Invalid hidden move");
        return false;
      }
      let mut next_state = current_state.clone();
      next_state.turn = next_state.turn.other_player();
      next_state.is_duck_move = false;
      // Find all duck positions that interfere with this move.
      let mut duck_allowed = !(next_state.get_move_duck_block_mask(m) | next_state.get_occupied());
      // Find the first allowed location for the duck.
      let duck_from = crate::rules::get_square(current_state.ducks.0);
      // We now figure out a nice spot to put the duck.
      let mut duck_to = 0;
      while let Some(square) = crate::rules::iter_bits(&mut duck_allowed) {
        // Pick whichever move has a lower x-coordinate, and a y-coordinate closer to the middle.
        let better_y = ((square as i32 / 8) - 3).abs() < ((duck_to as i32 / 8) - 3).abs();
        if square % 8 < duck_to % 8 || better_y {
          duck_to = square;
        }
      }
      // Handle the initial placement of the duck here too.
      let move_duck = Move { from: if current_state.ducks.0 == 0 { duck_to } else { duck_from }, to: duck_to };
      // Move the duck out of the way first.
      log(&format!("Applying duck movement move: {:?} (duck_allowed: {:016x})", move_duck, duck_allowed));
      match self.engine.get_state_mut().apply_move::<false>(move_duck, None) {
        Ok(_) => {}
        Err(msg) => {
          log(&format!("Failed to apply move: {}", msg));
          return false;
        }
      }
      self.mcts.apply_move(move_duck);
    }

    match self.engine.get_state_mut().apply_move::<false>(m, None) {
      Ok(_) => {}
      Err(msg) => {
        log(&format!("Failed to apply move: {}", msg));
        return false;
      }
    }
    log(&format!("Applied move: {:?}", m));
    self.mcts.apply_move(m);
    use crate::inference::InferenceEngine;
    //self.inference_engine.clear();
    true
  }

  pub fn step_until_batch(&mut self, features_array: &mut [f32]) -> usize {
    for i in 0..MAX_STEPS_BEFORE_INFERENCE {
      self.mcts.step();
      if self.inference_engine.has_batch() {
        break;
      }
    }
    //log(&format!("In flight after stepping: {}", self.mcts.in_flight_count));
    //log(&format!(
    //  "Inference engine has batch: {}",
    //  self.inference_engine.has_batch()
    //));
    self.inference_engine.fetch_work(features_array)
  }

  pub fn give_answers(&mut self, policy_array: &[f32], value_array: &[f32]) {
    //log(&format!("In flight before give answers: {}", self.mcts.in_flight_count));
    use crate::inference::InferenceEngine;
    self.inference_engine.give_answers(policy_array, value_array);
    self.mcts.get_state().sanity_check().unwrap();
    self.inference_engine.predict(|inference_results| {
      for (i, (cookie, pending_path)) in inference_results.cookies.into_iter().enumerate() {
        self.mcts.process_path(pending_path.clone(), inference_results.get(i));
      }
    });
    //log(&format!("In flight after give answers: {}", self.mcts.in_flight_count));
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
    engine:           search::Engine::new(seed, WEB_TRANSPOSITION_TABLE_SIZE),
    mcts:             mcts::Mcts::new(0, seed, tfjs_inference_engine, SearchParams::default()),
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

fn run_perft<const NNUE: bool, const EVAL: bool>() -> usize {
  use crate::rules::{Move, State};

  struct StackEntry {
    depth: usize,
    state: State,
    m:     Move,
  }

  //let start_time = std::time::Instant::now();
  let mut nodes = 0;
  let mut moves = vec![];
  let mut stack = vec![StackEntry {
    depth: 0,
    state: State::starting_state(),
    m:     Move::from_uci("e2e4").unwrap(),
  }];
  let mut nnue = crate::nnue::Nnue::new(&State::starting_state(), crate::nnue::BUNDLED_NETWORK);
  while let Some(mut entry) = stack.pop() {
    nodes += 1;
    let adjustment = entry.state.apply_move::<NNUE>(entry.m, Some(&mut nnue)).unwrap();
    if EVAL {
      crate::eval::basic_eval(&entry.state);
    }
    if NNUE {
      nnue.evaluate(&entry.state);
    }
    if entry.depth == 4 {
      // ...
    } else {
      entry.state.move_gen::<false>(&mut moves);
      for (i, m) in moves.iter().enumerate() {
        //let mut new_state = entry.state.clone();
        //let undo_cookie = new_state.apply_move::<false>(*m, None).unwrap();
        stack.push(StackEntry {
          depth: entry.depth + 1,
          state: entry.state.clone(),
          m:     *m,
        });
      }
      moves.clear();
    }
    if NNUE {
      // FIXME: This is WRONG! I need to pass in the parent state!
      nnue.apply_adjustment::<true>(&entry.state, &adjustment);
    }
  }
  nodes

  //log(&format!("nnue={} Perft took {}ms, {} nodes", NNUE, start_time.elapsed().as_millis(), nodes));
  //log(&format!("{} nodes", nodes));
  //log(&format!("{} seconds", start_time.elapsed().as_secs_f64()));
  //log(&format!("{} Mnodes/second", 1e-6 * nodes as f64 / start_time.elapsed().as_secs_f64()));
}

#[wasm_bindgen]
pub fn perft() -> usize {
  run_perft::<false, false>()
}

#[wasm_bindgen]
pub fn perft_eval() -> usize {
  run_perft::<false, true>()
}

#[wasm_bindgen]
pub fn perft_nnue() -> usize {
  run_perft::<true, false>()
}

#[wasm_bindgen]
pub fn test_simd() {
  let mut v = [i16x8(0, 1, 2, 3, 4, 5, 6, 7); 8];
  for i in 0..8 {
    v[i] = i16x8_add(v[i], i16x8_splat(i as i16));
  }
  log(&format!("{:?}", v));
}

#[wasm_bindgen]
pub fn test_threads() {
  let mut threads = vec![];
  for i in 0..4 {
    threads.push(std::thread::spawn(move || {
      log(&format!("Thread {} started", i));
      for j in 0..1000000 {
        if j % 100000 == 0 {
          log(&format!("Thread {} at {}", i, j));
        }
      }
      log(&format!("Thread {} done", i));
    }));
  }
  for t in threads {
    t.join().unwrap();
  }
}

#[wasm_bindgen]
pub fn test_shared_mem(shared_mem: Int32Array, my_value: i32) {
  log("Version 3");
  let loaded_value = shared_mem.get_index(0);
  log(&format!(
    "[worker={}] Loaded value: {}",
    my_value, loaded_value
  ));
  shared_mem.set_index(0, my_value);
  /*
  // Cast the shared memory to slice of std::sync::atomic::AtomicI32.
  let shared_mem = unsafe {
    std::slice::from_raw_parts(shared_mem.as_ptr() as *const std::sync::atomic::AtomicI32, shared_mem.len())
  };
  let loaded_value = shared_mem[0].load(std::sync::atomic::Ordering::SeqCst);
  log(&format!("[{}] test_shared_mem sees: {:?}", my_value, loaded_value));
  shared_mem[0].store(my_value, std::sync::atomic::Ordering::SeqCst);
  */
}
