use std::arch::wasm32::*;

use js_sys::{Array, Atomics, Int32Array, SharedArrayBuffer, Uint8Array};
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use slotmap::SlotMap;

use crate::inference::{FEATURES_SIZE, POLICY_LEN};
use crate::inference_web::MAX_BATCH_SIZE;
use crate::mcts::{PendingPath, SearchParams};
use crate::{
  inference, inference_web, mcts,
  rules::{Move, Player},
  search,
  rules,
};

const MAX_STEPS_BEFORE_INFERENCE: usize = 40 * MAX_BATCH_SIZE;
const WEB_TRANSPOSITION_TABLE_SIZE: usize = 1 << 20;

#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = console)]
  pub fn log(s: &str);
}


slotmap::new_key_type! {
  pub struct GameNodeId;
}

#[derive(Clone, Copy, Serialize)]
pub struct TreeEdge {
  m: Move,
  child: GameNodeId,
}

#[derive(Serialize)]
pub struct NodeEval {
  white_perspective_wdl: [f32; 3],
  top_moves: Vec<(Move, f32)>,
  steps: usize,
}

#[derive(Serialize)]
pub struct GameNode {
  state: rules::State,
  parent: Option<GameNodeId>,
  legal_moves: Vec<Move>,
  legal_duck_skipping_moves: Vec<Move>,
  outgoing_edges: Vec<TreeEdge>,
  evaluation: NodeEval,
}

impl GameNode {
  fn new(state: rules::State, parent: Option<GameNodeId>) -> Self {
    let mut legal_moves = Vec::new();
    let mut legal_duck_skipping_moves = Vec::new();
    state.move_gen::<false>(&mut legal_moves);
    if state.is_duck_move && state.get_outcome().is_none() {
      let mut next_state = state.clone();
      next_state.turn = next_state.turn.other_player();
      next_state.is_duck_move = false;
      // Remove the duck, because we could put it somewhere to not interfere.
      next_state.ducks.0 = 0;
      next_state.move_gen::<false>(&mut legal_duck_skipping_moves);
    }
    Self {
      state,
      parent,
      legal_moves,
      legal_duck_skipping_moves,
      outgoing_edges: Vec::new(),
      evaluation: NodeEval {
        white_perspective_wdl: [0.0; 3],
        top_moves: vec![],
        steps: 0,
      },
    }
  }
}

#[wasm_bindgen]
pub struct GameTree {
  nodes: SlotMap<GameNodeId, GameNode>,
  cursor: GameNodeId,
  root: GameNodeId,
}

#[wasm_bindgen]
impl GameTree {
  pub fn new() -> Self {
    let mut gt = Self {
      nodes: SlotMap::with_key(),
      cursor: GameNodeId::default(),
      root: GameNodeId::default(),
    };
    gt.cursor = gt.new_node(rules::State::starting_state(), None);
    gt.root = gt.cursor;
    gt
  }

  fn new_node(&mut self, state: rules::State, parent: Option<GameNodeId>) -> GameNodeId {
    self.nodes.insert_with_key(|id| GameNode::new(state, parent))
  }

  fn inner_make_move(&mut self, m: Move) -> bool {
    let node = &mut self.nodes[self.cursor];
    if !node.legal_moves.contains(&m) {
      log(&format!("Illegal move: {:?}", m));
      return false;
    }
    // Check if we have an edge for this move already.
    for edge in &node.outgoing_edges {
      if edge.m == m {
        self.cursor = edge.child;
        return true;
      }
    }
    let mut next_state = node.state.clone();
    next_state.apply_move::<false>(m, None);
    let child = self.new_node(next_state, Some(self.cursor));
    self.nodes[self.cursor].outgoing_edges.push(TreeEdge { m, child });
    self.cursor = child;
    true
  }

  pub fn make_move(&mut self, m: JsValue, is_duck_skipping: bool) -> bool {
    let m: Move = serde_wasm_bindgen::from_value(m).unwrap_or_else(|e| {
      log(&format!("Failed to deserialize move: {}", e));
      panic!("Failed to deserialize move: {}", e);
    });
    let node = &mut self.nodes[self.cursor];
    if is_duck_skipping {
      let duck_move = match node.state.compute_duck_skip_move(m) {
        Some(m) => m,
        None => {
          log(&format!("Illegal duck skipping move: {:?}", m));
          return false;
        }
      };
      self.inner_make_move(duck_move);
    }
    self.inner_make_move(m)
  }

  pub fn click_to_id(&mut self, click: JsValue) -> bool {
    let click: GameNodeId = serde_wasm_bindgen::from_value(click).unwrap_or_else(|e| {
      log(&format!("Failed to deserialize node id: {}", e));
      panic!("Failed to deserialize node id: {}", e);
    });
    self.inner_click_to_id(click)
  }

  pub fn delete_by_id(&mut self, id: JsValue) -> bool {
    let id: GameNodeId = serde_wasm_bindgen::from_value(id).unwrap_or_else(|e| {
      log(&format!("Failed to deserialize node id: {}", e));
      panic!("Failed to deserialize node id: {}", e);
    });
    if id == self.root {
      log("Cannot delete root node");
      return false;
    }
    let parent = self.nodes[id].parent.unwrap();
    // Remove the pointer from the parent.
    let parent_node = &mut self.nodes[parent];
    parent_node.outgoing_edges.retain(|edge| edge.child != id);
    fn recursive_free_nodes(nodes: &mut SlotMap<GameNodeId, GameNode>, id: GameNodeId) {
      // FIXME: unwrap -> expect, for debugging purposes
      let node = nodes.remove(id).unwrap();
      for edge in node.outgoing_edges {
        recursive_free_nodes(nodes, edge.child);
      }
    }
    recursive_free_nodes(&mut self.nodes, id);
    // If our cursor now points to a deleted node, move it to the parent.
    if !self.nodes.contains_key(self.cursor) {
      self.cursor = parent;
    }    
    true
  }

  pub fn promote_by_id(&mut self, id: JsValue) -> bool {
    let id: GameNodeId = serde_wasm_bindgen::from_value(id).unwrap_or_else(|e| {
      log(&format!("Failed to deserialize node id: {}", e));
      panic!("Failed to deserialize node id: {}", e);
    });
    // While we have a parent, make sure we're first in the parent's list of
    // outgoing edges, then move up.
    let mut current = id;
    while let Some(parent) = self.nodes[current].parent {
      let parent_node = &mut self.nodes[parent];
      let our_index = parent_node.outgoing_edges.iter().position(|edge| edge.child == current).unwrap();
      parent_node.outgoing_edges.swap(0, our_index);
      current = parent;
    }
    true
  }

  fn inner_click_to_id(&mut self, click: GameNodeId) -> bool {
    // Check if the node is in the tree.
    let have_node = self.nodes.contains_key(click);
    if have_node {
      self.cursor = click;
    } else {
      log(&format!("Node {:?} not found", click));
    }
    have_node
  }

  /// Traverses to the parent.
  pub fn history_back(&mut self) -> bool {
    if let Some(parent) = self.nodes[self.cursor].parent {
      self.cursor = parent;
      true
    } else {
      false
    }
  }

  /// Traverses to the first child, if present.
  pub fn history_forward(&mut self) -> bool {
    if let Some(&edge) = self.nodes[self.cursor].outgoing_edges.first() {
      self.cursor = edge.child;
      true
    } else {
      false
    }
  }

  fn delete_subtree(&mut self, node_id: GameNodeId) -> Option<()> {
    // Remove the node from the slotmap.
    let node = self.nodes.remove(node_id)?;
    // If we have a parent node, remove the edge to this node.
    if let Some(parent) = node.parent {
      let parent_node = &mut self.nodes[parent];
      parent_node.outgoing_edges.retain(|edge| edge.child != node_id);
    }
    // Delete all children.
    for edge in node.outgoing_edges {
      self.delete_subtree(edge.child);
    }
    Some(())
  }

  pub fn get_serialized_state(&self) -> JsValue {
    // Serialize the current node.
    let node = &self.nodes[self.cursor];

    #[derive(Serialize)]
    pub struct Info {
      pub id: GameNodeId,
      pub edges: Vec<(Move, String, Info)>,
    }

    // Serialize the entire tree of moves and IDs.
    fn make_tree(nodes: &SlotMap<GameNodeId, GameNode>, node_id: GameNodeId) -> Info {
      let node = &nodes[node_id];
      let mut edges = Vec::new();
      for edge in &node.outgoing_edges {
        edges.push((
          edge.m,
          node.state.get_move_name(edge.m),
          make_tree(nodes, edge.child),
        ));
      }
      Info {
        id: node_id,
        edges,
      }
    }
    let info = make_tree(&self.nodes, self.root);
    serde_wasm_bindgen::to_value(&(node, info, self.cursor)).unwrap_or_else(|e| {
      log(&format!("Failed to serialize state: {}", e));
      panic!("Failed to serialize state: {}", e);
    })
  }
}

#[wasm_bindgen]
pub fn new_game_tree() -> GameTree {
  GameTree::new()
}

#[wasm_bindgen]
pub struct Engine {
  inference_engine: &'static inference_web::TensorFlowJsEngine<(usize, PendingPath)>,
  //engine:           search::Engine,
  mcts:             mcts::Mcts<'static, inference_web::TensorFlowJsEngine<(usize, PendingPath)>>,
  //input_array: Box<[f32; MAX_BATCH_SIZE * FEATURES_SIZE]>,
  //policy_array: Box<[f32; MAX_BATCH_SIZE * POLICY_LEN]>,
  //value_array: Box<[f32; MAX_BATCH_SIZE]>,
}

#[wasm_bindgen]
impl Engine {
  pub fn get_state(&self) -> JsValue {
    serde_wasm_bindgen::to_value(self.mcts.get_state()).unwrap_or_else(|e| {
      log(&format!("Failed to serialize state: {}", e));
      JsValue::NULL
    })
  }

  pub fn set_state(&mut self, state: JsValue) {
    panic!("Not implemented");
    //let new_state = serde_wasm_bindgen::from_value(state).unwrap_or_else(|e| {
    //  log(&format!("Failed to deserialize state: {}", e));
    //  panic!("Failed to deserialize state: {}", e);
    //});
    //self.mcts.set_state(new_state);
  }

  pub fn get_moves(&self) -> JsValue {
    let mut moves = Vec::new();
    let mut optional_moves = Vec::new();
    let current_state = self.mcts.get_state();
    //let current_state = self.engine.get_state();
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
      match self.mcts.get_state().compute_duck_skip_move(m) {
        Some(duck_move) => {
          self.mcts.apply_move(duck_move);
          //self.engine.apply_move(duck_move);
        }
        None => {
          log("Invalid duck move");
          return false;
        }
      }
    }

    //match self.engine.get_state_mut().apply_move::<false>(m, None) {
    //  Ok(_) => {}
    //  Err(msg) => {
    //    log(&format!("Failed to apply move: {}", msg));
    //    return false;
    //  }
    //}
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
    let (root_score, nodes) = self.mcts.get_root_score();
    serde_wasm_bindgen::to_value(&(pv, root_score, nodes)).unwrap_or_else(|e| {
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
pub struct Pvs {
  engine: search::Engine,
}

#[wasm_bindgen]
impl Pvs {
  pub fn apply_move(&mut self, m: JsValue, is_hidden: bool) -> bool {
    let m: Move = serde_wasm_bindgen::from_value(m).unwrap_or_else(|e| {
      log(&format!("Failed to deserialize move: {}", e));
      panic!("Failed to deserialize move: {}", e);
    });
    if is_hidden {
      // FIXME: I need to make this properly insert duck moves.
      match self.engine.get_state().compute_duck_skip_move(m) {
        Some(duck_move) => {
          match self.engine.get_state_mut().apply_move::<false>(duck_move, None) {
            Ok(_) => {}
            Err(msg) => {
              log(&format!("Failed to apply move: {}", msg));
              return false;
            }
          }
          //self.engine.apply_move(duck_move);
        }
        None => {
          log("Invalid duck move");
          return false;
        }
      }
    }
    match self.engine.get_state_mut().apply_move::<false>(m, None) {
      Ok(_) => {}
      Err(msg) => {
        log(&format!("Failed to apply move: {}", msg));
        return false;
      }
    }
    true
  }

  pub fn mate_search(&mut self, depth: u16) -> JsValue {
    //// We can only do mate searches at non-duck moves.
    //if self.engine.get_state().is_duck_move {
    //  return JsValue::NULL;
    //}

    // Note that only a non-duck move can capture a king.
    // Therefore it only makes sense to search to even depths from duck moves,
    // and odd depths from non-duck moves.
    let depth = match self.engine.get_state().is_duck_move {
      true => 2 * depth,
      false => 2 * depth + 1,
    };
    let (eval, moves) = self.engine.mate_search(depth);
    // Change the eval to be from white's perspective.
    let eval = match self.engine.get_state().turn {
      Player::White => eval,
      Player::Black => -eval,
    };
    serde_wasm_bindgen::to_value(&(eval, moves, self.engine.nodes_searched)).unwrap_or_else(|e| {
      log(&format!("Failed to serialize score: {}", e));
      JsValue::NULL
    })
  }
}

#[wasm_bindgen]
pub fn new_pvs(seed: u64) -> Pvs {
  Pvs {
    engine: search::Engine::new(seed, WEB_TRANSPOSITION_TABLE_SIZE),
  }
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
  //let mut nnue = crate::nnue::Nnue::new(&State::starting_state(), crate::nnue::BUNDLED_NETWORK);
  // FIXME: Support NNUE again!
  assert!(!NNUE);
  while let Some(mut entry) = stack.pop() {
    nodes += 1;
    //let adjustment = entry.state.apply_move::<NNUE>(entry.m, Some(&mut nnue)).unwrap();
    let adjustment = entry.state.apply_move::<NNUE>(entry.m, None).unwrap();
    if EVAL {
      crate::eval::basic_eval(&entry.state);
    }
    //if NNUE {
    //  nnue.evaluate(&entry.state);
    //}
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
    //if NNUE {
    //  // FIXME: This is WRONG! I need to pass in the parent state!
    //  nnue.apply_adjustment::<true>(&entry.state, &adjustment);
    //}
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
