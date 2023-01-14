use core::panic;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

use slotmap::SlotMap;

use crate::inference::{
  Evaluation, FullPrecisionModelOutputs, InferenceEngine, ModelOutputs, POLICY_LEN,
};
use crate::rng::Rng;
//use crate::inference::{ModelOutputs, POLICY_LEN};
use crate::rules::{Move, State, RepetitionState, GameOutcome, Player};

//const EXPLORATION_ALPHA: f32 = 1.0;
//const DUCK_EXPLORATION_ALPHA: f32 = 0.5;
//const FIRST_PLAY_URGENCY: f32 = 0.2;
const ROOT_SOFTMAX_TEMP: f32 = 1.1;
const DIRICHLET_ALPHA: f32 = 0.1;

// For the in-browser search we use pretty light dirichlet noise.
#[cfg(target_arch = "wasm32")]
const DIRICHLET_WEIGHT: f32 = 0.15;

// For game generation we use the same value AlphaZero did.
#[cfg(not(target_arch = "wasm32"))]
const DIRICHLET_WEIGHT: f32 = 0.25;

// Create slotmap keys
slotmap::new_key_type! {
  pub struct NodeIndex;
  //pub struct EdgeIndex;
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SearchParams {
  pub exploration_alpha:      f32,
  pub duck_exploration_alpha: f32,
  pub first_play_urgency:     f32,
}

// These parameters were tuned with search_param_opt.py
// I tuned them against both 200 visit games, and 1000 visit games.
// They seemed to behave very similarly. In both cases it seems like
// it wants an even lower alpha than these values, but I'm uncomfortable
// going quite as low as seems to be best.
impl Default for SearchParams {
  fn default() -> Self {
    Self {
      exploration_alpha:      0.5,
      duck_exploration_alpha: 0.25,
      first_play_urgency:     0.2,
    }
  }
}

impl std::str::FromStr for SearchParams {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    let mut params = Self::default();
    if s == "default" {
      return Ok(params);
    }
    for part in s.split(':') {
      let mut parts = part.split('=');
      let key = parts.next().ok_or_else(|| format!("Missing key in {}", part))?;
      let value = parts.next().ok_or_else(|| format!("Missing value in {}", part))?;
      if parts.next().is_some() {
        return Err(format!("Extra = in {}", part));
      }
      let value_f32: f32 =
        value.parse().map_err(|_| format!("Invalid value for {}: {}", key, value))?;
      match key {
        "alpha" => params.exploration_alpha = value_f32,
        "duckalpha" => params.duck_exploration_alpha = value_f32,
        "fpu" => params.first_play_urgency = value_f32,
        _ => return Err(format!("Invalid key: {}", key)),
      }
    }
    Ok(params)
  }
}

#[inline(always)]
fn state_to_default_full_prec_model_outputs(state: &State) -> FullPrecisionModelOutputs {
  let mut value = Evaluation::EVEN_EVAL;
  let mut white_wdl = [0.0, 1.0, 0.0];
  match state.get_outcome() {
    None => {}
    Some(GameOutcome::Draw) => {
      value.is_exact = true;
    }
    Some(GameOutcome::Win(player)) => {
      value.expected_score = 1.0;
      value.perspective_player = player;
      value.is_exact = true;
      white_wdl = match player {
        Player::White => [1.0, 0.0, 0.0],
        Player::Black => [0.0, 0.0, 1.0],
      };
    }
  }
  FullPrecisionModelOutputs {
    policy: Box::new([1.0; POLICY_LEN]),
    value,
    white_wdl,
  }
}

#[derive(Clone)]
pub struct EdgeEntry {
  node: NodeIndex,
  causes_threefold: bool,
}

#[derive(Clone)]
pub struct MctsNode {
  pub depth:             u32,
  pub state:             State,
  //pub hash:              u64,
  pub moves:             Vec<Move>,
  pub outputs:           ModelOutputs,
  pub dirichlet_applied: bool,
  pub needs_eval:        bool,
  pub total_score:       f32,
  pub total_white_wdl:         [f32; 3],
  pub exact_score_and_move:    Option<(f32, Move)>,
  pub visits:            u32,
  pub tail_visits:       u32,
  pub in_flight:         u32,
  pub policy_explored:   f32,
  pub outgoing_edges:    HashMap<Move, EdgeEntry>,
  pub gc_state:          u32,
  pub propagated:        bool,
}

impl MctsNode {
  fn new(state: State, depth: u32) -> Self {
    //let hash = state.get_transposition_table_hash();
    let needs_eval = state.get_outcome().is_none();
    let mut moves = vec![];
    state.move_gen::<false>(&mut moves);
    // FIXME: in terminal positions I need to make up a correct WDL value.
    let outputs = ModelOutputs::quantize_from(
      state_to_default_full_prec_model_outputs(&state),
      &moves,
    );
    //outputs.renormalize(&moves);
    Self {
      depth,
      state,
      //hash,
      moves,
      outputs,
      dirichlet_applied: false,
      needs_eval,
      total_score: 0.0,
      total_white_wdl: [0.0; 3],
      exact_score_and_move: None,
      visits: 0,
      tail_visits: 0,
      in_flight: 0,
      policy_explored: 0.0,
      outgoing_edges: HashMap::new(),
      gc_state: 0,
      propagated: false,
    }
  }

  fn get_subtree_value(&self) -> Evaluation {
    if let Some((exact_score, _)) = self.exact_score_and_move {
      return Evaluation {
        expected_score: exact_score,
        perspective_player: self.state.turn,
        is_exact: true,
      };
    }
    // Implement so called "virtual losses" -- we pretend each in flight evaluation
    // will evaluate to the worst possible outcome, to encourage diversity of paths.
    Evaluation {
      expected_score:     self.total_score / (self.visits + self.in_flight) as f32,
      perspective_player: self.state.turn,
      is_exact:        false,
    }
  }

  fn adjust_score(&mut self, eval: Evaluation, white_wdl: &[f32; 3]) {
    let new_score = eval.expected_score_for_player(self.state.turn);
    self.visits += 1;
    self.total_score += new_score; //eval.expected_score_for_player(self.state.turn);
    self.total_white_wdl[0] += white_wdl[0];
    self.total_white_wdl[1] += white_wdl[1];
    self.total_white_wdl[2] += white_wdl[2];
    // FIXME: I need to figure out what to do here.
    // If
  }

  fn total_action_score(
    &self,
    search_params: &SearchParams,
    sqrt_policy_explored: f32,
    nodes: &SlotMap<NodeIndex, MctsNode>,
    m: Move,
  ) -> f32 {
    // FIXME: Implement sticky mates here.
    let effective_visits = self.visits + self.in_flight;
    let (u, q) = match self.outgoing_edges.get(&m) {
      None => (
        (effective_visits as f32).sqrt(),
        (self.get_subtree_value().expected_score_for_player(self.state.turn)
          - search_params.first_play_urgency * sqrt_policy_explored)
          .max(0.0),
      ),
      Some(edge) => {
        let child = &nodes[edge.node];
        let subtree_value = child.get_subtree_value();
        let perspective_score = subtree_value.expected_score_for_player(self.state.turn);
        //// If the child has an exact winning/losing score, then we amplify it enormously.
        //// TODO: What do I do with exact drawn scores?
        //if subtree_value.is_exact && perspective_score.abs() > 0.99 {
        //  return (perspective_score - 0.5) * 1000.0;
        //}
        (
          (effective_visits as f32).sqrt() / (1.0 + (child.visits + child.in_flight) as f32),
          perspective_score,
        )
      }
    };
    let alpha = match self.state.is_duck_move {
      true => search_params.duck_exploration_alpha,
      false => search_params.exploration_alpha,
    };
    let u = alpha * self.posterior(m) * u;
    q + u
  }

  fn select_action(
    &self,
    search_params: &SearchParams,
    nodes: &SlotMap<NodeIndex, MctsNode>,
  ) -> Option<Move> {
    //println!("select_action from: {:?}", self.depth);
    if self.moves.is_empty() {
      return None;
    }
    if let Some((_, m)) = self.exact_score_and_move {
      return Some(m);
    }
    let sqrt_policy_explored = self.policy_explored.sqrt();
    let mut best_score = -std::f32::INFINITY;
    let mut best_move = None;
    for m in &self.moves {
      let score = self.total_action_score(search_params, sqrt_policy_explored, nodes, *m);
      //println!(
      //  "    score: {:?} for move: {:?}  (posterior: {})",
      //  score,
      //  m,
      //  self.posterior(*m)
      //);
      if score > best_score {
        best_score = score;
        best_move = Some(*m);
      }
    }
    best_move
  }

  fn new_child(&mut self, m: Move, causes_threefold: bool, child_index: NodeIndex) {
    // FIXME: Maybe this should be a debug assert?
    assert!(!self.outgoing_edges.contains_key(&m));
    // We now guarantee that this move is actually legal here.
    assert!(self.moves.contains(&m));
    self.outgoing_edges.insert(m, EdgeEntry {
      node: child_index,
      causes_threefold,
    });
    // FIXME: I need to track the policy_explored more carefully, as evals might not be filled in yet!
    self.policy_explored = (self.policy_explored + self.posterior(m)).min(1.0);
  }

  fn add_dirichlet_noise(&mut self, rng: &mut Rng) {
    assert!(!self.dirichlet_applied);
    self.dirichlet_applied = true;

    if self.moves.is_empty() {
      return;
    }

    //let backup_policy = self.outputs.policy.clone();

    //use rand_distr::Distribution;
    //let mut thread_rng = rand::thread_rng();
    //let dist = rand_distr::Gamma::new(DIRICHLET_ALPHA, 1.0).unwrap();
    // Generate noise.
    let mut noise = [0.0; POLICY_LEN];
    let mut noise_sum = 0.0;
    let mut policy_sum = 0.0;
    for m in &self.moves {
      let idx = m.to_index() as usize;
      // Generate a gamma-distributed noise value.
      let new_noise = rng.generate_gamma_variate(DIRICHLET_ALPHA);
      noise[idx] = new_noise;
      noise_sum += new_noise;
      // Apply the new policy softmax temperature.
      let new_policy = self.outputs.get_policy(idx).powf(1.0 / ROOT_SOFTMAX_TEMP);
      self.outputs.set_policy(idx, new_policy);
      policy_sum += new_policy;
    }
    // Mix policy with noise.
    for i in 0..noise.len() {
      let mixed_policy = (1.0 - DIRICHLET_WEIGHT) * self.outputs.get_policy(i) / policy_sum
        + DIRICHLET_WEIGHT * noise[i] / noise_sum;
      self.outputs.set_policy(i, mixed_policy);
    }
    //// Make sure the policy is zero on illegal moves.
    //for i in 0..noise.len() {
    //  let mut is_move = false;
    //  for m in &self.moves {
    //    if m.to_index() as usize == i {
    //      is_move = true;
    //      break;
    //    }
    //  }
    //  if !is_move {
    //    if self.outputs.get_policy(i) != 0.0 {
    //      println!(
    //        "Invalid policy: {:?} at move {:?}",
    //        self.outputs.get_policy(i),
    //        Move::from_index(i as u16)
    //      );
    //      println!("  sums: {:?} {:?}", policy_sum, noise_sum);
    //      println!("  moves: {:?}", self.moves);
    //      println!("  state: {:?}", self.state);
    //      println!("  policy: {:?}", self.outputs.policy);
    //      println!("  backup: {:?}", backup_policy);
    //      panic!()
    //    }
    //    assert_eq!(self.outputs.get_policy(i), 0.0);
    //  }
    //}
    // Assert normalization.
    //let sum = self.outputs.quantized_policy.iter().sum::<u32>();
    //let count_nonzero = self.outputs.policy.iter().filter(|&&x| x != 0.0).count();
    //// FIXME: Something is busted here, normalization keeps failing dramatically!
    //if (sum - 1.0).abs() >= 1e-3 {
    //  println!(
    //    "\x1b[91mWARNING:\x1b[0m policy sum: {}  nonzero: {}",
    //    sum, count_nonzero
    //  );
    //  println!("  moves: {:?}", self.moves);
    //  println!("  state: {:?}", self.state);
    //  //println!("  policy: {:?}", self.outputs.policy);
    //}
    //debug_assert!((sum - 1.0).abs() < 1e-3);
  }

  pub fn posterior(&self, m: Move) -> f32 {
    self.outputs.get_policy(m.to_index() as usize)
  }
}

//fn get_transposition_table_key(state: &State, depth: u32) -> (u64, u32) {
//  // We include the depth in the key to keep the tree a DAG.
//  // This misses some transposition opportunities, but is very cheap.
//  (state.get_transposition_table_hash(), depth)
//}

//slotmap::new_key_type! {
//  pub struct PendingIndex;
//}

#[derive(Clone)]
pub struct PendingPath {
  path: Vec<NodeIndex>,
}

pub struct Mcts<'a, Infer: InferenceEngine<(usize, PendingPath)>> {
  // This ID is used to identify evaluation requests from this MCTS instance.
  pub id:                  usize,
  pub search_params:       SearchParams,
  pub inference_engine:    &'a Infer,
  pub in_flight_count:     usize,
  pub rng:                 Rng,
  pub root:                NodeIndex,
  pub nodes:               SlotMap<NodeIndex, MctsNode>,
  pub transposition_table: HashMap<(u64, u32), NodeIndex>,
  pub root_repetition_state:  RepetitionState,
  pub mark_state_counter:  u32,
  //pending_paths:       SlotMap<PendingIndex, PendingPath>,
}

// TODO: Implement speculatively finding good candidate states to evaluate and cache.

impl<'a, Infer: InferenceEngine<(usize, PendingPath)>> Mcts<'a, Infer> {
  pub fn new(
    id: usize,
    seed: u64,
    inference_engine: &'a Infer,
    search_params: SearchParams,
    state: State,
  ) -> Mcts<Infer> {
    let mut this = Self {
      id,
      search_params,
      inference_engine,
      in_flight_count: 0,
      rng: Rng::new(seed),
      root: NodeIndex::default(),
      nodes: SlotMap::with_key(),
      transposition_table: HashMap::new(),
      //pending_paths: SlotMap::with_key(),
      root_repetition_state: RepetitionState::new(),
      mark_state_counter: 0,
    };
    this.root = this.add_child_and_adjust_scores(vec![], None, state, 0);
    this
  }

  pub fn get_state(&self) -> &State {
    &self.nodes[self.root].state
  }

  pub fn any_in_flight(&self) -> bool {
    self.in_flight_count > 0
  }

  pub fn get_root_score(&self) -> (f32, u32) {
    // FIXME: I want to average over children here.
    let root = &self.nodes[self.root];
    (
      root.get_subtree_value().expected_score_for_player(crate::rules::Player::White),
      root.visits,
    )
  }

  pub fn get_root_children_visit_count(&self) -> u32 {
    let root = &self.nodes[self.root];
    let mut total_visits = 0;
    for (m, edge) in &root.outgoing_edges {
      let node = &self.nodes[edge.node];
      total_visits += node.visits;
    }
    total_visits
  }

  /// Gets a (value for white, wdl for white, visit count) triple.
  /// Currently this averages over children scaled by how likely we are to make those moves.
  /// This therefore doesn't quite match get_root_score.
  pub fn get_gui_evaluation(&self) -> (f32, [f32; 3], Vec<(Move, f32)>, u32) {
    let root = &self.nodes[self.root];
    let mut total_visits = 0;
    let mut total_weight: f32 = 0.0;
    // Sum up squares of visit counts for each child.
    for (m, edge) in &root.outgoing_edges {
      let node = &self.nodes[edge.node];
      let weight = node.visits as f32;
      total_visits += node.visits;
      total_weight += weight * weight;
    }
    let mut white_wdl = [0.0, 0.0, 0.0];
    let mut top_moves = Vec::new();
    // Compute weighted average of white WDLs.
    for (m, edge) in &root.outgoing_edges {
      let node = &self.nodes[edge.node];
      let mut weight = node.visits as f32;
      weight = weight * weight / total_weight;
      top_moves.push((*m, node.visits as f32 / total_visits as f32));
      white_wdl[0] += weight * node.total_white_wdl[0] / node.visits as f32;
      white_wdl[1] += weight * node.total_white_wdl[1] / node.visits as f32;
      white_wdl[2] += weight * node.total_white_wdl[2] / node.visits as f32;
    }
    //// Assert approximate normalization.
    //assert!((white_wdl[0] + white_wdl[1] + white_wdl[2] - 1.0).abs() < 1e-3);
    // Compute expected score for white.
    let score = white_wdl[0] + white_wdl[1] * 0.5;
    // Reweight any mates in one to be maximum weight,
    // and reweight moves that hang mate in one to be zero weight.
    let mut move_gen_scratch0 = vec![];
    let mut move_gen_scratch1 = vec![];
    for (m, weight) in &mut top_moves {
      let mut state_after_move = root.state.clone();
      state_after_move.apply_move::<false>(*m, None).unwrap();
      // Check if we just won.
      if state_after_move.get_outcome() == Some(GameOutcome::Win(root.state.turn)) {
        //crate::log(&format!("Found mate in one: {:?}", m));
        *weight = 100.0;
        // FIXME: Adjust the score here.
        break;
      }

      fn check_if_position_has_move_producing_outcome(
        position: &State,
        move_gen_scratch: &mut Vec<Move>,
        sought_outcome: Option<GameOutcome>,
      ) -> bool {
        //if position.is_duck_move {
        //  crate::log("check_if_position_has_move_producing_outcome: position is duck move");
        //}
        assert!(!position.is_duck_move);
        move_gen_scratch.clear();
        // Set quiescence because we're looking for a capture.
        position.move_gen::<true>(move_gen_scratch);
        for m in move_gen_scratch {
          let mut state_after_move = position.clone();
          state_after_move.apply_move::<false>(*m, None).expect("mate-in-1-check-position-apply-move");
          if state_after_move.get_outcome() == sought_outcome {
            return true;
          }
        }
        false
      }

      let loss_outcome = Some(GameOutcome::Win(root.state.turn.other_player()));

      // Check if this move hangs mate in one.
      let mut hangs_mate;
      if root.state.is_duck_move {
        // If we just made a duck move, then make sure that they don't have any moves that kill us.
        hangs_mate = check_if_position_has_move_producing_outcome(
          &state_after_move,
          &mut move_gen_scratch0,
          loss_outcome,
        );
      } else {
        // If we just made a non-duck move, make sure that we have at least one duck move that doesn't lose.
        hangs_mate = true;
        move_gen_scratch0.clear();
        state_after_move.move_gen::<false>(&mut move_gen_scratch0);
        for duck_followup in &move_gen_scratch0 {
          let mut state_after_duck_followup = state_after_move.clone();
          state_after_duck_followup.apply_move::<false>(*duck_followup, None).expect("mate-in-1-duck-followup-apply-move");
          if !check_if_position_has_move_producing_outcome(
            &state_after_duck_followup,
            &mut move_gen_scratch1,
            loss_outcome,
          ) {
            hangs_mate = false;
            break;
          }
        }
      }
      if hangs_mate {
        //crate::log(&format!("Found hanging mate in one: {:?}", m));
        *weight = 0.0;
      }
    }
    // Normalize weights.
    let total_weight: f32 = top_moves.iter().map(|(_, w)| w).sum();
    for (_, weight) in &mut top_moves {
      *weight /= total_weight;
    }
    // Sort the moves by weight.
    top_moves.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    return (score, white_wdl, top_moves, total_visits);
  }

  // FIXME: These names here are so bad.
  pub fn get_pv(&self) -> Vec<Move> {
    let mut pv = vec![];
    let mut node_index = self.root;
    loop {
      let node = &self.nodes[node_index];
      let best_move = match node
        .outgoing_edges
        .iter()
        .max_by_key(|(_, edge)| self.nodes[edge.node].visits)
        .map(|(m, _)| *m)
      {
        Some(m) => m,
        None => break,
      };
      pv.push(best_move);
      node_index = node.outgoing_edges[&best_move].node;
    }
    pv
  }

  fn select_principal_variation(&self, best: bool) -> (Vec<NodeIndex>, Option<Move>) {
    let mut node = &self.nodes[self.root];
    let mut nodes = vec![self.root];
    loop {
      let m: Option<Move> = match best {
        // If we're picking the best PV, just take the most visited.
        true => node
          .outgoing_edges
          .iter()
          .max_by_key(|(_, edge)| self.nodes[edge.node].visits)
          .map(|(m, _)| *m),
        // Otherwise select according to PUCT.
        false => node.select_action(&self.search_params, &self.nodes),
      };
      match m {
        // None means we have no legal moves at the leaf (terminal).
        None => return (nodes, None),
        // Some means we have legal moves.
        Some(m) => {
          // Try to find this edge.
          let child = match node.outgoing_edges.get(&m) {
            None => return (nodes, Some(m)),
            Some(edge) => edge.node,
          };
          nodes.push(child);
          node = &self.nodes[child];
        }
      }
    }
  }

  /*
          // Add virtual visits to everyone along the path.
  for node_index in pv_nodes {
    self.nodes[node_index].in_flight += 1;
  }
          // Append the new child to the PV.
  leaf_node.new_child(m, child_index);
  pv_nodes.push(child_index);
  self.pending_paths.push(PendingPath { path: pv_nodes });
   */

  fn add_child_and_adjust_scores(
    &mut self,
    mut path: Vec<NodeIndex>,
    m: Option<Move>,
    state: State,
    depth: u32,
  ) -> NodeIndex {
    let new_hash = state.get_transposition_table_hash();
    // // We check if this path results in a threefold repetition.
    // let mut repetition_state = self.root_repetition_state.clone();
    // for node_index in &path {
    //   let repetition_along_path = repetition_state.add(self.nodes[*node_index].hash);
    //   assert!(!repetition_along_path);
    // }
    // // Finally, check if adding the new state results in a threefold repetition.
    // let threefold_repetition = repetition_state.would_adding_cause_threefold(new_hash);
    let threefold_repetition = false;

    //let parent_repetition_state = match path.last() {
    //  None => &self.empty_repetition_state,
    //  Some(parent_index) => &self.nodes[*parent_index].repetition_state,
    //};
    //println!("Adding child at depth {} (path={:?}).", depth, path);
    // Check our transposition table to see if this new state has already been reached.
    let transposition_table_key = (new_hash, depth);
    // We get a node, possibly new.
    let last_node_index = match self.transposition_table.entry(transposition_table_key) {
      Entry::Occupied(mut entry) => {
        // To avoid making illegal moves due to hash collisions, we must check if the states match.
        let transposition_node_index = *entry.get();
        let transposition_node = &self.nodes[transposition_node_index];
        match transposition_node.state.equal_states(&state) {
          true => transposition_node_index,
          false => {
            use std::io::Write;
            println!("\x1b[91m!!!!!!!!!!!!!!! Hash collision!\x1b[0m");
            // Write a file to /tmp, because I *really* want to see this.
            {
              let mut f = std::fs::File::create("/tmp/mcts_hash_collision.txt").unwrap();
              writeln!(f, "State 1:").unwrap();
              writeln!(f, "{:?}", state).unwrap();
              writeln!(f, "State 2:").unwrap();
              writeln!(f, "{:?}", transposition_node.state).unwrap();
              writeln!(f, "Hash key: {:?}", transposition_table_key).unwrap();
            }
            let node = MctsNode::new(state, depth);
            let new_node_index = self.nodes.insert(node);
            entry.insert(new_node_index)
          }
        }
      }
      Entry::Vacant(entry) => {
        let node = MctsNode::new(state, depth);
        let new_node_index = self.nodes.insert(node);
        *entry.insert(new_node_index)
      }
    };
    match (path.last(), m) {
      // We use an empty path and empty move to indicate building the root.
      (None, None) => {}
      (Some(parent_node_index), Some(m)) => {
        let parent_node = &mut self.nodes[*parent_node_index];
        parent_node.new_child(m, threefold_repetition, last_node_index);
      }
      _ => unreachable!(),
    }
    path.push(last_node_index);
    // If the node is terminal then we can adjust scores immediately.
    if !self.nodes[last_node_index].needs_eval {
      self.adjust_scores_on_path::<false>(path, "terminal");
    } else {
      // Otherwise we need to queue it up as a pending path.
      for node_index in &path {
        self.nodes[*node_index].in_flight += 1;
      }
      self.inference_engine.add_work(
        &self.nodes[last_node_index].state,
        (self.id, PendingPath { path }),
      );
      self.in_flight_count += 1;
    }
    last_node_index
  }

  pub fn process_path(
    &mut self,
    pending_path: PendingPath,
    model_outputs: FullPrecisionModelOutputs,
  ) {
    assert!(self.in_flight_count > 0);
    // FIXME: I need to ignore the path if it includes GCed nodes.
    let node_index = pending_path.path.last().unwrap();
    // If the tail is GCed, then we ignore the whole thing.
    if !self.nodes.contains_key(*node_index) {
      self.in_flight_count -= 1;
      return;
    }
    let node = &mut self.nodes[*node_index];
    node.needs_eval = false;
    node.outputs = ModelOutputs::quantize_from(model_outputs, &node.moves);
    //crate::log(&format!("Outputs: {:?}", node.outputs.value));
    //node.outputs.renormalize(&node.moves);
    self.adjust_scores_on_path::<true>(pending_path.path, "inference");
    self.in_flight_count -= 1;
  }

  //pub fn predict_now(&mut self) {
  //  //self.inference_engine.predict(|inference_results| {
  //  //  for i in 0..inference_results.length {
  //  //    match self.pending_paths.remove(inference_results.cookies[i]) {
  //  //      None => {
  //  //        println!("WARNING: Got inference result for unknown cookie.");
  //  //        continue;
  //  //      }
  //  //      Some(pending_path) => {
  //  //        println!("Got inference result for path: {:?}.", pending_path.path);
  //  //        let node_index = pending_path.path.last().unwrap();
  //  //        let node = &mut self.nodes[*node_index];
  //  //        node.needs_eval = false;
  //  //        node.outputs = inference_results.get(i);
  //  //        //crate::log(&format!("Outputs: {:?}", node.outputs.value));
  //  //        //node.outputs.renormalize(&node.moves);
  //  //        self.adjust_scores_on_path::<true>(pending_path.path, "inference");
  //  //      }
  //  //    }
  //  //  }
  //  //});
  //}

  pub fn step(&mut self) {
    let root_visit_count = self.nodes[self.root].visits + self.nodes[self.root].in_flight;
    let (pv_nodes, pv_move) = self.select_principal_variation(false);
    match pv_move {
      // If the move is null then we have no legal moves, so just propagate the score again.
      None => self.adjust_scores_on_path::<false>(pv_nodes, "reprop"),
      // If the move is non-null then expand once at the leaf in that direction.
      Some(m) => {
        // Create the new child state.
        let leaf_node = &self.nodes[*pv_nodes.last().unwrap()];
        let mut state = leaf_node.state.clone();
        state.apply_move::<false>(m, None).unwrap();
        let new_depth = leaf_node.depth + 1;
        // Possibly create a new child node.
        self.add_child_and_adjust_scores(pv_nodes, Some(m), state, new_depth);
      }
    };
    // Assert that the root's edge visit count went up by one.
    assert_eq!(
      self.nodes[self.root].visits + self.nodes[self.root].in_flight,
      root_visit_count + 1
    );
  }

  pub fn adjust_scores_on_path<const DECREMENT_IN_FLIGHT: bool>(
    &mut self,
    path: Vec<NodeIndex>,
    cause: &str,
  ) {
    //crate::log(&format!("Adjusting scores on path: {:?}", path));
    if path.is_empty() {
      crate::log("WARNING: Got empty path.");
      crate::log(&format!("Cause: {}", cause));
    }
    let last_node_index = *path.last().unwrap();
    {
      //crate::log(&format!("Last node: {:?}", self.nodes.contains_key(last_node_index)));
      if !self.nodes.contains_key(last_node_index) {
        crate::log("WARNING: Got path with unknown tail.");
        crate::log(&format!("Cause: {}", cause));
      }
      let last_node = &mut self.nodes[last_node_index];
      if last_node.needs_eval {
        crate::log("WARNING: Got path with un-evaluated tail.");
        crate::log(&format!("Cause: {}", cause));
      }
      //if last_node.propagated && !last_node.state.is_game_over() {
      //  crate::log("WARNING: Got path with already propagated non-terminal tail.");
      //  crate::log(&format!("Cause: {}", cause));
      //}
      assert!(!last_node.needs_eval);
      //assert!(!last_node.propagated || last_node.state.is_game_over());
      last_node.propagated = true;
    }
    self.nodes[last_node_index].tail_visits += 1;
    let value_score = self.nodes[last_node_index].outputs.value;
    let white_wdl = self.nodes[last_node_index].outputs.white_wdl;
    // Adjust every node along the path, including the final node itself.
    for node_index in path {
      if !self.nodes.contains_key(node_index) {
        // FIXME: I'll probably comment this out, and just silently ignore missing nodes.
        crate::log("WARNING: Got path with unknown node.");
        crate::log(&format!("Cause: {}", cause));
        continue;
      }
      let node = &mut self.nodes[node_index];
      node.adjust_score(value_score, &white_wdl);
      if DECREMENT_IN_FLIGHT {
        assert!(node.in_flight > 0);
        node.in_flight -= 1;
      }
    }
  }

  pub fn have_reached_visit_count(&self, tree_size: u32) -> bool {
    self.get_sum_child_visits(self.root) as u32 >= tree_size
  }

  pub fn have_reached_visit_count_short_circuiting(&self, tree_size: u32) -> bool {
    // Find the most and second most visited children at the root.
    let mut best_child_visits = 0;
    let mut second_best_child_visits = 0;
    for (_, edge) in &self.nodes[self.root].outgoing_edges {
      let child_visits = self.nodes[edge.node].visits;
      if child_visits > best_child_visits {
        second_best_child_visits = best_child_visits;
        best_child_visits = child_visits;
      } else if child_visits > second_best_child_visits {
        second_best_child_visits = child_visits;
      }
    }
    // Compute the greatest possible number of additional steps we might take.
    let maximum_steps = tree_size as i64 - self.nodes[self.root].visits as i64;
    // Check if there are enough steps left for the second best to surpass the best.
    maximum_steps <= 0 || second_best_child_visits as i64 + maximum_steps < best_child_visits as i64
  }

  /*
  pub fn step_until(&mut self, tree_size: u32, early_out: bool) -> u32 {
    let mut steps_taken = 0;
    loop {
      // Find the most and second most visited children at the root.
      let mut best_child_visits = 0;
      let mut second_best_child_visits = 0;
      for (_, child_index) in &self.nodes[self.root].outgoing_edges {
        let child_visits = self.nodes[*child_index].visits;
        if child_visits > best_child_visits {
          second_best_child_visits = best_child_visits;
          best_child_visits = child_visits;
        } else if child_visits > second_best_child_visits {
          second_best_child_visits = child_visits;
        }
      }
      // Compute the greatest possible number of additional steps we might take.
      let maximum_steps = tree_size as i64 - self.nodes[self.root].visits as i64;
      if maximum_steps <= 0 {
        break;
      }
      // Check if there are enough steps left for the second best to surpass the best.
      if early_out && second_best_child_visits as i64 + maximum_steps < best_child_visits as i64 {
        break;
      }
      // Step.
      self.step();
      steps_taken += 1;
    }
    steps_taken
  }
  */

  fn get_sum_child_visits(&self, node_index: NodeIndex) -> i32 {
    // Naively we could use node.visits - 1, but due to transpositions
    // that might not actually be the right value.
    self.nodes[node_index]
      .outgoing_edges
      .values()
      .map(|edge| self.nodes[edge.node].visits as i32)
      .sum()
  }

  fn get_immediately_winning_move(&self) -> Option<Move> {
    let current_player = self.nodes[self.root].state.turn;
    for (m, edge) in &self.nodes[self.root].outgoing_edges {
      let eval = self.nodes[edge.node].outputs.value;
      if eval.is_exact && eval.expected_score_for_player(current_player) >= 1.0 {
        return Some(*m);
      }
    }
    None
  }

  pub fn get_train_distribution(&self) -> Vec<(Move, f32)> {
    // Check if we have any immediately winning children.
    if let Some(m) = self.get_immediately_winning_move() {
      return vec![(m, 1.0)];
    }

    let sum_child_visits = self.get_sum_child_visits(self.root) as f32;
    let mut distribution = Vec::new();
    for (m, edge) in &self.nodes[self.root].outgoing_edges {
      let child_visits = self.nodes[edge.node].visits;
      distribution.push((*m, child_visits as f32 / sum_child_visits));
    }
    distribution
  }

  pub fn sample_move_by_visit_count(&self, beta: u32) -> Option<Move> {
    //let sum_child_visits = self.get_sum_child_visits(self.root);
    // Naively we could use node.visits - 1, but due to transpositions
    // that might not actually be the right value.

    // Check if we have any immediately winning children.
    if let Some(m) = self.get_immediately_winning_move() {
      return Some(m);
    }

    let temperature_sum_child_visits: i64 = self.nodes[self.root]
      .outgoing_edges
      .values()
      .map(|edge| (self.nodes[edge.node].visits as i64).pow(beta))
      .sum();

    if temperature_sum_child_visits == 0 {
      if self.get_state().get_outcome().is_none() {
        crate::log("WARNING: No moves available!");
        self.print_tree_root();
        panic!("No moves??");
      }
      return None;
    }
    let mut visit_count =
      self.rng.generate_range_sketchy(temperature_sum_child_visits as u64) as i64;
    for (m, edge) in &self.nodes[self.root].outgoing_edges {
      visit_count -= (self.nodes[edge.node].visits as i64).pow(beta);
      if visit_count < 0 {
        return Some(*m);
      }
    }
    panic!("Failed to sample move by visit count");
  }

  pub fn apply_noise_to_root(&mut self) {
    self.nodes[self.root].add_dirichlet_noise(&mut self.rng);
  }

  pub fn apply_move(&mut self, m: Move) {
    //if self.any_in_flight() {
    //  crate::log("In-flight nodes when applying move!");
    //}
    //assert!(!self.any_in_flight());
    let root_node = &self.nodes[self.root];
    self.root = match root_node.outgoing_edges.get(&m) {
      // If we already have a node for this move, then just make it the new root.
      Some(edge) => edge.node,
      // Otherwise, we create a new node.
      None => {
        let mut new_state = root_node.state.clone();
        match new_state.apply_move::<false>(m, None) {
          Ok(_) => (),
          Err(e) => {
            crate::log(&format!(
              "Failed to apply move {:?} to state: {:?}",
              m, root_node.state
            ));
            panic!("Failed to apply move: {:?}", e);
            //panic!(e);
          }
        }
        self.add_child_and_adjust_scores(vec![], None, new_state, root_node.depth + 1)
      }
    };
    // FIXME: This policy of garbage collection is pretty arbitrary.
    if self.nodes.len() > 3_000 {
      self.garbage_collect();
    }
    //// FIXME: What do I do about pending paths?
    //assert!(self.pending_paths.is_empty());
  }

  pub fn garbage_collect(&mut self) {
    // This `mark_state` value need only be unique.
    self.mark_state_counter += 1;
    let mark_state = self.mark_state_counter;
    let mut stack = vec![self.root];
    while let Some(node_index) = stack.pop() {
      let node = &mut self.nodes[node_index];
      if node.gc_state != mark_state {
        for edge in node.outgoing_edges.values() {
          stack.push(edge.node);
        }
      }
      node.gc_state = mark_state;
    }
    // Drop nodes, and references to freed nodes in the transposition table.
    self.transposition_table.retain(|_, node_index| {
      self.nodes[*node_index].gc_state == mark_state
    });
    self.nodes.retain(|_, node| node.gc_state == mark_state);
  }

  pub fn reroot_tree(&mut self, new_state: &State) {
    let new_state_hash = new_state.get_transposition_table_hash();
    // First check if we have a child that matches this state.
    let root_node = &self.nodes[self.root];
    for (_, edge) in &root_node.outgoing_edges {
      let child_node = &self.nodes[edge.node];
      if child_node.state.get_transposition_table_hash() == new_state_hash {
        self.root = edge.node;
        // FIXME: This policy of garbage collection is pretty arbitrary.
        if self.nodes.len() > 3_000 {
          self.garbage_collect();
        }
        return;
      }
    }

    self.transposition_table.clear();
    self.nodes.clear();
    self.root = self.add_child_and_adjust_scores(vec![], None, new_state.clone(), new_state.plies);
  }

  pub fn print_tree_root(&self) {
    self.print_tree(self.root, 0);
  }

  fn print_tree(&self, root: NodeIndex, _desired_state: u32) {
    let mut already_printed_set = std::collections::HashSet::new();
    let mut stack = vec![(None, root, 0)];
    while !stack.is_empty() {
      let (m, node_index, depth) = stack.pop().unwrap();
      if depth > 3 {
        println!("... (depth {})", depth);
        continue;
      }
      let node = &self.nodes[node_index];
      let already_printed = already_printed_set.contains(&node_index);
      already_printed_set.insert(node_index);
      println!(
        "{}{}{}[{:?}] (vis={} tail={} iflght={} eval={} mean={} movs={}){}\x1b[0m",
        "  ".repeat(depth),
        match m {
          Some(m) => format!("{} -> ", m),
          None => "".to_string(),
        },
        "",
        //match node.gc_state == desired_state {
        //  true => "\x1b[91m",
        //  false => "\x1b[92m",
        //},
        node_index,
        node.visits,
        node.tail_visits,
        node.in_flight,
        node.outputs.value,
        node.get_subtree_value(),
        node.moves.len(),
        match already_printed {
          true => " (already printed)",
          false => "",
        },
      );
      if !already_printed {
        for (m, edge) in &node.outgoing_edges {
          stack.push((Some(*m), edge.node, depth + 1));
        }
      }
    }
  }

  pub fn print_tree_as_graphviz(&self) {
    println!("digraph {{");
    let mut already_printed_set = std::collections::HashSet::new();
    let mut stack = vec![(None, self.root, 0)];
    while !stack.is_empty() {
      let (_m, node_index, depth) = stack.pop().unwrap();
      let node = &self.nodes[node_index];
      let already_printed = already_printed_set.contains(&node_index);
      already_printed_set.insert(node_index);
      println!("  {:?} [label=\"\"]", node_index,);
      //println!(
      //  "  {:?} [label=\"{:?} (visits={} tail={} in-flight={} value={:?} mean={:?} moves={})\"]",
      //  node_index,
      //  node_index,
      //  node.visits,
      //  node.tail_visits,
      //  node.in_flight,
      //  node.outputs.value,
      //  node.get_subtree_value(),
      //  node.moves.len(),
      //);
      if !already_printed {
        for (m, edge) in &node.outgoing_edges {
          //println!("  {:?} -> {:?} [label=\"{:?}\"]", node_index, edge.node, m);
          println!("  {:?} -> {:?}", node_index, edge.node);
          stack.push((Some(*m), edge.node, depth + 1));
        }
      }
    }
    println!("}}");
  }
}
