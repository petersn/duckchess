use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::future::Pending;

use slotmap::SlotMap;

use crate::inference::{ModelOutputs, POLICY_LEN, InferenceEngine, PendingIndex};
use crate::rng::Rng;
//use crate::inference::{ModelOutputs, POLICY_LEN};
use crate::rules::{GameOutcome, Move, State};

const EXPLORATION_ALPHA: f32 = 1.0;
const FIRST_PLAY_URGENCY: f32 = 0.2;
const ROOT_SOFTMAX_TEMP: f32 = 1.2;
const DIRICHLET_ALPHA: f32 = 0.1;
const DIRICHLET_WEIGHT: f32 = 0.25;

// Create slotmap keys
slotmap::new_key_type! {
  pub struct NodeIndex;
  //pub struct EdgeIndex;
}

#[derive(Clone)]
struct MctsNode {
  depth:             u32,
  state:             State,
  moves:             Vec<Move>,
  outputs:           ModelOutputs,
  dirichlet_applied: bool,
  needs_eval:        bool,
  total_score:       f32,
  visits:            u32,
  in_flight:         u32,
  policy_explored:   f32,
  outgoing_edges:    HashMap<Move, NodeIndex>,
  gc_state:          u32,
  propagated:        bool,
}

impl MctsNode {
  fn new(state: State, depth: u32) -> Self {
    let turn_flip = if state.white_turn { 1.0 } else { -1.0 };
    let outcome = state.get_outcome();
    let mut outputs = ModelOutputs {
      policy: [1.0; POLICY_LEN],
      value:  match outcome {
        GameOutcome::Ongoing => 0.0,
        GameOutcome::WhiteWin => turn_flip,
        GameOutcome::BlackWin => -turn_flip,
        GameOutcome::Draw => 0.0,
      },
    };
    let mut moves = vec![];
    state.move_gen::<false>(&mut moves);
    outputs.renormalize(&moves);
    Self {
      depth,
      state,
      moves,
      outputs,
      dirichlet_applied: false,
      needs_eval: outcome == GameOutcome::Ongoing,
      total_score: 0.0,
      visits: 0,
      in_flight: 0,
      policy_explored: 0.0,
      outgoing_edges: HashMap::new(),
      gc_state: 0,
      propagated: false,
    }
  }

  fn get_subtree_value(&self) -> f32 {
    // Implement so called "virtual losses" -- we pretend each in flight evaluation
    // will evaluate to the worst possible outcome, to encourage diversity of paths.
    self.total_score / (self.visits + self.in_flight) as f32
  }

  fn adjust_score(&mut self, score: f32) {
    self.visits += 1;
    self.total_score += score;
  }

  fn total_action_score(
    &self,
    sqrt_policy_explored: f32,
    nodes: &SlotMap<NodeIndex, MctsNode>,
    m: Move,
  ) -> f32 {
    let effective_visits = self.visits + self.in_flight;
    let (u, q) = match self.outgoing_edges.get(&m) {
      None => (
        (effective_visits as f32).sqrt(),
        (self.get_subtree_value() - FIRST_PLAY_URGENCY * sqrt_policy_explored).max(0.0),
      ),
      Some(child_index) => {
        let child = &nodes[*child_index];
        (
          (effective_visits as f32).sqrt() / (1.0 + (child.visits + child.in_flight) as f32),
          child.get_subtree_value(),
        )
      }
    };
    let u = EXPLORATION_ALPHA * self.posterior(m) * u;
    q + u
  }

  fn select_action(&self, nodes: &SlotMap<NodeIndex, MctsNode>) -> Option<Move> {
    println!("select_action from: {:?}", self.depth);
    if self.moves.is_empty() {
      return None;
    }
    let sqrt_policy_explored = self.policy_explored.sqrt();
    let mut best_score = -std::f32::INFINITY;
    let mut best_move = None;
    for m in &self.moves {
      let score = self.total_action_score(sqrt_policy_explored, nodes, *m);
      println!(
        "    score: {:?} for move: {:?}  (posterior: {})",
        score,
        m,
        self.posterior(*m)
      );
      if score > best_score {
        best_score = score;
        best_move = Some(*m);
      }
    }
    best_move
  }

  fn new_child(&mut self, m: Move, child_index: NodeIndex) {
    debug_assert!(!self.outgoing_edges.contains_key(&m));
    self.outgoing_edges.insert(m, child_index);
    self.policy_explored += self.posterior(m);
  }

  fn add_dirichlet_noise(&mut self, rng: &mut Rng) {
    assert!(!self.dirichlet_applied);
    self.dirichlet_applied = true;

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
      let new_policy = self.outputs.policy[idx].powf(1.0 / ROOT_SOFTMAX_TEMP);
      self.outputs.policy[idx] = new_policy;
      policy_sum += new_policy;
    }
    // Mix policy with noise.
    for i in 0..noise.len() {
      self.outputs.policy[i] = (1.0 - DIRICHLET_WEIGHT) * self.outputs.policy[i] / policy_sum
        + DIRICHLET_WEIGHT * noise[i] / noise_sum;
    }
    // Assert normalization.
    let sum = self.outputs.policy.iter().sum::<f32>();
    debug_assert!((sum - 1.0).abs() < 1e-3);
  }

  fn posterior(&self, m: Move) -> f32 {
    self.outputs.policy[m.to_index() as usize]
  }
}

fn get_transposition_table_key(state: &State, depth: u32) -> (u64, u32) {
  // We include the depth in the key to keep the tree a DAG.
  // This misses some transposition opportunities, but is very cheap.
  (state.get_transposition_table_hash(), depth)
}

struct PendingPath {
  path: Vec<NodeIndex>,
}

pub struct Mcts<'a, Infer: InferenceEngine> {
  inference_engine: &'a Infer,              
  rng:                 Rng,
  root:                NodeIndex,
  nodes:               SlotMap<NodeIndex, MctsNode>,
  transposition_table: HashMap<(u64, u32), NodeIndex>,
  pending_paths:       SlotMap<PendingIndex, PendingPath>,
}

// TODO: Implement speculatively finding good candidate states to evaluate and cache.

impl<'a, Infer: InferenceEngine> Mcts<'a, Infer> {
  pub fn new(seed: u64, inference_engine: &'a Infer) -> Mcts<Infer> {
    let mut this = Self {
      inference_engine,
      rng:                 Rng::new(seed),
      root:                NodeIndex::default(),
      nodes:               SlotMap::with_key(),
      transposition_table: HashMap::new(),
      pending_paths:       SlotMap::with_key(),
    };
    this.root = this.add_child_and_adjust_scores(vec![], None, State::starting_state(), 0);
    this
  }

  pub fn get_state(&self) -> &State {
    &self.nodes[self.root].state
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
          .max_by_key(|(_, node_index)| self.nodes[**node_index].visits)
          .map(|(m, _)| *m),
        // Otherwise select according to PUCT.
        false => node.select_action(&self.nodes),
      };
      match m {
        // None means we have no legal moves at the leaf (terminal).
        None => return (nodes, None),
        // Some means we have legal moves.
        Some(m) => {
          // Try to find this edge.
          let child = match node.outgoing_edges.get(&m) {
            None => return (nodes, Some(m)),
            Some(child) => *child,
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
    println!("Adding child at depth {} (path={:?}).", depth, path);
    // Check our transposition table to see if this new state has already been reached.
    let transposition_table_key = get_transposition_table_key(&state, depth);
    // We get a node, possibly new.
    let last_node_index = match self.transposition_table.entry(transposition_table_key) {
      // If we have a transposition, just use it.
      Entry::Occupied(entry) => *entry.get(),
      Entry::Vacant(entry) => {
        let node = MctsNode::new(state, depth);
        let new_node_index = self.nodes.insert(node);
        match (path.last(), m) {
          (None, None) => {}
          (Some(parent_node_index), Some(m)) => {
            let parent_node = &mut self.nodes[*parent_node_index];
            parent_node.new_child(m, new_node_index);
          }
          _ => unreachable!(),
        }
        *entry.insert(new_node_index)
      }
    };
    path.push(last_node_index);
    // If the node is terminal then we can adjust scores immediately.
    if !self.nodes[last_node_index].needs_eval {
      self.adjust_scores_on_path::<false>(path);
    } else {
      // Otherwise we need to queue it up as a pending path.
      for node_index in &path {
        self.nodes[*node_index].in_flight += 1;
      }
      self.inference_engine.add_work(
        &self.nodes[last_node_index].state,
        self.pending_paths.insert(PendingPath { path }),
      );
    }
    last_node_index
  }

  pub fn predict_now(&mut self) {
    self.inference_engine.predict(|inference_results| {
      for i in 0..inference_results.length {
        match self.pending_paths.remove(inference_results.cookies[i]) {
          None => {
            println!("WARNING: Got inference result for unknown cookie.");
            continue;
          }
          Some(pending_path) => {
            println!("Got inference result for path: {:?}.", pending_path.path);
            let node_index = pending_path.path.last().unwrap();
            let node = &mut self.nodes[*node_index];
            node.needs_eval = false;
            node.outputs = inference_results.get(i);
            //node.outputs.renormalize(&node.moves);
            self.adjust_scores_on_path::<true>(pending_path.path);
          }
        }
      }
    });
  }

  pub fn step(&mut self) {
    let root_visit_count = self.nodes[self.root].visits + self.nodes[self.root].in_flight;
    let (pv_nodes, pv_move) = self.select_principal_variation(false);
    match pv_move {
      // If the move is null then we have no legal moves, so just propagate the score again.
      None => self.adjust_scores_on_path::<false>(pv_nodes),
      // If the move is non-null then expand once at the leaf in that direction.
      Some(m) => {
        // Create the new child state.
        let leaf_node = &self.nodes[*pv_nodes.last().unwrap()];
        let mut state = leaf_node.state.clone();
        state.apply_move(m).unwrap();
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

  pub fn adjust_scores_on_path<const DECREMENT_IN_FLIGHT: bool>(&mut self, path: Vec<NodeIndex>) {
    let last_node_index = *path.last().unwrap();
    {
      let last_node = &mut self.nodes[last_node_index];
      assert!(!last_node.needs_eval);
      assert!(!last_node.propagated);
      last_node.propagated = true;
    }
    let value_score = (self.nodes[last_node_index].outputs.value + 1.0) / 2.0;
    let is_from_whites_perspective = self.nodes[last_node_index].state.white_turn;
    // Adjust every node along the path, including the final node itself.
    for node_index in path {
      let node = &mut self.nodes[node_index];
      let player_perspective_score = match node.state.white_turn == is_from_whites_perspective {
        true => value_score,
        false => 1.0 - value_score,
      };
      node.adjust_score(player_perspective_score);
      if DECREMENT_IN_FLIGHT {
        assert!(node.in_flight > 0);
        node.in_flight -= 1;
      }
    }
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
      .map(|child_index| self.nodes[*child_index].visits as i32)
      .sum()
  }

  pub fn get_train_distribution(&self) -> Vec<(Move, f32)> {
    let sum_child_visits = self.get_sum_child_visits(self.root) as f32;
    let mut distribution = Vec::new();
    for (m, child_index) in &self.nodes[self.root].outgoing_edges {
      let child_visits = self.nodes[*child_index].visits;
      distribution.push((*m, child_visits as f32 / sum_child_visits));
    }
    distribution
  }

  pub fn sample_move_by_visit_count(&self) -> Option<Move> {
    let sum_child_visits = self.get_sum_child_visits(self.root);
    if sum_child_visits == 0 {
      return None;
    }
    let mut visit_count = self.rng.generate_range(sum_child_visits as u32) as i32;
    for (m, child_index) in &self.nodes[self.root].outgoing_edges {
      visit_count -= self.nodes[*child_index].visits as i32;
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
    let root_node = &self.nodes[self.root];
    self.root = match root_node.outgoing_edges.get(&m) {
      // If we already have a node for this move, then just make it the new root.
      Some(child_index) => *child_index,
      // Otherwise, we create a new node.
      None => {
        let mut new_state = root_node.state.clone();
        new_state.apply_move(m).unwrap();
        self.add_child_and_adjust_scores(vec![], None, new_state, root_node.depth + 1)
      }
    };
    // Next we garbage collect.
    // This `mark_state` value need only be unique.
    let mark_state = 100 + self.nodes[self.root].depth;
    let mut stack = vec![self.root];
    while let Some(node_index) = stack.pop() {
      let node = &mut self.nodes[node_index];
      if node.gc_state != mark_state {
        for child_index in node.outgoing_edges.values() {
          stack.push(*child_index);
        }
      }
      node.gc_state = mark_state;
    }
    self.nodes.retain(|_, node| node.gc_state == mark_state);
    // Finally, we clear out the transposition table, because it may
    // still contain references to freed nodes.
    self.transposition_table.clear();
  }

  pub fn print_tree_root(&self) {
    self.print_tree(self.root, 0);
  }

  fn print_tree(&self, root: NodeIndex, desired_state: u32) {
    let mut already_printed_set = std::collections::HashSet::new();
    let mut stack = vec![(None, root, 0)];
    while !stack.is_empty() {
      let (m, node_index, depth) = stack.pop().unwrap();
      let node = &self.nodes[node_index];
      let already_printed = already_printed_set.contains(&node_index);
      already_printed_set.insert(node_index);
      println!(
        "{}{}{}[{:?}] (visits={} in-flight={} value={} mean={} moves={}){}\x1b[0m",
        "  ".repeat(depth),
        match m {
          Some(m) => format!("{:?} -> ", m),
          None => "".to_string(),
        },
        match node.gc_state == desired_state {
          true => "\x1b[91m",
          false => "\x1b[92m",
        },
        node_index,
        node.visits,
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
        for (m, child_index) in &node.outgoing_edges {
          stack.push((Some(*m), *child_index, depth + 1));
        }
      }
    }
  }
}
