use std::collections::hash_map::Entry;
use std::collections::HashMap;

use slotmap::SlotMap;

use crate::inference::{InferenceEngine, ModelOutputs, POLICY_LEN};
use crate::rules::{GameOutcome, Move, State};

const EXPLORATION_ALPHA: f32 = 1.0;
const FIRST_PLAY_URGENCY: f32 = 0.2;
const ROOT_SOFTMAX_TEMP: f32 = 1.2;
const DIRICHLET_ALPHA: f32 = 0.1;
const DIRICHLET_WEIGHT: f32 = 0.25;

#[derive(Clone)]
struct Evals {
  moves:             Vec<Move>,
  outputs:           ModelOutputs,
  dirichlet_applied: bool,
}

impl Evals {
  async fn create(inference_engine: &InferenceEngine, state: &State) -> Self {
    let turn_flip = if state.white_turn { 1.0 } else { -1.0 };
    let mut outputs = match state.get_outcome() {
      GameOutcome::Ongoing => inference_engine.predict(state).await,
      GameOutcome::WhiteWin => ModelOutputs {
        policy: [0.0; POLICY_LEN],
        value:  turn_flip,
      },
      GameOutcome::BlackWin => ModelOutputs {
        policy: [0.0; POLICY_LEN],
        value:  -turn_flip,
      },
      GameOutcome::Draw => ModelOutputs {
        policy: [0.0; POLICY_LEN],
        value:  0.0,
      },
    };
    let mut moves = vec![];
    state.move_gen::<false>(&mut moves);
    outputs.renormalize(&moves);
    Self {
      moves,
      outputs,
      dirichlet_applied: false,
    }
  }

  fn add_dirichlet_noise(&mut self) {
    assert!(!self.dirichlet_applied);
    self.dirichlet_applied = true;

    use rand_distr::Distribution;
    let mut thread_rng = rand::thread_rng();
    let dist = rand_distr::Gamma::new(DIRICHLET_ALPHA, 1.0).unwrap();
    // Generate noise.
    let mut noise = [0.0; POLICY_LEN];
    let mut noise_sum = 0.0;
    let mut policy_sum = 0.0;
    for m in &self.moves {
      let idx = m.to_index() as usize;
      // Generate a gamma-distributed noise value.
      let new_noise = dist.sample(&mut thread_rng);
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

// Create slotmap keys
slotmap::new_key_type! {
  pub struct NodeIndex;
  //pub struct EdgeIndex;
}

//#[derive(Clone)]
//struct MctsEdge {
//  parent:      NodeIndex,
//  child:       NodeIndex,
//}

/*
impl MctsEdge {
  fn get_edge_score(&self) -> f32 {
    match self.visits == 0 {
      true => 0.0,
      false => self.total_score / self.visits as f32,
    }
  }

  fn adjust_edge_score(&mut self, score: f32) {
    self.visits += 1;
    self.total_score += score;
  }
}
*/

#[derive(Clone)]
struct MctsNode {
  depth:           u32,
  state:           State,
  evals:           Evals,
  total_score:     f32,
  visits:          u32,
  policy_explored: f32,
  outgoing_edges:  HashMap<Move, NodeIndex>,
  gc_state:        u32,
}

impl MctsNode {
  async fn create(depth: u32, inference_engine: &InferenceEngine, state: State) -> Self {
    let evals = Evals::create(inference_engine, &state).await;
    Self {
      depth,
      state,
      evals,
      total_score: 0.0,
      visits: 0,
      policy_explored: 0.0,
      outgoing_edges: HashMap::new(),
      gc_state: 0,
    }
  }

  fn get_subtree_value(&self) -> f32 {
    self.total_score / self.visits as f32
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
    let (u, q) = match self.outgoing_edges.get(&m) {
      None => (
        (self.visits as f32).sqrt(),
        (self.get_subtree_value() - FIRST_PLAY_URGENCY * sqrt_policy_explored).max(0.0),
      ),
      Some(child_index) => {
        let child = &nodes[*child_index];
        (
          (self.visits as f32).sqrt() / (1.0 + child.visits as f32),
          child.get_subtree_value(),
        )
      }
    };
    let u = EXPLORATION_ALPHA * self.evals.posterior(m) * u;
    q + u
  }

  fn select_action(&self, nodes: &SlotMap<NodeIndex, MctsNode>) -> Option<Move> {
    if self.evals.moves.is_empty() {
      return None;
    }
    let sqrt_policy_explored = self.policy_explored.sqrt();
    let mut best_score = -std::f32::INFINITY;
    let mut best_move = None;
    for m in &self.evals.moves {
      let score = self.total_action_score(sqrt_policy_explored, nodes, *m);
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
    self.policy_explored += self.evals.posterior(m);
  }
}

fn get_transposition_table_key(state: &State, depth: u32) -> (u64, u32) {
  // We include the depth in the key to keep the tree a DAG.
  // This misses some transposition opportunities, but is very cheap.
  (state.get_transposition_table_hash(), depth)
}

pub struct Mcts<'a> {
  inference_engine:    &'a InferenceEngine,
  root:                NodeIndex,
  nodes:               SlotMap<NodeIndex, MctsNode>,
  transposition_table: HashMap<(u64, u32), NodeIndex>,
}

impl<'a> Mcts<'a> {
  pub async fn create(inference_engine: &'a InferenceEngine) -> Mcts<'a> {
    let mut nodes = SlotMap::with_key();
    let root = nodes.insert(MctsNode::create(0, inference_engine, State::starting_state()).await);
    Self {
      inference_engine,
      root,
      nodes,
      transposition_table: HashMap::new(),
    }
  }

  async fn select_principal_variation(
    &self,
    best: bool,
  ) -> (NodeIndex, Vec<NodeIndex>, Option<Move>) {
    let mut node_index = self.root;
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
        None => return (node_index, nodes, None),
        // Some means we have legal moves.
        Some(m) => {
          // Try to find this edge.
          let child = match node.outgoing_edges.get(&m) {
            None => return (node_index, nodes, Some(m)),
            Some(child) => *child,
          };
          nodes.push(child);
          node_index = child;
          node = &self.nodes[child];
        }
      }
    }
  }

  pub fn get_state(&self) -> &State {
    &self.nodes[self.root].state
  }

  pub async fn create_node(&mut self, state: State, depth: u32) -> NodeIndex {
    // Check our transposition table to see if this new state has already been reached.
    let transposition_table_key = get_transposition_table_key(&state, depth);
    match self.transposition_table.entry(transposition_table_key) {
      // If we have a transposition, just use it.
      Entry::Occupied(entry) => *entry.get(),
      Entry::Vacant(entry) => *entry
        .insert(self.nodes.insert(MctsNode::create(depth, &self.inference_engine, state).await)),
    }
  }

  pub async fn step(&mut self) {
    let root_visit_count = self.nodes[self.root].visits;
    let (pv_leaf, mut pv_nodes, pv_move) = self.select_principal_variation(false).await;
    let child_index = match pv_move {
      // If the move is null then we have no legal moves, so just propagate the score again.
      None => pv_leaf,
      // If the move is non-null then expand once at the leaf in that direction.
      Some(m) => {
        // Create the new child state.
        let mut state = self.nodes[pv_leaf].state.clone();
        state.apply_move(m).unwrap();
        let new_depth = self.nodes[pv_leaf].depth + 1;
        let child_index = self.create_node(state, new_depth).await;
        // Append the new child to the PV.
        self.nodes[pv_leaf].new_child(m, child_index);
        pv_nodes.push(child_index);
        child_index
      }
    };
    let value_score = (self.nodes[child_index].evals.outputs.value + 1.0) / 2.0;
    let is_from_whites_perspective = self.nodes[child_index].state.white_turn;
    for node_index in pv_nodes {
      let node = &mut self.nodes[node_index];
      let player_perspective_score = match node.state.white_turn == is_from_whites_perspective {
        true => value_score,
        false => 1.0 - value_score,
      };
      node.adjust_score(player_perspective_score);
    }
    // Assert that the root's edge visit count went up by one.
    assert_eq!(self.nodes[self.root].visits, root_visit_count + 1);
  }

  pub async fn step_until(&mut self, tree_size: u32, early_out: bool) -> u32 {
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
      self.step().await;
      steps_taken += 1;
    }
    steps_taken
  }

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
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let sum_child_visits = self.get_sum_child_visits(self.root);
    if sum_child_visits == 0 {
      return None;
    }
    let mut visit_count = rng.gen_range(0..sum_child_visits);
    for (m, child_index) in &self.nodes[self.root].outgoing_edges {
      visit_count -= self.nodes[*child_index].visits as i32;
      if visit_count < 0 {
        return Some(*m);
      }
    }
    panic!("Failed to sample move by visit count");
  }

  pub fn apply_noise_to_root(&mut self) {
    self.nodes[self.root].evals.add_dirichlet_noise();
  }

  pub async fn apply_move(&mut self, m: Move) {
    let root = &self.nodes[self.root];
    self.root = match root.outgoing_edges.get(&m) {
      // If we already have a node for this move, then just make it the new root.
      Some(child_index) => *child_index,
      // Otherwise, we create a new node.
      None => {
        let mut new_state = root.state.clone();
        new_state.apply_move(m).unwrap();
        self.create_node(new_state, root.depth + 1).await
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

  pub fn print_tree(&self, root: NodeIndex, desired_state: u32) {
    let mut already_printed_set = std::collections::HashSet::new();
    let mut stack = vec![(None, root, 0)];
    while !stack.is_empty() {
      let (m, node_index, depth) = stack.pop().unwrap();
      let node = &self.nodes[node_index];
      let already_printed = already_printed_set.contains(&node_index);
      already_printed_set.insert(node_index);
      println!(
        "{}{}{}node{:?} (visits={} value={} mean={}){}\x1b[0m",
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
        node.evals.outputs.value,
        node.get_subtree_value(),
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
