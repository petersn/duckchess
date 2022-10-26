use std::{
  cell::{Cell, UnsafeCell},
  collections::HashMap,
  future::Future,
  rc::{Rc, Weak},
};

use slotmap::SlotMap;

use crate::inference::{InferenceEngine, ModelOutputs};
use crate::rules::{GameOutcome, Move, State};

const EXPLORATION_ALPHA: f32 = 1.0;
const DIRICHLET_ALPHA: f32 = 0.15;
const DIRICHLET_WEIGHT: f32 = 0.25;

/*
async fn evaluate_network(inference_engine: &InferenceEngine, state: &State, evals: &mut Evals) {
  let turn_flip = if state.white_turn { 1.0 } else { -1.0 };
  evals.outcome = state.get_outcome();
  match evals.outcome {
    GameOutcome::Ongoing => {}
    GameOutcome::WhiteWin => return evals.value = turn_flip,
    GameOutcome::BlackWin => return evals.value = -turn_flip,
    GameOutcome::Draw => return evals.value = 0.0,
  }
  state.move_gen::<false>(&mut evals.moves);

  //inference_engine.predict(state, &mut evals.policy, &mut evals.value).await;

  //// Make the posterior uniform over all legal moves.
  //let value = 1.0 / evals.moves.len() as f32;
  //for m in &evals.moves {
  //  evals.policy[(m.from % 64) as usize * 64 + m.to as usize] = value;
  //}

  // TODO: Actually pass data to TensorFlow here.
  // For now, just wait a small random amount.
  //let delay = rand::random::<u64>() % 100;
  //tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
}
*/

#[derive(Clone)]
struct Evals {
  outcome: GameOutcome,
  moves:   Vec<Move>,
  outputs: ModelOutputs,
}

impl Evals {
  async fn create(worker_id: usize, inference_engine: &InferenceEngine, state: &State) -> Self {
    let mut outputs = inference_engine.predict(worker_id, state).await;
    let mut moves = vec![];
    state.move_gen::<false>(&mut moves);
    outputs.renormalize(&moves);
    // TODO: The rest of the stuff.
    Self {
      outcome: state.get_outcome(),
      moves,
      outputs,
    }
  }

  fn add_dirichlet_noise(&mut self) {
    use rand_distr::Distribution;
    let mut thread_rng = rand::thread_rng();
    let dist = rand_distr::Gamma::new(DIRICHLET_ALPHA, 1.0).unwrap();
    // Generate noise.
    let mut noise = [0.0; 64 * 64];
    for i in 0..noise.len() {
      noise[i] = dist.sample(&mut thread_rng);
    }
    // Normalize noise.
    let sum = noise.iter().sum::<f32>();
    for i in 0..noise.len() {
      noise[i] /= sum;
    }
    // Mix policy with noise.
    for i in 0..noise.len() {
      self.outputs.policy[i] =
        (1.0 - DIRICHLET_WEIGHT) * self.outputs.policy[i] + DIRICHLET_WEIGHT * noise[i];
    }
    // Assert normalization.
    let sum = self.outputs.policy.iter().sum::<f32>();
    debug_assert!((sum - 1.0).abs() < 1e-2);
  }

  fn posterior(&self, m: Move) -> f32 {
    self.outputs.policy[(m.from % 64) as usize * 64 + m.to as usize]
  }
}

// Create slotmap keys
slotmap::new_key_type! {
  pub struct NodeIndex;
  pub struct EdgeIndex;
}

#[derive(Clone)]
struct MctsEdge {
  visits:      u32,
  total_score: f32,
  parent:      NodeIndex,
  child:       NodeIndex,
}

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

#[derive(Clone)]
struct MctsNode {
  state:           State,
  evals:           Evals,
  //incoming: Option<EdgeIndex>,
  all_edge_visits: u32,
  outgoing_edges:  HashMap<Move, EdgeIndex>,
}

impl MctsNode {
  async fn create(worker_id: usize, inference_engine: &InferenceEngine, state: State) -> Self {
    let evals = Evals::create(worker_id, inference_engine, &state).await;
    Self {
      state,
      evals,
      //incoming,
      all_edge_visits: 0,
      outgoing_edges: HashMap::new(),
    }
  }

  fn total_action_score(&self, edges: &SlotMap<EdgeIndex, MctsEdge>, m: Move) -> f32 {
    let (u, q) = match self.outgoing_edges.get(&m) {
      None => ((1.0 + self.all_edge_visits as f32).sqrt(), 0.0),
      Some(edge_index) => {
        let edge = &edges[*edge_index];
        //println!("edge stats: {} / {}", edge.visits, self.all_edge_visits);
        (
          (1.0 + self.all_edge_visits as f32).sqrt() / (1.0 + edge.visits as f32),
          edge.get_edge_score(),
        )
      }
    };
    if self.evals.posterior(m) > 0.1 {
      //println!("u: {}, q: {}, posterior: {}", u, q, self.evals.posterior(m));
    }
    let u = EXPLORATION_ALPHA * self.evals.posterior(m) * u;
    q + u
  }

  fn select_action(&self, edges: &SlotMap<EdgeIndex, MctsEdge>) -> Option<Move> {
    if self.evals.moves.is_empty() {
      return None;
    }
    let mut best_score = -std::f32::INFINITY;
    let mut best_move = None;
    //println!("-------------");
    for m in &self.evals.moves {
      let score = self.total_action_score(edges, *m);
      //println!("{:?}: {}", m, score);
      if score > best_score {
        best_score = score;
        best_move = Some(*m);
      }
    }
    best_move
  }
}

pub struct Mcts<'a> {
  worker_id:        usize,
  inference_engine: &'a InferenceEngine,
  root:             NodeIndex,
  nodes:            SlotMap<NodeIndex, MctsNode>,
  edges:            SlotMap<EdgeIndex, MctsEdge>,
}

impl<'a> Mcts<'a> {
  pub async fn create(worker_id: usize, inference_engine: &'a InferenceEngine) -> Mcts<'a> {
    let mut node = MctsNode::create(worker_id, inference_engine, State::starting_state()).await;
    node.evals.add_dirichlet_noise();
    let mut nodes = SlotMap::with_key();
    let root = nodes.insert(node);
    Self {
      worker_id,
      inference_engine,
      root,
      nodes,
      edges: SlotMap::with_key(),
    }
  }

  async fn select_principal_variation(
    &self,
    best: bool,
  ) -> (NodeIndex, Vec<EdgeIndex>, Option<Move>) {
    let mut node_index = self.root;
    let mut node = &self.nodes[self.root];
    let mut edges = vec![];
    loop {
      //println!("got here");
      let m: Option<Move> = match best {
        true => node
          .outgoing_edges
          .iter()
          .max_by_key(|(_, edge_index)| self.edges[**edge_index].visits)
          .map(|(m, _)| *m),
        false => node.select_action(&self.edges),
      };
      match m {
        None => return (node_index, edges, None),
        Some(m) => {
          // Try to find this edge.
          let edge_index = match node.outgoing_edges.get(&m) {
            None => return (node_index, edges, Some(m)),
            Some(edge_index) => edge_index,
          };
          edges.push(*edge_index);
          let edge = &self.edges[*edge_index];
          node_index = edge.child;
          node = &self.nodes[edge.child];
        }
      }
    }
  }

  pub fn get_state(&self) -> &State {
    &self.nodes[self.root].state
  }

  pub async fn step(&mut self) {
    let root_all_edge_visits = self.nodes[self.root].all_edge_visits;
    let (pv_leaf, mut pv_edges, pv_move) = self.select_principal_variation(false).await;
    let new_node = match pv_move {
      None => {
        // If the move is null then we have no legal moves, so just propagate the score again.
        self.nodes[pv_leaf].all_edge_visits += 1;
        pv_leaf
      }
      Some(m) => {
        // If the move is non-null then expand once at the leaf.
        //println!("Expanding depth={} move={:?}", pv_edges.len(), m);
        let mut state = self.nodes[pv_leaf].state.clone();
        state.apply_move(m);
        let child =
          self.nodes.insert(MctsNode::create(self.worker_id, &self.inference_engine, state).await);
        let edge_index = self.edges.insert(MctsEdge {
          visits: 0,
          total_score: 0.0,
          parent: pv_leaf,
          child,
        });
        self.nodes[pv_leaf].outgoing_edges.insert(m, edge_index);
        pv_edges.push(edge_index);
        child
      }
    };
    let mut value_score = (self.nodes[new_node].evals.outputs.value + 1.0) / 2.0;
    for edge_index in pv_edges.iter().rev() {
      // Alternate the value score, since it's from the current player's perspective.
      value_score = 1.0 - value_score;
      let edge = &mut self.edges[*edge_index];
      edge.adjust_edge_score(value_score);
      let parent = &mut self.nodes[edge.parent];
      parent.all_edge_visits += 1;
    }
    // Assert that the root's edge visit count went up by one.
    assert_eq!(self.nodes[self.root].all_edge_visits, root_all_edge_visits + 1);
  }

  pub async fn step_until(&mut self, tree_size: u32) -> u32 {
    let mut steps_taken = 0;
    loop {
      // Find the most and second most visited edge counts at the root.
      let mut best_edge_visits = 0;
      let mut second_best_edge_visits = 0;
      for (_, edge_index) in &self.nodes[self.root].outgoing_edges {
        let edge = &self.edges[*edge_index];
        if edge.visits > best_edge_visits {
          second_best_edge_visits = best_edge_visits;
          best_edge_visits = edge.visits;
        } else if edge.visits > second_best_edge_visits {
          second_best_edge_visits = edge.visits;
        }
      }
      // Compute the greatest possible number of additional steps we might take.
      let maximum_steps = tree_size as i64 - self.nodes[self.root].all_edge_visits as i64;
      if maximum_steps > 2000 {
        panic!("maximum_steps={}", maximum_steps);
      }
      if maximum_steps <= 0 {
        break;
      }
      // Check if there are enough steps left for the second best to surpass the best.
      if second_best_edge_visits as i64 + maximum_steps < best_edge_visits as i64 {
        break;
      }
      // Step.
      self.step().await;
      steps_taken += 1;
      if steps_taken > 2000 {
        panic!("steps_taken={}", steps_taken);
      }
    }
    steps_taken
  }

  pub fn select_train_move(&self) -> Option<Move> {
    // Pick the most viisted move.
    let mut best_visits = 0;
    let mut best_move = None;
    for (m, edge_index) in &self.nodes[self.root].outgoing_edges {
      let edge = &self.edges[*edge_index];
      if edge.visits > best_visits {
        best_visits = edge.visits;
        best_move = Some(*m);
      }
    }
    best_move
  }

  pub fn sample_move_by_visit_count(&self) -> Option<Move> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut total_visits: i32 = 0;
    //println!("-------- {}", self.nodes[self.root.0].outgoing_edges.len());
    for (_, edge_index) in &self.nodes[self.root].outgoing_edges {
      total_visits += self.edges[*edge_index].visits as i32;
    }
    if total_visits == 0 {
      return None;
    }
    let mut visit_count = rng.gen_range(0..total_visits);
    for (m, edge_index) in &self.nodes[self.root].outgoing_edges {
      visit_count -= self.edges[*edge_index].visits as i32;
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
    match self.nodes[self.root].outgoing_edges.get(&m) {
      // If we already have a node for this move, then just make it the new root.
      Some(edge_index) => {
        let pre_gc_size = self.nodes.len();
        let new_root = self.edges[*edge_index].child;
        // We now perform garbage collection by recursively deleting
        // everything, but stopping at the new root.
        fn delete(this: &mut Mcts, new_root: NodeIndex, node_index: NodeIndex) {
          if node_index == new_root {
            return;
          }
          let edges_to_delete =
            this.nodes[node_index].outgoing_edges.values().copied().collect::<Vec<_>>();
          for delete_edge_index in edges_to_delete {
            let edge = &this.edges[delete_edge_index];
            delete(this, new_root, edge.child);
            this.edges.remove(delete_edge_index);
          }
          this.nodes.remove(node_index);
        }
        delete(self, new_root, self.root);
        self.root = new_root;
        //println!("GC'd {} nodes", pre_gc_size - self.nodes.len());
      }
      // Otherwise, we throw everything away.
      None => {
        let mut new_state = self.nodes[self.root].state.clone();
        new_state.apply_move(m);
        self.nodes.clear();
        self.edges.clear();
        self.root = self
          .nodes
          .insert(MctsNode::create(self.worker_id, &mut self.inference_engine, new_state).await);
      }
    }
  }

  pub fn print_tree(&self) {
    let mut queue = vec![(self.root, 0)];
    while !queue.is_empty() {
      let (node_index, depth) = queue.pop().unwrap();
      let node = &self.nodes[node_index];
      println!("{}node{:?}", "  ".repeat(depth), node_index);
      for (m, edge_index) in &node.outgoing_edges {
        let edge = &self.edges[*edge_index];
        println!(
          "{}edge={:?} move={:?} visits={} score={}",
          "  ".repeat(depth + 1),
          edge_index.0,
          m,
          edge.visits,
          edge.total_score / (edge.visits as f32 + 1e-6),
        );
        queue.push((edge.child, depth + 1));
      }
    }
  }
}
