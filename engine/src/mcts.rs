use std::{sync::{Weak, Arc}, collections::HashMap, future::Future};

use crate::rules::{Move, State, GameOutcome};

const EXPLORATION_ALPHA: f32 = 5.0;

async fn evaluate_network(state: &State, evals: &mut Evals) {
  state.move_gen::<false>(&mut evals.moves);
  // TODO: Actually pass data to TensorFlow here.
  // For now, just wait a small random amount.
  let delay = rand::random::<u64>() % 100;
  tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
}

struct Evals {
  outcome: GameOutcome,
  moves: Vec<Move>,
  policy: [f32; 64 * 64],
  value: f32,
}

impl Evals {
  fn new() -> Self {
    Self {
      outcome: GameOutcome::Ongoing,
      moves: Vec::new(),
      policy: [0.0; 64 * 64],
      value: 0.0,
    }
  }

  fn posterior(&self, m: Move) -> f32 {
    self.policy[(m.from % 64) as usize * 64 + m.to as usize]
  }
}

struct MctsEdge {
  visits: u32,
  total_score: f32,
  child: Arc<MctsNode>,
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

struct MctsNode {
  state: State,
  evals: Option<Evals>,
  parent: Weak<MctsNode>,
  all_edge_visits: u32,
  outgoing_edges: HashMap<Move, MctsEdge>,
}

impl MctsNode {
  fn total_action_score(&self, m: Move) -> f32 {
    let (u, Q) = match self.outgoing_edges.get(&m) {
      None => (
        (1.0 + self.all_edge_visits as f32).sqrt(),
        0.0,
      ),
      Some(edge) => (
        (1.0 + self.all_edge_visits as f32).sqrt() / (1.0 + edge.visits as f32),
        edge.get_edge_score(),
      ),
    };
    let u = EXPLORATION_ALPHA * self.evals.unwrap().posterior(m) * u;
    Q + u
  }

  async fn get_evals(&mut self) -> &Evals {
    match self.evals.get_or_insert_with(Evals::new) {
      None => {
        let e = self.evals.insert(Evals::new());
        evaluate_network(&self.state, e).await;
        e
      }
      Some(e) => e,
    }
  }
}

struct Mcts {
  root: MctsNode,
  
}
