use std::sync::{Weak, Arc};

use crate::{Move, State};

async fn evaluate_network(state: &State, evals: &mut evals) {
  self.state.move_gen::<false>(evals.moves);
  // TODO: Actually pass data to TensorFlow here.
  // For now, just wait a small random amount.
  let delay = rand::random::<u64>() % 100;
  tokio::time::sleep(Duration::from_millis(delay)).await;
}

struct Evals {
  outcome: GameOutcome,
  moves: Vec<Move>,
  policy: [f32; 64 * 64],
  value: f32,
}

impl Evals {
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
    match visits == 0 {
      true => 0.0,
      false => total_score / visits as f32,
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
    let (u, Q) = match outgoing_edges.get(m) {
      None => (
        (1.0 + self.all_edge_visits).sqrt(),
        0.0,
      ),
      Some(edge) => (
        (1.0 + self.all_edge_visits).sqrt() / (1.0 + edge.visits),
        edge.get_edge_score(),
      ),
    };
    let u = EXPLORATION_ALPHA * self.evals.unwrap().posterior(m) * u;
    Q + u
  }

  fn populate_evals(&mut self) ->  {
    if self.evals.is_none() {
      self.evals = Some(self.state.evaluate());
    }
  }
}

struct Mcts {
  root: MctsNode,
  
}
