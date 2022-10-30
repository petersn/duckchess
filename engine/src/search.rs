use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};

use crate::rng::Rng;
use crate::rules::{iter_bits, GameOutcome, Move, State};

type Evaluation = i32;

const VERY_NEGATIVE_EVAL: Evaluation = -1_000_000_000;
const VERY_POSITIVE_EVAL: Evaluation = 1_000_000_000;

#[rustfmt::skip]
const PAWN_MIDDLEGAME_PST: [Evaluation; 64] = [
   0,  0,  0,  0,  0,  0,  0,  0,
  50, 50, 50, 50, 50, 50, 50, 50,
  10, 10, 20, 30, 30, 20, 10, 10,
   5,  5, 10, 25, 25, 10,  5,  5,
   0,  0,  0, 20, 20,  0,  0,  0,
   5, -5,-10,  0,  0,-10, -5,  5,
   5, 10, 10,-20,-20, 10, 10,  5,
   0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const PAWN_ENDGAME_PST: [Evaluation; 64] = [
   0,  0,  0,  0,  0,  0,  0,  0,
  99, 99, 99, 99, 99, 99, 99, 99,
  40, 40, 60, 70, 70, 60, 40, 40,
  35, 35, 40, 55, 55, 40, 35, 35,
  20, 20, 20, 40, 40, 20, 20, 20,
  15,  5,  0, 10, 10,  0,  5, 15,
   5, 10, 10,-20,-20, 10, 10,  5,
   0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const KNIGHT_PST: [Evaluation; 64] = [
  -50,-40,-30,-30,-30,-30,-40,-50,
  -40,-20,  0,  0,  0,  0,-20,-40,
  -30,  0, 10, 15, 15, 10,  0,-30,
  -30,  5, 15, 20, 20, 15,  5,-30,
  -30,  0, 15, 20, 20, 15,  0,-30,
  -30,  5, 10, 15, 15, 10,  5,-30,
  -40,-20,  0,  5,  5,  0,-20,-40,
  -50,-40,-30,-30,-30,-30,-40,-50,
];

#[rustfmt::skip]
const BISHOP_PST: [Evaluation; 64] = [
  -20,-10,-10,-10,-10,-10,-10,-20,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -10,  0,  5, 10, 10,  5,  0,-10,
  -10,  5,  5, 10, 10,  5,  5,-10,
  -10,  0, 10, 10, 10, 10,  0,-10,
  -10, 10, 10, 10, 10, 10, 10,-10,
  -10,  5,  0,  0,  0,  0,  5,-10,
  -20,-10,-10,-10,-10,-10,-10,-20,
];

#[rustfmt::skip]
const ROOK_PST: [Evaluation; 64] = [
   0,  0,  0,  0,  0,  0,  0,  0,
   5, 10, 10, 10, 10, 10, 10,  5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
   0,  0,  0,  5,  5,  0,  0,  0,
];

#[rustfmt::skip]
const QUEEN_PST: [Evaluation; 64] = [
  -20,-10,-10, -5, -5,-10,-10,-20,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -10,  0,  5,  5,  5,  5,  0,-10,
   -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
  -10,  5,  5,  5,  5,  5,  0,-10,
  -10,  0,  5,  0,  0,  0,  0,-10,
  -20,-10,-10, -5, -5,-10,-10,-20,
];

#[rustfmt::skip]
const KING_MIDDLEGAME_PST: [Evaluation; 64] = [
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -20,-30,-30,-40,-40,-30,-30,-20,
  -10,-20,-20,-20,-20,-20,-20,-10,
   20, 20,  0,  0,  0,  0, 20, 20,
   20, 30, 10,  0,  0, 10, 30, 20,
];

#[rustfmt::skip]
const KING_ENDGAME_PST: [Evaluation; 64] = [
  -50,-40,-30,-20,-20,-30,-40,-50,
  -30,-20,-10,  0,  0,-10,-20,-30,
  -30,-10, 20, 30, 30, 20,-10,-30,
  -30,-10, 30, 40, 40, 30,-10,-30,
  -30,-10, 30, 40, 40, 30,-10,-30,
  -30,-10, 20, 30, 30, 20,-10,-30,
  -30,-30,  0,  0,  0,  0,-30,-30,
  -50,-30,-30,-30,-30,-30,-30,-50,
];

pub fn evaluate_state(state: &State) -> Evaluation {
  let mut score = 0;
  // endgame factor is 0 for middlegame, 1 for endgame
  let endgame_factor = 1.0
    - (2 * state.queens[0].0.count_ones()
      + 2 * state.queens[1].0.count_ones()
      + state.rooks[0].0.count_ones()
      + state.rooks[1].0.count_ones()) as f32
      / 8.0;
  //let king_pst = if is_endgame {
  //  KING_ENDGAME_PST
  //} else {
  //  KING_MIDDLEGAME_PST
  //};
  for (pst_mult, piece_value, pst, piece_array) in [
    (1.0 - endgame_factor, 50, PAWN_MIDDLEGAME_PST, &state.pawns),
    (endgame_factor, 50, PAWN_ENDGAME_PST, &state.pawns),
    (1.0, 450, KNIGHT_PST, &state.knights),
    (1.0, 250, BISHOP_PST, &state.bishops),
    (1.0, 500, ROOK_PST, &state.rooks),
    (1.0, 900, QUEEN_PST, &state.queens),
    (
      1.0 - endgame_factor,
      1_000_000,
      KING_MIDDLEGAME_PST,
      &state.kings,
    ),
    (endgame_factor, 1_000_000, KING_ENDGAME_PST, &state.kings),
  ] {
    let (mut us, mut them, pst_xor) = match state.white_turn {
      //let (mut us, mut them, pst_xor) = match true {
      true => (piece_array[1].0, piece_array[0].0, 0),
      false => (piece_array[0].0, piece_array[1].0, 56),
    };
    score += us.count_ones() as Evaluation * piece_value;
    score -= them.count_ones() as Evaluation * piece_value;
    while let Some(pos) = iter_bits(&mut us) {
      score += (pst_mult * pst[(pos ^ pst_xor ^ 56) as usize] as f32) as Evaluation;
    }
    while let Some(pos) = iter_bits(&mut them) {
      score -= (pst_mult * pst[(pos ^ pst_xor) as usize] as f32) as Evaluation;
    }
  }
  score + 25
}

const QUIESCENCE_DEPTH: u16 = 10;

fn make_terminal_scores_slightly_less_extreme<T>(p: (Evaluation, T)) -> (Evaluation, T) {
  let (score, m) = p;
  let score = if score < -100_000 {
    score + 1
  } else if score > 100_000 {
    score - 1
  } else {
    score
  };
  (score, m)
}

fn make_terminal_scores_much_less_extreme<T>(p: (Evaluation, T)) -> (Evaluation, T) {
  let (score, m) = p;
  let score = if score < -100_000 {
    score + 100
  } else if score > 100_000 {
    score - 100
  } else {
    score
  };
  (score, m)
}

pub struct Engine {
  nodes_searched:   u64,
  rng:              Rng,
  state:            State,
  move_order_table: HashMap<u64, Move>,
  killer_moves:     [Option<Move>; 100],
}

impl Engine {
  pub fn new(seed: u64) -> Self {
    Self {
      nodes_searched:   0,
      rng:              Rng::new(seed),
      state:            State::starting_state(),
      move_order_table: HashMap::new(),
      killer_moves:     [None; 100],
    }
  }

  pub fn get_state(&self) -> &State {
    &self.state
  }

  pub fn get_state_mut(&mut self) -> &mut State {
    &mut self.state
  }

  pub fn set_state(&mut self, state: State) {
    self.state = state;
  }

  pub fn apply_move(&mut self, m: Move) -> Result<(), &'static str> {
    self.state.apply_move(m)
  }

  pub fn get_moves(&self) -> Vec<Move> {
    let mut moves = vec![];
    self.state.move_gen::<false>(&mut moves);
    moves
  }

  pub fn get_outcome(&self) -> GameOutcome {
    self.state.get_outcome()
  }

  pub fn run(&mut self, depth: u16) -> (Evaluation, (Option<Move>, Option<Move>)) {
    self.nodes_searched = 0;
    let start_state = self.state.clone();
    // Apply iterative deepening.
    let mut p = (0, (None, None));
    for d in 1..=depth {
      p = self.pvs::<false>(d, &start_state, VERY_NEGATIVE_EVAL, VERY_POSITIVE_EVAL);
      //log(&format!(
      //  "Depth {}: {} (nodes={})",
      //  d, p.0, self.nodes_searched
      //));
    }
    p
  }

  fn pvs<const QUIESCENCE: bool>(
    &mut self,
    depth: u16,
    state: &State,
    mut alpha: Evaluation,
    beta: Evaluation,
  ) -> (Evaluation, (Option<Move>, Option<Move>)) {
    self.nodes_searched += 1;
    let game_over = state.is_game_over();
    let random_bonus = (self.rng.next_random() & 0xf) as Evaluation;
    match (game_over, depth, QUIESCENCE) {
      (true, _, _) => return (evaluate_state(state) + random_bonus, (None, None)),
      (_, 0, true) => return (evaluate_state(state) + random_bonus, (None, None)),
      (_, 0, false) => {
        return make_terminal_scores_much_less_extreme(self.pvs::<true>(
          QUIESCENCE_DEPTH,
          state,
          alpha,
          beta,
        ))
      }
      _ => {}
    }

    let mut moves = Vec::new();
    state.move_gen::<QUIESCENCE>(&mut moves);

    // If we're in a quiescence search and have quiesced, then return.
    if QUIESCENCE && moves.is_empty() {
      return (evaluate_state(state) + random_bonus, (None, None));
    }
    assert!(!moves.is_empty());

    // Reorder based on our move order table.
    let state_hash: u64 = state.get_transposition_table_hash();

    let mot_move = match QUIESCENCE {
      false => self.move_order_table.get(&state_hash),
      true => None,
    };
    let killer_move = match QUIESCENCE {
      false => self.killer_moves[depth as usize],
      true => None,
    };

    if mot_move.is_some() || killer_move.is_some() {
      moves.sort_by(|a, b| {
        let mut a_score = 0;
        let mut b_score = 0;
        if let Some(mot_move) = mot_move {
          if a == mot_move {
            a_score += 1;
          }
          if b == mot_move {
            b_score += 1;
          }
        }
        if let Some(killer_move) = killer_move {
          if *a == killer_move {
            a_score += 1;
          }
          if *b == killer_move {
            b_score += 1;
          }
        }
        b_score.cmp(&a_score)
      });
    }
    //moves.sort_by(|_, _| {
    //  self.next_random().cmp(&self.next_random())
    //});

    //log(&format!("pvs({}, {}, {}) moves={}", depth, alpha, beta, moves.len()));
    let mut best_score = VERY_NEGATIVE_EVAL;
    let mut best_pair = (None, None);

    // If we're in a QUIESCENCE search then we're allowed to pass.
    if QUIESCENCE {
      alpha = alpha.max(evaluate_state(state) + random_bonus);
      if alpha >= beta {
        moves.clear();
      }
    }

    let mut first = true;
    for m in moves {
      let mut new_state = state.clone();
      new_state.apply_move(m).unwrap();

      let mut score;
      let mut next_pair;

      // Two cases:
      // If new_state is a duck move state, we *don't* invert the score, as we take the next move.

      if new_state.is_duck_move {
        if first {
          (score, next_pair) = self.pvs::<QUIESCENCE>(depth - 1, &new_state, alpha, beta);
        } else {
          (score, next_pair) = self.pvs::<QUIESCENCE>(depth - 1, &new_state, alpha, alpha + 1);
          if alpha < score && score < beta {
            (score, next_pair) = self.pvs::<QUIESCENCE>(depth - 1, &new_state, score, beta);
          }
        }
      } else {
        if first {
          (score, next_pair) = self.pvs::<QUIESCENCE>(depth - 1, &new_state, -beta, -alpha);
          score *= -1;
        } else {
          (score, next_pair) = self.pvs::<QUIESCENCE>(depth - 1, &new_state, -alpha - 1, -alpha);
          score *= -1;
          if alpha < score && score < beta {
            (score, next_pair) = self.pvs::<QUIESCENCE>(depth - 1, &new_state, -beta, -score);
            score *= -1;
          }
        }
      }
      // TODO: For some reason adding the random bonus in here makes play awful??
      let comparison_score = score; // + random_bonus;
      if comparison_score > best_score {
        best_score = comparison_score;
        best_pair = (Some(m), next_pair.0);
      }
      if score > alpha && !QUIESCENCE {
        self.move_order_table.insert(state_hash, m);
      }
      alpha = alpha.max(score);
      if alpha >= beta {
        self.killer_moves[depth as usize] = Some(m);
        break;
      }
      first = false;
    }

    make_terminal_scores_slightly_less_extreme((alpha, best_pair))
  }
}
