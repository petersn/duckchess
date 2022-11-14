use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};

use crate::nnue::{Nnue, UndoCookie};
use crate::rng::Rng;
use crate::rules::{iter_bits, GameOutcome, Move, Player, State};

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

// #[inline(never)]
// pub fn evaluate_state(state: &State) -> Evaluation {
//   let endgame_factor = 1.0
//     - (2 * state.queens[0].0.count_ones()
//       + 2 * state.queens[1].0.count_ones()
//       + state.rooks[0].0.count_ones()
//       + state.rooks[1].0.count_ones()) as f32
//       / 8.0;
//   let adjust = (endgame_factor * 10.0) as i32;
//   return state.pawns[0].0.count_ones() as i32 - state.pawns[1].0.count_ones() as i32 + adjust;
// }

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
    let (mut us, mut them, pst_xor) = match state.turn {
      //let (mut us, mut them, pst_xor) = match true {
      Player::White => (piece_array[1].0, piece_array[0].0, 0),
      Player::Black => (piece_array[0].0, piece_array[1].0, 56),
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

enum NodeType {
  Exact,
  LowerBound,
  UpperBound,
}

struct TTEntry {
  zobrist:   u64,
  depth:     u16,
  score:     Evaluation,
  best_move: Option<Move>,
  node_type: NodeType,
}

pub struct Engine {
  pub nodes_searched:  u64,
  pub total_eval:      f32,
  rng:                 Rng,
  state:               State,
  transposition_table: Vec<TTEntry>,
  killer_moves:        [Option<Move>; 100],
}

impl Engine {
  pub fn new(seed: u64, tt_size: usize) -> Self {
    let state = State::starting_state();
    let nnue = Nnue::new(&state);
    let mut transposition_table = Vec::with_capacity(tt_size);
    for _ in 0..tt_size {
      transposition_table.push(TTEntry {
        zobrist:   0,
        depth:     0,
        score:     0,
        best_move: None,
        node_type: NodeType::Exact,
      });
    }
    Self {
      nodes_searched: 0,
      total_eval: 0.0,
      rng: Rng::new(seed),
      state,
      transposition_table,
      killer_moves: [None; 100],
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

  pub fn apply_move(&mut self, m: Move) -> Result<UndoCookie, &'static str> {
    self.state.apply_move::<false>(m, None)
  }

  pub fn get_moves(&self) -> Vec<Move> {
    let mut moves = vec![];
    self.state.move_gen::<false>(&mut moves);
    moves
  }

  pub fn get_outcome(&self) -> Option<GameOutcome> {
    self.state.get_outcome()
  }

  pub fn run(&mut self, depth: u16) -> (Evaluation, (Option<Move>, Option<Move>)) {
    self.nodes_searched = 0;
    let start_state = self.state.clone();
    let mut nnue = Nnue::new(&start_state);
    // Apply iterative deepening.
    let mut p = (0, (None, None));
    //for d in 1..=depth {
    for d in 1..=depth {
      p = self.pvs::<false, false>(
        d,
        &start_state,
        &mut nnue,
        VERY_NEGATIVE_EVAL,
        VERY_POSITIVE_EVAL,
      );
      //log(&format!(
      //  "Depth {}: {} (nodes={})",
      //  d, p.0, self.nodes_searched
      //));
    }
    p
  }

  fn tt_lookup(&self, state: &State) -> Option<&TTEntry> {
    let index = (state.zobrist % self.transposition_table.len() as u64) as usize;
    let entry = &self.transposition_table[index];
    match entry.zobrist == state.zobrist {
      true => Some(entry),
      false => None,
    }
  }

  fn tt_insert(
    &mut self,
    zobrist: u64,
    depth: u16,
    score: Evaluation,
    best_move: Option<Move>,
    node_type: NodeType,
  ) {
    let index = (zobrist % self.transposition_table.len() as u64) as usize;
    let entry = &mut self.transposition_table[index];
    entry.zobrist = zobrist;
    entry.depth = depth;
    entry.score = score;
    entry.best_move = best_move;
    entry.node_type = node_type;
  }

  fn pvs<const QUIESCENCE: bool, const NNUE: bool>(
    &mut self,
    depth: u16,
    state: &State,
    nnue: &mut Nnue,
    mut alpha: Evaluation,
    beta: Evaluation,
  ) -> (Evaluation, (Option<Move>, Option<Move>)) {
    // Evaluate the nnue to get a score.
    let nnue_evaluation = if NNUE {
      nnue.evaluate(state);
      nnue.value
    } else {
      0
    };
    let get_eval = || {
      if NNUE {
        nnue_evaluation
      } else {
        evaluate_state(state)
      }
    };

    let game_over = state.get_outcome().is_some();
    let random_bonus = (self.rng.next_random() & 0xf) as Evaluation;
    match (game_over, depth, QUIESCENCE) {
      (true, _, _) => return (get_eval() + random_bonus, (None, None)),
      (_, 0, true) => return (get_eval() + random_bonus, (None, None)),
      (_, 0, false) => {
        return make_terminal_scores_much_less_extreme(self.pvs::<true, NNUE>(
          QUIESCENCE_DEPTH,
          state,
          nnue,
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
      return (get_eval() + random_bonus, (None, None));
    }
    assert!(!moves.is_empty());

    // Reorder based on our move order table.
    let state_hash: u64 = state.get_transposition_table_hash();

    let mot_move: Option<&Move> = None;
    //let mot_move = match QUIESCENCE {
    //  false => self.move_order_table.get(&state_hash),
    //  true => None,
    //};
    let killer_move = match QUIESCENCE {
      false => self.killer_moves[depth as usize],
      true => None,
    };

    //if mot_move.is_some() || killer_move.is_some() {
    moves.sort_by(|a, b| {
      let mut a_score = (100.0 * nnue.outputs[a.to as usize]) as i32;
      let mut b_score = (100.0 * nnue.outputs[b.to as usize]) as i32;
      if let Some(mot_move) = mot_move {
        if a == mot_move {
          a_score += 10_000;
        }
        if b == mot_move {
          b_score += 10_000;
        }
      }
      if let Some(killer_move) = killer_move {
        if *a == killer_move {
          a_score += 10_000;
        }
        if *b == killer_move {
          b_score += 10_000;
        }
      }
      b_score.cmp(&a_score)
    });
    //moves.sort_by(|_, _| {
    //  self.next_random().cmp(&self.next_random())
    //});

    //log(&format!("pvs({}, {}, {}) moves={}", depth, alpha, beta, moves.len()));
    let mut best_score = VERY_NEGATIVE_EVAL;
    let mut best_pair = (None, None);

    // If we're in a QUIESCENCE search then we're allowed to pass.
    if QUIESCENCE {
      alpha = alpha.max(nnue_evaluation + random_bonus);
      if alpha >= beta {
        moves.clear();
      }
    }

    let mut first = true;
    for m in moves {
      self.nodes_searched += 1;
      let mut new_state = state.clone();
      let undo_cookie = if NNUE {
        new_state.apply_move::<true>(m, Some(nnue)).unwrap()
      } else {
        new_state.apply_move::<false>(m, None).unwrap()
      };
      //nnue.undo(undo_cookie);
      //let debugging_hash = nnue.get_debugging_hash();
      //println!("Undo debugging hash: {:016x}", debugging_hash);

      let mut score;
      let mut next_pair;

      // Two cases:
      // If new_state is a duck move state, we *don't* invert the score, as we take the next move.

      if new_state.is_duck_move {
        if first {
          (score, next_pair) =
            self.pvs::<QUIESCENCE, NNUE>(depth - 1, &new_state, nnue, alpha, beta);
        } else {
          (score, next_pair) =
            self.pvs::<QUIESCENCE, NNUE>(depth - 1, &new_state, nnue, alpha, alpha + 1);
          if alpha < score && score < beta {
            (score, next_pair) =
              self.pvs::<QUIESCENCE, NNUE>(depth - 1, &new_state, nnue, score, beta);
          }
        }
      } else {
        if first {
          (score, next_pair) =
            self.pvs::<QUIESCENCE, NNUE>(depth - 1, &new_state, nnue, -beta, -alpha);
          score *= -1;
        } else {
          (score, next_pair) =
            self.pvs::<QUIESCENCE, NNUE>(depth - 1, &new_state, nnue, -alpha - 1, -alpha);
          score *= -1;
          if alpha < score && score < beta {
            (score, next_pair) =
              self.pvs::<QUIESCENCE, NNUE>(depth - 1, &new_state, nnue, -beta, -score);
            score *= -1;
          }
        }
      }

      if NNUE {
        nnue.undo(undo_cookie);
      }
      //assert_eq!(debugging_hash, nnue.get_debugging_hash());
      //if state.is_duck_move {
      //  let eval = nnue.evaluate().expected_score;
      //  self.total_eval += eval;
      //}

      // TODO: For some reason adding the random bonus in here makes play awful??
      let comparison_score = score; // + random_bonus;
      if comparison_score > best_score {
        best_score = comparison_score;
        best_pair = (Some(m), next_pair.0);
      }
      if score > alpha && !QUIESCENCE {
        //self.tt_insert(state_hash, m, score, depth);
        //self.move_order_table.insert(state_hash, m);
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
