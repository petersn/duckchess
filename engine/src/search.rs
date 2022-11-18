use crate::nnue::{Nnue, NnueAdjustment};
use crate::rng::Rng;
use crate::rules::{iter_bits, GameOutcome, Move, Player, State};

// Represents an evaluation in a position from the perspective of the player to move.
pub type IntEvaluation = i32;

const EVAL_DRAW: IntEvaluation = 0;
const EVAL_VERY_NEGATIVE: IntEvaluation = -1_000_000_000;
const EVAL_VERY_POSITIVE: IntEvaluation = 1_000_000_000;
const EVAL_LOSS: IntEvaluation = -1_000_000;
const EVAL_WIN: IntEvaluation = 1_000_000;

pub fn eval_terminal_state(state: &State) -> Option<IntEvaluation> {
  Some(match (state.get_outcome()?, state.turn) {
    (GameOutcome::Draw, _) => EVAL_DRAW,
    (GameOutcome::Win(Player::White), Player::White) => EVAL_WIN,
    (GameOutcome::Win(Player::White), Player::Black) => EVAL_LOSS,
    (GameOutcome::Win(Player::Black), Player::White) => EVAL_LOSS,
    (GameOutcome::Win(Player::Black), Player::Black) => EVAL_WIN,
  })
}

pub fn make_mate_score_slightly_less_extreme(eval: IntEvaluation) -> IntEvaluation {
  if eval > 100_000 {
    eval - 1
  } else if eval < -100_000 {
    eval + 1
  } else {
    eval
  }
}

pub fn make_mate_score_much_less_extreme(eval: IntEvaluation) -> IntEvaluation {
  if eval > 100_000 {
    eval - 100
  } else if eval < -100_000 {
    eval + 100
  } else {
    eval
  }
}

#[rustfmt::skip]
const PAWN_MIDDLEGAME_PST: [IntEvaluation; 64] = [
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
const PAWN_ENDGAME_PST: [IntEvaluation; 64] = [
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
const KNIGHT_PST: [IntEvaluation; 64] = [
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
const BISHOP_PST: [IntEvaluation; 64] = [
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
const ROOK_PST: [IntEvaluation; 64] = [
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
const QUEEN_PST: [IntEvaluation; 64] = [
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
const KING_MIDDLEGAME_PST: [IntEvaluation; 64] = [
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
const KING_ENDGAME_PST: [IntEvaluation; 64] = [
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
// pub fn evaluate_state(state: &State) -> IntEvaluation {
//   let endgame_factor = 1.0
//     - (2 * state.queens[0].0.count_ones()
//       + 2 * state.queens[1].0.count_ones()
//       + state.rooks[0].0.count_ones()
//       + state.rooks[1].0.count_ones()) as f32
//       / 8.0;
//   let adjust = (endgame_factor * 10.0) as i32;
//   return state.pawns[0].0.count_ones() as i32 - state.pawns[1].0.count_ones() as i32 + adjust;
// }

pub fn evaluate_state(state: &State) -> IntEvaluation {
  if let Some(terminal_eval) = eval_terminal_state(state) {
    return terminal_eval;
  }

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
    score += us.count_ones() as IntEvaluation * piece_value;
    score -= them.count_ones() as IntEvaluation * piece_value;
    while let Some(pos) = iter_bits(&mut us) {
      score += (pst_mult * pst[(pos ^ pst_xor ^ 56) as usize] as f32) as IntEvaluation;
    }
    while let Some(pos) = iter_bits(&mut them) {
      score -= (pst_mult * pst[(pos ^ pst_xor) as usize] as f32) as IntEvaluation;
    }
  }
  score + 25
}

const QUIESCENCE_DEPTH: u16 = 10;

enum NodeType {
  Exact,
  LowerBound,
  UpperBound,
}

struct MoveOrderEntry {
  m: Move,
}

struct TTEntry {
  zobrist:   u64,
  depth:     u16,
  score:     IntEvaluation,
  best_move: Option<Move>,
  node_type: NodeType,
}

pub struct Engine {
  pub nodes_searched:  u64,
  pub total_eval:      f32,
  rng:                 Rng,
  state:               State,
  move_order_table:    Vec<MoveOrderEntry>,
  transposition_table: Vec<TTEntry>,
  killer_moves:        [Option<Move>; 100],
}

impl Engine {
  pub fn new(seed: u64, tt_size: usize) -> Self {
    let state = State::starting_state();
    let mut move_order_table = Vec::with_capacity(tt_size);
    for _ in 0..tt_size {
      move_order_table.push(MoveOrderEntry {
        m: Move::INVALID,
      });
    }
    let mut transposition_table = Vec::with_capacity(tt_size);
    for _ in 0..tt_size {
      transposition_table.push(TTEntry {
        zobrist:   0,
        depth:     0,
        score:     EVAL_DRAW,
        best_move: None,
        node_type: NodeType::Exact,
      });
    }
    Self {
      nodes_searched: 0,
      total_eval: 0.0,
      rng: Rng::new(seed),
      state,
      move_order_table,
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

  pub fn apply_move(&mut self, m: Move) -> Result<NnueAdjustment, &'static str> {
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

  pub fn run(&mut self, depth: u16, use_nnue: bool) -> (IntEvaluation, (Option<Move>, Option<Move>)) {
    self.nodes_searched = 0;
    let start_state = self.state.clone();
    let mut nnue = Nnue::new(&start_state, crate::nnue::BUNDLED_NETWORK);
    // Apply iterative deepening.
    let mut p = (EVAL_DRAW, (None, None));
    //for d in 1..=depth {
    for d in 1..=depth {
      let nnue_hash = nnue.get_debugging_hash();
      p = match use_nnue {
        true => self.pvs::<false, true>(
          d,
          &start_state,
          &mut nnue,
          EVAL_VERY_NEGATIVE,
          EVAL_VERY_POSITIVE,
        ),
        false => self.pvs::<false, false>(
          d,
          &start_state,
          &mut nnue,
          EVAL_VERY_NEGATIVE,
          EVAL_VERY_POSITIVE,
        ),
      };
      assert_eq!(nnue_hash, nnue.get_debugging_hash());
      //log(&format!(
      //  "Depth {}: {} (nodes={})",
      //  d, p.0, self.nodes_searched
      //));
    }
    p
  }

  fn probe_tt(&self, state: &State) -> Option<&TTEntry> {
    let index = (state.zobrist % self.transposition_table.len() as u64) as usize;
    let entry = &self.transposition_table[index];
    match entry.zobrist == state.zobrist {
      true => Some(entry),
      false => None,
    }
  }

  fn probe_move_order_table(&self, state: &State) -> Option<Move> {
    let index = (state.zobrist % self.move_order_table.len() as u64) as usize;
    let entry = &self.move_order_table[index];
    match entry.m == Move::INVALID {
      true => None,
      false => Some(entry.m),
    }
  }

  fn move_order_table_insert(&mut self, state: &State, m: Move) {
    let index = (state.zobrist % self.move_order_table.len() as u64) as usize;
    self.move_order_table[index].m = m;
  }

  fn probe_killer_moves(&self, depth: u16) -> Option<Move> {
    self.killer_moves.get(depth as usize).copied().unwrap_or(None)
  }

  fn tt_insert(
    &mut self,
    zobrist: u64,
    depth: u16,
    score: IntEvaluation,
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
    mut alpha: IntEvaluation,
    beta: IntEvaluation,
  ) -> (IntEvaluation, (Option<Move>, Option<Move>)) {
    // Check the transposition table.
    // if let Some(entry) = self.probe_tt(state) {
    //   if entry.depth >= depth {
    //     match entry.node_type {
    //       NodeType::Exact => return (entry.score, (entry.best_move, None)),
    //       NodeType::LowerBound => {
    //         alpha = max(alpha, entry.score);
    //       }
    //       NodeType::UpperBound => {
    //         if entry.score <= alpha {
    //           return (entry.score, (entry.best_move, None));
    //         }
    //       }
    //     }
    //   }
    // }
  
    // Evaluate the nnue to get a score.
    let get_eval = || {
      if NNUE {
        //(nnue.evaluate(state).expected_score * 1000.0) as i32
        nnue.evaluate(state)
      } else {
        evaluate_state(state)
      }
    };

    let game_over = state.get_outcome().is_some();
    let random_bonus = (self.rng.next_random() & 0xf) as i32;
    match (game_over, depth, QUIESCENCE) {
      (true, _, _) => return (get_eval() + random_bonus, (None, None)),
      (_, 0, true) => return (get_eval() + random_bonus, (None, None)),
      (_, 0, false) => {
        let (score, move_pair) = self.pvs::<true, NNUE>(
          QUIESCENCE_DEPTH,
          state,
          nnue,
          alpha,
          beta,
        );
        return (make_mate_score_much_less_extreme(score), move_pair);
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
    //let state_hash: u64 = state.get_transposition_table_hash();

    let mot_move = match QUIESCENCE {
      false => self.probe_move_order_table(&state),
      true => None,
    };
    let killer_move = match QUIESCENCE {
      false => self.probe_killer_moves(depth),
      true => None,
    };

    //if mot_move.is_some() || killer_move.is_some() {
    moves.sort_by(|&a, &b| {
      //let mut a_score = (100.0 * nnue.outputs[a.to as usize]) as i32;
      //let mut b_score = (100.0 * nnue.outputs[b.to as usize]) as i32;
      let mut a_score = 0;
      let mut b_score = 0;
      if let Some(mot_move) = mot_move {
        if a == mot_move {
          a_score += 10_000;
        }
        if b == mot_move {
          b_score += 10_000;
        }
      }
      if let Some(killer_move) = killer_move {
        if a == killer_move {
          a_score += 10_000;
        }
        if b == killer_move {
          b_score += 10_000;
        }
      }
      b_score.cmp(&a_score)
    });
    //moves.sort_by(|_, _| {
    //  self.next_random().cmp(&self.next_random())
    //});

    //log(&format!("pvs({}, {}, {}) moves={}", depth, alpha, beta, moves.len()));
    let mut best_score = EVAL_VERY_NEGATIVE;
    let mut best_pair = (None, None);

    // If we're in a QUIESCENCE search then we're allowed to pass.
    if QUIESCENCE {
      alpha = alpha.max(get_eval());
      if alpha >= beta {
        moves.clear();
      }
    }

    let mut first = true;
    for m in moves {
      self.nodes_searched += 1;
      let mut new_state = state.clone();
      let adjustment = if NNUE {
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
        // Must pass in the old state, as we're undoing the move.
        nnue.apply_adjustment::<true>(&state, &adjustment);
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
        //self.move_order_table_insert(&state, m);
      }
      alpha = alpha.max(score);
      if alpha >= beta {
        self.killer_moves[depth as usize] = Some(m);
        break;
      }
      first = false;
    }

    let score = make_mate_score_slightly_less_extreme(alpha);
    self.tt_insert(state.zobrist, depth, score, best_pair.0, NodeType::Exact);
    (score, best_pair)
  }
}
