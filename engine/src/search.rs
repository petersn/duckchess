use crate::eval::basic_eval;
use crate::nnue::{Nnue, NnueAdjustment};
use crate::rng::Rng;
use crate::rules::{get_square, iter_bits, GameOutcome, Move, State};

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
    (GameOutcome::Win(a), b) if a == b => EVAL_WIN,
    _ => EVAL_LOSS,
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

const QUIESCENCE_DEPTH: u16 = 10;
const KILLER_MOVE_COUNT: usize = 64;

//#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
//pub struct MovePair {
//  regular: Move,
//  duck:    Move,
//}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeType {
  Exact,
  LowerBound,
  UpperBound,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PrincipalVariation {
  pub eval: IntEvaluation,
  pub moves: Vec<Move>,
}

impl PrincipalVariation {
  fn new() -> Self {
    Self {
      eval: EVAL_DRAW,
      moves: vec![],
    }
  }

  fn from_one_move(eval: IntEvaluation, m: Move) -> Self {
    Self {
      eval,
      moves: vec![m],
    }
  }
}

struct TTEntry {
  zobrist:   u64,
  depth:     u16,
  score:     IntEvaluation,
  best_move: Move,
  node_type: NodeType,
}

pub struct Engine {
  pub nodes_searched:  u64,
  pub total_eval:      f32,
  rng:                 Rng,
  state:               State,
  transposition_table: Vec<TTEntry>,
  killer_moves:        [Move; KILLER_MOVE_COUNT],
}

impl Engine {
  pub fn new(seed: u64, tt_size: usize) -> Self {
    let state = State::starting_state();
    let mut transposition_table = Vec::with_capacity(tt_size);
    for _ in 0..tt_size {
      transposition_table.push(TTEntry {
        zobrist:   0,
        depth:     0,
        score:     EVAL_DRAW,
        best_move: Move::INVALID,
        node_type: NodeType::Exact,
      });
    }
    Self {
      nodes_searched: 0,
      total_eval: 0.0,
      rng: Rng::new(seed),
      state,
      transposition_table,
      killer_moves: [Move::INVALID; KILLER_MOVE_COUNT],
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

  pub fn run(&mut self, depth: u16, use_nnue: bool) -> PrincipalVariation {
    self.nodes_searched = 0;
    let start_state = self.state.clone();
    let mut nnue = Nnue::new(&start_state, crate::nnue::BUNDLED_NETWORK);
    // Apply iterative deepening.
    let mut pv = PrincipalVariation::new();
    //for d in 1..=depth {
    for d in 1..=depth {
      let nnue_hash = nnue.get_debugging_hash();
      let (eval, m) = match use_nnue {
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
      pv = PrincipalVariation {
        eval,
        moves: m.into_iter().collect(),
      };
      //log(&format!(
      //  "Depth {}: {} (nodes={})",
      //  d, pv.0, self.nodes_searched
      //));
    }
    pv
  }

  pub fn mate_search(&mut self, depth: u16) -> (IntEvaluation, Option<(Move, Move)>) {
    self.nodes_searched = 0;
    let start_state = self.state.clone();
    //self.mate_search_inner(depth, &start_state, EVAL_VERY_NEGATIVE, EVAL_VERY_POSITIVE)
    self.mate_search_inner(depth, &start_state, -1000, 1000)
  }

  fn mate_search_inner(
    &mut self,
    depth: u16,
    state: &State,
    mut alpha: i32,
    beta: i32,
    //mut alpha: IntEvaluation,
    //beta: IntEvaluation,
  ) -> (IntEvaluation, Option<(Move, Move)>) {
    //println!("mate_search_inner: depth={}", depth);
    match (state.get_outcome(), state.turn) {
      (Some(GameOutcome::Draw), _) => return (0, None),
      (Some(GameOutcome::Win(a)), b) => return (if a == b { 1000 } else { -1000 }, None),
      _ => {}
    }
    if depth == 0 {
      // If we're at depth 0 then we should be in a duck move.
      if !state.is_duck_move {
        crate::log("mate_search_inner: depth=0 but not in duck move");
      }
      assert!(state.is_duck_move);
      return (0, None);
    }

    // Search over all moves here.
    let mut moves = vec![];
    state.move_gen::<false>(&mut moves);
    assert!(!moves.is_empty());
    let mut best_score = -2000;
    let mut best_pair = None;

    for m in moves {
      self.nodes_searched += 1;
      let mut new_state = state.clone();
      new_state.apply_move::<false>(m, None).unwrap();
      // We must match on is_duck_move, because we only negate scores when switching between players.
      let (score, inner_moves) = match state.is_duck_move {
        // Must negate in this case.
        true => {
          let (score, inner_moves) = self.mate_search_inner(depth - 1, &new_state, -beta, -alpha);
          (-score, inner_moves)
        }
        false => self.mate_search_inner(depth - 1, &new_state, alpha, beta),
      };
      let move_pair = match inner_moves {
        Some((a, _)) => (m, a),
        None => (m, Move::INVALID),
      };
      if score >= beta {
        return (score, Some(move_pair));
      }
      alpha = alpha.max(score);
      if score > best_score {
        best_score = score;
        best_pair = Some(move_pair);
      }
    }
    (best_score, best_pair)

    /*
    for m in moves {
      self.nodes_searched += 1;
      let mut new_state = state.clone();
      new_state.apply_move::<false>(m, None).unwrap();

      let (score, inner_moves) = self.mate_search_inner(depth - 1, &new_state, -beta, -alpha);
      let score = -score;
      if score >= beta {
        return (score, match inner_moves {
          Some((m1, _)) => Some((m, m1)),
          None => Some((m, Move::INVALID)),
        });
      }
      if score > alpha {
        alpha = score;
        best_pair = match inner_moves {
          Some((m1, _)) => Some((m, m1)),
          None => Some((m, Move::INVALID)),
        };
      }
    }
    let score = make_mate_score_slightly_less_extreme(alpha);
    (score, best_pair)
    */
  }

  fn probe_tt(&mut self, hash: u64) -> Option<&mut TTEntry> {
    let index = (hash % self.transposition_table.len() as u64) as usize;
    let entry = &mut self.transposition_table[index];
    match entry.zobrist == hash {
      true => Some(entry),
      false => {
        //println!("TT miss");
        None
      }
    }
  }

  fn insert_tt(
    &mut self,
    hash: u64,
    depth: u16,
    score: IntEvaluation,
    best_move: Move,
    node_type: NodeType,
  ) {
    //println!("TT insert: {} {} {} {:?} {:?}", hash, depth, score, best_move, node_type);
    let index = (hash % self.transposition_table.len() as u64) as usize;
    let entry = &mut self.transposition_table[index];
    entry.zobrist = hash;
    entry.depth = depth;
    entry.score = score;
    entry.best_move = best_move;
    entry.node_type = node_type;
  }

  fn probe_killer_move(&self, state: &State) -> Move {
    // Here we divide by two because the game rules count regular move and duck move as separate plies.
    let index = (state.plies / 2) % KILLER_MOVE_COUNT as u32;
    self.killer_moves[index as usize]
  }

  fn insert_killer_move(&mut self, state: &State, m: Move) {
    // Here we divide by two because the game rules count regular move and duck move as separate plies.
    let index = (state.plies / 2) % KILLER_MOVE_COUNT as u32;
    self.killer_moves[index as usize] = m;
  }

  fn pvs<const QUIESCENCE: bool, const NNUE: bool>(
    &mut self,
    depth: u16,
    state: &State,
    nnue: &mut Nnue,
    mut alpha: IntEvaluation,
    mut beta: IntEvaluation,
  ) -> (IntEvaluation, Option<Move>) {
    let state_hash = state.get_transposition_table_hash();
    let mut tt_move = Move::INVALID;
    // Check the transposition table.
    if let Some(entry) = self.probe_tt(state_hash) {
      tt_move = entry.best_move;
      //println!("TT hit: {} {} {} {:?} {:?}", entry.zobrist, entry.depth, entry.score, entry.best_move, entry.node_type);
      if entry.depth >= depth {
        match entry.node_type {
          NodeType::Exact => return (entry.score, Some(entry.best_move)),
          NodeType::LowerBound => alpha = alpha.max(entry.score),
          NodeType::UpperBound => beta = beta.min(entry.score),
        }
        if alpha >= beta {
          return (entry.score, Some(entry.best_move));
        }
      }
      /*
      // We use deeper TT entries, even though this can result in instability due to
      // evals at different depths being less than totally comparable.
      if entry.depth >= depth {
        match entry.node_type {
          NodeType::Exact => return (entry.score, entry.best_move),
          NodeType::LowerBound => {
            alpha = alpha.max(entry.score);
            if alpha >= beta {
              return (alpha, entry.best_move);
            }
          }
          NodeType::UpperBound => {
            beta = beta.min(entry.score);
            if alpha >= beta {
              return (beta, entry.best_move);
            }
          }
        }
      }
      */
    }

    let get_eval = || {
      //let random_bonus = (self.rng.next_random() & 0xf) as i32;
      let random_bonus = 0;
      random_bonus
        + if NNUE {
          nnue.evaluate_value(state)
        } else {
          basic_eval(state)
        }
    };

    let game_over = state.get_outcome().is_some();
    match (game_over, depth, QUIESCENCE) {
      (true, _, _) => return (get_eval(), None),
      (_, 0, true) => return (get_eval(), None),
      (_, 0, false) => {
        let (score, move_pair) = self.pvs::<true, NNUE>(QUIESCENCE_DEPTH, state, nnue, alpha, beta);
        return (make_mate_score_much_less_extreme(score), move_pair);
      }
      _ => {}
    }

    // We now generate all moves.
    let mut moves_to_search = vec![];
    match state.is_duck_move {
      false => state.move_gen::<QUIESCENCE>(&mut moves_to_search),
      true => {
        // For duck moves we generate only those that block an opponent move, plus one more.
        // This isn't totally sound, but it helps reduce the branching factor enormously.
        let occupied = state.get_occupied();
        let mut state_copy = state.clone();
        state_copy.is_duck_move = false;
        state_copy.turn = state_copy.turn.other_player();
        let mut block_mask: u64 = state_copy.generate_duck_block_mask::<QUIESCENCE>();
        if block_mask.count_ones() <= 1 {
          //println!("Only one blockable move after {}  QUIESCENCE = {}", m, QUIESCENCE);
          if !QUIESCENCE {
            panic!("This should be extremely rare, so I want to know if it happens for now.");
          }
          let other_free_spots = !occupied & !block_mask;
          assert!(other_free_spots != 0);
          // Find the bottom set bit in other_free_spots.
          let first_free_spot = other_free_spots & other_free_spots.wrapping_neg();
          debug_assert!(first_free_spot.count_ones() == 1);
          block_mask |= first_free_spot;
          //// FIXME: Do something less wasteful here!
          //block_mask = u64::MAX;
        }
        block_mask &= !occupied;
        let current_duck_pos = get_square(state.ducks.0);
        while let Some(pos) = iter_bits(&mut block_mask) {
          moves_to_search.push(Move {
            // Our encoding of the initial duck placement move requires this match.
            from: match state.ducks.0 == 0 {
              true => pos,
              false => current_duck_pos,
            },
            to:   pos,
          });
        }
      }
    }

    // If we're in a quiescence search and have quiesced, then return.
    if QUIESCENCE && moves_to_search.is_empty() {
      return (get_eval(), None);
    }
    assert!(!moves_to_search.is_empty());

    // We now sort the moves.
    let mut nnue_policy_from_scores = [0i16; 64];
    let mut nnue_policy_to_scores = [0i16; 64];
    if NNUE && depth >= 2 {
      nnue.evaluate_policy(state, &mut nnue_policy_from_scores, &mut nnue_policy_to_scores);
    }
    let killer_move = self.probe_killer_move(state);
    //if mot_move.is_some() || killer_move.is_some() {
    moves_to_search.sort_by(|&a, &b| {
      //let mut a_score = (100.0 * nnue.outputs[a.to as usize]) as i32;
      //let mut b_score = (100.0 * nnue.outputs[b.to as usize]) as i32;
      let mut a_score =
        nnue_policy_from_scores[a.from as usize] as i32 +
        nnue_policy_to_scores[a.to as usize] as i32;
      let mut b_score =
        nnue_policy_from_scores[b.from as usize] as i32 +
        nnue_policy_to_scores[b.to as usize] as i32;
      macro_rules! adjust_scores {
        ($good_move:expr, $score:expr) => {
          if $good_move == a {
            a_score += $score;
          }
          if $good_move == b {
            b_score += $score;
          }
        };
      }
      adjust_scores!(killer_move, 10000);
      adjust_scores!(tt_move, 15000);
      b_score.cmp(&a_score)
    });

    //log(&format!("pvs({}, {}, {}) moves={}", depth, alpha, beta, moves.len()));
    let mut best_score = EVAL_VERY_NEGATIVE;
    let mut best_move = None;

    // If we're in a quiescence search then we're allowed to pass.
    if QUIESCENCE {
      alpha = alpha.max(get_eval());
      if alpha >= beta {
        return (alpha, None);
      }
    }

    let mut node_type = NodeType::UpperBound;
    let mut first = true;
    for search_move in moves_to_search {
      self.nodes_searched += 1;
      let mut new_state = state.clone();
      let adjustment = if NNUE {
        new_state.apply_move::<NNUE>(search_move, Some(nnue)).unwrap()
      } else {
        new_state.apply_move::<false>(search_move, None).unwrap()
      };
      //nnue.undo(undo_cookie);
      //let debugging_hash = nnue.get_debugging_hash();
      //println!("Undo debugging hash: {:016x}", debugging_hash);

      let mut score;
      let mut next_move;

      // Two cases:
      // If new_state is a duck move state, we *don't* invert the score, as we take the next move.

      if new_state.is_duck_move {
        if first {
          (score, next_move) = self.pvs::<QUIESCENCE, NNUE>(depth - 1, &new_state, nnue, alpha, beta);
        } else {
          (score, next_move) = self.pvs::<QUIESCENCE, NNUE>(depth - 1, &new_state, nnue, alpha, alpha + 1);
          if alpha < score && score < beta {
            (score, next_move) = self.pvs::<QUIESCENCE, NNUE>(depth - 1, &new_state, nnue, score, beta);
          }
        }
      } else {
        if first {
          (score, next_move) = self.pvs::<QUIESCENCE, NNUE>(depth - 1, &new_state, nnue, -beta, -alpha);
          score *= -1;
        } else {
          (score, next_move) = self.pvs::<QUIESCENCE, NNUE>(depth - 1, &new_state, nnue, -alpha - 1, -alpha);
          score *= -1;
          if alpha < score && score < beta {
            (score, next_move) = self.pvs::<QUIESCENCE, NNUE>(depth - 1, &new_state, nnue, -beta, -score);
            score *= -1;
          }
        }
      }

      if NNUE {
        // Must pass in the old state, as we're undoing the move.
        //// FIXME: This is WRONG! I need to pass in the intermediate state!
        //// I really need to fix this up before I try to use the NNUE eval.
        nnue.apply_adjustment::<true>(&state, &adjustment);
      }
      //assert_eq!(debugging_hash, nnue.get_debugging_hash());
      //if state.is_duck_move {
      //  let eval = nnue.evaluate().expected_score;
      //  self.total_eval += eval;
      //}

      if score > best_score {
        best_score = score;
        best_move = Some(search_move);
      }
      if score > alpha && !QUIESCENCE {
        node_type = NodeType::Exact;
        //self.tt_insert(state_hash, m, score, depth);
        //self.move_order_table.insert(state_hash, m);
        //self.move_order_table_insert(&state, m);
      }
      alpha = alpha.max(score);
      if alpha >= beta {
        self.insert_killer_move(state, best_move.unwrap());
        node_type = NodeType::LowerBound;
        break;
      }
      first = false;
    }

    let score = make_mate_score_slightly_less_extreme(alpha);
    //self.tt_insert(state.zobrist, depth, score, best_pair.0, NodeType::Exact);
    if !QUIESCENCE && let Some(m) = best_move {
      // FIXME: I disabled the TT.
      self.insert_tt(state_hash, depth, score, m, node_type);
    }
    (score, best_move)
  }
}
