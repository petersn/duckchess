use crate::eval::basic_eval;
use crate::nnue::{Nnue, NnueAdjustment};
use crate::rng::Rng;
use crate::rules::{GameOutcome, Move, State, get_square, iter_bits};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct MovePair {
  regular: Move,
  duck: Move,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeType {
  Exact,
  LowerBound,
  UpperBound,
}

struct TTEntry {
  zobrist:   u64,
  depth:     u16,
  score:     IntEvaluation,
  best_move: Option<MovePair>,
  node_type: NodeType,
}

pub struct Engine {
  pub nodes_searched:  u64,
  pub total_eval:      f32,
  rng:                 Rng,
  state:               State,
  transposition_table: Vec<TTEntry>,
  killer_moves:        [MovePair; KILLER_MOVE_COUNT],
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
      killer_moves: [MovePair { regular: Move::INVALID, duck: Move::INVALID }; KILLER_MOVE_COUNT],
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

  pub fn run(&mut self, depth: u16, use_nnue: bool) -> (IntEvaluation, Option<MovePair>) {
    self.nodes_searched = 0;
    let start_state = self.state.clone();
    let mut nnue = Nnue::new(&start_state, crate::nnue::BUNDLED_NETWORK);
    // Apply iterative deepening.
    let mut p = (EVAL_DRAW, None);
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

  pub fn mate_search(&mut self, depth: u16) -> (IntEvaluation, Option<(Move, Move)>) {
    self.nodes_searched = 0;
    let start_state = self.state.clone();
    self.mate_search_inner(depth, &start_state, EVAL_VERY_NEGATIVE, EVAL_VERY_POSITIVE)
  }

  fn mate_search_inner(
    &mut self,
    depth: u16,
    state: &State,
    mut alpha: IntEvaluation,
    beta: IntEvaluation,
  ) -> (IntEvaluation, Option<(Move, Move)>) {
    let eval = eval_terminal_state(state);
    match eval {
      Some(score) => return (score, None),
      None => (),
    }
    if depth == 0 {
      return (EVAL_DRAW, None);
    }

    // Search over all moves here.
    let mut moves = vec![];
    state.move_gen::<false>(&mut moves);
    assert!(!moves.is_empty());
    let mut best_pair = None;

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
    best_move: Option<MovePair>,
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


  fn probe_killer_move(&self, state: &State) -> MovePair {
    // Here we divide by two because the game rules count regular move and duck move as separate plies.
    let index = (state.plies / 2) % KILLER_MOVE_COUNT as u32;
    self.killer_moves[index as usize]
  }

  fn insert_killer_move(&mut self, state: &State, m: MovePair) {
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
  ) -> (IntEvaluation, Option<MovePair>) {
    let state_hash = state.get_transposition_table_hash();
    let mut tt_move_pair = MovePair { regular: Move::INVALID, duck: Move::INVALID };
    // Check the transposition table.
    if let Some(entry) = self.probe_tt(state_hash) {
      tt_move_pair = entry.best_move.unwrap_or(tt_move_pair);
      //println!("TT hit: {} {} {} {:?} {:?}", entry.zobrist, entry.depth, entry.score, entry.best_move, entry.node_type);
      if entry.depth >= depth {
        match entry.node_type {
          NodeType::Exact => return (entry.score, entry.best_move),
          NodeType::LowerBound => alpha = alpha.max(entry.score),
          NodeType::UpperBound => beta = beta.min(entry.score),
        }
        if alpha >= beta {
          return (entry.score, entry.best_move);
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
      let random_bonus = (self.rng.next_random() & 0xf) as i32;
      random_bonus + if NNUE {
        nnue.evaluate(state)
      } else {
        basic_eval(state)
      }
    };

    let game_over = state.get_outcome().is_some();
    match (game_over, depth, QUIESCENCE) {
      (true, _, _) => return (get_eval(), None),
      (_, 0, true) => return (get_eval(), None),
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

    // We now generate (regular, duck) move pairs, and sort all of them.
    // For regular search we apply some light pruning on duck moves.
    // For quiescence search we prune much more aggressively on both.
    //let mut naive_move_pairs = 0;
    let mut move_pairs = vec![];
    {
      let mut single_moves = Vec::new();
      state.move_gen::<QUIESCENCE>(&mut single_moves);
      for m in single_moves {
        let mut child = state.clone();
        child.apply_move::<false>(m, None).unwrap();
        // If the child is terminal then generate just a single dummy duck move.
        if child.get_outcome().is_some() {
          move_pairs.push(MovePair {
            regular: m,
            duck: Move::INVALID,
          });
          continue;
        }
        let child_occupied = child.get_occupied();
        // Forcibly skip over our duck move.
        assert!(child.is_duck_move);
        child.is_duck_move = false;
        child.turn = child.turn.other_player();
        // Do move-gen for the opponent, to find candidate moves for us to block.
        let mut block_mask: u64 = child.generate_duck_block_mask::<QUIESCENCE>();
        // If they only have one or zero blockable moves then make up another place to put the duck,
        // or we might (with absolutely tiny probability) end up stalemating them, or in quiescence
        // search we might force the duck somewhere silly.
        if block_mask.count_ones() <= 1 {
          //println!("Only one blockable move after {}  QUIESCENCE = {}", m, QUIESCENCE);
          if !QUIESCENCE {
            panic!("This should be extremely rare, so I want to know if it happens for now.");
          }
          let other_free_spots = !child_occupied & !block_mask;
          assert!(other_free_spots != 0);
          // Find the bottom set bit in other_free_spots.
          let first_free_spot = other_free_spots & other_free_spots.wrapping_neg();
          debug_assert!(first_free_spot.count_ones() == 1);
          block_mask |= first_free_spot;
          //// FIXME: Do something less wasteful here!
          //block_mask = u64::MAX;
        }
        block_mask &= !child_occupied;
        //block_mask = !child.get_occupied();
        //naive_move_pairs += child_occupied.count_zeros() - 1;
        let current_duck_pos = get_square(child.ducks.0);
        while let Some(pos) = iter_bits(&mut block_mask) {
          let m_duck = Move {
            // Our encoding of the initial duck placement move requires this match.
            from: match child.ducks.0 == 0 {
              true => pos,
              false => current_duck_pos,
            },
            to: pos,
          };
          move_pairs.push(MovePair {
            regular: m,
            duck: m_duck,
          });
        }
      }
      //println!("{} move pairs (naive {})", move_pairs.len(), naive_move_pairs);
    }

    // If we're in a quiescence search and have quiesced, then return.
    if QUIESCENCE && move_pairs.is_empty() {
      return (get_eval(), None);
    }
    assert!(!move_pairs.is_empty());

    // Reorder based on our move order table.
    //let state_hash: u64 = state.get_transposition_table_hash();

    let killer_move = self.probe_killer_move(state);

    //if mot_move.is_some() || killer_move.is_some() {
    move_pairs.sort_by(|&a, &b| {
      //let mut a_score = (100.0 * nnue.outputs[a.to as usize]) as i32;
      //let mut b_score = (100.0 * nnue.outputs[b.to as usize]) as i32;
      let mut a_score = 0;
      let mut b_score = 0;
      macro_rules! adjust_scores {
        ($good_move:expr, $score_regular:expr, $score_duck:expr) => {
          if $good_move.regular == a.regular {
            a_score += $score_regular;
          }
          if $good_move.regular == b.regular {
            b_score += $score_regular;
          }
          if $good_move.duck == a.duck {
            a_score += $score_duck;
          }
          if $good_move.duck == b.duck {
            b_score += $score_duck;
          }
        };
      }
      adjust_scores!(killer_move, 100, 50);
      adjust_scores!(tt_move_pair, 150, 120);
      b_score.cmp(&a_score)
    });
    //moves.sort_by(|_, _| {
    //  self.next_random().cmp(&self.next_random())
    //});

    //log(&format!("pvs({}, {}, {}) moves={}", depth, alpha, beta, moves.len()));
    let mut best_score = EVAL_VERY_NEGATIVE;
    let mut best_pair = None;

    // If we're in a quiescence search then we're allowed to pass.
    if QUIESCENCE {
      alpha = alpha.max(get_eval());
      if alpha >= beta {
        return (alpha, None);
      }
    }

    let mut node_type = NodeType::UpperBound;
    let mut first = true;
    for move_pair in move_pairs {
      self.nodes_searched += 1;
      let mut new_state = state.clone();
      let adjustments = if NNUE {
        let adjustment0 = new_state.apply_move::<true>(move_pair.regular, Some(nnue)).unwrap();
        if new_state.get_outcome().is_some() {
          // FIXME: I don't need to do PVS here!
          continue;
          panic!("This should be impossible.");
        }
        let adjustment1 = new_state.apply_move::<true>(move_pair.duck, Some(nnue)).unwrap();
        (adjustment0, adjustment1)
      } else {
        let adjustment0 = new_state.apply_move::<false>(move_pair.regular, None).unwrap();
        if new_state.get_outcome().is_some() {
          // FIXME: I don't need to do PVS here!
          continue;
          panic!("This should be impossible.");
        }
        let adjustment1 = new_state.apply_move::<false>(move_pair.duck, None).unwrap();
        (adjustment0, adjustment1)
      };
      //nnue.undo(undo_cookie);
      //let debugging_hash = nnue.get_debugging_hash();
      //println!("Undo debugging hash: {:016x}", debugging_hash);

      let mut score;
      let mut next_pair;

      // Two cases:
      // If new_state is a duck move state, we *don't* invert the score, as we take the next move.
      assert!(!new_state.is_duck_move);
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

      if NNUE {
        // Must pass in the old state, as we're undoing the move.
        // FIXME: This is WRONG! I need to pass in the intermediate state!
        // I really need to fix this up before I try to use the NNUE eval.
        nnue.apply_adjustment::<true>(&state, &adjustments.1);
        nnue.apply_adjustment::<true>(&state, &adjustments.0);
      }
      //assert_eq!(debugging_hash, nnue.get_debugging_hash());
      //if state.is_duck_move {
      //  let eval = nnue.evaluate().expected_score;
      //  self.total_eval += eval;
      //}

      if score > best_score {
        best_score = score;
        best_pair = Some(move_pair);
      }
      if score > alpha && !QUIESCENCE {
        node_type = NodeType::Exact;
        //self.tt_insert(state_hash, m, score, depth);
        //self.move_order_table.insert(state_hash, m);
        //self.move_order_table_insert(&state, m);
      }
      alpha = alpha.max(score);
      if alpha >= beta {
        self.insert_killer_move(state, move_pair);
        node_type = NodeType::LowerBound;
        break;
      }
      first = false;
    }

    let score = make_mate_score_slightly_less_extreme(alpha);
    //self.tt_insert(state.zobrist, depth, score, best_pair.0, NodeType::Exact);
    self.insert_tt(state_hash, depth, score, best_pair, node_type);
    (score, best_pair)
  }
}
