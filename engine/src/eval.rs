use crate::{
  rules::{iter_bits, Player, State},
  search::{eval_terminal_state, IntEvaluation},
};

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

pub fn basic_eval(state: &State) -> IntEvaluation {
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
