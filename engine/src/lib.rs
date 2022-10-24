use std::{
  cmp::Ordering,
  collections::{hash_map::DefaultHasher, HashMap},
  hash::{Hash, Hasher},
};

use serde::{ser::SerializeSeq, Deserialize, Serialize};
use wasm_bindgen::prelude::*;

mod mcts;

#[cfg(not(target_arch = "wasm32"))]
mod python;

include!(concat!(env!("OUT_DIR"), "/tables.rs"));

#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = console)]
  fn log(s: &str);
}

#[derive(Clone, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CastlingRights {
  pub king_side:  bool,
  pub queen_side: bool,
}

#[derive(Clone, Hash)]
pub struct BitBoard(pub u64);

impl Serialize for BitBoard {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    let mut bytes = serializer.serialize_seq(Some(8))?;
    for i in 0..8 {
      let byte = (self.0 >> (i * 8)) as u8;
      bytes.serialize_element(&byte)?;
    }
    bytes.end()
  }
}

impl<'de> Deserialize<'de> for BitBoard {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    let bytes = Vec::<u8>::deserialize(deserializer)?;
    let mut board = 0;
    for (i, byte) in bytes.iter().enumerate() {
      board |= (*byte as u64) << (i * 8);
    }
    Ok(BitBoard(board))
  }
}

#[derive(Clone, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct State {
  pub pawns:           [BitBoard; 2],
  pub knights:         [BitBoard; 2],
  pub bishops:         [BitBoard; 2],
  pub rooks:           [BitBoard; 2],
  pub queens:          [BitBoard; 2],
  pub kings:           [BitBoard; 2],
  pub ducks:           BitBoard,
  pub en_passant:      BitBoard,
  pub highlight:       BitBoard,
  pub castling_rights: [CastlingRights; 2],
  pub white_turn:      bool,
  pub is_duck_move:    bool,
}

pub enum GameOutcome {
  Ongoing,
  Draw,
  WhiteWin,
  BlackWin,
}

const ALL_BUT_A_FILE: u64 = 0xfefefefefefefefe;
const ALL_BUT_H_FILE: u64 = 0x7f7f7f7f7f7f7f7f;
const MIDDLE_SIX_RANKS: u64 = 0x00ffffffffffff00;

fn get_square(bitboard: u64) -> Square {
  bitboard.trailing_zeros() as Square
}

fn iter_bits(bitboard: &mut u64) -> Option<Square> {
  let pos = bitboard.trailing_zeros();
  if pos == 64 {
    return None;
  }
  *bitboard &= *bitboard - 1;
  Some(pos as Square)
}

// Our ray direction table:
//   0: hi: right
//   1: hi: up-right
//   2: hi: up
//   3: hi: up-left
//   4: lo: left
//   5: lo: down-left
//   6: lo: down
//   7: lo: down-right

macro_rules! ray_attack {
  ($occupancy:ident, $pos:ident, $ray_dirs:expr) => {{
    let mut attacks = 0;
    for ray_dir in $ray_dirs {
      let ray = RAYS[ray_dir as usize][$pos as usize];
      let blockers = $occupancy & ray;
      if blockers != 0 {
        // Ray directions less than 4 have their first hit in the lowest bit,
        // while ray directions 4 and up have their first hit in the highest bit.
        let blocker = match ray_dir < 4 {
          true => blockers.trailing_zeros(),
          false => 63 - blockers.leading_zeros(),
        };
        attacks |= ray & !RAYS[ray_dir as usize][blocker as usize];
      } else {
        attacks |= ray;
      }
    }
    attacks
  }};
}

fn rook_attacks(occupancy: u64, pos: Square) -> u64 {
  ray_attack!(occupancy, pos, [0, 2, 4, 6])
}

fn bishop_attacks(occupancy: u64, pos: Square) -> u64 {
  ray_attack!(occupancy, pos, [1, 3, 5, 7])
}

fn queen_attacks(occupancy: u64, pos: Square) -> u64 {
  ray_attack!(occupancy, pos, [0, 1, 2, 3, 4, 5, 6, 7])
}

fn log_bitboard(bits: u64) {
  for rank in (0..8).rev() {
    let mut line = String::new();
    for file in 0..8 {
      let bit = 1 << (rank * 8 + file);
      line.push(if bits & bit != 0 { '1' } else { '0' });
    }
    log(&format!("{}: {}", rank, line));
  }
}

impl State {
  fn starting_state() -> State {
    State {
      pawns:           [BitBoard(0x00ff000000000000), BitBoard(0x000000000000ff00)],
      knights:         [BitBoard(0x4200000000000000), BitBoard(0x0000000000000042)],
      bishops:         [BitBoard(0x2400000000000000), BitBoard(0x0000000000000024)],
      rooks:           [BitBoard(0x8100000000000000), BitBoard(0x0000000000000081)],
      queens:          [BitBoard(0x0800000000000000), BitBoard(0x0000000000000008)],
      kings:           [BitBoard(0x1000000000000000), BitBoard(0x0000000000000010)],
      ducks:           BitBoard(0),
      en_passant:      BitBoard(0),
      highlight:       BitBoard(0),
      castling_rights: [
        CastlingRights {
          king_side:  true,
          queen_side: true,
        },
        CastlingRights {
          king_side:  true,
          queen_side: true,
        },
      ],
      white_turn:      true,
      is_duck_move:    false,
    }
  }

  // Warning: This doesn't include stalemate.
  fn is_game_over(&self) -> bool {
    self.kings[0].0 == 0 || self.kings[1].0 == 0
  }

  fn move_gen_for_color<const QUIESCENCE: bool, const IS_WHITE: bool>(
    &self,
    moves: &mut Vec<Move>,
  ) {
    macro_rules! shift_backward {
      ($board:expr) => {
        if IS_WHITE {
          $board >> 8
        } else {
          $board << 8
        }
      };
    }
    let forward_one_row: SquareDelta = if IS_WHITE { 8 } else { 248 };
    let our_pawns = self.pawns[IS_WHITE as usize].0;
    let our_knights = self.knights[IS_WHITE as usize].0;
    let our_bishops = self.bishops[IS_WHITE as usize].0;
    let our_rooks = self.rooks[IS_WHITE as usize].0;
    let our_queens = self.queens[IS_WHITE as usize].0;
    let our_king = self.kings[IS_WHITE as usize].0;
    let our_pieces: u64 = our_pawns | our_knights | our_bishops | our_rooks | our_queens | our_king;
    let their_pieces: u64 = self.pawns[(!IS_WHITE) as usize].0
      | self.knights[(!IS_WHITE) as usize].0
      | self.bishops[(!IS_WHITE) as usize].0
      | self.rooks[(!IS_WHITE) as usize].0
      | self.queens[(!IS_WHITE) as usize].0
      | self.kings[(!IS_WHITE) as usize].0;
    let occupied: u64 = self.ducks.0 | our_pieces | their_pieces;

    // Handle the duck move part of the turn.
    if self.is_duck_move {
      let mut duck_moves = !occupied;
      while let Some(pos) = iter_bits(&mut duck_moves) {
        moves.push(Move {
          from:      get_square(self.ducks.0),
          to:        pos,
          promotion: None,
        });
      }
      return;
    }

    // Check for castling.
    if !QUIESCENCE {
      if self.castling_rights[IS_WHITE as usize].king_side {
        let movement_mask = if IS_WHITE {
          0b01100000
        } else {
          0b01100000 << 56
        };
        if occupied & movement_mask == 0 {
          moves.push(Move {
            from:      get_square(our_king),
            to:        get_square(our_king) + 2,
            promotion: None,
          });
        }
      }
      if self.castling_rights[IS_WHITE as usize].queen_side {
        let movement_mask = if IS_WHITE {
          0b00001110
        } else {
          0b00001110 << 56
        };
        if occupied & movement_mask == 0 {
          moves.push(Move {
            from:      get_square(our_king),
            to:        get_square(our_king) - 2,
            promotion: None,
          });
        }
      }
    }

    // Find all moves for pawns.
    let (second_rank, seventh_rank) = if IS_WHITE {
      (0x000000000000ff00, 0x00ff000000000000)
    } else {
      (0x00ff000000000000, 0x000000000000ff00)
    };
    let mut single_pawn_pushes = MIDDLE_SIX_RANKS & our_pawns & !shift_backward!(occupied);
    let mut double_pawn_pushes =
      (single_pawn_pushes & second_rank) & !shift_backward!(shift_backward!(occupied));

    let pawn_capturable = their_pieces | (self.en_passant.0 & !self.ducks.0);
    let mut pawn_capture_left =
      our_pawns & (shift_backward!(pawn_capturable & ALL_BUT_H_FILE) << 1);
    let mut pawn_capture_right =
      our_pawns & (shift_backward!(pawn_capturable & ALL_BUT_A_FILE) >> 1);

    let mut promote_single_pawn_pushes = single_pawn_pushes & seventh_rank;
    let mut promote_double_pawn_pushes = double_pawn_pushes & seventh_rank;
    let mut promote_pawn_capture_left = pawn_capture_left & seventh_rank;
    let mut promote_pawn_capture_right = pawn_capture_right & seventh_rank;
    single_pawn_pushes &= !seventh_rank;
    double_pawn_pushes &= !seventh_rank;
    pawn_capture_left &= !seventh_rank;
    pawn_capture_right &= !seventh_rank;

    macro_rules! add_pawn_moves {
      ("plain", $bits:ident, $moves:ident, $delta:expr) => {
        while let Some(pos) = iter_bits(&mut $bits) {
          $moves.push(Move {
            from:      pos,
            to:        pos.wrapping_add($delta),
            promotion: None,
          });
        }
      };
      ("promotion", $bits:ident, $moves:ident, $delta:expr) => {
        while let Some(pos) = iter_bits(&mut $bits) {
          for promotable_piece in [
            PromotablePiece::Queen,
            //PromotablePiece::Knight,
            //PromotablePiece::Rook,
            //PromotablePiece::Bishop,
          ] {
            $moves.push(Move {
              from:      pos,
              to:        pos.wrapping_add($delta),
              promotion: Some(promotable_piece),
            });
          }
        }
      };
    }
    add_pawn_moves!("plain", pawn_capture_left, moves, forward_one_row - 1);
    add_pawn_moves!("plain", pawn_capture_right, moves, forward_one_row + 1);
    add_pawn_moves!(
      "promotion",
      promote_pawn_capture_left,
      moves,
      forward_one_row - 1
    );
    add_pawn_moves!(
      "promotion",
      promote_pawn_capture_right,
      moves,
      forward_one_row + 1
    );
    add_pawn_moves!(
      "promotion",
      promote_single_pawn_pushes,
      moves,
      forward_one_row
    );
    add_pawn_moves!(
      "promotion",
      promote_double_pawn_pushes,
      moves,
      forward_one_row * 2
    );
    if !QUIESCENCE {
      add_pawn_moves!("plain", single_pawn_pushes, moves, forward_one_row);
      add_pawn_moves!(
        "plain",
        double_pawn_pushes,
        moves,
        forward_one_row.wrapping_mul(2)
      );
    }

    let moveable_mask = !(our_pieces | self.ducks.0);

    // Find all moves for knights.
    let mut knights = our_knights;
    while let Some(pos) = iter_bits(&mut knights) {
      let mut knight_moves = KNIGHT_MOVES[pos as usize];
      knight_moves &= moveable_mask;
      while let Some(knight_move) = iter_bits(&mut knight_moves) {
        if QUIESCENCE && their_pieces & (1 << knight_move) == 0 {
          continue;
        }
        moves.push(Move {
          from:      pos,
          to:        knight_move,
          promotion: None,
        });
      }
    }

    // Find all moves for kings.
    let mut kings = our_king;
    while let Some(pos) = iter_bits(&mut kings) {
      let mut king_moves = KING_MOVES[pos as usize];
      king_moves &= moveable_mask;
      while let Some(king_move) = iter_bits(&mut king_moves) {
        if QUIESCENCE && their_pieces & (1 << king_move) == 0 {
          continue;
        }
        moves.push(Move {
          from:      pos,
          to:        king_move,
          promotion: None,
        });
      }
    }

    // Find all moves for rooks.
    let mut rooks = our_rooks;
    while let Some(pos) = iter_bits(&mut rooks) {
      let attacked = rook_attacks(occupied, pos);
      let mut rook_moves = attacked & moveable_mask;
      while let Some(rook_move) = iter_bits(&mut rook_moves) {
        if QUIESCENCE && their_pieces & (1 << rook_move) == 0 {
          continue;
        }
        moves.push(Move {
          from:      pos,
          to:        rook_move,
          promotion: None,
        });
      }
    }

    // Find all moves for bishops.
    let mut bishops = our_bishops;
    while let Some(pos) = iter_bits(&mut bishops) {
      let attacked = bishop_attacks(occupied, pos);
      let mut bishop_moves = attacked & moveable_mask;
      while let Some(bishop_move) = iter_bits(&mut bishop_moves) {
        if QUIESCENCE && their_pieces & (1 << bishop_move) == 0 {
          continue;
        }
        moves.push(Move {
          from:      pos,
          to:        bishop_move,
          promotion: None,
        });
      }
    }

    // Find all moves for queens.
    let mut queens = our_queens;
    while let Some(pos) = iter_bits(&mut queens) {
      let attacked = queen_attacks(occupied, pos);
      let mut queen_moves = attacked & moveable_mask;
      while let Some(queen_move) = iter_bits(&mut queen_moves) {
        if QUIESCENCE && their_pieces & (1 << queen_move) == 0 {
          continue;
        }
        moves.push(Move {
          from:      pos,
          to:        queen_move,
          promotion: None,
        });
      }
    }
  }

  fn move_gen<const QUIESCENCE: bool>(&self, moves: &mut Vec<Move>) {
    match self.white_turn {
      true => self.move_gen_for_color::<QUIESCENCE, true>(moves),
      false => self.move_gen_for_color::<QUIESCENCE, false>(moves),
    }
  }

  fn apply_move(&mut self, m: &Move) -> bool {
    let from_mask = if m.from == 64 { 0 } else { 1 << m.from };
    let to_mask = 1 << m.to;
    if !self.is_duck_move {
      self.highlight.0 = 0;
    }
    self.highlight.0 |= from_mask | to_mask;
    // Handle duck moves.
    if self.is_duck_move {
      if self.ducks.0 & from_mask != 0 || m.from == 64 {
        self.ducks.0 ^= from_mask;
        self.ducks.0 |= to_mask;
        self.is_duck_move = false;
        self.white_turn = !self.white_turn;
        return true;
      }
      return false;
    }

    let moving_en_passant = self.en_passant.0 & to_mask != 0;
    self.en_passant.0 = 0;
    let mut remove_rooks = 0;
    let mut new_queens = 0;
    let mut new_rooks = 0;
    let mut new_bishops = 0;
    let mut new_knights = 0;

    enum PieceKind {
      Pawn,
      King,
      Rook,
      Other,
    }
    for (piece_kind, piece_array) in [
      (PieceKind::Pawn, &mut self.pawns),
      (PieceKind::Other, &mut self.knights),
      (PieceKind::Other, &mut self.bishops),
      (PieceKind::Rook, &mut self.rooks),
      (PieceKind::Other, &mut self.queens),
      (PieceKind::King, &mut self.kings),
    ] {
      let (a, b) = piece_array.split_at_mut(1);
      let (us, them) = match self.white_turn {
        true => (&mut b[0], &mut a[0]),
        false => (&mut a[0], &mut b[0]),
      };
      // Capture pieces on the target square.
      them.0 &= !to_mask;
      // Check if this is the kind of piece we're moving.
      if us.0 & from_mask != 0 {
        // Move our piece.
        us.0 ^= from_mask;
        us.0 |= to_mask;
        // Handle special rules for special pieces.
        match (piece_kind, m.from) {
          (PieceKind::Pawn, _) => {
            // Check if we're taking en passant.
            if moving_en_passant {
              match self.white_turn {
                true => them.0 &= !(1 << (m.to - 8)),
                false => them.0 &= !(1 << (m.to + 8)),
              }
            }
            // Setup the en passant state.
            let is_double_move = (m.from as i8 - m.to as i8).abs() == 16;
            match (is_double_move, self.white_turn) {
              (true, true) => self.en_passant.0 = to_mask >> 8,
              (true, false) => self.en_passant.0 = to_mask << 8,
              (false, _) => {}
            }
            // Check if we're promoting.
            match &m.promotion {
              Some(promotion) => {
                // Remove the pawn, and setup the promotion.
                us.0 &= !to_mask;
                match promotion {
                  PromotablePiece::Queen => new_queens = to_mask,
                  PromotablePiece::Rook => new_rooks = to_mask,
                  PromotablePiece::Knight => new_knights = to_mask,
                  PromotablePiece::Bishop => new_bishops = to_mask,
                }
              }
              None => {}
            }
          }
          (PieceKind::King, _) => {
            self.castling_rights[self.white_turn as usize] = CastlingRights {
              king_side:  false,
              queen_side: false,
            };
            // Check if we just castled.
            if (m.from as i8 - m.to as i8).abs() == 2 {
              let (rook_from, rook_to) = match (m.from, m.to) {
                (4, 6) => (7, 5),
                (4, 2) => (0, 3),
                (60, 62) => (63, 61),
                (60, 58) => (56, 59),
                _ => unreachable!(),
              };
              remove_rooks = 1 << rook_from;
              new_rooks = 1 << rook_to;
            }
          }
          (PieceKind::Rook, 0) | (PieceKind::Rook, 56) => {
            self.castling_rights[self.white_turn as usize].queen_side = false
          }
          (PieceKind::Rook, 7) | (PieceKind::Rook, 63) => {
            self.castling_rights[self.white_turn as usize].king_side = false
          }
          (PieceKind::Rook, _) => {}
          (PieceKind::Other, _) => {}
        }
      }
    }
    self.rooks[self.white_turn as usize].0 &= !remove_rooks;
    self.knights[self.white_turn as usize].0 |= new_knights;
    self.bishops[self.white_turn as usize].0 |= new_bishops;
    self.rooks[self.white_turn as usize].0 |= new_rooks;
    self.queens[self.white_turn as usize].0 |= new_queens;

    //self.white_turn = !self.white_turn;
    self.is_duck_move = true;
    true
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
enum PromotablePiece {
  // Ordered here based on likelihood of promotion.
  Queen,
  Knight,
  Rook,
  Bishop,
}

type Square = u8;
type SquareDelta = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Move {
  from:      Square,
  to:        Square,
  promotion: Option<PromotablePiece>,
}

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

fn evaluate_state(state: &State) -> Evaluation {
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

#[wasm_bindgen]
pub struct Engine {
  nodes_searched:   u64,
  seed:             u64,
  state:            State,
  move_order_table: HashMap<u64, Move>,
  killer_moves:     [Option<Move>; 100],
}

#[wasm_bindgen]
impl Engine {
  pub fn get_state(&self) -> JsValue {
    serde_wasm_bindgen::to_value(&self.state).unwrap_or_else(|e| {
      log(&format!("Failed to serialize state: {}", e));
      JsValue::NULL
    })
  }

  pub fn set_state(&mut self, state: JsValue) {
    self.state = serde_wasm_bindgen::from_value(state).unwrap_or_else(|e| {
      log(&format!("Failed to deserialize state: {}", e));
      panic!("Failed to deserialize state: {}", e);
    });
  }

  pub fn get_moves(&self) -> JsValue {
    let mut moves = Vec::new();
    self.state.move_gen::<false>(&mut moves);
    serde_wasm_bindgen::to_value(&moves).unwrap_or_else(|e| {
      log(&format!("Failed to serialize moves: {}", e));
      JsValue::NULL
    })
  }

  pub fn apply_move(&mut self, m: JsValue) -> bool {
    let m: Move = serde_wasm_bindgen::from_value(m).unwrap_or_else(|e| {
      log(&format!("Failed to deserialize move: {}", e));
      panic!("Failed to deserialize move: {}", e);
    });
    self.state.apply_move(&m)
  }

  pub fn run(&mut self, depth: u16) -> JsValue {
    self.nodes_searched = 0;
    let start_state = self.state.clone();
    // Apply iterative deepening.
    let mut p = (-1, (None, None));
    for d in 1..=depth {
      p = self.pvs::<false>(d, &start_state, VERY_NEGATIVE_EVAL, VERY_POSITIVE_EVAL);
      log(&format!(
        "Depth {}: {} (nodes={})",
        d, p.0, self.nodes_searched
      ));
    }
    serde_wasm_bindgen::to_value(&p).unwrap_or_else(|e| {
      log(&format!("Failed to serialize score: {}", e));
      JsValue::NULL
    })
  }

  fn next_random(&mut self) -> u64 {
    self.seed = self.seed.wrapping_add(1);
    let mut x = self.seed.wrapping_mul(0x243f6a8885a308d3);
    for _ in 0..4 {
      x ^= x >> 37;
      x = x.wrapping_mul(0x243f6a8885a308d3);
    }
    x
  }

  fn pvs<const QUIESCENCE: bool>(
    &mut self,
    depth: u16,
    state: &State,
    mut alpha: Evaluation,
    mut beta: Evaluation,
  ) -> (Evaluation, (Option<Move>, Option<Move>)) {
    self.nodes_searched += 1;
    let game_over = state.is_game_over();
    let random_bonus = (self.next_random() & 0xf) as Evaluation;
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
    let mut s = DefaultHasher::new();
    state.hash(&mut s);
    let state_hash: u64 = s.finish();

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
      new_state.apply_move(&m);

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

pub fn get_moves_internal(this: &Engine) -> Vec<Move> {
  let mut moves = Vec::new();
  this.state.move_gen::<false>(&mut moves);
  moves
}

pub fn apply_move_internal(this: &mut Engine, m: Move) -> bool {
  this.state.apply_move(&m)
}

pub fn run_internal(this: &mut Engine, depth: u16) -> (Evaluation, (Option<Move>, Option<Move>)) {
  this.nodes_searched = 0;
  let start_state = this.state.clone();
  // Apply iterative deepening.
  let mut p = (-1, (None, None));
  for d in 1..=depth {
    p = this.pvs::<false>(d, &start_state, VERY_NEGATIVE_EVAL, VERY_POSITIVE_EVAL);
    //log(&format!("Depth {}: {} (nodes={})", d, p.0, this.nodes_searched));
  }
  p
}

pub fn get_outcome_internal(this: &Engine) -> Option<&'static str> {
  match this.state.is_game_over() {
    true => Some(if this.state.kings[0].0 == 0 {
      "1-0"
    } else {
      "0-1"
    }),
    false => None,
  }
}

pub fn get_state_internal(this: &Engine) -> State {
  this.state.clone()
}

#[wasm_bindgen]
pub fn new_engine(seed: u64) -> Engine {
  Engine {
    nodes_searched: 0,
    seed,
    state: State::starting_state(),
    move_order_table: HashMap::new(),
    killer_moves: [None; 100],
  }
}
