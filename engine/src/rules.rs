include!(concat!(env!("OUT_DIR"), "/tables.rs"));

use std::{cmp::Ordering, collections::HashMap, hash::Hash};

use serde::{ser::SerializeSeq, Deserialize, Serialize};

const ALL_BUT_A_FILE: u64 = 0xfefefefefefefefe;
const ALL_BUT_H_FILE: u64 = 0x7f7f7f7f7f7f7f7f;
const MIDDLE_SIX_RANKS: u64 = 0x00ffffffffffff00;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PromotablePiece {
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
  pub from:      Square,
  pub to:        Square,
  pub promotion: Option<PromotablePiece>,
}

pub enum GameOutcome {
  Ongoing,
  Draw,
  WhiteWin,
  BlackWin,
}

impl GameOutcome {
  pub fn to_str(&self) -> Option<&'static str> {
    match self {
      GameOutcome::Ongoing => None,
      GameOutcome::Draw => Some("1/2-1/2"),
      GameOutcome::WhiteWin => Some("1-0"),
      GameOutcome::BlackWin => Some("0-1"),
    }
  }
}

fn get_square(bitboard: u64) -> Square {
  bitboard.trailing_zeros() as Square
}

pub fn iter_bits(bitboard: &mut u64) -> Option<Square> {
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

impl State {
  pub fn starting_state() -> State {
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

  // TODO: Implement stalemate.
  pub fn is_game_over(&self) -> bool {
    self.kings[0].0 == 0 || self.kings[1].0 == 0
  }

  // TODO: Implement stalemate.
  pub fn get_outcome(&self) -> GameOutcome {
    match (self.kings[0].0 == 0, self.kings[1].0 == 0) {
      (false, false) => GameOutcome::Ongoing,
      (true, false) => GameOutcome::WhiteWin,
      (false, true) => GameOutcome::BlackWin,
      (true, true) => unreachable!(),
    }
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

  pub fn move_gen<const QUIESCENCE: bool>(&self, moves: &mut Vec<Move>) {
    match self.white_turn {
      true => self.move_gen_for_color::<QUIESCENCE, true>(moves),
      false => self.move_gen_for_color::<QUIESCENCE, false>(moves),
    }
  }

  pub fn apply_move(&mut self, m: Move) -> bool {
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
