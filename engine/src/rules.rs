include!(concat!(env!("OUT_DIR"), "/tables.rs"));

use serde::{ser::SerializeSeq, Deserialize, Serialize};

use crate::nnue::{Nnue, UndoCookie, DUCK_LAYER};

pub const IS_DUCK_CHESS: bool = true;

pub const MOVE_HISTORY_LEN: usize = 4;

#[rustfmt::skip]
const ALL_BUT_A_FILE:   u64 = 0xfefefefefefefefe;
const ALL_BUT_H_FILE: u64 = 0x7f7f7f7f7f7f7f7f;
const MIDDLE_SIX_RANKS: u64 = 0x00ffffffffffff00;
const LAST_RANKS: u64 = 0xff000000000000ff;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Player {
  Black = 0,
  White = 1,
}

impl Player {
  pub fn other_player(self) -> Player {
    match self {
      Player::White => Player::Black,
      Player::Black => Player::White,
    }
  }
}

#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CastlingRights {
  pub king_side:  bool,
  pub queen_side: bool,
}

#[derive(Debug, Clone, Hash)]
pub struct BitBoard(pub u64);

impl BitBoard {
  pub fn vertically_mirror(self) -> BitBoard {
    BitBoard(self.0.swap_bytes())
  }
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
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
  pub castling_rights: [CastlingRights; 2],
  pub turn:            Player,
  pub is_duck_move:    bool,
  pub move_history:    [Option<Move>; MOVE_HISTORY_LEN],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Move {
  pub from: Square,
  pub to:   Square,
  //pub promotion: Option<PromotablePiece>,
}

impl Move {
  pub fn to_index(&self) -> u16 {
    let from = self.from as u16;
    let to = self.to as u16;
    from * 64 + to
  }

  pub fn from_index(index: u16) -> Self {
    let from = (index / 64) as Square;
    let to = (index % 64) as Square;
    Move { from, to }
  }

  pub fn to_uci(&self) -> String {
    let from = self.from as usize;
    let to = self.to as usize;
    format!(
      "{}{}{}{}",
      ((from % 8) as u8 + b'a') as char,
      ((from / 8) as u8 + b'1') as char,
      ((to % 8) as u8 + b'a') as char,
      ((to / 8) as u8 + b'1') as char,
    )
  }

  pub fn from_uci(uci: &str) -> Option<Self> {
    let mut chars = uci.chars();
    let from = chars.next()?;
    let from = from as u8 - b'a';
    let from = from + 8 * (chars.next()? as u8 - b'1');
    let to = chars.next()?;
    let to = to as u8 - b'a';
    let to = to + 8 * (chars.next()? as u8 - b'1');
    Some(Move { from, to })
  }
}

// Implement Display for Move, printing as SAN, like e2e4.
// FIXME: Repetition here with the above.
impl std::fmt::Display for Move {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    let from = self.from;
    let to = self.to;
    write!(
      f,
      "{}{}{}{}",
      (b'a' + (from % 8)) as char,
      (b'1' + (from / 8)) as char,
      (b'a' + (to % 8)) as char,
      (b'1' + (to / 8)) as char,
    )
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameOutcome {
  Win(Player),
  Draw,
}

impl GameOutcome {
  pub fn to_str(&self) -> &'static str {
    match self {
      GameOutcome::Win(Player::White) => "1-0",
      GameOutcome::Win(Player::Black) => "0-1",
      GameOutcome::Draw => "1/2-1/2",
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
      turn:            Player::White,
      is_duck_move:    false,
      move_history:    [None; MOVE_HISTORY_LEN],
    }
  }

  pub fn get_transposition_table_hash(&self) -> u64 {
    const MULT: u64 = 0x243f6a8885a308d3;
    let mut x = 0;
    macro_rules! hash_u64 {
      ($x:expr, $value:expr) => {{
        $x ^= $value;
        let x = x.wrapping_mul(MULT);
        $x = x ^ (x >> 37);
      }};
    }
    macro_rules! hash_piece {
      ($x:ident, $piece:expr) => {{
        for i in [0, 1] {
          hash_u64!($x, $piece[i].0);
        }
      }};
    }
    hash_piece!(x, self.pawns);
    hash_piece!(x, self.knights);
    hash_piece!(x, self.bishops);
    hash_piece!(x, self.rooks);
    hash_piece!(x, self.queens);
    hash_piece!(x, self.kings);
    hash_u64!(x, self.ducks.0);
    hash_u64!(x, self.en_passant.0);
    // Intentionally skip move history.
    let bits = (self.castling_rights[0].king_side as u64)
      ^ (self.castling_rights[0].queen_side as u64) << 8
      ^ (self.castling_rights[1].king_side as u64) << 16
      ^ (self.castling_rights[1].queen_side as u64) << 24
      ^ (self.turn as u64) << 32
      ^ (self.is_duck_move as u64) << 40;
    hash_u64!(x, bits);
    x
  }

  // TODO: Implement stalemate.
  pub fn get_outcome(&self) -> Option<GameOutcome> {
    match (
      self.kings[Player::Black as usize].0 == 0,
      self.kings[Player::White as usize].0 == 0,
    ) {
      (false, false) => None,
      (true, false) => Some(GameOutcome::Win(Player::White)),
      (false, true) => Some(GameOutcome::Win(Player::Black)),
      (true, true) => {
        // Print out the entire state.
        println!("{:?}", self);
        panic!("Both kings gone??");
      }
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
      let duck_square = get_square(self.ducks.0);
      while let Some(pos) = iter_bits(&mut duck_moves) {
        moves.push(Move {
          // Encode the initial duck placement as from == to.
          from: if self.ducks.0 == 0 { pos } else { duck_square },
          to:   pos,
          //promotion: None,
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
            from: get_square(our_king),
            to:   get_square(our_king) + 2,
            //promotion: None,
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
            from: get_square(our_king),
            to:   get_square(our_king) - 2,
            //promotion: None,
          });
        }
      }
    }

    // Find all moves for pawns.
    let (second_rank, seventh_rank): (u64, u64) = if IS_WHITE {
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

    //let mut promote_single_pawn_pushes = single_pawn_pushes & seventh_rank;
    //let mut promote_double_pawn_pushes = double_pawn_pushes & seventh_rank;
    //let mut promote_pawn_capture_left = pawn_capture_left & seventh_rank;
    //let mut promote_pawn_capture_right = pawn_capture_right & seventh_rank;

    // TODO: I should disable movement from the seventh rank again once I separate out promotions.
    //single_pawn_pushes &= !seventh_rank;
    //double_pawn_pushes &= !seventh_rank;
    //pawn_capture_left &= !seventh_rank;
    //pawn_capture_right &= !seventh_rank;

    macro_rules! add_pawn_moves {
      ("plain", $bits:ident, $moves:ident, $delta:expr) => {
        while let Some(pos) = iter_bits(&mut $bits) {
          $moves.push(Move {
            from: pos,
            to:   pos.wrapping_add($delta),
            //promotion: None,
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
    // add_pawn_moves!(
    //   "promotion",
    //   promote_pawn_capture_left,
    //   moves,
    //   forward_one_row - 1
    // );
    // add_pawn_moves!(
    //   "promotion",
    //   promote_pawn_capture_right,
    //   moves,
    //   forward_one_row + 1
    // );
    // add_pawn_moves!(
    //   "promotion",
    //   promote_single_pawn_pushes,
    //   moves,
    //   forward_one_row
    // );
    // add_pawn_moves!(
    //   "promotion",
    //   promote_double_pawn_pushes,
    //   moves,
    //   forward_one_row * 2
    // );
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
          from: pos,
          to:   knight_move,
          //promotion: None,
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
          from: pos,
          to:   king_move,
          //promotion: None,
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
          from: pos,
          to:   rook_move,
          //promotion: None,
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
          from: pos,
          to:   bishop_move,
          //promotion: None,
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
          from: pos,
          to:   queen_move,
          //promotion: None,
        });
      }
    }
  }

  pub fn move_gen<const QUIESCENCE: bool>(&self, moves: &mut Vec<Move>) {
    if self.get_outcome().is_some() {
      return;
    }
    match self.turn {
      Player::White => self.move_gen_for_color::<QUIESCENCE, true>(moves),
      Player::Black => self.move_gen_for_color::<QUIESCENCE, false>(moves),
    }
  }

  pub fn apply_move<const NNUE: bool>(
    &mut self,
    m: Move,
    nnue: Option<&mut Nnue>,
  ) -> Result<UndoCookie, &'static str> {
    let mut undo_cookie = UndoCookie::new();
    let nnue = if NNUE {
      nnue.unwrap()
    } else {
      // Make a fake dangling referenc that we won't use.
      unsafe { &mut *(0x1 as *mut Nnue) }
    };
    if m.from > 63 || m.to > 63 {
      return Err("from/to out of range");
    }

    // Advance the move history by one position.
    for i in (1..MOVE_HISTORY_LEN).rev() {
      self.move_history[i] = self.move_history[i - 1];
    }
    self.move_history[0] = Some(m);

    let from_mask = 1 << m.from;
    let to_mask = 1 << m.to;
    //if !self.is_duck_move {
    //  self.highlight.0 = 0;
    //}
    //self.highlight.0 |= from_mask | to_mask;
    // Handle duck moves.
    if self.is_duck_move {
      if self.ducks.0 & from_mask != 0 || (self.ducks.0 == 0 && m.from == m.to) {
        let extant_duck = self.ducks.0 != 0;
        self.ducks.0 &= !from_mask;
        self.ducks.0 |= to_mask;
        if NNUE {
          assert_eq!(undo_cookie.sub_layers[0], u16::MAX);
          assert_eq!(undo_cookie.add_layers[0], u16::MAX);
          if extant_duck {
            undo_cookie.sub_layers[0] = 64 * DUCK_LAYER as u16 + m.from as u16;
            nnue.sub_layer(undo_cookie.sub_layers[0]);
          }
          undo_cookie.add_layers[0] = 64 * DUCK_LAYER as u16 + m.to as u16;
          nnue.add_layer(undo_cookie.add_layers[0]);
        }
        self.is_duck_move = false;
        self.turn = self.turn.other_player();
        return Ok(undo_cookie);
      }
      return Err("no duck at from position");
    }

    let moving_en_passant = self.en_passant.0 & to_mask != 0;
    self.en_passant.0 = 0;
    let mut remove_rooks = 0;
    let mut new_queens = 0;
    let mut new_rooks = 0;
    //let mut new_bishops = 0;
    //let mut new_knights = 0;

    #[derive(Debug)]
    enum PieceKind {
      Pawn,
      King,
      Rook,
      Knight,
      Bishop,
      Queen,
    }
    let mut piece_moved = false;
    for (piece_layer_number, (piece_kind, piece_array)) in [
      (PieceKind::Pawn, &mut self.pawns),
      (PieceKind::Knight, &mut self.knights),
      (PieceKind::Bishop, &mut self.bishops),
      (PieceKind::Rook, &mut self.rooks),
      (PieceKind::Queen, &mut self.queens),
      (PieceKind::King, &mut self.kings),
    ]
    .into_iter()
    .enumerate()
    {
      let our_nnue_layer = piece_layer_number + 6 * self.turn as usize;
      let their_nnue_layer = piece_layer_number + 6 * self.turn.other_player() as usize;

      let (a, b) = piece_array.split_at_mut(1);
      let (us, them) = match self.turn {
        Player::White => (&mut b[0], &mut a[0]),
        Player::Black => (&mut a[0], &mut b[0]),
      };
      // Capture pieces on the target square.
      them.0 &= !to_mask;
      if NNUE && them.0 & to_mask != 0 {
        assert_eq!(undo_cookie.sub_layers[1], u16::MAX);
        undo_cookie.sub_layers[1] = 64 * their_nnue_layer as u16 + m.to as u16;
        nnue.sub_layer(undo_cookie.sub_layers[1]);
      }
      // Check if this is the kind of piece we're moving.
      if us.0 & from_mask != 0 {
        piece_moved = true;
        // Move our piece.
        us.0 &= !from_mask;
        us.0 |= to_mask;
        if NNUE {
          assert_eq!(undo_cookie.sub_layers[0], u16::MAX);
          assert_eq!(undo_cookie.add_layers[0], u16::MAX);
          undo_cookie.sub_layers[0] = 64 * our_nnue_layer as u16 + m.from as u16;
          undo_cookie.add_layers[0] = 64 * our_nnue_layer as u16 + m.to as u16;
          nnue.sub_add_layers(undo_cookie.sub_layers[0], undo_cookie.add_layers[0]);
        }
        // Handle special rules for special pieces.
        match (piece_kind, m.from) {
          (PieceKind::Pawn, _) => {
            // Check if we're taking en passant.
            if moving_en_passant {
              match self.turn {
                Player::White => {
                  them.0 &= !(1 << (m.to - 8));
                  if NNUE {
                    assert_eq!(undo_cookie.sub_layers[1], u16::MAX);
                    undo_cookie.sub_layers[1] = 64 * their_nnue_layer as u16 + (m.to - 8) as u16;
                    nnue.sub_layer(undo_cookie.sub_layers[1]);
                  }
                }
                Player::Black => {
                  them.0 &= !(1 << (m.to + 8));
                  if NNUE {
                    assert_eq!(undo_cookie.sub_layers[1], u16::MAX);
                    undo_cookie.sub_layers[1] = 64 * their_nnue_layer as u16 + (m.to + 8) as u16;
                    nnue.sub_layer(undo_cookie.sub_layers[1]);
                  }
                }
              }
            }
            // Setup the en passant state.
            let is_double_move = (m.from as i8 - m.to as i8).abs() == 16;
            match (is_double_move, self.turn) {
              (true, Player::White) => self.en_passant.0 = to_mask >> 8,
              (true, Player::Black) => self.en_passant.0 = to_mask << 8,
              (false, _) => {}
            }
            // Check if we're promoting.
            //match &m.promotion {
            //  Some(promotion) => {
            // Prevent promotions that aren't on the first or last rank.
            // This allows our move encoding to be lazy, and just say every move is promoting to queen.
            let promotion_mask = to_mask & LAST_RANKS;
            // Remove the pawn, and setup the promotion.
            us.0 &= !promotion_mask;
            new_queens = promotion_mask;
            if NNUE {
              let our_queen_layer = 2 * 4 + self.turn as usize;
              assert_eq!(undo_cookie.sub_layers[1], u16::MAX); // FIXME: This one can fail!
              assert_eq!(undo_cookie.add_layers[1], u16::MAX);
              undo_cookie.sub_layers[1] = 64 * our_nnue_layer as u16 + m.to as u16;
              undo_cookie.add_layers[1] = 64 * our_queen_layer as u16 + m.to as u16;
              nnue.sub_add_layers(undo_cookie.sub_layers[1], undo_cookie.add_layers[1]);
            }
            //match promotion {
            //  PromotablePiece::Queen => new_queens = promotion_mask,
            //  PromotablePiece::Rook => new_rooks = promotion_mask,
            //  PromotablePiece::Knight => new_knights = promotion_mask,
            //  PromotablePiece::Bishop => new_bishops = promotion_mask,
            //}
            //  }
            //  None => {}
            //}
          }
          (PieceKind::King, _) => {
            self.castling_rights[self.turn as usize] = CastlingRights {
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
                _ => panic!("Invalid castling move"),
              };
              remove_rooks = 1 << rook_from;
              new_rooks = 1 << rook_to;
              if NNUE {
                let our_rook_layer = 2 * 3 + self.turn as usize;
                assert_eq!(undo_cookie.sub_layers[1], u16::MAX);
                assert_eq!(undo_cookie.add_layers[1], u16::MAX);
                undo_cookie.sub_layers[1] = 64 * our_rook_layer as u16 + rook_from as u16;
                undo_cookie.add_layers[1] = 64 * our_rook_layer as u16 + rook_to as u16;
                nnue.sub_add_layers(undo_cookie.sub_layers[1], undo_cookie.add_layers[1]);
              }
            }
          }
          (PieceKind::Rook, 0) | (PieceKind::Rook, 56) => {
            self.castling_rights[self.turn as usize].queen_side = false
          }
          (PieceKind::Rook, 7) | (PieceKind::Rook, 63) => {
            self.castling_rights[self.turn as usize].king_side = false
          }
          (PieceKind::Rook, _) => {}
          (PieceKind::Knight, _) => {}
          (PieceKind::Bishop, _) => {}
          (PieceKind::Queen, _) => {}
        }
      }
    }
    if !piece_moved {
      return Err("no piece at from position");
    }
    self.rooks[self.turn as usize].0 &= !remove_rooks;
    //self.knights[self.turn as usize].0 |= new_knights;
    //self.bishops[self.turn as usize].0 |= new_bishops;
    self.rooks[self.turn as usize].0 |= new_rooks;
    self.queens[self.turn as usize].0 |= new_queens;

    if IS_DUCK_CHESS {
      self.is_duck_move = true;
    } else {
      self.turn = self.turn.other_player();
    }
    Ok(undo_cookie)
  }

  pub fn sanity_check(&self) -> Result<(), &'static str> {
    let mut all_pieces = self.ducks.0;
    for player in [Player::Black, Player::White] {
      for pieces in [
        &self.pawns[player as usize],
        &self.knights[player as usize],
        &self.bishops[player as usize],
        &self.rooks[player as usize],
        &self.queens[player as usize],
        &self.kings[player as usize],
      ] {
        if all_pieces & pieces.0 != 0 {
          return Err("pieces overlap");
        }
        all_pieces |= pieces.0;
      }
    }
    Ok(())
  }
}
