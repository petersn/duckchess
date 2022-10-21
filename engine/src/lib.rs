use serde::{ser::SerializeSeq, Deserialize, Serialize};
use wasm_bindgen::prelude::*;

include!(concat!(env!("OUT_DIR"), "/tables.rs"));

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[derive(Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct CastlingRights {
    king_side:  bool,
    queen_side: bool,
}

#[derive(Hash)]
struct BitBoard(u64);

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

#[derive(Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct State {
    pawns:           [BitBoard; 2],
    knights:         [BitBoard; 2],
    bishops:         [BitBoard; 2],
    rooks:           [BitBoard; 2],
    queens:          [BitBoard; 2],
    kings:           [BitBoard; 2],
    ducks:           BitBoard,
    en_passant:      BitBoard,
    highlight:       BitBoard,
    castling_rights: [CastlingRights; 2],
    white_turn:      bool,
    is_duck_move:    bool,
}

const ALL_BUT_A_FILE: u64 = 0xfefefefefefefefe;
const ALL_BUT_H_FILE: u64 = 0x7f7f7f7f7f7f7f7f;
const ALL_BUT_END_RANKS: u64 = 0x00ffffffffffff00;

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

impl State {
    fn starting_state() -> State {
        State {
            pawns:           [BitBoard(0x00ff000000000000), BitBoard(0x000000000000ff00)],
            knights:         [BitBoard(0x4200000000000000), BitBoard(0x0000000000000042)],
            bishops:         [BitBoard(0x2400000000000000), BitBoard(0x0000000000000024)],
            rooks:           [BitBoard(0x8100000000000000), BitBoard(0x0000000000000081)],
            queens:          [BitBoard(0x0800000000000000), BitBoard(0x0000000000000008)],
            kings:           [BitBoard(0x1000000000000000), BitBoard(0x0000000000000010)],
            ducks:           BitBoard(0x0000000000010000),
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

    fn move_gen_for_color<const is_white: bool>(&self, moves: &mut Vec<Move>) {
        macro_rules! shift_backward {
            ($board:expr) => {
                if is_white {
                    $board >> 8
                } else {
                    $board << 8
                }
            };
        }
        let forward_one_row: SquareDelta = if is_white { 8 } else { 248 };
        let our_pawns = self.pawns[is_white as usize].0;
        let our_knights = self.knights[is_white as usize].0;
        let our_bishops = self.bishops[is_white as usize].0;
        let our_rooks = self.rooks[is_white as usize].0;
        let our_queens = self.queens[is_white as usize].0;
        let our_king = self.kings[is_white as usize].0;
        let mut our_pieces: u64 =
            our_pawns | our_knights | our_bishops | our_rooks | our_queens | our_king;
        let mut their_pieces: u64 = self.pawns[(!is_white) as usize].0
            | self.knights[(!is_white) as usize].0
            | self.bishops[(!is_white) as usize].0
            | self.rooks[(!is_white) as usize].0
            | self.queens[(!is_white) as usize].0
            | self.kings[(!is_white) as usize].0;
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

        // Find all moves for pawns.
        let (first_rank, seventh_rank) = if is_white {
            (0x000000000000ff00, 0x00ff000000000000)
        } else {
            (0x00ff000000000000, 0x000000000000ff00)
        };
        let mut single_pawn_pushes = ALL_BUT_END_RANKS & our_pawns & !shift_backward!(occupied);
        let mut double_pawn_pushes =
            (single_pawn_pushes & first_rank) & !shift_backward!(shift_backward!(occupied));

        let pawn_capturable = their_pieces | self.en_passant.0;
        let mut pawn_capture_left =
            (our_pawns & ALL_BUT_A_FILE) & (shift_backward!(pawn_capturable & ALL_BUT_A_FILE) << 1);
        let mut pawn_capture_right =
            (our_pawns & ALL_BUT_H_FILE) & (shift_backward!(pawn_capturable & ALL_BUT_H_FILE) >> 1);
        let mut pawn_promotions =
            (single_pawn_pushes & seventh_rank) | pawn_capture_left | pawn_capture_right;
        while let Some(pos) = iter_bits(&mut single_pawn_pushes) {
            moves.push(Move {
                from:      pos,
                to:        pos + forward_one_row,
                promotion: None,
            });
        }
        while let Some(pos) = iter_bits(&mut double_pawn_pushes) {
            moves.push(Move {
                from:      pos,
                to:        pos + forward_one_row + forward_one_row,
                promotion: None,
            });
        }
        while let Some(pos) = iter_bits(&mut pawn_capture_left) {
            moves.push(Move {
                from:      pos,
                to:        pos + forward_one_row - 1,
                promotion: None,
            });
        }
        while let Some(pos) = iter_bits(&mut pawn_capture_right) {
            moves.push(Move {
                from:      pos,
                to:        pos + forward_one_row + 1,
                promotion: None,
            });
        }
        while let Some(pos) = iter_bits(&mut pawn_promotions) {
            for promotable_piece in [
                PromotablePiece::Queen,
                PromotablePiece::Knight,
                PromotablePiece::Rook,
                PromotablePiece::Bishop,
            ] {
                moves.push(Move {
                    from:      pos,
                    to:        pos + forward_one_row,
                    promotion: Some(promotable_piece),
                });
            }
        }

        // Find all moves for knights.
        let mut knights = our_knights;
        while let Some(pos) = iter_bits(&mut knights) {
            let mut knight_moves = KNIGHT_MOVES[pos as usize];
            knight_moves &= !our_pieces;
            while let Some(knight_move) = iter_bits(&mut knight_moves) {
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
            king_moves &= !our_pieces;
            while let Some(king_move) = iter_bits(&mut king_moves) {
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
            let mut rook_moves = attacked & !our_pieces;
            while let Some(rook_move) = iter_bits(&mut rook_moves) {
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
            let mut bishop_moves = attacked & !our_pieces;
            while let Some(bishop_move) = iter_bits(&mut bishop_moves) {
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
            let mut queen_moves = attacked & !our_pieces;
            while let Some(queen_move) = iter_bits(&mut queen_moves) {
                moves.push(Move {
                    from:      pos,
                    to:        queen_move,
                    promotion: None,
                });
            }
        }
    }

    fn move_gen(&self, moves: &mut Vec<Move>) {
        match self.white_turn {
            true => self.move_gen_for_color::<true>(moves),
            false => self.move_gen_for_color::<false>(moves),
        }
    }
}

#[derive(Hash, serde::Serialize, serde::Deserialize)]
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

#[derive(Hash, serde::Serialize, serde::Deserialize)]
struct Move {
    from:      Square,
    to:        Square,
    promotion: Option<PromotablePiece>,
}

struct MoveGen<'a> {
    state: &'a State,
}

impl<'a> MoveGen<'a> {
    fn new(state: &'a State) -> MoveGen<'a> {
        MoveGen { state }
    }
}

impl<'a> Iterator for MoveGen<'a> {
    type Item = Move;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

#[wasm_bindgen]
pub struct Engine {
    state: State,
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
        self.state = serde_wasm_bindgen::from_value(state).expect("serialization");
    }

    pub fn get_moves(&self) -> JsValue {
        let mut moves = Vec::new();
        self.state.move_gen(&mut moves);
        serde_wasm_bindgen::to_value(&moves).unwrap_or_else(|e| {
            log(&format!("Failed to serialize moves: {}", e));
            JsValue::NULL
        })
    }
}

#[wasm_bindgen]
pub fn new_engine() -> Engine {
    Engine {
        state: State::starting_state(),
    }
}
