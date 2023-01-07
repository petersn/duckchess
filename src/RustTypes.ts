// Generated by concat_types.py, from the output of https://github.com/Aleph-Alpha/ts-rs
// Run `cargo test` then `python concat_types.py` to rebuild.

export interface State {
  pawns: Array<BitBoard>,
  knights: Array<BitBoard>,
  bishops: Array<BitBoard>,
  rooks: Array<BitBoard>,
  queens: Array<BitBoard>,
  kings: Array<BitBoard>,
  ducks: BitBoard,
  enPassant: BitBoard,
  castlingRights: Array<CastlingRights>,
  turn: Player,
  isDuckMove: boolean,
  moveHistory: Array<Move | null>,
  plies: number,
}

export interface CastlingRights {
  kingSide: boolean,
  queenSide: boolean,
}

export interface Move {
  from: number,
  to: number,
}

export type Player = 'black' | 'white';

export type BitBoard = number[];


