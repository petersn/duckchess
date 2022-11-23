
export type MessageToEngineWorker = {
  type: 'init';
} | {
  type: 'applyMove';
  move: any;
  isHidden: boolean;
};

export type MessageFromEngineWorker = {
  type: 'initted';
} | {
  type: 'board';
  board: any;
  moves: any[];
  nextMoves: any[];
} | {
  type: 'evaluation';
  whiteWinProb: number;
  nodes: number;
  pv: {
    from: number;
    to: number;
  }[];
}
