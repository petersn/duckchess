
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
  evaluation: number;
  pv: {
    from: number;
    to: number;
  }[];
}
