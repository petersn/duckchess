
export type MessageToEngineWorker = {
  type: 'init';
} | {
  type: 'applyMove';
  move: any;
  isHidden: boolean;
} | {
  type: 'setRunEngine';
  runEngine: boolean;
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
};

export type MessageToSearchWorker = {
  type: 'init';
} | {
  type: 'applyMove';
  move: any;
  isHidden: boolean;
} | {
  type: 'setRunEngine';
  runEngine: boolean;
};

export type MessageFromSearchWorker = {
  type: 'initted';
};

