
type BasicMessages = {
  type: 'init';
} | {
  type: 'applyMove';
  move: any;
  isHidden: boolean;
} | {
  type: 'historyJump';
  index: number;
} | {
  type: 'setRunEngine';
  runEngine: boolean;
};

export type MessageToEngineWorker = BasicMessages;

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

export type MessageToSearchWorker = BasicMessages | {
  type: 'runAlphaBetaBenchmark';
};

export type MessageFromSearchWorker = {
  type: 'initted';
} | {
  type: 'alphaBetaBenchmarkResults';
  results: AlphaBetaBenchmarkResults;
} | {
  type: 'mateSearch';
  mateEval: number;
  nextMoves: any[];
};

export interface AlphaBetaBenchmarkResults {
  megaNodesPerSecondRaw: number;
  megaNodesPerSecondEval: number;
  megaNodesPerSecondNnue: number;
}
