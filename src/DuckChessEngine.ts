
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

/*
export class BoardState {
  
}

export class TreeEdge {
  move: any;
  child: MoveNode;

  constructor(move: any, child: MoveNode) {
    this.move = move;
    this.child = child;
  }
}

export class MoveNode {
  board: BoardState;
  children: TreeEdge[];

  constructor(board: BoardState) {
    this.board = board;
    this.children = [];
  }
}

export class DuckChessEngine {
  engineWorker: Worker;
  searchWorker: Worker;
  moveTree: MoveNode;

  constructor() {
    this.engineWorker = new Worker(new URL('./EngineWorker.ts', import.meta.url));
    this.engineWorker.onmessage = this.onEngineMessage;
    this.searchWorker = new Worker(new URL('./SearchWorker.ts', import.meta.url));
    this.searchWorker.onmessage = this.onSearchMessage;
    this.moveTree = new MoveNode();
  }
}
*/
