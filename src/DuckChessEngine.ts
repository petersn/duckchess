import init, { GameTree, new_game_tree } from "engine";
import { PieceKind } from "./ChessBoard";

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

export function parseRustBoardState(state: any): PieceKind[][] {
  let board: PieceKind[][] = [];
  for (let y = 0; y < 8; y++)
    board.push([null, null, null, null, null, null, null, null]);
  for (let y = 0; y < 8; y++) {
    for (let player = 0; player < 2; player++) {
      for (const name of ['pawns', 'knights', 'bishops', 'rooks', 'queens', 'kings', 'ducks']) {
        let byte;
        // Special handling for the duck
        if (name === 'ducks') {
          if (player === 1)
            continue;
          byte = state.ducks[7 - y];
        } else if (name === 'enpassants') {
          if (player === 1)
            continue;
          byte = state.enPassant[7 - y];
        } else {
          byte = state[name][player][7 - y];
        }
        for (let x = 0; x < 8; x++) {
          const hasPiece = byte & 1;
          byte = byte >> 1;
          if (!hasPiece)
            continue;
          let piece = name.replace('knight', 'night').slice(0, 1).toUpperCase();
          if (player === 1)
            piece = 'w' + piece;
          else
            piece = 'b' + piece;
          piece = piece.replace('bD', 'duck');
          board[y][x] = piece as PieceKind;
        }
      }
    }
  }
  return board;
}

export class DuckChessEngine {
  forceUpdateCallback: () => void;
  engineWorker: Worker;
  searchWorker: Worker;
  gameTree: GameTree;
  initFlags: boolean[] = [false, false];
  resolveInitPromise: (value: null) => void = null as any;
  initPromise: Promise<null>;

  constructor(forceUpdateCallback: () => void) {
    this.forceUpdateCallback = forceUpdateCallback;
    this.engineWorker = new Worker(new URL('./EngineWorker.ts', import.meta.url));
    this.engineWorker.onmessage = this.onEngineMessage;
    this.searchWorker = new Worker(new URL('./SearchWorker.ts', import.meta.url));
    this.searchWorker.onmessage = this.onSearchMessage;
    for (const worker of [this.engineWorker, this.searchWorker]) {
      worker.postMessage({ type: 'init' });
    }
    this.gameTree = new_game_tree();
    this.initPromise = new Promise((resolve) => {
      this.resolveInitPromise = resolve;
    });
  }

  setInitFlag(workerIndex: number) {
    this.initFlags[workerIndex] = true;
    if (this.initFlags[0] && this.initFlags[1])
      this.resolveInitPromise(null);
  }

  historyBack() {
    this.gameTree.history_back();
    this.forceUpdateCallback();
  }

  historyForward() {
    this.gameTree.history_forward();
    this.forceUpdateCallback();
  }

  setRunEngine(runEngine: boolean) {
    this.engineWorker.postMessage({ type: 'setRunEngine', runEngine });
  }

  runAlphaBetaBenchmark(callback: (results: AlphaBetaBenchmarkResults) => void) {
    //this.searchWorker.postMessage({ type: 'runAlphaBetaBenchmark' });
  }

  onEngineMessage = (e: MessageEvent<MessageFromEngineWorker>) => {
    //console.log('Main thread got:', e.data);
    switch (e.data.type) {
      case 'initted':
        this.setInitFlag(0);
        break;
      case 'evaluation':
        const whiteWinProb = e.data.whiteWinProb;
        const Q = 2 * whiteWinProb - 1;
        //this.evaluation = 1.11714640912 * Math.tan(1.5620688421 * Q);
        //this.pv = e.data.pv;
        //this.nodes = e.data.nodes;
        break;
    }
    this.forceUpdateCallback();
  }

  onSearchMessage = (e: MessageEvent<MessageFromSearchWorker>) => {
    console.log('Main thread got:', e.data);
    switch (e.data.type) {
      case 'initted':
        this.setInitFlag(1);
        break;
      case 'alphaBetaBenchmarkResults':
        //this.benchmarkCallback(e.data.results);
        //this.benchmarkCallback = () => {};
        break;
    }
    this.forceUpdateCallback();
  }
}

export async function createDuckChessEngine(
  forceUpdate: () => void,
): Promise<DuckChessEngine> {
  await init();
  const engine = new DuckChessEngine(forceUpdate);
  await engine.initPromise;
  return engine;
}
