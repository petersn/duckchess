import init, { GameTree, new_game_tree, parse_pgn4 } from "engine";
import { PieceKind } from "./ChessBoard";

type BasicMessages = {
  type: 'init';
} | {
  type: 'setState';
  state: any;
} | {
  type: 'setRunEngine';
  runEngine: boolean;
};

export type MessageToEngineWorker = BasicMessages;

export type MessageFromEngineWorker = {
  type: 'initted';
} | {
  type: 'modelLoadProgress';
  progress: number;
} | {
  type: 'engineOutput';
  engineOutput: any;
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
  engineOutput: any;
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

export type PlayModeInfo = {
  player: 'white' | 'black';
  steps: number;
} | null;

export class DuckChessEngine {
  loadProgressCallback: (progress: number) => void;
  forceUpdateCallback: () => void;
  engineWorker: Worker;
  searchWorker: Worker;
  gameTree: GameTree;
  initFlags: boolean[] = [false, false];
  resolveInitPromise: (value: null) => void = null as any;
  initPromise: Promise<null>;
  playMode: PlayModeInfo = null;
  thinkProgress: number = 0;

  constructor(
    loadProgressCallback: (progress: number) => void,
    forceUpdateCallback: () => void,
  ) {
    this.loadProgressCallback = loadProgressCallback;
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
    this.sendBoardToEngine();
    this.forceUpdateCallback();
  }

  historyForward() {
    this.gameTree.history_forward();
    this.sendBoardToEngine();
    this.forceUpdateCallback();
  }

  sendBoardToEngine() {
    const { state } = this.gameTree.get_serialized_state()[0];
    this.engineWorker.postMessage({ type: 'setState', state });
    this.searchWorker.postMessage({ type: 'setState', state });
  }

  setPgn4(pgn4: string) {
    const result = parse_pgn4(pgn4);
    console.log('Parsing result:', result);
    if (result.Err !== undefined) {
      window.alert('Error parsing PGN4: ' + result.Err);
      return;
    }
    const { headers, moves } = result.Ok;
    // Update the tree.
    this.gameTree = new_game_tree();
    for (let i = 0; i < moves.length; i++) {
      const move = moves[i];
      const success = this.gameTree.make_move(move, false);
      if (!success) {
        window.alert(`Error parsing PGN4: invalid move ${JSON.stringify(move)} at index ${i}`);
        return;
      }
    }
    // Update the engine.
    this.sendBoardToEngine();
    this.forceUpdateCallback();
  }

  setRunEngine(runEngine: boolean) {
    for (const worker of [this.engineWorker, this.searchWorker])
      worker.postMessage({ type: 'setRunEngine', runEngine });
  }

  setPlayMode(playMode: PlayModeInfo) {
    this.playMode = playMode;
    if (this.playMode === null)
      return;
    this.engineWorker.postMessage({ type: 'setRunEngine', runEngine: true });
  }

  runAlphaBetaBenchmark(callback: (results: AlphaBetaBenchmarkResults) => void) {
    //this.searchWorker.postMessage({ type: 'runAlphaBetaBenchmark' });
  }

  maybeMakeEngineMove() {
    if (this.playMode === null)
      return;
    const move = this.gameTree.get_play_move(
      this.playMode.player,
      this.playMode.steps,
    );
    // Numbers represent progress.
    if (typeof move === 'number') {
      if (move !== this.thinkProgress) {
        this.thinkProgress = move;
        this.forceUpdateCallback();
      }
    } else if (move !== null) {
      // Non-null represents a move.
      console.log('Making engine move:', move);
      this.gameTree.make_move(move, false)
      this.sendBoardToEngine();
    }
  }

  onEngineMessage = (e: MessageEvent<MessageFromEngineWorker>) => {
    //console.log('Main thread got:', e.data);
    switch (e.data.type) {
      case 'initted':
        this.setInitFlag(0);
        break;
      case 'modelLoadProgress':
        this.loadProgressCallback(e.data.progress);
        break;
      case 'engineOutput':
        const engineOutput = e.data.engineOutput;
        this.gameTree.apply_engine_output(engineOutput);
        this.maybeMakeEngineMove();
        break;
      //case 'evaluation':
      //  const whiteWinProb = e.data.whiteWinProb;
      //  const Q = 2 * whiteWinProb - 1;
      //  //this.evaluation = 1.11714640912 * Math.tan(1.5620688421 * Q); // 4 * tan(3.14 * x - 1.57)
      //  //this.pv = e.data.pv;
      //  //this.nodes = e.data.nodes;
      //  break;
    }
    this.forceUpdateCallback();
  }

  onSearchMessage = (e: MessageEvent<MessageFromSearchWorker>) => {
    //console.log('Main thread got:', e.data);
    switch (e.data.type) {
      case 'initted':
        this.setInitFlag(1);
        break;
      case 'mateSearch':
        const engineOutput = e.data.engineOutput;
        this.gameTree.apply_engine_output(engineOutput);
        this.maybeMakeEngineMove();
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
  loadProgressCallback: (progress: number) => void,
  forceUpdate: () => void,
): Promise<DuckChessEngine> {
  await init();
  const engine = new DuckChessEngine(loadProgressCallback, forceUpdate);
  await engine.initPromise;
  return engine;
}
