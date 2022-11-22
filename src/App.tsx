import { engine } from '@tensorflow/tfjs';
import React from 'react';
import './App.css';
import { MessageFromEngineWorker } from './EngineWorkerMessages';
import { ChessBoard, ChessPiece, PieceKind } from './ChessBoard';

class EngineWorker {
  worker: Worker;
  initCallback: (ew: EngineWorker) => void;
  boardState: any;
  legalMoves: any[];
  nextMoves: any[];
  pv: any[] = [];
  evaluation: number = 0;
  forceUpdateCallback: () => void;

  constructor(initCallback: () => void, forceUpdateCallback: () => void) {
    this.worker = new Worker(new URL('./EngineWorker.ts', import.meta.url));
    this.worker.onmessage = this.onMessage;
    this.initCallback = initCallback;
    this.forceUpdateCallback = forceUpdateCallback;
    this.worker.postMessage({ type: 'init' });
    this.boardState = null;
    this.legalMoves = [];
    this.nextMoves = [];
  }

  onMessage = (e: MessageEvent<MessageFromEngineWorker>) => {
    console.log('Main thread got:', e.data);
    switch (e.data.type) {
      case 'initted':
        this.initCallback(this);
        break;
      case 'board':
        this.boardState = e.data.board;
        this.legalMoves = e.data.moves;
        this.nextMoves = e.data.nextMoves;
        break;
      case 'evaluation':
        this.evaluation = e.data.evaluation;
        this.pv = e.data.pv;
        break;
    }
    this.forceUpdateCallback();
  }

  getState() {
    return this.boardState;
  }

  getMoves(): any[] {
    return this.legalMoves;
  }

  applyMove(move: any, isHidden: boolean) {
    this.worker.postMessage({ type: 'applyMove', move, isHidden });
  }
}

function App(props: {}) {
  const [selectedSquare, setSelectedSquare] = React.useState<[number, number] | null>(null);
  const [forceUpdateCounter, setForceUpdateCounter] = React.useState(0);
  const [pair, setPair] = React.useState<any>(null);

  const [engineWorker, setEngineWorker] = React.useState<EngineWorker | null>(null);
  React.useEffect(() => {
    console.log('Initializing worker...');
    const worker = new EngineWorker(
      () => {
        console.log('Worker initialized!');
        setEngineWorker(worker);
      },
      () => {
        setTimeout(() => setForceUpdateCounter(Math.random()), 10);
//        setForceUpdateCounter(forceUpdateCounter + 1);
      },
    );
  }, []);

  // let board: React.ReactNode[][] = [];
  // for (let y = 0; y < 8; y++)
  //   board.push([null, null, null, null, null, null, null, null]);
  // let legalMoves: any[] = [];
  // if (engineWorker !== null) {
  //   board = engineWorker.getState() || board;
  //   legalMoves = engineWorker.getMoves() || legalMoves;
  // }

  let board: React.ReactNode[][] = [];
  for (let y = 0; y < 8; y++)
    board.push([null, null, null, null, null, null, null, null]);
  let legalMoves: any[] = [];
  let hiddenLegalMoves: any[] = [];
  if (engineWorker !== null) {
    const state = engineWorker.getState();
    legalMoves = engineWorker.getMoves();
    hiddenLegalMoves = engineWorker.nextMoves;
    console.log('HIDDEN LEGAL MOVES', hiddenLegalMoves);
    if (state !== null) {
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
              board[y][x] = piece;
            }
          }
        }
      }
    }
  }

  //React.useEffect(() => {
  //  engineWorker.forceUpdateCallback = () => {
  //    setForceUpdateCounter(Math.random());
  //  };
  //}, []);

  //const { engineWorker } = props;
  //const state = engineWorker.getState();
  //const moves: any[] = engineWorker.getMoves();
  //React.useEffect(() => {
  //  //for (let i = 0; i < 100; i++) {
  //  //  const start = performance.now();
  //  //  const tfjsResult = tfjsModel.predict(tf.ones([64, 8, 8, 22])) as tf.Tensor[];
  //  //  tfjsResult[0].data()
  //  //  const end = performance.now();
  //  //  console.log('tfjs prediction took', end - start, 'ms');
  //  //}
  //  //console.log(tfjsResult[0], tfjsResult[1]);
  //  let pair = engineWorker.run(4);
  //  setPair(pair);
  //  setTimeout(() => {
  //    if (pair && pair[1][0]) {
  //      engine.apply_move(pair[1][0]);
  //      setForceUpdateCounter(forceUpdateCounter + 1);
  //    }
  //  }, 1000000);
  //}, [forceUpdateCounter]);

  //if (state === null) {
  //  return <div>Loading...</div>;
  //}

  /*
  function clickOn(x: number, y: number) {
    if (selectedSquare === null) {
      // Check if this is the initial duck placement.
      const m = moves.find(m => m.from === 64 && m.to === x + (7 - y) * 8);
      if (m) {
        engineWorker.applyMove(m);
        setSelectedSquare(null);
        setForceUpdateCounter(forceUpdateCounter + 1);
      } else {
        setSelectedSquare([x, y]);
      }
    } else {
      let [fromX, fromY] = selectedSquare;
      fromY = 7 - fromY;
      y = 7 - y;
      const encodedFrom = 8 * fromY + fromX;
      const encodedTo = 8 * y + x;
      // Find the first move that matches the selected square and the clicked square.
      const m = moves.find((m: any) => m.from === encodedFrom && m.to === encodedTo);
      if (m) {
        engineWorker.applyMove(m);
        setForceUpdateCounter(forceUpdateCounter + 1);
      }
      setSelectedSquare(null);
    }
  }
  */


  /*
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
          let piece = name.slice(0, -1);
          if (player === 1)
            piece = piece.toUpperCase();
          board[y][x] = <span style={{
            fontSize: '40px',
            userSelect: 'none',
            fontFamily: 'Arial Unicode MS',
          }}>
            <ChessPiece piece={piece as PieceKind} />
          </span>;
        }
      }
    }
  }
  */

  /*
  //let showMoves: any[] = (pair && pair[1][0]) ? pair[1] : [];
  console.log('----PV:', engineWorker.pv);
  let showMoves: any[] = engineWorker.pv;
  if ((state.isDuckMove || true) && showMoves) {
    showMoves = showMoves.slice(0, 1);
  }
  */

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
    }}>
      <div style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <div style={{
          flex: 1,
          height: 600,
          minHeight: 600,
        }}>
          <div style={{
            marginLeft: 'auto',
            height: '100%',
            maxWidth: 400,
            border: '1px solid #eee',
            padding: 10,
            boxSizing: 'border-box',
            overflow: 'scroll',
          }}>
            <div style={{ fontWeight: 'bold', fontSize: '120%', marginBottom: 10 }}>Duck Chess Analysis Board</div>

            <a href="https://duckchess.com/">Duck chess</a> is a variant in which there is one extra piece, the duck, which is shared between the two players and cannot be captured.
            Each turn consists of making a regular move, and then moving the duck to any empty square (the duck may not stay in place).
            There is no check or checkmate, you win by capturing the opponent's king.
          </div>
        </div>
        <ChessBoard
          board={board as any}
          legalMoves={legalMoves}
          hiddenLegalMoves={hiddenLegalMoves}
          onMove={(move, isHidden) => {
            if (engineWorker !== null) {
              console.log('[snp1] move', move, isHidden);
              engineWorker.applyMove(move, isHidden);
            }
          }}
        />
        <div style={{
          flex: 1,
          height: 600,
          minHeight: 600,
        }}>
          <div style={{
            height: '100%',
            maxWidth: 400,
            border: '1px solid #eee',
            padding: 10,
            boxSizing: 'border-box',
          }}>
            <div style={{ fontWeight: 'bold', fontSize: '120%', marginBottom: 10 }}>Engine</div>
          </div>
        </div>
      </div>

      <div style={{ margin: 20, textAlign: 'center' }}>
        Created by Peter Schmidt-Nielsen
        (<a href="https://twitter.com/ptrschmdtnlsn">Twitter</a>, <a href="https://peter.website">Website</a>)<br/>
        Engine + web interface: <a href="https://github.com/petersn/duckchess">github.com/petersn/duckchess</a><br/>
        <span style={{ fontSize: '50%', opacity: 0.5 }}>Piece SVGs: Cburnett (CC BY-SA 3), Duck SVG + all code: my creation (CC0)</span>
      </div>
    </div>
  );
}

export default App;
