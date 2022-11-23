import { engine } from '@tensorflow/tfjs';
import React from 'react';
import './App.css';
import { MessageFromEngineWorker, MessageFromSearchWorker } from './WorkerMessages';
import { ChessBoard, ChessPiece, PieceKind } from './ChessBoard';

class Workers {
  engineWorker: Worker;
  searchWorker: Worker;
  initCallback: (ew: Workers) => void;
  boardState: any;
  legalMoves: any[];
  nextMoves: any[];
  pv: any[] = [];
  evaluation: number = 0;
  nodes: number = 0;
  initFlags: boolean[] = [false, false];
  forceUpdateCallback: () => void;

  constructor(initCallback: () => void, forceUpdateCallback: () => void) {
    this.engineWorker = new Worker(new URL('./EngineWorker.ts', import.meta.url));
    this.engineWorker.onmessage = this.onEngineMessage;
    this.searchWorker = new Worker(new URL('./SearchWorker.ts', import.meta.url));
    this.searchWorker.onmessage = this.onSearchMessage;
    this.initCallback = initCallback;
    this.forceUpdateCallback = forceUpdateCallback;
    for (const worker of [this.engineWorker, this.searchWorker]) {
      worker.postMessage({ type: 'init' });
    }
    this.boardState = null;
    this.legalMoves = [];
    this.nextMoves = [];
  }

  setInitFlag(workerIndex: number) {
    this.initFlags[workerIndex] = true;
    if (this.initFlags[0] && this.initFlags[1])
      this.initCallback(this);
  }

  onEngineMessage = (e: MessageEvent<MessageFromEngineWorker>) => {
    console.log('Main thread got:', e.data);
    switch (e.data.type) {
      case 'initted':
        this.setInitFlag(0);
        break;
      case 'board':
        this.boardState = e.data.board;
        this.legalMoves = e.data.moves;
        this.nextMoves = e.data.nextMoves;
        break;
      case 'evaluation':
        const whiteWinProb = e.data.whiteWinProb;
        const Q = 2 * whiteWinProb - 1;
        this.evaluation = 1.11714640912 * Math.tan(1.5620688421 * Q);
        this.pv = e.data.pv;
        this.nodes = e.data.nodes;
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
    for (const worker of [this.engineWorker, this.searchWorker]) {
      worker.postMessage({ type: 'applyMove', move, isHidden });
    }
  }

  setRunEngine(runEngine: boolean) {
    for (const worker of [this.engineWorker, this.searchWorker]) {
      worker.postMessage({ type: 'setRunEngine', runEngine });
    }
  }
}

function App(props: {}) {
  const [selectedSquare, setSelectedSquare] = React.useState<[number, number] | null>(null);
  const [forceUpdateCounter, setForceUpdateCounter] = React.useState(0);
  const [pair, setPair] = React.useState<any>(null);
  const [duckFen, setDuckFen] = React.useState<string>('');
  const [pgn, setPgn] = React.useState<string>('');
  const [runEngine, setRunEngine] = React.useState<boolean>(false);

  const [workers, setWorkers] = React.useState<Workers | null>(null);
  React.useEffect(() => {
    console.log('Initializing worker...');
    const workers = new Workers(
      () => {
        console.log('Worker initialized!');
        // FIXME: There's a race if you toggle runEngine before the engine is created.
        setWorkers(workers);
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
  if (workers !== null) {
    const state = workers.getState();
    legalMoves = workers.getMoves();
    hiddenLegalMoves = workers.nextMoves;
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
  //  workers.forceUpdateCallback = () => {
  //    setForceUpdateCounter(Math.random());
  //  };
  //}, []);

  //const { workers } = props;
  //const state = workers.getState();
  //const moves: any[] = workers.getMoves();
  //React.useEffect(() => {
  //  //for (let i = 0; i < 100; i++) {
  //  //  const start = performance.now();
  //  //  const tfjsResult = tfjsModel.predict(tf.ones([64, 8, 8, 22])) as tf.Tensor[];
  //  //  tfjsResult[0].data()
  //  //  const end = performance.now();
  //  //  console.log('tfjs prediction took', end - start, 'ms');
  //  //}
  //  //console.log(tfjsResult[0], tfjsResult[1]);
  //  let pair = workers.run(4);
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
        workers.applyMove(m);
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
        workers.applyMove(m);
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
  console.log('----PV:', workers.pv);
  let showMoves: any[] = workers.pv;
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
            if (workers !== null) {
              console.log('[snp1] move', move, isHidden);
              workers.applyMove(move, isHidden);
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
            <input type="checkbox" checked={runEngine} onChange={e => {
              setRunEngine(e.target.checked);
              if (workers !== null) {
                workers.setRunEngine(e.target.checked);
              }
            }} /> Run engine

            {workers !== null && <div>
              Evaluation: {workers.evaluation}<br/>
              Nodes: {workers.nodes}<br/>
              PV: {workers.pv.map((m: any) => m.from + ' ' + m.to).join(' ')}
            </div>}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 10, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <input type="text" value={duckFen} onChange={e => setDuckFen(e.target.value)} style={{
          width: 400,
          backgroundColor: '#445',
          color: '#eee',
        }} />
        <textarea value={pgn} onChange={e => setPgn(e.target.value)} style={{
          marginTop: 5,
          width: 400,
          height: 100,
          backgroundColor: '#445',
          color: '#eee',
        }} placeholder="Paste PGN here..." />
      </div>

      <div style={{ marginTop: 10, textAlign: 'center' }}>
        Created by Peter Schmidt-Nielsen
        (<a href="https://twitter.com/ptrschmdtnlsn">Twitter</a>, <a href="https://peter.website">Website</a>)<br/>
        Engine + web interface: <a href="https://github.com/petersn/duckchess">github.com/petersn/duckchess</a><br/>
        <span style={{ fontSize: '50%', opacity: 0.5 }}>Piece SVGs: Cburnett (CC BY-SA 3), Duck SVG + all code: my creation (CC0)</span>
      </div>
    </div>
  );
}

export default App;
