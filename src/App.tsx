import { engine } from '@tensorflow/tfjs';
import React from 'react';
import './App.css';
import { MessageFromEngineWorker } from './EngineWorkerMessages';

class EngineWorker {
  worker: Worker;
  initCallback: (ew: EngineWorker) => void;
  boardState: any;
  moves: any[];
  pv: any[] = [];
  evaluation: number = 0;
  forceUpdateCallback: () => void;

  constructor(initCallback: () => void) {
    this.worker = new Worker(new URL('./EngineWorker.ts', import.meta.url));
    this.worker.onmessage = this.onMessage;
    this.initCallback = initCallback;
    this.worker.postMessage({ type: 'init' });
    this.boardState = null;
    this.moves = [];
    this.forceUpdateCallback = () => {};
  }

  onMessage = (e: MessageEvent<MessageFromEngineWorker>) => {
    console.log('Main thread got:', e.data);
    switch (e.data.type) {
      case 'initted':
        this.initCallback(this);
        break;
      case 'board':
        this.boardState = e.data.board;
        this.moves = e.data.moves;
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
    return this.moves;
  }

  applyMove(move: any) {
    this.worker.postMessage({ type: 'applyMove', move });
  }
}

function ChessPiece(props: { piece: string; }) {
  const pieceFileMapping = {
    'pawn':   'black_pawn.svg',   'PAWN':   'white_pawn.svg',
    'rook':   'black_rook.svg',   'ROOK':   'white_rook.svg',
    'knight': 'black_knight.svg', 'KNIGHT': 'white_knight.svg',
    'bishop': 'black_bishop.svg', 'BISHOP': 'white_bishop.svg',
    'queen':  'black_queen.svg',  'QUEEN':  'white_queen.svg',
    'king':   'black_king.svg',   'KING':   'duck.svg',
    'duck':   'unknown',
    'enpassant': 'unknown',
  };
  const pieceFile = (pieceFileMapping as any)[props.piece];
  return <img
    style={{
      width: '100%',
      height: '100%',
      objectFit: 'contain',
      userSelect: 'none',
    }}
    src={`${process.env.PUBLIC_URL}/icons/${pieceFile}`}
    draggable={false}
  />;
}

function AppWithEngineWorker(props: { engineWorker: EngineWorker }) {
  const [selectedSquare, setSelectedSquare] = React.useState<[number, number] | null>(null);
  const [forceUpdateCounter, setForceUpdateCounter] = React.useState(0);
  const [pair, setPair] = React.useState<any>(null);

  React.useEffect(() => {
    engineWorker.forceUpdateCallback = () => {
      setForceUpdateCounter(Math.random());
    };
  }, []);

  const { engineWorker } = props;
  const state = engineWorker.getState();
  const moves: any[] = engineWorker.getMoves();
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

  if (state === null) {
    return <div>Loading...</div>;
  }

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

  const board: React.ReactNode[][] = [];
  for (let y = 0; y < 8; y++)
    board.push([null, null, null, null, null, null, null, null]);

  const unicodeChessPieces = {
    'pawn':   '♟', 'PAWN':   '♙',
    'rook':   '♜', 'ROOK':   '♖',
    'knight': '♞', 'KNIGHT': '♘',
    'bishop': '♝', 'BISHOP': '♗',
    'queen':  '♛', 'QUEEN':  '♕',
    'king':   '♚', 'KING':   '♔',
    'duck':   '🦆',
    'enpassant': '👻',
  };

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
            <ChessPiece piece={piece} />
          </span>;
        }
      }
    }
  }

  //let showMoves: any[] = (pair && pair[1][0]) ? pair[1] : [];
  console.log('----PV:', engineWorker.pv);
  let showMoves: any[] = engineWorker.pv;
  if ((state.isDuckMove || true) && showMoves) {
    showMoves = showMoves.slice(0, 1);
  }

  const arrows = [];
  let k = 0;
  for (const move of showMoves) {
    if (!move)
      continue;
    const fromX = move.from % 8;
    const fromY = 7 - Math.floor(move.from / 8);
    const toX = move.to % 8;
    const toY = 7 - Math.floor(move.to / 8);
    let dx = toX - fromX;
    let dy = toY - fromY;
    const length = 1e-6 + Math.sqrt(dx * dx + dy * dy);
    dx /= length;
    dy /= length;
    const endX = toX * 50 + 25 - 10 * dx;
    const endY = toY * 50 + 25 - 10 * dy;
    if (move.from === 64) {
      const arrow = <circle
        key={k++}
        cx={toX * 50 + 25}
        cy={toY * 50 + 25}
        r={10}
        stroke="red"
        strokeWidth="5"
        fill="red"
      />;
      arrows.push(arrow);
      continue;
    }
    let d = `M ${fromX * 50 + 25} ${fromY * 50 + 25} L ${endX} ${endY}`;
    d += ` L ${endX + 5 * dy} ${endY - 5 * dx} L ${endX + 10 * dx} ${endY + 10 * dy} L ${endX - 5 * dy} ${endY + 5 * dx} L ${endX} ${endY} Z`;
    const arrow = <path
      key={k++}
      d={d}
      stroke="red"
      strokeWidth="5"
      fill="red"
    />;
    arrows.push(arrow);
  }

  const boardDiv = (
    <div style={{ margin: 10, position: 'relative', width: 600, height: 600, minWidth: 600, minHeight: 600 }}>
      <svg
        viewBox="0 0 400 400"
        style={{ width: 600, height: 600, position: 'absolute', zIndex: 1, pointerEvents: 'none' }}
      >
        {arrows}
      </svg>

      <div style={{ position: 'absolute' }}>
        <table style={{ borderCollapse: 'collapse', border: '1px solid #eee' }}>
          <tbody>
            {board.map((row, y) => (
              <tr key={y}>
                {row.map((piece, x) => {
                  const isSelected = selectedSquare !== null && selectedSquare[0] === x && selectedSquare[1] === y;
                  let backgroundColor = (x + y) % 2 === 0 ? '#eca' : '#b97';
                  //if (state.highlight[7 - y] & (1 << x))
                  //  backgroundColor = (x + y) % 2 === 0 ? '#dd9' : '#aa6';
                  if (isSelected)
                    backgroundColor = '#7f7';
                  return <td key={x} style={{ margin: 0, padding: 0 }}>
                    <div
                      style={{
                        width: 75,
                        maxWidth: 75,
                        height: 75,
                        maxHeight: 75,
                        backgroundColor,
                        textAlign: 'center',
                      }}
                      onClick={() => clickOn(x, y)}
                    >
                      {piece}
                    </div>
                  </td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  return (
    <div style={{
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
        }}>
          <span style={{ fontWeight: 'bold', fontSize: '120%' }}>Duck Chess Analysis Board</span><br/>

          <a href="https://duckchess.com/">Duck chess</a> is a variant in which there is one extra piece (the duck) that cannot be captured or moved through.
          Each turn consists of making regular move, and then moving the duck to any empty square (the duck may not stay in place).
        </div>
      </div>
      {boardDiv}
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
          <span style={{ fontWeight: 'bold', fontSize: '120%' }}>Engine</span>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [engineWorker, setEngineWorker] = React.useState<EngineWorker | null>(null);
  React.useEffect(() => {
    console.log('Initializing worker...');
    const worker = new EngineWorker(() => {
      console.log('Worker initialized!');
      setEngineWorker(worker);
    });
  }, []);
  return engineWorker ? <AppWithEngineWorker engineWorker={engineWorker} /> : <div>Loading engine...</div>;
}

/*
let globalBatchSize = 16;

function App() {
  const [model, setModel] = React.useState<tf.LayersModel | null>(null);
  const [batchSize, setBatchSize] = React.useState('16');
  const [fut1Latency, setFut1Latency] = React.useState(1000);
  const [fut2Latency, setFut2Latency] = React.useState(1000);

  React.useEffect(() => {
    const myWorker = new Worker(new URL('./worker.tsx', import.meta.url));
    myWorker.onmessage = (e) => {
      console.log('Message received from worker');
    };
    myWorker.postMessage({ type: 'init' });

    console.log('Initializing tfjs...');
    createConvModel()
      .then((model) => {
        setModel(model);
        // Start a loop of evaluating the model.
        const loop = async () => {
          (window as any).tensor = tf.randomUniform([1, 8, 8, 12]);
          return;
          let inp1 = tf.randomUniform([globalBatchSize, 22, 8, 8]);
          let inp2 = tf.randomUniform([globalBatchSize, 22, 8, 8]);
          let fut1LaunchTime = performance.now();
          let fut1 = model.predict(inp1) as tf.Tensor[];
          let fut2LaunchTime = performance.now();
          let fut2 = model.predict(inp2) as tf.Tensor[];
          // Keep two computations in flight at all times.
          while (true) {
            console.log('EVALUATED');
            fut1LaunchTime = performance.now();
            fut1 = model.predict(inp1) as tf.Tensor[];
            await fut2[0].data();
            await fut2[1].data();
            setFut2Latency(performance.now() - fut2LaunchTime);
            fut2[0].dispose();
            fut2[1].dispose();
            fut2LaunchTime = performance.now();
            fut2 = model.predict(inp2) as tf.Tensor[];
            const d = await fut1[0].data();
            await fut1[1].data();
            const fut1FinishTime = performance.now();
            setFut1Latency(fut1FinishTime - fut1LaunchTime);
            fut1[0].dispose();
            fut1[1].dispose();
            if (inp1.shape[0] !== globalBatchSize && Number.isInteger(globalBatchSize) && globalBatchSize > 0) {
              console.log('Resizing inputs to:', globalBatchSize);
              inp1.dispose();
              inp2.dispose();
              inp1 = tf.randomUniform([globalBatchSize, 22, 8, 8]);
              inp2 = tf.randomUniform([globalBatchSize, 22, 8, 8]);
            }
          }
        }
        loop();
      })
      .catch(console.error);
  }, []);

  return (
    <div style={{ margin: 30 }}>
      <h1>tfjs performance tester</h1>
      <p>
        {model ? 'Loaded tfjs model.' : 'Loading tfjs model...'}
      </p>
      <p>
        <label>Batch size: <input
          type="number"
          value={batchSize}
          onChange={(e) => {
            setBatchSize(e.target.value);
            const value = parseInt(e.target.value);
            globalBatchSize = value;
          }}
        /></label>
      </p>
      <p>
        <label>Future 1 latency: {fut1Latency.toFixed(1)} ms</label>
      </p>
      <p>
        <label>Future 2 latency: {fut2Latency.toFixed(1)} ms</label>
      </p>
      <p>
        Batch size: {globalBatchSize}
      </p>
      <p>
        Nodes per second: {(2 * globalBatchSize * 1000 / ((fut1Latency + fut2Latency) / 2)).toFixed(1)}
      </p>
    </div>
  );
}
*/

export default App;
