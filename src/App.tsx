import React from 'react';
import './App.css';
import init, { new_engine, Engine } from 'engine';
import * as tf from '@tensorflow/tfjs';

async function createConvModel(): Promise<tf.LayersModel> {
  const model = await tf.loadLayersModel('http://localhost:3000/model.json');
  return model;
  //const model = tf.sequential();
  //model.add(tf.layers.conv2d({
  //  inputShape: [28, 28, 1],
  //  kernelSize: 3,
  //  filters: 16,
  //  activation: 'relu'
  //}));
  //model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
  //model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
  //model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
  //model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
  //model.add(tf.layers.flatten({}));
  //model.add(tf.layers.dense({units: 64, activation: 'relu'}));
  //model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
  //model.loadWeights('http://localhost:8000/model.json');
  //return model;
}

function AppWithComputation(props: { computation: Computation }) {
  const [selectedSquare, setSelectedSquare] = React.useState<[number, number] | null>(null);
  const [forceUpdateCounter, setForceUpdateCounter] = React.useState(0);
  const [pair, setPair] = React.useState<any>(null);

  const { engine, tfjsModel } = props.computation;
  const state = engine.get_state();
  const moves: any[] = engine.get_moves();
  React.useEffect(() => {
    for (let i = 0; i < 100; i++) {
      const start = performance.now();
      const tfjsResult = tfjsModel.predict(tf.ones([64, 8, 8, 22])) as tf.Tensor[];
      tfjsResult[0].data()
      const end = performance.now();
      console.log('tfjs prediction took', end - start, 'ms');
    }
    //console.log(tfjsResult[0], tfjsResult[1]);
    let pair = engine.run(4);
    setPair(pair);
    setTimeout(() => {
      if (pair && pair[1][0]) {
        engine.apply_move(pair[1][0]);
        setForceUpdateCounter(forceUpdateCounter + 1);
      }
    }, 1000000);
  }, [forceUpdateCounter]);

  console.log('Pair:', pair);

  function clickOn(x: number, y: number) {
    if (selectedSquare === null) {
      // Check if this is the initial duck placement.
      const m = moves.find(m => m.from === 64 && m.to === x + (7 - y) * 8);
      if (m) {
        engine.apply_move(m);
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
        engine.apply_move(m);
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
            {(unicodeChessPieces as any)[piece]}
          </span>;
        }
      }
    }
  }

  let showMoves: any[] = (pair && pair[1][0]) ? pair[1] : [];
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

  return (
    <div style={{ margin: 30, position: 'relative', width: 400, height: 400, minWidth: 400, minHeight: 400 }}>
      <svg
        viewBox="0 0 400 400"
        style={{ width: 400, height: 400, position: 'absolute', zIndex: 1, pointerEvents: 'none' }}
      >
        {arrows}
      </svg>

      <table style={{ position: 'absolute', borderCollapse: 'collapse', border: '1px solid black' }}>
        <tbody>
          {board.map((row, y) => (
            <tr key={y}>
              {row.map((piece, x) => {
                const isSelected = selectedSquare !== null && selectedSquare[0] === x && selectedSquare[1] === y;
                let backgroundColor = (x + y) % 2 === 0 ? '#eca' : '#b97';
                if (state.highlight[7 - y] & (1 << x))
                  backgroundColor = (x + y) % 2 === 0 ? '#dd9' : '#aa6';
                if (isSelected)
                  backgroundColor = '#7f7';
                return <td key={x} style={{ margin: 0, padding: 0 }}>
                  <div
                    style={{
                      width: 50,
                      maxWidth: 50,
                      height: 50,
                      maxHeight: 50,
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
  );
}

interface Computation {
  engine: Engine;
  tfjsModel: tf.LayersModel;
}

function OldApp() {
  const [computation, setComputation] = React.useState<Computation | null>(null);
  React.useEffect(() => {
    console.log('Initializing wasm...');
    const seed = Math.floor(Math.random() * 1e9);
    init()
      .then(() => createConvModel()
        .then((tfjsModel) => setComputation({
          engine: new_engine(BigInt(seed)),
          tfjsModel,
        }))
      ).catch(console.error);
  }, []);
  return computation ? <AppWithComputation computation={computation} /> : <div>Loading WASM...</div>;
}

let globalBatchSize = 16;

function App() {
  const [model, setModel] = React.useState<tf.LayersModel | null>(null);
  const [batchSize, setBatchSize] = React.useState('16');
  const [fut1Latency, setFut1Latency] = React.useState(1000);
  const [fut2Latency, setFut2Latency] = React.useState(1000);

  React.useEffect(() => {
    console.log('Initializing tfjs...');
    createConvModel()
      .then((model) => {
        setModel(model);
        // Start a loop of evaluating the model.
        const loop = async () => {
          let inp1 = tf.randomUniform([globalBatchSize, 8, 8, 22]);
          let inp2 = tf.randomUniform([globalBatchSize, 8, 8, 22]);
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
            await fut1[0].data();
            await fut1[1].data();
            const fut1FinishTime = performance.now();
            setFut1Latency(fut1FinishTime - fut1LaunchTime);
            fut1[0].dispose();
            fut1[1].dispose();
            if (inp1.shape[0] !== globalBatchSize && Number.isInteger(globalBatchSize) && globalBatchSize > 0) {
              console.log('Resizing inputs to:', globalBatchSize);
              inp1.dispose();
              inp2.dispose();
              inp1 = tf.randomUniform([globalBatchSize, 8, 8, 22]);
              inp2 = tf.randomUniform([globalBatchSize, 8, 8, 22]);
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

export default App;
