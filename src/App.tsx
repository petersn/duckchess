import React from 'react';
import './App.css';
import init, { new_engine, Engine } from 'engine';

function AppWithEngine(props: { engine: Engine }) {
  const [selectedSquare, setSelectedSquare] = React.useState<[number, number] | null>(null);
  const [forceUpdateCounter, setForceUpdateCounter] = React.useState(0);

  const state = props.engine.get_state();
  const moves: any[] = props.engine.get_moves();

  function clickOn(x: number, y: number) {
    if (selectedSquare === null) {
      // Check if this is the initial duck placement.
      const m = moves.find(m => m.from === 64 && m.to === x + (7 - y) * 8);
      if (m) {
        props.engine.apply_move(m);
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
        props.engine.apply_move(m);
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

  const arrows = [];
  let k = 0;
  for (const move of moves) {
    const fromX = move.from % 8;
    const fromY = 7 - Math.floor(move.from / 8);
    const toX = move.to % 8;
    const toY = 7 - Math.floor(move.to / 8);
    let dx = toX - fromX;
    let dy = toY - fromY;
    const length = Math.sqrt(dx * dx + dy * dy);
    dx /= length;
    dy /= length;
    const endX = toX * 50 + 25 - 10 * dx;
    const endY = toY * 50 + 25 - 10 * dy;
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
        {/*arrows*/}
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

      {JSON.stringify(moves)}
    </div>
  );
}

function App() {
  const [engine, setEngine] = React.useState<Engine | null>(null);
  React.useEffect(() => {
    console.log('Initializing wasm...');
    init()
      .then(() => setEngine(new_engine()))
      .catch(console.error);
  }, []);
  return engine ? <AppWithEngine engine={engine} /> : <div>Loading WASM...</div>;
}

export default App;
