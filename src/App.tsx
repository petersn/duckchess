import React from 'react';
import './App.css';
import init, { new_engine, Engine } from 'engine';

function AppWithWasm() {
  const [engine, setEngine] = React.useState<Engine | null>(null);
  React.useEffect(() => {
    setEngine(new_engine());
  }, []);
  if (engine === null)
    return <div>Initializing engine...</div>;

  const state = engine.get_state();
  const moves = engine.get_moves();

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
  };

  for (let y = 0; y < 8; y++) {
    for (let player = 0; player < 2; player++) {
      for (const name of ['pawns', 'knights', 'bishops', 'rooks', 'queens', 'kings', 'ducks']) {
        let byte;
        // Special handling for the duck
        if (name === 'ducks') {
          if (player === 0)
            continue;
          byte = state.ducks[y];
        } else {
          byte = state[name][player][y];
        }
        for (let x = 0; x < 8; x++) {
          const hasPiece = byte & 1;
          byte = byte >> 1;
          if (!hasPiece)
            continue;
          let piece = name.slice(0, -1);
          if (player === 0)
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

  return (
    <div style={{ margin: 30 }}>
      <table style={{ borderCollapse: 'collapse', border: '1px solid black' }}>
        <tbody>
          {board.map((row, y) => (
            <tr key={y}>
              {row.map((piece, x) => (
                <td key={x} style={{ margin: 0, padding: 0 }}>
                  <div style={{
                    width: 50,
                    maxWidth: 50,
                    height: 50,
                    maxHeight: 50,
                    backgroundColor: (x + y) % 2 === 0 ? '#eca' : '#b97',
                    textAlign: 'center',
                  }}>
                    {piece}
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      {JSON.stringify(moves)}
    </div>
  );
}

function App() {
  const [initialized, setInitialized] = React.useState(false);
  React.useEffect(() => {
    console.log('Initializing wasm...');
    init()
      .then(() => setInitialized(true))
      .catch(console.error);
  }, []);
  return initialized ? <AppWithWasm /> : <div>Loading WASM...</div>;
}

export default App;
