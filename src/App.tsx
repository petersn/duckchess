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

  const board = [];
  for (let y = 0; y < 8; y++)
    board.push([null, null, null, null, null, null, null, null]);

  for (let y = 0; y < 8; y++) {
    for (let player = 0; player < 2; player++) {
      for (const name of ['pawns', 'knights', 'bishops', 'rooks', 'queens', 'kings']) {
        let byte = state[name][player][y];
        for (let x = 0; x < 8; x++) {
          const hasPiece = byte & 1;
          byte = byte >> 1;
          if (!hasPiece)
            continue;
        }
      }
    }
  }

  return (
    <div>
      {JSON.stringify(state)}
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
