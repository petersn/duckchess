import React from 'react';
import './App.css';
import init, { rust_func } from 'engine';

function AppWithWasm() {
  rust_func();
  return (
    <div>Hello!</div>
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
  return initialized ? <AppWithWasm /> : <div>Loading...</div>;
}

export default App;
