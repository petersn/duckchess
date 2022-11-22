
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

export {};
