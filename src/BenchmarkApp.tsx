import React from 'react';
import * as tf from '@tensorflow/tfjs';

// This is the number of input feature layers to the model.
const featureCount = 29;

let modelSmall: tf.LayersModel | null = null;
let modelMedium: tf.LayersModel | null = null;

async function launchAnother(
  model: tf.LayersModel | null,
  input: tf.Tensor,
  callback: (latency: number) => void,
) {
  const start = performance.now();
  const done = () => {
    const end = performance.now();
    const latency = end - start;
    callback(latency);
  };
  if (model == null) {
    setTimeout(done, 150);
    return;
  }
  const fut = model.predict(input) as tf.Tensor[];
  const policy = await fut[0].data();
  const value = await fut[1].data();
  fut[0].dispose();
  fut[1].dispose();
  done();
}

async function evalLoop(app: BenchmarkApp) {
  let input: tf.Tensor | null = null;
  let currentlyInFlight = 0;
  let wakeUpResolve: () => void = () => {};
  let wakeUpFuture: Promise<void> | null = null;

  //const input = tf.randomUniform([globalBatchSize, 29, 8, 8]);
  while (app.keepRunning) {
    // Check if we need to regenerate the input.
    const batchSize = Number(app.state.batchSize);
    if (input == null || (input.shape[0] !== batchSize && !Number.isNaN(batchSize)) && batchSize > 1 && batchSize <= 10000) {
      input = tf.randomUniform([batchSize, featureCount, 8, 8]);
    }

    // Wait until our wake up signal.
    //console.log('[bench] Waiting for wake up signal...');
    await wakeUpFuture;
    //console.log('[bench] Got wake up signal!');
    wakeUpFuture = new Promise(resolve => {
      wakeUpResolve = resolve;
      setTimeout(resolve, 500);
    });

    // Launch more while we don't have enough in flight.
    while (currentlyInFlight < Number(app.state.maxInFlight)) {
      currentlyInFlight++;
      const model = app.state.runTfjs ? (app.state.modelSize === 'small' ? modelSmall : modelMedium) : null;
      launchAnother(model, input, (latency: number) => {
        wakeUpResolve();
        currentlyInFlight--;
        if (model !== null)
          app.addLatency(latency);
      });
    }
    /*
      fut2 = model.predict(inp2) as tf.Tensor[];
      const d = await fut1[0].data();
      await fut1[1].data();
      const model = app.state.model === 'small' ? modelSmall : modelMedium;
      if (model == null) {
        throw new Error('Model is null');
      }
      model.predict(input).then(() => {
        currentlyInFlight--;
        wakeUpResolve();
      });
    }
      currentlyInFlight++;
      const model = app.state.modelSize === 'small' ? modelSmall : modelMedium;
      if (model == null) {
        throw new Error('Model not loaded');
      }
      model.predict(input).then(() => {
        currentlyInFlight--;
        wakeUpResolve();
      });
      */
  }
}

/*

    console.log('Initializing tfjs...');
    tf.loadLayersModel('/duck-chess/model-small/model.json')
      .then((model) => {
        setModel(model);
        // Start a loop of evaluating the model.
        const loop = async () => {
          let inp1 = tf.randomUniform([globalBatchSize, 29, 8, 8]);
          let inp2 = tf.randomUniform([globalBatchSize, 29, 8, 8]);
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
              inp1 = tf.randomUniform([globalBatchSize, 29, 8, 8]);
              inp2 = tf.randomUniform([globalBatchSize, 29, 8, 8]);
            }
          }
        }
        loop();
      })
      .catch(console.error);
  }, []);
*/

interface BenchmarkAppState {
  batchSize: string;
  maxInFlight: string;
  modelSize: string;
  latencies: number[];
  runTfjs: boolean;
  loadedSmall: boolean;
  loadedMedium: boolean;
}

export class BenchmarkApp extends React.PureComponent<{}, BenchmarkAppState> {
  keepRunning: boolean = true;

  constructor(props: {}) {
    super(props);
    this.state = {
      batchSize: '16',
      maxInFlight: '1',
      modelSize: 'small',
      latencies: [],
      runTfjs: false,
      loadedSmall: false,
      loadedMedium: false,
    };
  }

  addLatency(latency: number) {
    this.setState((state) => {
      const latencies = state.latencies.slice();
      latencies.push(latency);
      if (latencies.length > 10) {
        latencies.shift();
      }
      return {
        ...state,
        latencies,
      };
    });
  }

  componentDidMount() {
    if (modelSmall == null) {
      tf.loadLayersModel('/duck-chess/model-small/model.json')
        .then((model) => {
          modelSmall = model;
          this.setState({ loadedSmall: true });
        })
        .catch(console.error);
    }
    if (modelMedium == null) {
      tf.loadLayersModel('/duck-chess/model-medium/model.json')
        .then((model) => {
          modelMedium = model;
          this.setState({ loadedMedium: true });
        })
        .catch(console.error);
    }
    evalLoop(this);
  }

  componentWillUnmount() {
    this.keepRunning = false;
  }

  render() {
    const model = this.state.modelSize === 'small' ? modelSmall : modelMedium;
    const meanLatency = this.state.latencies.reduce((a, b) => a + b, 0) / (1e-6 + this.state.latencies.length);
    const batchesPerSecond = Number(this.state.maxInFlight) * 1e3 / meanLatency;
    const nps = Number(this.state.batchSize) * batchesPerSecond;
    return (
      <div style={{ width: 550, margin: 10, textAlign: 'center' }}>
        <h1>Performance tester</h1>
        <p>
          This duck chess engine has two parts: a neural network engine (trained from self-play à la AlphaZero, or lc0)
          and a custom alpha-beta engine
          (using <a href="https://en.wikipedia.org/wiki/Principal_variation_search">PVS</a>,
          with a custom <a href="https://en.wikipedia.org/wiki/Efficiently_updatable_neural_network">NNUE</a> for evaluation).
          The neural network engine is evaluated using TensorFlow.js, using your GPU via WebGL, while the alpha-beta engine
          runs on CPU in WebAssembly.
          This page allows for testing your browser's performance on both parts of the engine.
        </p>
        <h2>TensorFlow.js benchmark</h2>
        <p>
          {model ? 'Loaded tfjs model.' : 'Loading tfjs model...'}
        </p>
        <p>
          {/* On/off switch */}
          <label>
            <input
              type="checkbox"
              checked={this.state.runTfjs}
              onChange={(e) => this.setState({ runTfjs: e.target.checked })}
            />
            Run TensorFlow.js benchmark
          </label><br/>
          {/* Adjust batch size */}
          <label>Batch size: <input
            type="number"
            value={this.state.batchSize}
            onChange={(e) => this.setState({ batchSize: e.target.value })}
          /></label><br/>
          {/* Adjust in-flight count */}
          <label>Parallel batches in flight: <input
            type="number"
            value={this.state.maxInFlight}
            onChange={(e) => this.setState({ maxInFlight: e.target.value })}
          /></label><br/>
          {/* Adjust model size */}
          <label>Model size: <select
            value={this.state.modelSize}
            onChange={(e) => this.setState({ modelSize: e.target.value })}
          >
            <option value="small">Small</option>
            <option value="medium">Medium</option>
          </select></label>
        </p>
        <p>
          Latencies:<br/>
          <div style={{ display: 'inline-flex' }}>
            {this.state.latencies.slice(5).map((latency, i) => (
              <div style={{ display: 'inline-block', width: 50 }} key={i}>{latency.toFixed(1)} ms</div>
            ))}
          </div>
        </p>
        <p>
          Batch size: {Number(this.state.batchSize)} -
          Parallel batches in flight: {Number(this.state.maxInFlight)} -
          Model size: {this.state.modelSize}
        </p>
        <p>
          Mean latency: {meanLatency.toFixed(1)} ms<br/>
          Nodes per second: {nps.toFixed(1)}
        </p>
      </div>
    );
  }
}

//
/*
export function BenchmarkApp() {
  const [batchSize, setBatchSize] = React.useState('16');
  const [maxInFlight, setInFlightCount] = React.useState('3');
  const [modelSize, setModelSize] = React.useState('small');
  const [futureLatencies, setFutureLatencies] = React.useState<number[]>([1000, 1000, 1000]);

  React.useEffect(() => {
    keepRunning = true;
    evalLoop();
    return () => {
      keepRunning = false;
    };
  }, []);

  const model = modelSize === 'small' ? modelSmall : modelMedium;
  const fut1Latency = 0;
  const fut2Latency = 0;
}
*/
