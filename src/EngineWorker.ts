import init, { new_engine, max_batch_size, channel_count, parse_pgn4, Engine, perft, perft_nnue, perft_eval, test_threads, test_simd, test_shared_mem } from 'engine';
import * as tf from '@tensorflow/tfjs';
import { MessageToEngineWorker } from './EngineWorkerMessages';
import { threads } from 'wasm-feature-detect';

let model: tf.LayersModel;
let engine: Engine;
let worker: Worker;

function sendBoardState() {
  postMessage({ type: 'board', board: engine.get_state(), moves: engine.get_moves() });
}

const onSearchWorkerMessage = (e: MessageEvent<any>) => {
  console.log(e);
}

function workLoop() {
  return;
  setTimeout(workLoop, 10000);
  let inputArray = new Float32Array(max_batch_size() * channel_count() * 8 * 8);
  const batchSize = engine.step_until_batch(inputArray);
  //const batchSize = 1 as any;
  if (batchSize === 0) {
    return;
  }
  inputArray = inputArray.slice(0, batchSize * channel_count() * 8 * 8);
  // Compute how many entries in this array are nonzero.
  let nonZeroCount = 0;
  for (let i = 0; i < inputArray.length; i++) {
    if (inputArray[i] !== 0) {
      nonZeroCount++;
    }
  }
  console.log('Nonzero', nonZeroCount, 'of', inputArray.length);
  //console.log('batchSize', batchSize);
  const inp = tf.tensor4d(inputArray, [batchSize, channel_count(), 8, 8]);
  const [policy, value] = model.predict(inp) as [tf.Tensor, tf.Tensor];
  const policyData = policy.dataSync() as Float32Array;
  const valueData = value.dataSync() as Float32Array;
  // memcpy into policy_array and value_array
  //policyArray.set(policyData);
  //valueArray.set(valueData);
  engine.give_answers(policyData, valueData);
  // Delete the inputs.
  inp.dispose();
  policy.dispose();
  value.dispose();
  const evaluation = 9.99;
  const pv = engine.get_principal_variation();
  postMessage({ type: 'evaluation', evaluation, pv });
  // FIXME: Why do I need as any here?
  //await (engine as any).step(array);
}

async function initWorker() {
  const hasThreads = await threads();
  console.log('Has threads:', hasThreads);
  const sharedArrayBuffer = new SharedArrayBuffer(4 * 1024 * 1024);
  const wasm = await init();

  for (let i = 0; i < 0; i++) {
    //const start1 = performance.now();
    //const nodes1 = perft();
    //const end1 = performance.now();
    //console.log('Perft', nodes1, 'nodes in', end1 - start1, 'ms');
    //console.log('mega nodes per second', nodes1 / (end1 - start1) / 1000);

    const start2 = performance.now();
    const nodes2 = perft_nnue();
    const end2 = performance.now();
    console.log('Perft NNUE', nodes2, 'nodes in', end2 - start2, 'ms');
    console.log('mega nodes per second', nodes2 / (end2 - start2) / 1000);

    //const start3 = performance.now();
    //const nodes3 = perft_eval();
    //const end3 = performance.now();
    //console.log('Perft eval', nodes3, 'nodes in', end3 - start3, 'ms');
    //console.log('mega nodes per second', nodes3 / (end3 - start3) / 1000);
    console.log('------------------');
  }

  let array = new Int32Array(sharedArrayBuffer);
  setInterval(() => {
    test_shared_mem(array, 1);
  }, 1000);

  test_simd();

  const seed = Math.floor(Math.random() * 1e9);
  engine = new_engine(BigInt(seed));

  //model = await tf.loadLayersModel('/duck-chess-engine/model.json')
  //postMessage({ type: 'initted' });
  //sendBoardState();

  // Create a search worker.
  worker = new Worker(new URL('./SearchWorker.ts', import.meta.url));
  worker.onmessage = onSearchWorkerMessage;
  worker.postMessage({ type: 'init', mem: sharedArrayBuffer });

  workLoop();
}

onmessage = function(e: MessageEvent<MessageToEngineWorker>) {
  console.log('Web worker got:', e.data);
  switch (e.data.type) {
    case 'init':
      initWorker();
      break;
    case 'applyMove':
      engine.apply_move(e.data.move);
      sendBoardState();
      break;
  }
}
