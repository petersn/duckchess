import init, { new_engine, max_batch_size, channel_count, Engine } from 'engine';
import * as tf from '@tensorflow/tfjs';
import { MessageToEngineWorker } from './EngineWorkerMessages';

let model: tf.LayersModel;
let engine: Engine;

function sendBoardState() {
  postMessage({ type: 'board', board: engine.get_state(), moves: engine.get_moves() });
}

function workLoop() {
  setTimeout(workLoop, 2000);
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
  const evaluation = 0.37;
  const pv = engine.get_principal_variation();
  postMessage({ type: 'evaluation', evaluation, pv });
  // FIXME: Why do I need as any here?
  //await (engine as any).step(array);
}

async function initWorker() {
  await init();

  const seed = Math.floor(Math.random() * 1e9);
  engine = new_engine(BigInt(seed));

  model = await tf.loadLayersModel('/duck-chess-engine/model.json')
  postMessage({ type: 'initted' });
  sendBoardState();

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
