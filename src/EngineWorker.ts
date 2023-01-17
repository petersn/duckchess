import init, { new_engine, max_batch_size, channel_count, parse_pgn4, Engine, perft, perft_nnue, perft_eval, new_game_tree } from 'engine';
import * as tf from '@tensorflow/tfjs';
import { MessageFromEngineWorker, MessageToEngineWorker, ModelName } from './DuckChessEngine';
import { threads } from 'wasm-feature-detect';

// Declare the type of postMessage
declare function postMessage(message: MessageFromEngineWorker): void;

// In order to allow this file to be used either as a Web Worker, or a module from the
// main thread, we add a hook that allows one to redirect the postMessage calls.
let postMessageAlternative: ((message: MessageFromEngineWorker) => void) | null = null;
export function setPostMessageAlternative(f: (message: MessageFromEngineWorker) => void) {
  postMessageAlternative = f;
}

function postMessageWrapper(message: MessageFromEngineWorker) {
  if (postMessageAlternative !== null) {
    postMessageAlternative(message);
  } else {
    postMessage(message);
  }
}

let currentModel: string = 'none';
const modelRegistry: Map<string, tf.LayersModel | 'loading'> = new Map();
let engine: Engine;
let runEngine: boolean = false;

async function setModel(modelName: ModelName) {
  console.log('Setting model to', modelName);
  currentModel = modelName;
  // Don't start loading again.
  if (modelRegistry.has(modelName))
    return;
  modelRegistry.set(modelName, 'loading');
  console.log('Loading model', modelName);

  let path = '';
  switch (modelName) {
    case 'medium-001-128x10':
      path = '/models/run-016-step-233-medium/model.json';
      break;
    case 'large-001-256x20':
      path = '/models/run-016-step-200/model.json';
      break;
    // case 'medium-002-128x10':
    //   path = '/models/run-016-step-290-medium/model.json';
    //   break;
    case 'large-002-256x20':
      path = '/models/run-016-step-290/model.json';
      break;
    // case 'tiny-001-32x6':
    //   break;
    default:
      postMessageWrapper({
        type: 'error',
        message: 'Unknown model name: ' + modelName,
      });
      throw new Error('Unknown model name: ' + modelName);
  }
  let model;
  try {
    model = await tf.loadLayersModel(
      process.env.PUBLIC_URL + path,
      {
        onProgress: (progress) => {
          console.log('Loading', modelName, 'progress', progress);
          postMessageWrapper({
            type: 'modelLoadProgress',
            progress,
            modelName,
          });
        },
      },
    );
  } catch (e) {
    console.error('Failed to load model', modelName, e);
    postMessageWrapper({
      type: 'error',
      message: 'Failed to load model ' + modelName + ': ' + e,
    });
    return;
  }
  console.log('Loaded model', modelName);
  modelRegistry.set(modelName, model);
}

function workLoop() {
  const channelCount = channel_count();
  try {
    if (!runEngine)
      return;
    const model = modelRegistry.get(currentModel);
    if (model === undefined || model === 'loading')
      return;
    let inputArray = new Float32Array(max_batch_size() * channelCount * 8 * 8);
    const batchSize = engine.step_until_batch(inputArray);
    //const batchSize = 1 as any;
    if (batchSize === 0) {
      return;
    }
    inputArray = inputArray.slice(0, batchSize * channelCount * 8 * 8);
    //// Compute how many entries in this array are nonzero.
    //let nonZeroCount = 0;
    //for (let i = 0; i < inputArray.length; i++) {
    //  if (inputArray[i] !== 0) {
    //    nonZeroCount++;
    //  }
    //}
    //console.log('Nonzero', nonZeroCount, 'of', inputArray.length);
    //console.log('batchSize', batchSize);
    const inp = tf.tensor4d(inputArray, [batchSize, channelCount, 8, 8]);
    const [policy, wdl] = model.predict(inp) as [tf.Tensor, tf.Tensor];
    const policyData = policy.dataSync() as Float32Array;
    const wdlData = wdl.dataSync() as Float32Array;
    // memcpy into policy_array and value_array
    //policyArray.set(policyData);
    //valueArray.set(valueData);
    engine.give_answers(policyData, wdlData);
    // Delete the inputs.
    inp.dispose();
    policy.dispose();
    wdl.dispose();

    const engineOutput = engine.get_engine_output();
    postMessageWrapper({ type: 'engineOutput', engineOutput });

    //const [pv, whiteWinProb, nodes] = engine.get_principal_variation();
    //postMessageWrapper({ type: 'evaluation', whiteWinProb, pv, nodes });
    // FIXME: Why do I need as any here?
    //await (engine as any).step(array);
  } finally {
    setTimeout(workLoop, runEngine ? 1 : 50);
  }
}

async function initWorker(requireWebGL: boolean) {
  const success = await tf.setBackend('webgl');
  if (!success) {
    console.error('Failed to init WebGL backend!');
    postMessageWrapper({
      type: 'backendFailure',
      message: "WebGL failed -- site will run at maybe 5% of full speed. If you're using Safari, try Chrome or Firefox instead.",
      requireWebGL,
    });
    if (requireWebGL)
      return;
  }

  const hasThreads = await threads();
  console.log('Has threads:', hasThreads);
  //const sharedArrayBuffer = new SharedArrayBuffer(4 * 1024 * 1024);
  const wasm = await init();
  console.log('Channels:', channel_count());

  const gt = new_game_tree();
  const ss = gt.get_serialized_state();
  console.log('Serialized state', ss);

  for (let i = 0; i < 0; i++) {
    const start1 = performance.now();
    const nodes1 = perft();
    const end1 = performance.now();
    console.log('Perft', nodes1, 'nodes in', end1 - start1, 'ms');
    console.log('mega nodes per second', nodes1 / (end1 - start1) / 1000);

    const start2 = performance.now();
    const nodes2 = perft_nnue();
    const end2 = performance.now();
    console.log('Perft NNUE', nodes2, 'nodes in', end2 - start2, 'ms');
    console.log('mega nodes per second', nodes2 / (end2 - start2) / 1000);

    const start3 = performance.now();
    const nodes3 = perft_eval();
    const end3 = performance.now();
    console.log('Perft eval', nodes3, 'nodes in', end3 - start3, 'ms');
    console.log('mega nodes per second', nodes3 / (end3 - start3) / 1000);
    console.log('------------------');
  }

  // let array = new Int32Array(sharedArrayBuffer);
  // setInterval(() => {
  //   test_shared_mem(array, 1);
  // }, 1000);

  // test_simd();

  const seed = Math.floor(Math.random() * 1e9);
  engine = new_engine(BigInt(seed));

  //model = await tf.loadLayersModel(process.env.PUBLIC_URL + '/model-small/model.json')
  //model = await tf.loadLayersModel(process.env.PUBLIC_URL + '/model-medium/model.json')
  //model = await tf.loadLayersModel(process.env.PUBLIC_URL + '/run-016-model-058/model.json')
  //model = await tf.loadLayersModel(process.env.PUBLIC_URL + '/run-016-model-220/model.json')
  postMessageWrapper({ type: 'initted' });
  // Make an 8x8 array of all nulls.
  //const fakeState = Array(8).fill(null).map(() => Array(8).fill(null));
  //postMessageWrapper({ type: 'board', board: fakeState, moves: [] });
  //sendBoardState();
  workLoop();

  //// Create a search worker.
  //worker = new Worker(new URL('./SearchWorker.ts', import.meta.url));
  //worker.onmessage = onSearchWorkerMessage;
  //worker.postMessageWrapper({ type: 'init', mem: sharedArrayBuffer });
}

export function handleMessage(msg: MessageToEngineWorker) {
  switch (msg.type) {
    case 'init':
      initWorker(msg.requireWebGL);
      break;
    case 'setState':
      engine.set_state(msg.state, msg.repetitionHashes);
      break;
    case 'setModel':
      setModel(msg.modelName);
      break;
    case 'setSearchParams':
      const success = engine.set_search_params(msg.searchParams);
      if (!success) {
        console.error('Failed to set search params!');
        postMessageWrapper({ type: 'error', message: 'Failed to set search params!' });
      }
      break;
    //case 'applyMove':
    //  console.log('Applying move', msg.move, 'isHidden', msg.isHidden);
    //  engine.apply_move(msg.move, msg.isHidden);
    //  sendBoardState();
    //  break;
    case 'setRunEngine':
      runEngine = msg.runEngine;
      break;
  }
}

onmessage = function(e: MessageEvent<MessageToEngineWorker>) {
  //console.log('Web worker got:', e.data);
  handleMessage(e.data);
}
