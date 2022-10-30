import init, { new_engine, modify_array, Engine } from 'engine';
import * as tf from '@tensorflow/tfjs';
import { MessageToEngineWorker } from './EngineWorkerMessages';

let model: tf.LayersModel;
let engine: Engine;
let array: Float32Array;

function sendBoardState() {
  postMessage({ type: 'board', board: engine.get_state(), moves: engine.get_moves() });
}

async function workLoop() {
  setTimeout(workLoop, 500);
  // FIXME: Why do I need as any here?
  await (engine as any).step(array);
}

async function initWorker() {
  array = new Float32Array(8 * 8 * 12);
  await init();
  const seed = Math.floor(Math.random() * 1e9);
  engine = await new_engine(BigInt(seed));
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
