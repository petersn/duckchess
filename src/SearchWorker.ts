import init, { perft, perft_eval, Pvs, new_pvs } from 'engine';
import { MessageFromSearchWorker } from './DuckChessEngine';

// Declare the type of postMessage
declare function postMessage(message: MessageFromSearchWorker): void;

let pvs: Pvs;
let runEngine: boolean = false;

async function initSearchWorker(sharedArrayBuffer: SharedArrayBuffer) {
  await init();
  console.log('Search worker initialized');
  const seed = Math.floor(Math.random() * 1e9);
  pvs = new_pvs(BigInt(seed));
  postMessage({ type: 'initted' });
  //const array = new Int32Array(sharedArrayBuffer);
  //setInterval(() => {
  //  test_shared_mem(array, 2);
  //}, 1000);
}

async function runAlphaBetaBenchmark() {
  const start1 = performance.now();
  const nodes1 = perft();
  const end1 = performance.now();
  console.log('Perft', nodes1, 'nodes in', end1 - start1, 'ms');
  console.log('mega nodes per second', nodes1 / (end1 - start1) / 1000);

  //const start2 = performance.now();
  //const nodes2 = perft_nnue();
  //const end2 = performance.now();
  //console.log('Perft NNUE', nodes2, 'nodes in', end2 - start2, 'ms');
  //console.log('mega nodes per second', nodes2 / (end2 - start2) / 1000);

  const start3 = performance.now();
  const nodes3 = perft_eval();
  const end3 = performance.now();
  console.log('Perft eval', nodes3, 'nodes in', end3 - start3, 'ms');
  console.log('mega nodes per second', nodes3 / (end3 - start3) / 1000);
  console.log('------------------');

  const megaNodesPerSecondRaw = nodes1 / (end1 - start1) / 1000;
  //const megaNodesPerSecondNnue = nodes2 / (end2 - start2) / 1000;
  const megaNodesPerSecondNnue = 0;
  const megaNodesPerSecondEval = nodes3 / (end3 - start3) / 1000;
  postMessage({ type: 'alphaBetaBenchmarkResults', results: {
    megaNodesPerSecondRaw, megaNodesPerSecondNnue, megaNodesPerSecondEval,
  } });
}

onmessage = function(e: MessageEvent<any>) {
  //console.log('Search worker got:', e.data);
  switch (e.data.type) {
    case 'init':
      initSearchWorker(e.data.mem);
      break;
    case 'setState':
      pvs.set_state(e.data.state);
      if (runEngine) {
        for (const d of [1, 2]) {
          const engineOutput = pvs.mate_search(d);
          postMessage({ type: 'mateSearch', engineOutput });
        }
      }
      //pvs.apply_move(e.data.move, e.data.isHidden);
      //const result = pvs.mate_search(2);
      //console.log('Mate search result', result[0][0], 'nodes:', result[1]);
      //postMessage({ type: 'mateSearch', mateEval: result[0][0], nextMoves: result[0][1] });
      break;
    case 'setRunEngine':
      runEngine = e.data.runEngine;
      break;
    case 'runAlphaBetaBenchmark':
      runAlphaBetaBenchmark();
      break;
  }
}
