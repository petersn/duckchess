import init, { new_engine, max_batch_size, channel_count, parse_pgn4, Engine, perft, perft_nnue, perft_eval, test_threads, test_simd, test_shared_mem } from 'engine';

async function initSearchWorker(sharedArrayBuffer: SharedArrayBuffer) {
  await init();
  console.log('Search worker initialized');
  const array = new Int32Array(sharedArrayBuffer);
  setInterval(() => {
    test_shared_mem(array, 2);
  }, 1000);
}

onmessage = function(e: MessageEvent<any>) {
  console.log('Search worker got:', e.data);
  switch (e.data.type) {
    case 'init':
      initSearchWorker(e.data.mem);
  }
}
