
export type MessageToEngineWorker = {
  type: 'init';
} | {
  type: 'applyMove';
  move: any;
};

export type MessageFromEngineWorker = {
  type: 'initted';
} | {
  type: 'board';
  board: any;
  moves: any[];
} | {
  type: 'evaluation';
  evaluation: number;
  bestMove: {
    from: number;
    to: number;
  };
} 
