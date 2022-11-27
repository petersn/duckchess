import { engine } from '@tensorflow/tfjs';
import React from 'react';
import './App.css';
import { AlphaBetaBenchmarkResults, MessageFromEngineWorker, MessageFromSearchWorker } from './WorkerMessages';
import { ChessBoard, ChessPiece, PieceKind } from './ChessBoard';
import { BrowserRouter as Router, Route, Link, Routes, Navigate } from 'react-router-dom';
import { BenchmarkApp } from './BenchmarkApp';

// FIXME: This is a mildly hacky way to get the router location...
function getRouterPath(): string {
  let routerPath = window.location.pathname;
  if (routerPath.includes('/'))
    routerPath = routerPath.substring(routerPath.lastIndexOf('/') + 1);
  return routerPath;
}

export class Workers {
  engineWorker: Worker;
  searchWorker: Worker;
  initCallback: (ew: Workers) => void;
  boardState: any;
  legalMoves: any[];
  nextMoves: any[];
  pv: any[] = [];
  evaluation: number = 0;
  nodes: number = 0;
  initFlags: boolean[] = [false, false];
  forceUpdateCallback: () => void;
  benchmarkCallback: (results: AlphaBetaBenchmarkResults) => void;

  constructor(initCallback: () => void, forceUpdateCallback: () => void) {
    this.engineWorker = new Worker(new URL('./EngineWorker.ts', import.meta.url));
    this.engineWorker.onmessage = this.onEngineMessage;
    this.searchWorker = new Worker(new URL('./SearchWorker.ts', import.meta.url));
    this.searchWorker.onmessage = this.onSearchMessage;
    this.initCallback = initCallback;
    this.forceUpdateCallback = forceUpdateCallback;
    this.benchmarkCallback = () => {};
    for (const worker of [this.engineWorker, this.searchWorker]) {
      worker.postMessage({ type: 'init' });
    }
    this.boardState = null;
    this.legalMoves = [];
    this.nextMoves = [];
  }

  setInitFlag(workerIndex: number) {
    this.initFlags[workerIndex] = true;
    if (this.initFlags[0] && this.initFlags[1])
      this.initCallback(this);
  }

  onEngineMessage = (e: MessageEvent<MessageFromEngineWorker>) => {
    console.log('Main thread got:', e.data);
    switch (e.data.type) {
      case 'initted':
        this.setInitFlag(0);
        break;
      case 'board':
        this.boardState = e.data.board;
        this.legalMoves = e.data.moves;
        this.nextMoves = e.data.nextMoves;
        break;
      case 'evaluation':
        const whiteWinProb = e.data.whiteWinProb;
        const Q = 2 * whiteWinProb - 1;
        this.evaluation = 1.11714640912 * Math.tan(1.5620688421 * Q);
        this.pv = e.data.pv;
        this.nodes = e.data.nodes;
        break;
    }
    this.forceUpdateCallback();
  }

  onSearchMessage = (e: MessageEvent<MessageFromSearchWorker>) => {
    console.log('Main thread got:', e.data);
    switch (e.data.type) {
      case 'initted':
        this.setInitFlag(1);
        break;
      case 'alphaBetaBenchmarkResults':
        this.benchmarkCallback(e.data.results);
        this.benchmarkCallback = () => {};
        break;
    }
    this.forceUpdateCallback();
  }

  getState() {
    return this.boardState;
  }

  getMoves(): any[] {
    return this.legalMoves;
  }

  applyMove(move: any, isHidden: boolean) {
    for (const worker of [this.engineWorker, this.searchWorker]) {
      worker.postMessage({ type: 'applyMove', move, isHidden });
    }
  }

  setRunEngine(runEngine: boolean) {
    for (const worker of [this.engineWorker, this.searchWorker]) {
      worker.postMessage({ type: 'setRunEngine', runEngine });
    }
  }

  runAlphaBetaBenchmark(callback: (results: AlphaBetaBenchmarkResults) => void) {
    this.searchWorker.postMessage({ type: 'runAlphaBetaBenchmark' });
    this.benchmarkCallback = callback;
  }
}

function TopBar(props: { isMobile: boolean }) {
  const barEntryStyle: React.CSSProperties = {
    paddingLeft: 20,
    paddingRight: 20,
    borderRight: '1px solid black',
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    color: '#ddd',
    textDecoration: 'none',
  };
  const selectedTab: React.CSSProperties = {
    ...barEntryStyle,
    backgroundColor: '#225',
  };
  const unselectedTab: React.CSSProperties = {
    ...barEntryStyle,
    backgroundColor: '#447',
  };
  const path = getRouterPath();
  return (
    <div style={{
      width: '100%',
      height: 40,
      backgroundColor: '#336',
      borderBottom: '1px solid black',
      display: 'flex',
      flexDirection: 'row',
      alignItems: 'center',
      fontSize: props.isMobile ? '100%' : '130%',
      fontWeight: 'bold',
    }}>
      {props.isMobile || <div style={barEntryStyle}>
        Duck Chess Engine
      </div>}

      <Link to={'/analysis'} style={path === 'analysis' ? selectedTab : unselectedTab} replace>
        Analysis
      </Link>

      <Link to={'/benchmark'} style={path === 'benchmark' ? selectedTab : unselectedTab} replace>
        Benchmark
      </Link>

      <Link to={'/info'} style={path === 'info' ? selectedTab : unselectedTab} replace>
        Info
      </Link>
    </div>
  );
}

function AnalysisPage(props: { isMobile: boolean, workers: Workers | null }) {
  const [selectedSquare, setSelectedSquare] = React.useState<[number, number] | null>(null);
  const [pair, setPair] = React.useState<any>(null);
  const [duckFen, setDuckFen] = React.useState<string>('');
  const [pgn, setPgn] = React.useState<string>('');
  const [runEngine, setRunEngine] = React.useState<boolean>(false);

  // Stop the engine when this subpage isn't active.
  React.useEffect(() => {
    if (props.workers) {
      props.workers.setRunEngine(runEngine);
    }
    return () => {
      if (props.workers) {
        props.workers.setRunEngine(false);
      }
    };
  }, [props.workers]);

  React.useEffect(() => {
    const shortcutsHandler = (event: KeyboardEvent) => {
      if (event.key === ' ') {
        // Make the top move.
        if (props.workers !== null && props.workers.pv.length !== 0) {
          props.workers.applyMove(props.workers.pv[0], false);
        }
      }
    };
    document.addEventListener('keydown', shortcutsHandler);
    return () => document.removeEventListener('keydown', shortcutsHandler);
  }, [props.workers]);

  // let board: React.ReactNode[][] = [];
  // for (let y = 0; y < 8; y++)
  //   board.push([null, null, null, null, null, null, null, null]);
  // let legalMoves: any[] = [];
  // if (engineWorker !== null) {
  //   board = engineWorker.getState() || board;
  //   legalMoves = engineWorker.getMoves() || legalMoves;
  // }

  let board: React.ReactNode[][] = [];
  for (let y = 0; y < 8; y++)
    board.push([null, null, null, null, null, null, null, null]);
  let legalMoves: any[] = [];
  let hiddenLegalMoves: any[] = [];
  if (props.workers !== null) {
    const state = props.workers.getState();
    legalMoves = props.workers.getMoves();
    hiddenLegalMoves = props.workers.nextMoves;
    console.log('HIDDEN LEGAL MOVES', hiddenLegalMoves);
    if (state !== null) {
      for (let y = 0; y < 8; y++) {
        for (let player = 0; player < 2; player++) {
          for (const name of ['pawns', 'knights', 'bishops', 'rooks', 'queens', 'kings', 'ducks']) {
            let byte;
            // Special handling for the duck
            if (name === 'ducks') {
              if (player === 1)
                continue;
              byte = state.ducks[7 - y];
            } else if (name === 'enpassants') {
              if (player === 1)
                continue;
              byte = state.enPassant[7 - y];
            } else {
              byte = state[name][player][7 - y];
            }
            for (let x = 0; x < 8; x++) {
              const hasPiece = byte & 1;
              byte = byte >> 1;
              if (!hasPiece)
                continue;
              let piece = name.replace('knight', 'night').slice(0, 1).toUpperCase();
              if (player === 1)
                piece = 'w' + piece;
              else
                piece = 'b' + piece;
              piece = piece.replace('bD', 'duck');
              board[y][x] = piece;
            }
          }
        }
      }
    }
  }

  //React.useEffect(() => {
  //  props.workers.forceUpdateCallback = () => {
  //    setForceUpdateCounter(Math.random());
  //  };
  //}, []);

  //const { props.workers } = props;
  //const state = props.workers.getState();
  //const moves: any[] = props.workers.getMoves();
  //React.useEffect(() => {
  //  //for (let i = 0; i < 100; i++) {
  //  //  const start = performance.now();
  //  //  const tfjsResult = tfjsModel.predict(tf.ones([64, 8, 8, 22])) as tf.Tensor[];
  //  //  tfjsResult[0].data()
  //  //  const end = performance.now();
  //  //  console.log('tfjs prediction took', end - start, 'ms');
  //  //}
  //  //console.log(tfjsResult[0], tfjsResult[1]);
  //  let pair = props.workers.run(4);
  //  setPair(pair);
  //  setTimeout(() => {
  //    if (pair && pair[1][0]) {
  //      engine.apply_move(pair[1][0]);
  //      setForceUpdateCounter(forceUpdateCounter + 1);
  //    }
  //  }, 1000000);
  //}, [forceUpdateCounter]);

  //if (state === null) {
  //  return <div>Loading...</div>;
  //}

  /*
  function clickOn(x: number, y: number) {
    if (selectedSquare === null) {
      // Check if this is the initial duck placement.
      const m = moves.find(m => m.from === 64 && m.to === x + (7 - y) * 8);
      if (m) {
        props.workers.applyMove(m);
        setSelectedSquare(null);
        setForceUpdateCounter(forceUpdateCounter + 1);
      } else {
        setSelectedSquare([x, y]);
      }
    } else {
      let [fromX, fromY] = selectedSquare;
      fromY = 7 - fromY;
      y = 7 - y;
      const encodedFrom = 8 * fromY + fromX;
      const encodedTo = 8 * y + x;
      // Find the first move that matches the selected square and the clicked square.
      const m = moves.find((m: any) => m.from === encodedFrom && m.to === encodedTo);
      if (m) {
        props.workers.applyMove(m);
        setForceUpdateCounter(forceUpdateCounter + 1);
      }
      setSelectedSquare(null);
    }
  }
  */


  /*
  for (let y = 0; y < 8; y++) {
    for (let player = 0; player < 2; player++) {
      for (const name of ['pawns', 'knights', 'bishops', 'rooks', 'queens', 'kings', 'ducks']) {
        let byte;
        // Special handling for the duck
        if (name === 'ducks') {
          if (player === 1)
            continue;
          byte = state.ducks[7 - y];
        } else if (name === 'enpassants') {
          if (player === 1)
            continue;
          byte = state.enPassant[7 - y];
        } else {
          byte = state[name][player][7 - y];
        }
        for (let x = 0; x < 8; x++) {
          const hasPiece = byte & 1;
          byte = byte >> 1;
          if (!hasPiece)
            continue;
          let piece = name.slice(0, -1);
          if (player === 1)
            piece = piece.toUpperCase();
          board[y][x] = <span style={{
            fontSize: '40px',
            userSelect: 'none',
            fontFamily: 'Arial Unicode MS',
          }}>
            <ChessPiece piece={piece as PieceKind} />
          </span>;
        }
      }
    }
  }
  */

  /*
  //let showMoves: any[] = (pair && pair[1][0]) ? pair[1] : [];
  console.log('----PV:', props.workers.pv);
  let showMoves: any[] = props.workers.pv;
  if ((state.isDuckMove || true) && showMoves) {
    showMoves = showMoves.slice(0, 1);
  }
  */

  let topMoves = [];
  if (props.workers !== null && props.workers.boardState !== null) {
    // If it's a duck move, only show the next 1 move.
    const moveCount = props.workers.boardState.isDuckMove ? 1 : 2;
    topMoves = props.workers.pv.slice(0, moveCount);
  }

  return <>
    <div style={{
      width: '100%',
      height: '100%',
      display: 'flex',
      flexDirection: props.isMobile ? 'column' : 'row',
      alignItems: 'center',
      justifyContent: 'center',
    }}>
      {props.isMobile || <div style={{
        flex: 1,
        height: 600,
        minHeight: 600,
      }}>
        <div style={{
          marginLeft: 'auto',
          height: '100%',
          maxWidth: 400,
          border: '1px solid #222',
          backgroundColor: '#33333f',
          padding: 10,
          boxSizing: 'border-box',
          overflow: 'scroll',
        }}>
          <div style={{ fontWeight: 'bold', fontSize: '120%', marginBottom: 10 }}>Duck Chess Analysis Board</div>

          <a href="https://duckchess.com/">Duck chess</a> is a variant in which there is one extra piece, the duck, which is shared between the two players and cannot be captured.
          Each turn consists of making a regular move, and then moving the duck to any empty square (the duck may not stay in place).
          There is no check or checkmate, you win by capturing the opponent's king.
        </div>
      </div>}
      <ChessBoard
        isMobile={props.isMobile}
        board={board as any}
        legalMoves={legalMoves}
        hiddenLegalMoves={hiddenLegalMoves}
        topMoves={topMoves}
        onMove={(move, isHidden) => {
          if (props.workers !== null) {
            console.log('[snp1] move', move, isHidden);
            props.workers.applyMove(move, isHidden);
          }
        }}
        style={{ margin: 10 }}
      />
      <div style={{
        flex: props.isMobile ? undefined : 1,
        height: props.isMobile ? undefined : 600,
        minHeight: props.isMobile ? undefined : 600,
        width: props.isMobile ? '100%' : undefined,
      }}>
        <div style={{
          height: '100%',
          maxWidth: 400,
          border: '1px solid #222',
          backgroundColor: '#33333f',
          padding: 10,
          boxSizing: 'border-box',
        }}>
          <div style={{ fontWeight: 'bold', fontSize: '120%', marginBottom: 10 }}>Engine</div>
          <input type="checkbox" checked={runEngine} onChange={e => {
            setRunEngine(e.target.checked);
            if (props.workers !== null) {
              props.workers.setRunEngine(e.target.checked);
            }
          }} /> Run engine

          {props.workers !== null && <div>
            Evaluation: {props.workers.evaluation}<br/>
            Nodes: {props.workers.nodes}<br/>
            PV: {props.workers.pv.map((m: any) => m.from + ' ' + m.to).join(' ')}
          </div>}

        </div>
      </div>
    </div>

    <div style={{ marginTop: 10, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <input type="text" value={duckFen} onChange={e => setDuckFen(e.target.value)} style={{
        width: 400,
        backgroundColor: '#445',
        color: '#eee',
      }} />
      <textarea value={pgn} onChange={e => setPgn(e.target.value)} style={{
        marginTop: 5,
        width: 400,
        height: 100,
        backgroundColor: '#445',
        color: '#eee',
      }} placeholder="Paste PGN here..." />
    </div>

    <div style={{ marginTop: 10, textAlign: 'center', width: '95vw' }}>
      Created by Peter Schmidt-Nielsen
      (<a href="https://twitter.com/ptrschmdtnlsn">Twitter</a>, <a href="https://peter.website">Website</a>)<br/>
      Engine + web interface: <a href="https://github.com/petersn/duckchess">github.com/petersn/duckchess</a><br/>
      <span style={{ fontSize: '50%', opacity: 0.5 }}>Piece SVGs: Cburnett (CC BY-SA 3), Duck SVG + all code: my creation (CC0)</span>
    </div>
  </>;
}

function BenchmarkPage(props: { isMobile: boolean, workers: Workers | null }) {
  if (props.workers === null)
    return <div>Loading...</div>;
  return <BenchmarkApp isMobile={props.isMobile} workers={props.workers} />;
}

function InfoPage(props: { isMobile: boolean }) {
  return (
    <>Hello</>
  );
}

function App() {
  const [workers, setWorkers] = React.useState<Workers | null>(null);
  const setForceUpdateCounter = React.useState(0)[1];
  const [width, setWidth] = React.useState<number>(window.innerWidth);

  React.useEffect(() => {
    console.log('Initializing worker...');
    const workers = new Workers(
      () => {
        console.log('Worker initialized!');
        // FIXME: There's a race if you toggle runEngine before the engine is created.
        setWorkers(workers);
      },
      () => {
        setTimeout(() => setForceUpdateCounter(Math.random()), 1);
      },
    );
  }, []);

  function handleWindowSizeChange() {
    setWidth(window.innerWidth);
  }
  React.useEffect(() => {
    window.addEventListener('resize', handleWindowSizeChange);
    return () => window.removeEventListener('resize', handleWindowSizeChange);
  }, []);

  const isMobile = width <= 768;

  console.log('process.env.PUBLIC_URL:', process.env.PUBLIC_URL);

  function navigation(element: React.ReactNode): React.ReactNode {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}>
        <TopBar isMobile={isMobile} />
        {element}
      </div>
    );
  }

  return (
    <Router basename={process.env.PUBLIC_URL}>
      <Routes>
        <Route path="/analysis" element={navigation(<AnalysisPage isMobile={isMobile} workers={workers} />)} />
        <Route path="/benchmark" element={navigation(<BenchmarkPage isMobile={isMobile} workers={workers} />)} />
        <Route path="/info" element={navigation(<InfoPage isMobile={isMobile} />)} />
        <Route path="/" element={navigation(<Navigate to={"/analysis"} replace />)} />
      </Routes>
    </Router>
  );
}

export default App;
