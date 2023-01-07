import { engine } from '@tensorflow/tfjs';
import React from 'react';
import './App.css';
import init, { new_game_tree, GameTree } from 'engine';
import { AlphaBetaBenchmarkResults, MessageFromEngineWorker, MessageFromSearchWorker } from './DuckChessEngine';
import { ChessBoard, ChessPiece, PieceKind, BOARD_MAX_SIZE, Move } from './ChessBoard';
import { BrowserRouter as Router, Route, Link, Routes, Navigate } from 'react-router-dom';
import { BenchmarkApp } from './BenchmarkApp';

/*
{row.moves.map((entry, j) =>
          <span key={j} style={{ paddingLeft: 10, paddingRight: 10, color: '#ddd', padding: 1 }}>
            {entry!.name}
          </span>
        )}
*/

// FIXME: This is a mildly hacky way to get the router location...
function getRouterPath(): string {
  let routerPath = window.location.pathname;
  if (routerPath.includes('/'))
    routerPath = routerPath.substring(routerPath.lastIndexOf('/') + 1);
  return routerPath;
}

function parseRustBoardState(state: any): PieceKind[][] {
  let board: PieceKind[][] = [];
  for (let y = 0; y < 8; y++)
    board.push([null, null, null, null, null, null, null, null]);
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
          board[y][x] = piece as PieceKind;
        }
      }
    }
  }
  return board;
}

export class Workers {
  //engineWorker: Worker;
  //searchWorker: Worker;
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
    //this.engineWorker = new Worker(new URL('./EngineWorker.ts', import.meta.url));
    //this.engineWorker.onmessage = this.onEngineMessage;
    //this.searchWorker = new Worker(new URL('./SearchWorker.ts', import.meta.url));
    //this.searchWorker.onmessage = this.onSearchMessage;
    this.initCallback = initCallback;
    this.forceUpdateCallback = forceUpdateCallback;
    this.benchmarkCallback = () => {};
    //for (const worker of [this.engineWorker, this.searchWorker]) {
    //  worker.postMessage({ type: 'init' });
    //}
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
    //for (const worker of [this.engineWorker, this.searchWorker]) {
    //  worker.postMessage({ type: 'applyMove', move, isHidden });
    //}
  }

  historyJump(index: number) {
    //for (const worker of [this.engineWorker, this.searchWorker]) {
    //  worker.postMessage({ type: 'historyJump', index });
    //}
  }

  setRunEngine(runEngine: boolean) {
    //for (const worker of [this.engineWorker, this.searchWorker]) {
    //  worker.postMessage({ type: 'setRunEngine', runEngine });
    //}
  }

  runAlphaBetaBenchmark(callback: (results: AlphaBetaBenchmarkResults) => void) {
    //this.searchWorker.postMessage({ type: 'runAlphaBetaBenchmark' });
    this.benchmarkCallback = callback;
  }
}

function TopBar(props: { screenWidth: number, isMobile: boolean }) {
  // If the window is really narrow we need a different layout.
  const narrow = props.screenWidth < 320;
  const barEntryStyle: React.CSSProperties = {
    paddingLeft: narrow ? 10 : 20,
    paddingRight: narrow ? 10 : 20,
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

interface Info {
  id: any;
  edges: [Move, string, Info][];
}

function AnalysisPage(props: { isMobile: boolean, gameTree: GameTree, workers: Workers | null }) {
  const [selectedSquare, setSelectedSquare] = React.useState<[number, number] | null>(null);
  const [pair, setPair] = React.useState<any>(null);
  const [duckFen, setDuckFen] = React.useState<string>('');
  const [pgn, setPgn] = React.useState<string>('');
  const [runEngine, setRunEngine] = React.useState<boolean>(false);
  const [forceUpdate, setForceUpdate] = React.useState<number>(0);
  const [nodeContextMenu, setNodeContextMenu] = React.useState<number | null>(null);

  const [ser, a, cursor]: [any, Info, any] = props.gameTree.get_serialized_state();
  console.log('Got:', ser);
  console.log('Got:', a);
  console.log('Got:', cursor);

  interface MoveRowEntry {
    name: string;
    id: any;
  }

  interface MoveRow {
    moves: (MoveRowEntry | null)[];
    smallRowContent: React.ReactNode | null;
  }

  const moveRows: MoveRow[] = [];
  let info = a;
  let moveNum = 0;

  function generateSmallRow(
    info: Info,
    moveNum: number,
    prefix: string,
    startingSmallRow: boolean
  ) {
    const output = [];
    for (let i = 1; i < info.edges.length + (1 - +startingSmallRow); i++) {
      const effectiveI = i % info.edges.length;
      const [move, moveName, child] = info.edges[effectiveI];
      output.push(<React.Fragment key={i}>
        {effectiveI == 0 || '('} {moveSpan({
          name: prefix + moveName,
          id: child.id,
        }, false)} {generateSmallRow(child, moveNum + 1, '', false)} {effectiveI == 0 || ')'}
      </React.Fragment>);
    }
    return output;
  }

  while (info.edges.length > 0) {
    if (moveNum % 4 === 0 ) {
      moveRows.push({ moves: [null, null, null, null], smallRowContent: null });
    }
    // If there are side branches then insert a little indicator for them.
    let prefix = '';
    if (info.edges.length > 1) {
      // We add a prefix if the move is for black.
      if (moveNum % 4 >= 2)
        prefix = '...';
      const smallRow: MoveRow = {
        moves: [],
        smallRowContent: generateSmallRow(info, moveNum, prefix, true)
      }
      moveRows.push(smallRow);
      moveRows.push({ moves: [null, null, null, null], smallRowContent: null });
    }
    const [move, moveName, child] = info.edges[0];
    const moveRow = moveRows[moveRows.length - 1];
    moveRow.moves[moveNum % 4] = {
      name: prefix + moveName,
      id: child.id,
    };
    info = child;
    moveNum++;
  }

  function moveSpan(move: MoveRowEntry | null, isFull: boolean) {
    const moveSpanStyle: React.CSSProperties = isFull ? {
      display: 'flex',
      color: '#ddd',
      cursor: 'pointer',
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
    } : {
      color: '#ddd',
      cursor: 'pointer',
    }
    if (move === null) {
      return <span style={moveSpanStyle}></span>;
    }
    let contextMenu: React.ReactNode = null;
    if (nodeContextMenu === move.id.idx) {
      contextMenu = <div style={{
        position: 'absolute',
        backgroundColor: '#336',
        border: '1px solid black',
        padding: 4,
        zIndex: 1,
        transform: 'translate(-100%, 0)',
      }}>
        <div className='contextMenuButton' onClick={() => {
          props.gameTree.delete_by_id(move.id);
          setNodeContextMenu(null);
          setForceUpdate(Math.random());
        }}>Delete</div>
        {isFull || <div className='contextMenuButton' onClick={() => {
          props.gameTree.promote_by_id(move.id);
          setNodeContextMenu(null);
          setForceUpdate(Math.random());
        }}>Promote</div>}
      </div>;
    }
  
    return <span
      style={{
        ...moveSpanStyle,
        color: move.id.idx === nodeContextMenu ? 'yellow' :
        (move.id.idx === cursor.idx ? 'red' : '#ddd'),
      }}
      onClick={() => {
        props.gameTree.click_to_id(move.id);
        setNodeContextMenu(null);
        setForceUpdate(Math.random());
      }}
      onContextMenu={(e) => {
        e.preventDefault();
        setNodeContextMenu(move.id.idx);
      }}
    >{move.name}{contextMenu}</span>;
  }

  const moveHalfBoxStyle: React.CSSProperties = {
    display: 'flex',
    flex: 1,
    padding: 4,
    margin: 1,
    height: 20,
    alignItems: 'center',
  };

  const renderedMoveRows = moveRows.map((row, i) => {
    if (row.smallRowContent !== null) {
      return <div key={i} style={{
        fontSize: '80%',
        opacity: 0.8,
        backgroundColor: '#777',
        margin: 1,
      }}>
        {row.smallRowContent}
      </div>;
    }
    if (row.moves.every(m => m === null)) {
      return null;
    }
    return <div key={i} style={{ display: 'flex', flexDirection: 'row' }}>
      <div style={{ ...moveHalfBoxStyle, backgroundColor: '#667' }}>
        {moveSpan(row.moves[0], true)}
        {moveSpan(row.moves[1], true)}
      </div>
      <div style={{ ...moveHalfBoxStyle, backgroundColor: '#445' }}>
        {moveSpan(row.moves[2], true)}
        {moveSpan(row.moves[3], true)}
      </div>
    </div>;
  });

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
      } else if (event.key === 'Escape') {
        // Clear the selected square.
        setSelectedSquare(null);
        setNodeContextMenu(null);
      } else if (event.key === 'ArrowLeft') {
        props.gameTree.history_back();
        setForceUpdate(Math.random());
        setNodeContextMenu(null);
        // Go back one move.
        if (props.workers !== null) {
          props.workers.applyMove({ type: 'undo' }, false);
        }
      } else if (event.key === 'ArrowRight') {
        props.gameTree.history_forward();
        setForceUpdate(Math.random());
        setNodeContextMenu(null);
        // Go forward one move.
        if (props.workers !== null) {
          props.workers.applyMove({ type: 'redo' }, false);
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

  const marginTop = props.isMobile ? 3 : 10;

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
        height: BOARD_MAX_SIZE,
        minHeight: BOARD_MAX_SIZE,
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
        highlightDuck={ser.state.isDuckMove}
        board={parseRustBoardState(ser.state)}
        legalMoves={ser.legal_moves}
        hiddenLegalMoves={ser.legal_duck_skipping_moves}
        topMoves={topMoves}
        onMove={(move, isHidden) => {
          //if (props.workers !== null) {
            console.log('[snp1] move', move, isHidden);
            props.gameTree.make_move(move, isHidden);
            setForceUpdate(Math.random());
            // Force a rerender.
            //props.workers.applyMove(move, isHidden);
          //}
        }}
        style={{ margin: 10 }}
      />
      <div style={{
        flex: props.isMobile ? undefined : 1,
        height: props.isMobile ? undefined : BOARD_MAX_SIZE,
        minHeight: props.isMobile ? undefined : BOARD_MAX_SIZE,
        width: props.isMobile ? '100%' : undefined,
      }}>
        <div style={{
          height: '100%',
          maxWidth: 400,
          border: '1px solid #222',
          backgroundColor: '#33333f',
          padding: 10,
          boxSizing: 'border-box',
          margin: props.isMobile ? 'auto' : undefined,
        }}>
          <div style={{ fontWeight: 'bold', fontSize: '120%', marginBottom: 10 }}>Engine</div>
          <input type="checkbox" checked={runEngine} onChange={e => {
            setRunEngine(e.target.checked);
            if (props.workers !== null) {
              props.workers.setRunEngine(e.target.checked);
            }
          }} /> Run engine

          <div>
            {renderedMoveRows}
          </div>

          {props.workers !== null && <div>
            Evaluation: {props.workers.evaluation.toFixed(3)}<br/>
            Nodes: {props.workers.nodes}<br/>
            PV: {props.workers.pv.map((m: any) => m.from + ' ' + m.to).join(' ')}
          </div>}

        </div>
      </div>
    </div>

    <div style={{ width: '100%', marginTop, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <input type="text" value={duckFen} onChange={e => setDuckFen(e.target.value)} style={{
        //width: 400,
        width: '96%',
        maxWidth: 400,
        backgroundColor: '#445',
        color: '#eee',
      }} />
      <textarea value={pgn} onChange={e => setPgn(e.target.value)} style={{
        marginTop: 5,
        //width: 400,
        width: '96%',
        maxWidth: 400,
        height: 100,
        backgroundColor: '#445',
        color: '#eee',
      }} placeholder="Paste PGN here..." />
    </div>

    <div style={{ marginTop, textAlign: 'center', width: '95vw' }}>
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
  const [gameTree, setGameTree] = React.useState<GameTree | null>(null);
  const [workers, setWorkers] = React.useState<Workers | null>(null);
  const setForceUpdateCounter = React.useState(0)[1];
  const [width, setWidth] = React.useState<number>(window.innerWidth);

  React.useEffect(() => {
    init().then(() => setGameTree(new_game_tree()));
  }, []);

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

  const isMobile = width < 768;

  console.log('process.env.PUBLIC_URL:', process.env.PUBLIC_URL);

  function navigation(element: React.ReactNode): React.ReactNode {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}>
        <TopBar screenWidth={width} isMobile={isMobile} />
        {element}
      </div>
    );
  }

  if (gameTree === null)
    return <div>Loading...</div>;

  return (
    <Router basename={process.env.PUBLIC_URL}>
      <Routes>
        <Route path="/analysis" element={navigation(<AnalysisPage isMobile={isMobile} gameTree={gameTree} workers={workers} />)} />
        <Route path="/benchmark" element={navigation(<BenchmarkPage isMobile={isMobile} workers={workers} />)} />
        <Route path="/info" element={navigation(<InfoPage isMobile={isMobile} />)} />
        <Route path="/" element={navigation(<Navigate to={"/analysis"} replace />)} />
      </Routes>
    </Router>
  );
}

export default App;
