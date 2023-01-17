import { engine } from '@tensorflow/tfjs';
import React from 'react';
import './App.css';
import init, { new_game_tree, GameTree, get_visit_limit, get_wasm_version } from 'engine';
import { AlphaBetaBenchmarkResults, createDuckChessEngine, DuckChessEngine, MessageFromEngineWorker, MessageFromSearchWorker, ModelName, parseRustBoardState } from './DuckChessEngine';
import { ChessBoard, ChessPiece, PieceKind, BOARD_MAX_SIZE, Move } from './ChessBoard';
import { BrowserRouter as Router, Route, Link, Routes, Navigate } from 'react-router-dom';
import { BenchmarkApp } from './BenchmarkApp';

const GUI_VERSION = 'v0.1.10';

// FIXME: This is hacky.
// For cosmetic reasons I cap the visits I show, to hide the few visits over the limit we might do.
const VISIT_LIMIT = 50_000;

const DEFAULT_PGN4 = ``;


type ToolMode = 'analysis' | 'play100' | 'play1k' | 'play3k';

// FIXME: This is a mildly hacky way to get the router location...
function getRouterPath(): string {
  let routerPath = window.location.pathname;
  if (routerPath.includes('/'))
    routerPath = routerPath.substring(routerPath.lastIndexOf('/') + 1);
  return routerPath;
}

function LimitationsPage(props: { isMobile: boolean }) {
  return <div style={{ width: 600 }}>
    <h1>Limitations</h1>
    This engine currently has some major limitations:
    <ul>
      <li>Under-promotion isn't implemented quite yet, sorry. :(</li>
      <li>No 50-move rule is implemented, although threefold repetition is implemented.</li>
      <li>In some occasional circumstances the engine will fail to properly understand threefold, due to specifics of how threefold is dealt with in the transposition table.
        The engine might either play a move blundering a threefold from a winning position, or be artifically scared of a non-existent threefold.
        This is fixable, but it's unfortunately rather annoying, so I might not do it immediately.</li>
      <li>The alpha-beta mate search doesn't know about threefold repetition at all yet, so the engine might report a mate that is ruled out by threefold.</li>
      <li>The network was trained with very little discounting in time, so it has a tendency to faff about in the endgame even when it's overwhelmingly winning.</li>
      <li>Likewise, if the engine is overwhelmingly losing it will sometimes start to play sub-optimal moves, as MCTS barely distinguishes a 0% chance of winning from a 1% chance.</li>
    </ul>
    If you find bugs, please report them on <a href="https://github.com/petersn/duckchess/issues">GitHub</a>!
    <div style={{ marginTop: 20, fontSize: '150%' }}>
      <Link to={'/'}>
        Back
      </Link>
    </div>
  </div>;
}

function SettingsPage(props: { isMobile: boolean, engine: DuckChessEngine }) {
  // Get our search params from local storage.
  const [searchParams, setSearchParams] = React.useState(
    localStorage.getItem('duckChessSearchParams') || 'default'
  );

  return <div style={{ width: 600 }}>
    <h1>Settings</h1>

    <p>
      My goal is to allow fine-adjustment of the engine here soon.
      The deep search parameters promote searching variations deeply, rather than searching broadly (this is equivalent to lowering cpuct in lc0).
      My testing seems to indicate that the deep search parameters outperform default by ~100 elo points, but the values seem so
      absurd (they're about 10x lower than what lc0 settled on!) that I'm scared to deploy them as the default.
    </p>

    <div style={{ marginTop: 20 }}>
      <label style={{ marginRight: 10 }}>Search params:</label>
      <select value={searchParams} onChange={(e) => {
        setSearchParams(e.target.value);
        localStorage.setItem('duckChessSearchParams', e.target.value);
        props.engine.reloadSearchParams();
      }}>
        <option value="default">Default</option>
        <option value="alpha=0.25:duckalpha=0.125:fpu=0.2">Deep</option>
      </select><br/>
    </div>

    <div style={{ marginTop: 20 }}>
      (Coming soon, a contempt setting, to force the engine to avoid draws.)
    </div>

    <div style={{ marginTop: 20, fontSize: '150%' }}>
      <Link to={'/'}>
        Back
      </Link>
    </div>
  </div>;
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

function WinDrawLossIndicator(props: { isMobile: boolean, boardFlipped: boolean, wdl: number[] }) {
  let firstColor = '#444';
  let lastColor = '#eee';
  let wdl = props.wdl;
  if (props.boardFlipped) {
    firstColor = '#eee';
    lastColor = '#444';
    wdl = [wdl[2], wdl[1], wdl[0]];
  }

  const { isMobile } = props;
  return (
    <div style={{
      width: isMobile ? 399 : 20,
      height: isMobile ? 20 : 399,
      backgroundColor: '#aaa',
      overflow: 'hidden',
      position: 'relative',
    }}>
      {/* Black wins */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: isMobile ? 399 * (wdl[0] + wdl[1]) : 0,
        width: isMobile ? 399 * wdl[2] : 20,
        height: isMobile ? 20 : 399 * wdl[2],
        backgroundColor: firstColor,
      }} />
      {/* Draw */}
      <div style={{
        position: 'absolute',
        top: isMobile ? 0 : 399 * wdl[2],
        left: isMobile ? 399 * wdl[0] : 0,
        width: isMobile ? 399 * wdl[1] : 20,
        height: isMobile ? 20 : 399 * wdl[1],
        backgroundColor: '#aaa',
      }} />
      {/* White wins */}
      <div style={{
        position: 'absolute',
        top: isMobile ? 0 : 399 * (wdl[2] + wdl[1]),
        left: 0,
        width: isMobile ? 399 * wdl[0] : 20,
        height: isMobile ? 20 : 399 * wdl[0],
        backgroundColor: lastColor,
      }} />
    </div>
  );
}

interface Info {
  id: any;
  evaluation: {
    white_perspective_score: number;
    white_perspective_wdl: [number, number, number];
    mate_score: number | null;
    top_moves: [Move, number][];
    steps: number;
  };
  edges: [Move, string, Info][];
  turn: 'white' | 'black';
}

function getDefaultModelName(): ModelName {
  return localStorage.getItem('modelName') as ModelName || 'medium-001-128x10';
}

let globalBoardFlipped = false;

function AnalysisPage(props: { isMobile: boolean, engine: DuckChessEngine }) {
  const { engine } = props;
  //const [selectedSquare, setSelectedSquare] = React.useState<[number, number] | null>(null);
  //const [pair, setPair] = React.useState<any>(null);
  //const [duckFen, setDuckFen] = React.useState<string>('');
  const [modelName, setModelName] = React.useState<ModelName>(getDefaultModelName());
  const [toolMode, setToolMode] = React.useState<ToolMode>('analysis');
  const [pgn, setPgn] = React.useState<string>(DEFAULT_PGN4);
  const [boardFlipped, setBoardFlipped] = React.useState<boolean>(globalBoardFlipped);
  //const [enginePlayer, setEnginePlayer] = React.useState<'white' | 'black' | null>(null);
  const [runEngine, setRunEngine] = React.useState<boolean>(true);
  const [nodeContextMenu, setNodeContextMenu] = React.useState<number | null>(null);

  const setForceUpdateCounter = React.useState(0)[1];
  const forceUpdate = () => {
    setForceUpdateCounter(Math.random());
  };

  function adjustModelName(modelName: ModelName) {
    setModelName(modelName);
    localStorage.setItem('modelName', modelName);
    engine.setModel(modelName);
  }

  function adjustToolMode(toolMode: ToolMode) {
    setToolMode(toolMode);
    // If the tool mode is set to one of the play modes, adjust the engine for this.
    const enginePlayer = boardFlipped ? 'white' : 'black';
    if (toolMode === 'play100') {
      engine.setPlayMode({ player: enginePlayer, steps: 100 });
    } else if (toolMode === 'play1k') {
      engine.setPlayMode({ player: enginePlayer, steps: 1000 });
    } else if (toolMode === 'play3k') {
      engine.setPlayMode({ player: enginePlayer, steps: 3000 });
    } else {
      engine.setPlayMode(null);
      engine.setRunEngine(runEngine);
      //setEnginePlayer(null);
    }
  }

  let [ser, info, cursor, pvLeaf, boardHash]: [any, Info, any, any, number] = engine.gameTree.get_serialized_state();
  //console.log('GOT:', cursor, pvLeaf);

  interface MoveRowEntry {
    name: string;
    pawn_score: number | null;
    mate_score: number | null;
    steps: number;
    id: any;
  }

  interface MoveRow {
    //pawn_score: number | null;
    //mate_score: number | null;
    moves: (MoveRowEntry | null)[];
    smallRowContent: React.ReactNode | null;
  }

  const moveRows: MoveRow[] = [];
  let moveNum = 0;

  let thisNodeInfo: (Info | null)[] = [null];
  // We walk the entire tree trying to find the current move.
  function walk(info: Info) {
    if (info.id.idx == cursor.idx) {
      thisNodeInfo[0] = info;
      return true;
    }
    for (const edge of info.edges) {
      if (walk(edge[2]))
        return true;
    }
    return false;
  }
  walk(info);

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
          pawn_score: child.evaluation.white_perspective_score,
          mate_score: child.evaluation.mate_score,
          steps: child.evaluation.steps,
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
      pawn_score: child.evaluation.white_perspective_score,
      mate_score: child.evaluation.mate_score,
      steps: child.evaluation.steps,
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
        top: 20,
        left: '50%',
        backgroundColor: '#336',
        border: '1px solid black',
        padding: 4,
        zIndex: 80,
        userSelect: 'none',
        fontSize: '14pt',
        textAlign: 'center',
        transform: 'translate(-50%, 0)',
        //transform: props.isMobile ? 'translate(25%, 0)' : 'translate(-100%, 0)',
      }}>
        <div className='contextMenuButton' onClick={() => {
          engine.gameTree.delete_by_id(move.id);
          engine.sendBoardToEngine();
          setNodeContextMenu(null);
          forceUpdate();
        }}>Delete</div>
        {isFull || <div className='contextMenuButton' onClick={() => {
          engine.gameTree.promote_by_id(move.id);
          engine.sendBoardToEngine();
          setNodeContextMenu(null);
          forceUpdate();
        }}>Promote</div>}
        <div className='contextMenuButton' style={{ fontSize: '80%' }} onClick={() => {
          setNodeContextMenu(null);
          forceUpdate();
        }}>Cancel</div>
      </div>;
    }

    let score = '';
    if (toolMode === 'analysis' && move.steps !== 0 && isFull) {
      if (move.mate_score != null) {
        // We now map the mate_score (which is plies to mate) to a standard mate.
        // So, a value of 1 means the player to move can immediately capture the king.
        // A value of 2 means that the player to move must move the duck, then get captured.
        // Therefore, mate_score in (1, 2, 3) corresponds to a "mate in 1",
        // while (4, 5, 6, 7) corresponds to a "mate in 2", and so on.
        // So we set:
        const sign = move.mate_score > 0 ? '' : '-';
        const mate_in = Math.floor(Math.abs(move.mate_score) / 4) + 1;
        score = 'M' + sign + mate_in;
      } else {
        score = Math.max(-99.9, Math.min(99.9, (4 * Math.tan(3.14 * move.pawn_score! - 1.57)))).toFixed(1);
      }
    }

    return <span
      style={{
        ...moveSpanStyle,
        position: 'relative',
        color: move.id.idx === nodeContextMenu ? 'yellow' :
        (move.id.idx === cursor.idx ? 'red' : '#ddd'),
      }}
      onClick={() => {
        engine.gameTree.click_to_id(move.id);
        engine.sendBoardToEngine();
        setNodeContextMenu(null);
        forceUpdate();
        engine.maybeMakeEngineMove();
      }}
      onContextMenu={(e) => {
        e.preventDefault();
        setNodeContextMenu(move.id.idx);
      }}
    >{score} {move.name}{contextMenu}</span>;
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
        color: '#ddd',
        //opacity: 0.8,
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
    if (engine) {
      engine.setRunEngine(runEngine);
    }
    return () => {
      if (engine) {
        engine.setRunEngine(false);
      }
    };
  }, [engine]);

  React.useEffect(() => {
    const shortcutsHandler = (event: KeyboardEvent) => {
      if (event.key === ' ') {
        engine.makeTopMove();
        setNodeContextMenu(null);
      } else if (event.key === 'Escape') {
        setNodeContextMenu(null);
      } else if (event.key === 'ArrowLeft') {
        engine.historyBack();
        setNodeContextMenu(null);
      } else if (event.key === 'ArrowRight') {
        engine.historyForward();
        setNodeContextMenu(null);
      } else if (event.key === 'f') {
        globalBoardFlipped = !globalBoardFlipped;
        setBoardFlipped(globalBoardFlipped);
        setNodeContextMenu(null);
      }
    };
    const globalClick = (event: MouseEvent) => {
      if (event.target instanceof HTMLElement) {
        if (event.target.closest('.contextMenuButton') === null) {
          setNodeContextMenu(null);
        }
      }
    };

    document.addEventListener('keydown', shortcutsHandler);
    document.addEventListener('click', globalClick);
    return () => {
      document.removeEventListener('keydown', shortcutsHandler);
      document.removeEventListener('click', globalClick);
    };
  }, [engine]);

  // let board: React.ReactNode[][] = [];
  // for (let y = 0; y < 8; y++)
  //   board.push([null, null, null, null, null, null, null, null]);
  // let legalMoves: any[] = [];
  // if (engineWorker !== null) {
  //   board = engineWorker.getState() || board;
  //   legalMoves = engineWorker.getMoves() || legalMoves;
  // }

  //React.useEffect(() => {
  //  engine.forceUpdateCallback = () => {
  //    setForceUpdateCounter(Math.random());
  //  };
  //}, []);

  //const { engine } = props;
  //const state = engine.getState();
  //const moves: any[] = engine.getMoves();
  //React.useEffect(() => {
  //  //for (let i = 0; i < 100; i++) {
  //  //  const start = performance.now();
  //  //  const tfjsResult = tfjsModel.predict(tf.ones([64, 8, 8, 22])) as tf.Tensor[];
  //  //  tfjsResult[0].data()
  //  //  const end = performance.now();
  //  //  console.log('tfjs prediction took', end - start, 'ms');
  //  //}
  //  //console.log(tfjsResult[0], tfjsResult[1]);
  //  let pair = engine.run(4);
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
        engine.applyMove(m);
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
        engine.applyMove(m);
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
  console.log('----PV:', engine.pv);
  let showMoves: any[] = engine.pv;
  if ((state.isDuckMove || true) && showMoves) {
    showMoves = showMoves.slice(0, 1);
  }
  */

  //let topMoves = [];
  // if (engine !== null && engine.boardState !== null) {
  //   // If it's a duck move, only show the next 1 move.
  //   const moveCount = engine.boardState.isDuckMove ? 1 : 2;
  //   topMoves = engine.pv.slice(0, moveCount);
  // }

  const marginTop = props.isMobile ? 3 : 10;
  let nodes = 0;
  let white_wdl = [0, 0, 0];
  let topMoves: [Move, number][] = [];
  if (toolMode === 'analysis' && thisNodeInfo[0] !== null) {
    nodes = thisNodeInfo[0].evaluation.steps;
    white_wdl = thisNodeInfo[0].evaluation.white_perspective_wdl;
    topMoves = thisNodeInfo[0].evaluation.top_moves.slice(0, 3);
    // Renormalize the weights across these three moves.
    let totalWeight = 1e-6;
    for (const move of topMoves) {
      totalWeight += move[1];
    }
    for (const move of topMoves) {
      move[1] = move[1] / totalWeight;
    }
  }
  let engineStatus = thisNodeInfo[0] === null ? 'BUG' : <>
    (Nodes: {Math.min(nodes, VISIT_LIMIT)})
  </>;

  const isHistoryMove = cursor.idx !== pvLeaf.idx;
  const isEnginesTurn = (!isHistoryMove) && thisNodeInfo[0] !== null && thisNodeInfo[0].turn === engine.playMode?.player;

  const modelProgress = globalModelLoadProgress[modelName];

  return <>
    {modelProgress !== undefined && modelProgress < 1 && <div style={{
      position: 'absolute',
      top: 5,
      left: 5,
      width: 200,
      height: 75,
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      border: '1px solid #222',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      pointerEvents: 'none',
      userSelect: 'none',
      zIndex: 100,
    }}>
      Loading model: {Math.floor(modelProgress * 100)}%
    </div>}

    <div style={{
      width: '100%',
      height: '100%',
      display: 'flex',
      flexDirection: props.isMobile ? 'column' : 'row',
      alignItems: 'center',
      justifyContent: 'center',
    }}>
      <div style={{ flex: 1 }} />
      {/*props.isMobile || <div style={{
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
      </div>*/}
      <ChessBoard
        isMobile={props.isMobile}
        isInitialDuckPlacementMove={ser.state.plies === 1}
        highlightDuck={ser.state.isDuckMove}
        boardFlipped={boardFlipped}
        board={parseRustBoardState(ser.state)}
        pastOccurrences={ser.past_occurrences}
        moveHistory={ser.state.moveHistory}
        isDuckMove={ser.state.isDuckMove}
        boardHash={boardHash}
        legalMoves={ser.legal_moves}
        hiddenLegalMoves={ser.legal_duck_skipping_moves}
        topMoves={topMoves} //topMoves}
        onMove={(move, isHidden) => {
          //if (engine !== null) {
            //console.log('[snp1] move', move, isHidden);
            engine.gameTree.make_move(move, isHidden);
            engine.sendBoardToEngine();
            forceUpdate();
            // Force a rerender.
            //engine.applyMove(move, isHidden);
          //}
        }}
        style={{ margin: 10 }}
      />
      {toolMode === 'analysis' &&
        <WinDrawLossIndicator isMobile={props.isMobile} boardFlipped={boardFlipped} wdl={white_wdl} />
      }
      <div style={{
        //flex: props.isMobile ? undefined : 1,
        height: props.isMobile ? undefined : BOARD_MAX_SIZE,
        minHeight: props.isMobile ? undefined : BOARD_MAX_SIZE,
        width: props.isMobile ? '100%' : 400,
      }}>
        <div style={{
          height: '100%',
          maxWidth: 400,
          maxHeight: BOARD_MAX_SIZE,
          border: '1px solid #222',
          backgroundColor: '#33333f',
          padding: 10,
          boxSizing: 'border-box',
          margin: props.isMobile ? 'auto' : undefined,
        }}>
          {/*<div style={{ fontWeight: 'bold', fontSize: '120%', marginBottom: 10 }}>Engine</div>*/}
          <div style={{ position: 'relative', width: '100%', height: 25 }}>
            {toolMode === 'analysis' ? <>
              <input type="checkbox" checked={runEngine} onChange={e => {
                setRunEngine(e.target.checked);
                if (engine !== null) {
                  engine.setRunEngine(e.target.checked);
                }
              }} /> Run engine {engineStatus}
              <div style={{ float: 'right' }} >
                <button onClick={() => {
                  if (props.isMobile) {
                    adjustToolMode('play100');
                  } else {
                    adjustToolMode('play1k');
                  }
                }}>Play mode</button>
              </div>
            </> : <div
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: (100 * Math.max(0.05, engine.thinkProgress)) + '%',
                height: 20,
                backgroundColor: isEnginesTurn ? '#944' : undefined,
                outline: isEnginesTurn ? '1px solid black' : undefined,
                whiteSpace: 'nowrap',
              }}
            >
              {isHistoryMove ? 'Game history' :
                (isEnginesTurn ? 'Thinking...' : 'Your move')}
            </div>}
          </div>

          <div style={{
            overflowY: 'scroll',
            height: BOARD_MAX_SIZE - 46,
          }}>
            {renderedMoveRows}
          </div>

          {/*engine !== null && <div>
            Evaluation: {engine.evaluation.toFixed(3)}<br/>
            Nodes: {engine.nodes}<br/>
            PV: {engine.pv.map((m: any) => m.from + ' ' + m.to).join(' ')}
        </div>*/}

        </div>
      </div>
      <div style={{ flex: 1 }} />
    </div>

    <div style={{ width: '100%', marginTop, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      {/*<input type="text" value={duckFen} onChange={e => setDuckFen(e.target.value)} style={{
        //width: 400,
        width: '96%',
        maxWidth: 400,
        backgroundColor: '#445',
        color: '#eee',
      }} />*/}
      <div style={{
        width: 400,
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'space-around',
      }}>
        {/*<label>Model: <select
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
        >
          <option value="medium-r16-s223-128x10">Medium</option>
          <option value="large-r16-s200-256x20">Large</option>
        </select></label>*/}

        <label>Mode: <select
          value={toolMode}
          onChange={(e) => adjustToolMode(e.target.value as ToolMode)}
        >
          <option value="analysis">Analysis</option>
          <option value="play100">Play (100 steps)</option>
          <option value="play1k">Play (1k steps)</option>
          <option value="play3k">Play (3k steps)</option>
        </select></label>

        <label>Model: <select
          value={modelName}
          onChange={(e) => adjustModelName(e.target.value as ModelName)}
        >
          <option value="medium-001-128x10">Medium</option>
          <option value="large-001-256x20">Large</option>
        </select></label>

        <button onClick={() => {
          if (engine !== null) {
            globalBoardFlipped = !globalBoardFlipped;
            setBoardFlipped(globalBoardFlipped);
            setNodeContextMenu(null);
          }
        }}>Flip board</button>

        <button onClick={() => {
          if (engine !== null) {
            engine.setPgn4(pgn);
            setNodeContextMenu(null);
          }
        }}>Load PGN</button>
      </div>
      <textarea
        value={pgn}
        onChange={e => setPgn(e.target.value)}
        style={{
          marginTop: 5,
          //width: 400,
          width: '96%',
          maxWidth: 400,
          height: 100,
          backgroundColor: '#445',
          color: '#eee',
        }}
        placeholder="Paste chess.com PGN here..."
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            e.stopPropagation();
            if (engine !== null) {
              engine.setPgn4(pgn);
            }
          }
        }}
      />
    </div>

    <div style={{ marginTop, textAlign: 'center', width: '95vw' }}>
      Created by Peter Schmidt-Nielsen
      (<a href="https://twitter.com/ptrschmdtnlsn">Twitter</a>, <a href="https://peter.website">Website</a>)<br/>
      Engine + web interface: <a href="https://github.com/petersn/duckchess">github.com/petersn/duckchess</a><br/>
      <span style={{ fontSize: '50%', opacity: 0.5 }}>{GUI_VERSION}-{get_wasm_version()} Piece SVGs: Cburnett (CC BY-SA 3), Duck SVG + all code: my creation (CC0)</span>
      <br/>

      <Link to={'/limitations'}>
        Click here for more info on limitations of the engine
      </Link>
      <br/>
      <Link to={'/settings'}>
        Click here for advanced settings
      </Link>
    </div>
  </>;
}

function InfoPage(props: { isMobile: boolean }) {
  return (
    <>Hello</>
  );
}

const globalModelLoadProgress: { [key: string]: number } = {};

function App() {
  const [engine, setEngine] = React.useState<DuckChessEngine | null>(null);
  const [width, setWidth] = React.useState<number>(window.innerWidth);

  const setForceUpdateCounter = React.useState(0)[1];
  const forceUpdate = () => {
    setForceUpdateCounter(Math.random());
  };

  React.useEffect(() => {
    createDuckChessEngine(
      (modelName: ModelName, progress: number) => {
        globalModelLoadProgress[modelName] = progress;
        forceUpdate();
      },
      forceUpdate,
      getDefaultModelName(),
    ).then(setEngine);
  }, []);

  React.useEffect(() => {
    console.log('Initializing worker...');
    /*
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
    */
  }, []);

  React.useEffect(() => {
    function handleWindowSizeChange() {
      setWidth(window.innerWidth);
    }
    window.addEventListener('resize', handleWindowSizeChange);
    return () => window.removeEventListener('resize', handleWindowSizeChange);
  }, []);

  const isMobile = width < 768;

  function navigation(element: React.ReactNode): React.ReactNode {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}>
        {/*<TopBar screenWidth={width} isMobile={isMobile} />*/}
        {element}
      </div>
    );
  }

  if (engine === null)
    return <div>Loading...</div>;

  return (
    <Router basename={process.env.PUBLIC_URL}>
      <Routes>
        <Route path="/analysis" element={navigation(<AnalysisPage isMobile={isMobile} engine={engine} />)} />
        <Route path="/benchmark" element={navigation(<BenchmarkApp isMobile={isMobile} engine={engine} />)} />
        <Route path="/info" element={navigation(<InfoPage isMobile={isMobile} />)} />
        <Route path="/limitations" element={navigation(<LimitationsPage isMobile={isMobile} />)} />
        <Route path="/settings" element={navigation(<SettingsPage isMobile={isMobile} engine={engine} />)} />
        <Route path="/" element={navigation(<Navigate to={"/analysis"} replace />)} />
      </Routes>
    </Router>
  );
}

export default App;
