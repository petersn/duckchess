import React from 'react';

export const BOARD_MAX_SIZE = 400;

export type PieceKind = 'wP' | 'wN' | 'wB' | 'wR' | 'wQ' | 'wK' | 'bP' | 'bN' | 'bB' | 'bR' | 'bQ' | 'bK' | 'duck' | null;

export interface Move {
  from: number;
  to: number;
}

function getChessPieceImagePath(piece: PieceKind): string {
  if (piece === null)
    throw new Error('Cannot get image path for null piece');
  const pieceFileMapping = {
    'bP': 'black_pawn.svg',   'wP': 'white_pawn.svg',
    'bR': 'black_rook.svg',   'wR': 'white_rook.svg',
    'bN': 'black_knight.svg', 'wN': 'white_knight.svg',
    'bB': 'black_bishop.svg', 'wB': 'white_bishop.svg',
    'bQ': 'black_queen.svg',  'wQ': 'white_queen.svg',
    'bK': 'black_king.svg',   'wK': 'white_king.svg',
    'duck':   'duck.svg',
  };
  const pieceFile = pieceFileMapping[piece];
  return `${process.env.PUBLIC_URL}/icons/${pieceFile}`;
}

export function ChessPiece(props: {
  hightlightDuck: boolean;
  piece: PieceKind;
  style?: React.CSSProperties;
}) {
  if (props.piece === null)
    return null;
  const pieceFileMapping = {
    'bP': 'black_pawn.svg',   'wP': 'white_pawn.svg',
    'bR': 'black_rook.svg',   'wR': 'white_rook.svg',
    'bN': 'black_knight.svg', 'wN': 'white_knight.svg',
    'bB': 'black_bishop.svg', 'wB': 'white_bishop.svg',
    'bQ': 'black_queen.svg',  'wQ': 'white_queen.svg',
    'bK': 'black_king.svg',   'wK': 'white_king.svg',
    'duck':   'duck.svg',
  };
  const pieceFile = pieceFileMapping[props.piece];
  return <img
    style={{
      width: '100%',
      height: '100%',
      objectFit: 'contain',
      userSelect: 'none',
      ...props.style,
    }}
    src={`${process.env.PUBLIC_URL}/icons/${pieceFile}`}
    draggable={false}
  />;
}

export interface ChessBoardProps {
  isMobile: boolean;
  isInitialDuckPlacementMove: boolean;
  highlightDuck: boolean;
  boardFlipped: boolean;
  board: PieceKind[][];
  pastOccurrences: number;
  moveHistory: (Move | undefined)[];
  isDuckMove: boolean;
  boardHash: number;
  legalMoves: Move[];
  hiddenLegalMoves: Move[];
  topMoves: [Move, number][];
  onMove: (move: Move, isHidden: boolean) => void;
  style?: React.CSSProperties;
}

interface ChessBoardState {
  selectedSquare: [number, number] | null;
  clickExempt: [number, number] | null;
  draggingPiece: PieceKind;
  skipSquare: [number, number];

  userDrawnArrows: [number, number, number, number][];
  userHighlightedSquares: [number, number][];
  arrowStart: [number, number] | null;
  arrowHover: [number, number] | null;
}

export class ChessBoard extends React.Component<ChessBoardProps, ChessBoardState> {
  rafHandle: number = -1;
  lastCoords: [number, number] = [0, 0];
  dragState: {
    offsetX: number;
    offsetY: number;
  } | null = null;
  draggableDivRef = React.createRef<HTMLDivElement>();

  constructor(props: ChessBoardProps) {
    super(props);
    this.state = {
      selectedSquare: null,
      clickExempt: null,
      draggingPiece: null,
      skipSquare: [-1, -1],
      userDrawnArrows: [],
      userHighlightedSquares: [],
      arrowStart: null,
      arrowHover: null,
    };
  }

  rafLoop = (time: number) => {
    this.rafHandle = window.requestAnimationFrame(this.rafLoop);
  }

  componentDidMount() {
    //this.rafHandle = window.requestAnimationFrame(this.rafLoop);
    document.addEventListener('mousemove', this.onMouseMove);
    document.addEventListener('click', this.onGlobalClick);
    document.addEventListener('mouseup', this.onGlobalMouseUp);
  }

  componentWillUnmount() {
    window.cancelAnimationFrame(this.rafHandle);
    document.removeEventListener('mousemove', this.onMouseMove);
    document.removeEventListener('click', this.onGlobalClick);
    document.removeEventListener('mouseup', this.onGlobalMouseUp);
  }

  onMouseMove = (event: MouseEvent) => {
    this.lastCoords = [event.clientX, event.clientY];
    this.applyCoords();
  }

  applyCoords() {
    if (this.dragState !== null && this.draggableDivRef.current !== null) {
      this.draggableDivRef.current.style.left = this.lastCoords[0] + 'px';
      this.draggableDivRef.current.style.top = this.lastCoords[1] + 'px';
    }
  }

  onGlobalClick = (event: MouseEvent) => {
    if (this.dragState !== null) {
      this.dragState = null;
      this.setState({ draggingPiece: null, skipSquare: [-1, -1] });
    }
  }

  onGlobalMouseUp = (event: MouseEvent) => {
    // If it's a right mouse up, cancel the arrow.
    if (event.button === 2) {
      this.setState({ arrowStart: null, arrowHover: null });
    }
  }

  clearAllAnnotations() {
    this.setState({
      selectedSquare: null,
      userDrawnArrows: [],
      userHighlightedSquares: [],
    });
  }

  // If the board prop changes, clear all annotations.
  componentDidUpdate(prevProps: ChessBoardProps) {
    if (prevProps.boardHash !== this.props.boardHash) {
      this.clearAllAnnotations();
    }
  }

  // Soft clicks don't ever select a square, but they can still complete a move.
  onClick(x: number, y: number, soft?: boolean) {
    const clickSquare = x + (7 - y) * 8;
    if (this.state.selectedSquare === null) {
      if (soft)
        return;
      const isHidden = this.props.hiddenLegalMoves.some(m => m.from === clickSquare && m.to === clickSquare);
      // Check if this is the initial duck placement.
      const m = [...this.props.legalMoves, ...this.props.hiddenLegalMoves].find(m => m.from === clickSquare && m.to === clickSquare);
      if (m) {
        //engineWorker.applyMove(m);
        //setSelectedSquare(null);
        //setForceUpdateCounter(forceUpdateCounter + 1);
        this.props.onMove(m, isHidden);
        this.setState({ selectedSquare: null });
      } else {
        this.setState({ selectedSquare: [x, y] });
      }
    } else {
      if (this.tryToCompleteMove(x, y))
        return;
      this.setState({ selectedSquare: null });
      // let [fromX, fromY] = this.state.selectedSquare;
      // fromY = 7 - fromY;
      // const encodedFrom = 8 * fromY + fromX;
      // const encodedTo = 8 * (7 - y) + x;
      // const isHidden = this.props.hiddenLegalMoves.some(m => m.from === encodedFrom && m.to === encodedTo);
      // // Find the first move that matches the selected square and the clicked square.
      // const m = [...this.props.legalMoves, ...this.props.hiddenLegalMoves].find((m: any) => m.from === encodedFrom && m.to === encodedTo);
      // if (m) {
      //   this.props.onMove(m, isHidden);
      //   //engineWorker.applyMove(m);
      //   //setForceUpdateCounter(forceUpdateCounter + 1);
      //   this.setState({ selectedSquare: null });
      // } else {
      //   this.setState({ selectedSquare: [x, y] });
      // }
      //setSelectedSquare(null);
    }
  }

  tryToCompleteMove(x: number, y: number): boolean {
    if (this.state.selectedSquare === null)
      return false;
    let [fromX, fromY] = this.state.selectedSquare;
    fromY = 7 - fromY;
    const encodedFrom = 8 * fromY + fromX;
    const encodedTo = 8 * (7 - y) + x;
    const isHidden = this.props.hiddenLegalMoves.some(m => m.from === encodedFrom && m.to === encodedTo);
    // Find the first move that matches the selected square and the clicked square.
    const m = [...this.props.legalMoves, ...this.props.hiddenLegalMoves].find((m: any) => m.from === encodedFrom && m.to === encodedTo);
    if (m) {
      this.props.onMove(m, isHidden);
      //engineWorker.applyMove(m);
      //setForceUpdateCounter(forceUpdateCounter + 1);
      this.setState({ selectedSquare: null });
      return true;
    }
    return false;
  }

  beginArrow(x: number, y: number) {
    this.setState({ arrowStart: [x, y], selectedSquare: null });
  }

  updateArrow(x: number, y: number) {
    this.setState({ arrowHover: [x, y] });
  }

  endArrow(x: number, y: number) {
    if (this.state.arrowStart) {
      const newArrows = this.state.userDrawnArrows.slice();
      // If the arrow starts and ends in the same square, it's a highlight.
      if (this.state.arrowStart[0] === x && this.state.arrowStart[1] === y) {
        this.highlightSquare(x, y);
        this.setState({ arrowStart: null });
        return;
      }
      // Check if we already have this exact arrow.
      const [sx, sy] = this.state.arrowStart;
      const index = newArrows.findIndex(a => a[0] === sx && a[1] === sy && a[2] === x && a[3] === y);
      if (index !== -1) {
        console.log('removing arrow');
        newArrows.splice(index, 1);
      } else {
        newArrows.push([...this.state.arrowStart, x, y]);
      }
      this.setState({ userDrawnArrows: newArrows, arrowStart: null });
    }
  }

  highlightSquare(x: number, y: number) {
    const newHighlightedSquares = this.state.userHighlightedSquares.slice();
    // Toggle the square.
    const index = newHighlightedSquares.findIndex(([hx, hy]) => hx === x && hy === y);
    if (index !== -1) {
      newHighlightedSquares.splice(index, 1);
    } else {
      newHighlightedSquares.push([x, y]);
    }
    this.setState({ userHighlightedSquares: newHighlightedSquares });
  }

  render() {
    // This constant is the size of one square in the SVG.
    const SQ = 50;
    const svgElements = [];

    const moveHistoryHighlight: number[] = [];
    // If we're on a duck move then highlight just the opponent's last non-duckmove.
    let historyMoveIndex = this.props.isDuckMove ? 0 : 1;
    const move = this.props.moveHistory[historyMoveIndex];
    if (move !== undefined)
      moveHistoryHighlight.push(move.from, move.to);

    // Draw the backdrop of squares.
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const isSelected = this.state.selectedSquare && this.state.selectedSquare[0] === x && this.state.selectedSquare[1] === y;
        let backgroundColor = (x + y) % 2 === 0 ? '#eca' : '#b97';
        let foregroundColor = (x + y) % 2 === 0 ? '#b97' : '#eca';
        //if (state.highlight[7 - y] & (1 << x))
        //  backgroundColor = (x + y) % 2 === 0 ? '#dd9' : '#aa6';
        if (moveHistoryHighlight.includes(x + (7 - y) * 8)) {
          backgroundColor = (x + y) % 2 === 0 ? '#efa' : '#bb7';
        }
        if (this.state.userHighlightedSquares.some(([hx, hy]) => hx === x && hy === y)) {
          backgroundColor = '#f77';
        }
        if (isSelected) {
          backgroundColor = '#7f7';
        }
        let fileLabel: React.ReactNode = null;
        let rankLabel: React.ReactNode = null;
        if (y == 7) {
          fileLabel = <text
            key={`file-label-${x}`}
            x={SQ * x + 0.09 * SQ}
            y={SQ * y + 0.91 * SQ}
            textAnchor="middle"
            dominantBaseline="middle"
            style={{ fontSize: 10, fontWeight: 'bold', userSelect: 'none', pointerEvents: 'none' }}
            fill={foregroundColor}
          >
            {String.fromCharCode('a'.charCodeAt(0) + x)}
          </text>;
        }
        if (x == 7) {
          rankLabel = <text
            key={`rank-label-${y}`}
            x={SQ * x + 0.91 * SQ}
            y={SQ * y + 0.14 * SQ}
            textAnchor="middle"
            dominantBaseline="middle"
            style={{ fontSize: 10, fontWeight: 'bold', userSelect: 'none', pointerEvents: 'none' }}
            fill={foregroundColor}
          >
            {8 - y}
          </text>;
        }
        const maybeFlippedX = this.props.boardFlipped ? 7 - x : x;
        const maybeFlippedY = this.props.boardFlipped ? 7 - y : y;
        svgElements.push(
          <rect
            key={x + y * 8}
            x={maybeFlippedX * SQ}
            y={maybeFlippedY * SQ}
            width={SQ}
            height={SQ}
            fill={backgroundColor}
            onClick={(e) => {
              // If this square is click exempt, skip it.
              if (
                this.state.clickExempt !== null &&
                this.state.clickExempt[0] === x &&
                this.state.clickExempt[1] === y
              )
                return;
              let soft = this.props.board[y][x] === null;
              // If this is the initial duck placement then it's not soft.
              if (this.props.isInitialDuckPlacementMove)
                soft = false;
              console.log('On click click')
              this.onClick(x, y, soft);
            }}
            onMouseDown={(e) => {
              if (e.button !== 0) {
                this.beginArrow(x, y);
                return;
              }
              // If the square is unoccupied, then we soft click it, preventing selection.
              const soft = this.props.board[y][x] === null;
              // Click this square, unless it's already selected.
              if (!this.state.selectedSquare || this.state.selectedSquare[0] !== x || this.state.selectedSquare[1] !== y) {
                console.log('Mouse down click');
                this.onClick(x, y, soft);
              }
              // Check if there's a piece here.
              if (this.props.board[y][x] === null)
                return;
              const hereSelected = (
                this.state.selectedSquare &&
                this.state.selectedSquare[0] === x &&
                this.state.selectedSquare[1] === y
              );
              this.setState({
                // Force selection to this square, if there's a piece here.
                selectedSquare: [x, y],
                // If you release on this square then it won't click.
                clickExempt: hereSelected ? null : [x, y],
              });
              //console.log('mouse down', x, y);
              // Begin dragging.
              const rect = e.currentTarget.getBoundingClientRect();
              this.dragState = {
                offsetX: (rect.left - e.clientX),
                offsetY: (rect.top - e.clientY),
              };
              this.setState({
                draggingPiece: this.props.board[y][x],
                skipSquare: [x, y],
              });
              this.applyCoords();
            }}
            onMouseUp={(e) => {
              if (e.button !== 0) {
                this.endArrow(x, y);
                return;
              }
              //console.log('mouse up', x, y);
              // Check for a drop.
              if (this.dragState) {
                this.tryToCompleteMove(x, y);
                //console.log('Mouse up click')
                //this.onClick(x, y, true);
              }
            }}
            onMouseEnter={(e) => {
              this.updateArrow(x, y);
            }}
            onContextMenu={(e) => {
              e.preventDefault();
            }}
          />,
          fileLabel,
          rankLabel,
        );
      }
    }

    // Add the pieces.
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        // If this piece is being dragged then don't draw it in the SVG.
        if (this.state.skipSquare[0] === x && this.state.skipSquare[1] === y)
          continue;
        const piece = this.props.board[y][x];
        if (piece === null)
          continue;
        const path = getChessPieceImagePath(piece);
        const maybeFlippedX = this.props.boardFlipped ? 7 - x : x;
        const maybeFlippedY = this.props.boardFlipped ? 7 - y : y;
        if (piece === 'duck' && this.props.highlightDuck) {
          svgElements.push(
            <circle
              style={{ userSelect: 'none', pointerEvents: 'none' }}
              key={x + y * 8 + 64 + 64}
              cx={maybeFlippedX * SQ + 0.5 * SQ}
              cy={maybeFlippedY * SQ + 0.5 * SQ}
              r={0.5 * SQ}
              fill="yellow"
              filter="url(#glow)"
            />
          );
        }
        svgElements.push(
          <image
            style={{ userSelect: 'none', pointerEvents: 'none' }}
            key={x + y * 8 + 64}
            x={maybeFlippedX * SQ}
            y={maybeFlippedY * SQ}
            width={SQ}
            height={SQ}
            href={path}
          />
        );
      }
    }

    // Add a circle for every legal move from the selected square.
    const circles: React.ReactNode[] = [];
    for (const move of [...this.props.legalMoves, ...this.props.hiddenLegalMoves]) {
      if (this.state.selectedSquare === null)
        continue;
      const [fromX, fromY] = this.state.selectedSquare;
      if (move.from !== fromX + (7 - fromY) * 8)
        continue;
      const toX = move.to % 8;
      const toY = 7 - Math.floor(move.to / 8);
      const mfToX = this.props.boardFlipped ? 7 - toX : toX;
      const mfToY = this.props.boardFlipped ? 7 - toY : toY;
      svgElements.push(<circle
        style={{ userSelect: 'none', pointerEvents: 'none' }}
        key={`circle-${move.to}`}
        cx={mfToX * 50 + 25} cy={mfToY * 50 + 25}
        r={10}
        fill="rgba(0, 255, 0, 0.4)"
      />);
    }

    if (this.props.isInitialDuckPlacementMove) {
      for (const move of this.props.legalMoves) {
        const toX = move.to % 8;
        const toY = 7 - Math.floor(move.to / 8);
        const mfToX = this.props.boardFlipped ? 7 - toX : toX;
        const mfToY = this.props.boardFlipped ? 7 - toY : toY;
        svgElements.push(<circle
          style={{ userSelect: 'none', pointerEvents: 'none' }}
          key={`duck-circle-${move.to}`}
          cx={mfToX * 50 + 25} cy={mfToY * 50 + 25}
          r={10}
          fill="rgba(0, 255, 0, 0.4)"
        />);
      }
    }

    function addArrow(
      key: string,
      fromX: number,
      toX: number,
      fromY: number,
      toY: number,
      colorPrefix: string,
      weight: number,
    ) {
      let dx = toX - fromX;
      let dy = toY - fromY;
      const length = 1e-6 + Math.sqrt(dx * dx + dy * dy);
      dx /= length;
      dy /= length;
      const endX = toX * 50 + 25 - 10 * dx;
      const endY = toY * 50 + 25 - 10 * dy;

      const sizing = 1.5 * Math.max(0.5, Math.pow(weight, 0.5));
      const opacity = 0.5 * (0.25 + 0.75 * weight);
      const w = 4 * sizing;
      const headW = 7 * sizing;
      const headL = 18 * sizing;
      let d = `M ${fromX * 50 + 25 - w * dy} ${fromY * 50 + 25 + w * dx} L ${endX - w * dy} ${endY + w * dx}`;
      d += ` L ${endX + (w + headW) * dy} ${endY - (w + headW) * dx} L ${endX + headL * dx} ${endY + headL * dy}`;
      d += ` L ${endX - (w + headW) * dy} ${endY + (w + headW) * dx} L ${endX + w * dy} ${endY - w * dx}`;
      d += ` L ${fromX * 50 + 25 + w * dy} ${fromY * 50 + 25 - w * dx} Z`;
      const arrow = <path
        style={{ userSelect: 'none', pointerEvents: 'none' }}
        key={`arrow-${key}-${fromX}-${fromY}-${toX}-${toY}`}
        d={d}
        //stroke="rgba(0, 0, 255, 0.35)"
        strokeWidth="5"
        fill={colorPrefix + opacity + ')'}
      />;
      svgElements.push(arrow);
    }

    // Add arrows for highlit moves.
    let isNumberOneMove = true;
    for (let [move, weight] of this.props.topMoves) {
      if (isNumberOneMove)
        weight = Math.min(1, weight + 0.15);
      else
        weight = Math.max(0, weight - 0.1);
      isNumberOneMove = false;
      //if (!move)
      //  continue;
      const fromX = move.from % 8;
      const fromY = 7 - Math.floor(move.from / 8);
      const toX = move.to % 8;
      const toY = 7 - Math.floor(move.to / 8);
      const mfFromX = this.props.boardFlipped ? 7 - fromX : fromX;
      const mfFromY = this.props.boardFlipped ? 7 - fromY : fromY;
      const mfToX = this.props.boardFlipped ? 7 - toX : toX;
      const mfToY = this.props.boardFlipped ? 7 - toY : toY;
      if (move.from === 64 || move.from === move.to) {
        const arrow = <circle
          style={{ userSelect: 'none', pointerEvents: 'none' }}
          key={`arrow-${move.from}-${move.to}`}
          cx={mfToX * 50 + 25} cy={mfToY * 50 + 25}
          r={13 * (1 + 0.5 * weight)}
          //stroke="rgba(0, 0, 255, 0.35)"
          strokeWidth="5"
          fill="rgba(0, 0, 255, 0.35)"
        />;
        svgElements.push(arrow);
        continue;
      }
      addArrow('move', mfFromX, mfToX, mfFromY, mfToY, 'rgba(0, 0, 255,', weight);
    }

    // Show all of the existing arrows.
    for (const arrow of this.state.userDrawnArrows) {
      const [fromX, fromY, toX, toY] = arrow;
      const mfFromX = this.props.boardFlipped ? 7 - fromX : fromX;
      const mfFromY = this.props.boardFlipped ? 7 - fromY : fromY;
      const mfToX = this.props.boardFlipped ? 7 - toX : toX;
      const mfToY = this.props.boardFlipped ? 7 - toY : toY;
      addArrow('', mfFromX, mfToX, mfFromY, mfToY, 'rgba(255, 0, 0,', 0.7);
    }

    // Show the current arrow that's being drawn.
    if (this.state.arrowStart && this.state.arrowHover) {
      const fromX = this.state.arrowStart[0];
      const fromY = this.state.arrowStart[1];
      const toX = this.state.arrowHover[0];
      const toY = this.state.arrowHover[1];
      const mfFromX = this.props.boardFlipped ? 7 - fromX : fromX;
      const mfFromY = this.props.boardFlipped ? 7 - fromY : fromY;
      const mfToX = this.props.boardFlipped ? 7 - toX : toX;
      const mfToY = this.props.boardFlipped ? 7 - toY : toY;
      addArrow('hover', mfFromX, mfToX, mfFromY, mfToY, 'rgba(255, 0, 0,', 0.7);
    }


    /*
    const pieces: React.ReactNode[] = [];
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const piece = this.props.board[y][x];
        if (piece === null)
          continue;
        pieces.push(<ChessPiece
          key={y * 8 + x}
          piece={piece}
          style={{
            position: 'absolute',
            left: x * scale,
            top: y * scale,
            width: scale,
            height: scale,
            userSelect: 'none',
            pointerEvents: 'none',
            zIndex: 2,
          }}
        />);
      }
    }
    */

    if (this.props.pastOccurrences > 0) {
      let text = 'Somehow more than three repetitions??';
      if (this.props.pastOccurrences === 1)
        text = 'Twofold';
      else if (this.props.pastOccurrences === 2)
        text = 'Threefold (draw)';
      svgElements.push(
        <text
          key="repetitions"
          x="0"
          y="0"
          textAnchor="left"
          alignmentBaseline="hanging"
          fill="yellow"
          style={{
            fontSize: 20,
            textShadow: '-1px 0 2px black, 0 1px 2px black, 1px 0 2px black, 0 -1px 2px black',
            pointerEvents: 'none',
            userSelect: 'none',
          }}
        >
          {text}
        </text>
      );
    }

    return <>
      <svg
        viewBox="0 0 400 400"
        style={{
          width: '100%',
          height: '100%',
          maxWidth: BOARD_MAX_SIZE,
          maxHeight: BOARD_MAX_SIZE,
          ...this.props.style,
        }}
        onMouseLeave={() => {
          this.setState({ arrowHover: null });
        }}
      >
        <filter id="glow">
          <feGaussianBlur in="SourceGraphic" stdDeviation="4" />
        </filter>
        {svgElements}
      </svg>

      <div
        ref={this.draggableDivRef}
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          transform: `translate(-50%, -50%)`,
          pointerEvents: 'none',
          zIndex: 50,
        }}
      >
        {this.state.draggingPiece !== null && <ChessPiece
          hightlightDuck={false}
          piece={this.state.draggingPiece}
        />}
      </div>
    </>;

    /*
    return (
      <div style={{
        margin: 10,
        position: 'relative',
        width: 8 * scale,
        height: 8 * scale,
        minWidth: 8 * scale,
        minHeight: 8 * scale,
      }}>
        <svg
          viewBox="0 0 400 400"
          style={{ width: 600, height: 600, position: 'absolute', zIndex: 1, pointerEvents: 'none' }}
        >

        </svg>

        <div style={{ position: 'absolute' }}>
          {pieces}

          <table style={{ borderCollapse: 'collapse', border: '1px solid #eee' }}>
            <tbody>
              {this.props.board.map((row, y) => (
                <tr key={y}>
                  {row.map((piece, x) => {
                    const isSelected = this.state.selectedSquare && this.state.selectedSquare[0] === x && this.state.selectedSquare[1] === y;
                    let backgroundColor = (x + y) % 2 === 0 ? '#eca' : '#b97';
                    let foregroundColor = (x + y) % 2 === 0 ? '#b97' : '#eca';
                    //if (state.highlight[7 - y] & (1 << x))
                    //  backgroundColor = (x + y) % 2 === 0 ? '#dd9' : '#aa6';
                    if (isSelected) {
                      backgroundColor = '#7f7';
                    }
                    let fileLabel: React.ReactNode = null;
                    let rankLabel: React.ReactNode = null;
                    if (y == 7) {
                      fileLabel = <div
                        style={{
                          position: 'absolute',
                          left: 3,
                          bottom: 1,
                          color: foregroundColor,
                          fontSize: 15,
                          fontWeight: 'bold',
                          userSelect: 'none',
                          pointerEvents: 'none',
                        }}
                      >
                        {String.fromCharCode('a'.charCodeAt(0) + x)}
                      </div>
                    }
                    if (x == 7) {
                      rankLabel = <div
                        style={{
                          position: 'absolute',
                          right: 1,
                          top: 3,
                          color: foregroundColor,
                          fontSize: 15,
                          fontWeight: 'bold',
                          userSelect: 'none',
                          pointerEvents: 'none',
                        }}
                      >
                        {8 - y}
                      </div>
                    }
                    //const isSelected = selectedSquare !== null && selectedSquare[0] === x && selectedSquare[1] === y;
                    return <td key={x} style={{ margin: 0, padding: 0 }}>
                      <div
                        style={{
                          position: 'relative',
                          width: scale,
                          maxWidth: scale,
                          height: scale,
                          maxHeight: scale,
                          backgroundColor,
                          textAlign: 'center',
                        }}
                        onClick={() => this.onClick(x, y)}
                      >
                        {fileLabel}
                        {rankLabel}
                      </div>
                    </td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
    */
  }
}
