import React from 'react';

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

export function ChessPiece(props: { piece: PieceKind; style?: React.CSSProperties }) {
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
  board: PieceKind[][];
  legalMoves: Move[];
  hiddenLegalMoves: Move[];
  topMoves: Move[];
  onMove: (move: Move, isHidden: boolean) => void;
}

interface ChessBoardState {
  selectedSquare: [number, number] | null;
}

export class ChessBoard extends React.Component<ChessBoardProps, ChessBoardState> {
  constructor(props: ChessBoardProps) {
    super(props);
    this.state = {
      selectedSquare: null,
    };
  }

  onClick(x: number, y: number) {
    const clickSquare = x + (7 - y) * 8;
    if (this.state.selectedSquare === null) {
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
      let [fromX, fromY] = this.state.selectedSquare;
      fromY = 7 - fromY;
      y = 7 - y;
      const encodedFrom = 8 * fromY + fromX;
      const encodedTo = 8 * y + x;
      const isHidden = this.props.hiddenLegalMoves.some(m => m.from === encodedFrom && m.to === encodedTo);
      // Find the first move that matches the selected square and the clicked square.
      const m = [...this.props.legalMoves, ...this.props.hiddenLegalMoves].find((m: any) => m.from === encodedFrom && m.to === encodedTo);
      if (m) {
        this.props.onMove(m, isHidden);
        //engineWorker.applyMove(m);
        //setForceUpdateCounter(forceUpdateCounter + 1);
      }
      //setSelectedSquare(null);
      this.setState({ selectedSquare: null });
    }
  }

  render() {
    // This constant is the size of one square in the SVG.
    const SQ = 50;
    const svgElements = [];

    // Draw the backdrop of squares.
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
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
        svgElements.push(
          <rect
            key={x + y * 8}
            x={x * SQ}
            y={y * SQ}
            width={SQ}
            height={SQ}
            fill={backgroundColor}
            onClick={() => this.onClick(x, y)}
          />,
          fileLabel,
          rankLabel,
        );
      }
    }

    // Add the pieces.
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const piece = this.props.board[y][x];
        if (piece === null)
          continue;
        const path = getChessPieceImagePath(piece);
        svgElements.push(
          <image
            style={{ userSelect: 'none', pointerEvents: 'none' }}
            key={x + y * 8 + 64}
            x={x * SQ}
            y={y * SQ}
            width={SQ}
            height={SQ}
            href={path}
          />
        );
      }
    }

    // Add arrows for highlit moves.
    for (const move of this.props.topMoves) {
      //if (!move)
      //  continue;
      const fromX = move.from % 8;
      const fromY = 7 - Math.floor(move.from / 8);
      const toX = move.to % 8;
      const toY = 7 - Math.floor(move.to / 8);
      let dx = toX - fromX;
      let dy = toY - fromY;
      const length = 1e-6 + Math.sqrt(dx * dx + dy * dy);
      dx /= length;
      dy /= length;
      const endX = toX * 50 + 25 - 10 * dx;
      const endY = toY * 50 + 25 - 10 * dy;
      if (move.from === 64 || move.from === move.to) {
        const arrow = <circle
          style={{ userSelect: 'none', pointerEvents: 'none' }}
          key={`arrow-${move.from}-${move.to}`}
          cx={toX * 50 + 25} cy={toY * 50 + 25}
          r={10}
          stroke="red"
          strokeWidth="5"
          fill="red"
        />;
        svgElements.push(arrow);
        continue;
      }
      let d = `M ${fromX * 50 + 25} ${fromY * 50 + 25} L ${endX} ${endY}`;
      d += ` L ${endX + 5 * dy} ${endY - 5 * dx} L ${endX + 10 * dx} ${endY + 10 * dy} L ${endX - 5 * dy} ${endY + 5 * dx} L ${endX} ${endY} Z`;
      const arrow = <path
        style={{ userSelect: 'none', pointerEvents: 'none' }}
        key={`arrow-${move.from}-${move.to}`}
        d={d}
        stroke="red"
        strokeWidth="5"
        fill="red"
      />;
      svgElements.push(arrow);
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
      svgElements.push(<circle
        style={{ userSelect: 'none', pointerEvents: 'none' }}
        key={`circle-${move.to}`}
        cx={toX * 50 + 25} cy={toY * 50 + 25}
        r={10}
        fill="rgba(0, 255, 0, 0.4)"
      />);
    }

    return (
      <svg
        viewBox="0 0 400 400"
        style={{
          width: '100%',
          height: '100%',
          maxWidth: 600,
          maxHeight: 600,
        }}
      >
        {svgElements}
      </svg>
    );

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
