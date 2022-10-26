import sys
import random
import json

import engine

games = []
for path in sys.argv[1:]:
    with open(path) as f:
        for line in f:
            games.append(json.loads(line))
print("Games:", len(games))

game = random.choice(games)

def render_state(state):
    piece_kinds = ["pawns", "knights", "bishops", "rooks", "queens", "kings"]
    board = [["."] * 8 for _ in range(8)]
    for color in (0, 1):
        for piece_index, piece in enumerate(piece_kinds):
            for rank in range(8):
                byte = state[piece][color][rank]
                for i in range(8):
                    present = (byte >> i) & 1
                    if present:
                        c = "PNBRQK"[piece_index]
                        if color == 1:
                            c = c.lower()
                            c = "\x1b[91m" + c + "\x1b[0m"
                        else:
                            c = "\x1b[92m" + c + "\x1b[0m"
                        board[rank][i] = c
    for rank in range(8):
        byte = state["ducks"][rank]
        for i in range(8):
            present = (byte >> i) & 1
            if present:
                board[rank][i] = "#"
    print("\n".join(" ".join(row) for row in board[::-1]))

e = engine.Engine(0)
for move in game["moves"]:
    e.apply_move(json.dumps(move))
    state = json.loads(e.get_state())
    render_state(state)
    input("> ")
