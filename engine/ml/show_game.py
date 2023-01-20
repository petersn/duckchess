import sys
import random
import json
import collections

import engine

#final_state = {'pawns': [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], 'knights': [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], 'bishops': [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], 'rooks': [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 32]], 'queens': [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], 'kings': [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2, 0]], 'ducks': [0, 0, 0, 0, 16, 0, 0, 0], 'enPassant': [0, 0, 0, 0, 0, 0, 0, 0], 'castlingRights': [{'kingSide': False, 'queenSide': False}, {'kingSide': False, 'queenSide': False}], 'turn': 'white', 'isDuckMove': True, 'moveHistory': [{'from': 41, 'to': 49}, {'from': 58, 'to': 36}, {'from': 57, 'to': 49}, {'from': 59, 'to': 58}], 'zobrist': 0, 'plies': 225}

move_squares = [
    b + a
    for a in "12345678"
    for b in "abcdefgh"
]

# This function pretty prints all of the differences between two JSON objects
# recursively, with the differences highlighted in red.
def diff_json(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in set(a.keys()) | set(b.keys()):
            if key not in a:
                print("\x1b[91m" + key + "\x1b[0m" + ": " + str(b[key]))
            elif key not in b:
                print("\x1b[91m" + key + "\x1b[0m" + ": " + str(a[key]))
            else:
                diff_json(a[key], b[key])
    elif isinstance(a, list) and isinstance(b, list):
        for i in range(max(len(a), len(b))):
            if i >= len(a):
                print("\x1b[91m" + str(i) + "\x1b[0m" + ": " + str(b[i]))
            elif i >= len(b):
                print("\x1b[91m" + str(i) + "\x1b[0m" + ": " + str(a[i]))
            else:
                diff_json(a[i], b[i])
    elif a != b:
        print("\x1b[91m" + str(a) + "\x1b[0m" + " != " + str(b))


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

if __name__ == "__main__":
    games = []
    for path in sys.argv[1:]:
        with open(path) as f:
            for line in f:
                if not line.strip(): continue
                games.append(json.loads(line))
    print("Games:", len(games))

    # # Find the game that matches our target final state.
    # for i, game in enumerate(games):
    #     if game["final_state"] == final_state:
    #         print("FOUND GAME", i)
    #         with open("/tmp/debug-game-step-235.json", "w") as f:
    #             json.dump(game, f, indent=2)
    #         break
    # else:
    #     print("Failed to find game")
    #     exit()

    game = random.choice(games)
    #game = games[21]
    #game = games[77]
    print("GAME LENGTH:", len(game["moves"]))
    print(game.keys())

    last_state = None

    repetition_counts = collections.Counter()

    e = engine.Engine(0, False)
    for i, move in enumerate(game["moves"]):
        #print(game["moves"][:i + 1])
        #print("vvvvvvvvvvvvvvvvvvvv")
        #print("ABOUT TO APPLY:", move_squares[move["from"]] + move_squares[move["to"]])
        #print("PRESTATE:", game["states"][i])
        #if game["states"][i] != last_state:
        #    print("\x1b[91mSTATE MISMATCH!!!!!!!!!!!!!\x1b[0m")
        #    diff_json(game["states"][i], last_state)
        #render_state(game["states"][i])
        #new_engine = engine.Engine(0)
        #moves = new_engine.get_moves()
        #print(" ".join(move_squares[m["from"]] + move_squares[m["to"]] for m in map(json.loads, moves)))
        #print("^^^^^^^^^^^^^^^^^^^^")
        ##new_engine.set_state(json.dumps(game["states"][i]))
        #
        r = e.apply_move(json.dumps(move))
        if r is not None:
            print("BAD:", r)
        d = json.loads(e.get_state())
        d.pop("moveHistory")
        d.pop("plies")
        key = str(d)
        repetition_counts[key] += 1
        print(repetition_counts[key])
        state = json.loads(e.get_state())
        #print("AFTER MOVE:")
        #print(state)
        render_state(state)
        #info_line = "[%3i] " % i + move_squares[move["from"]] + move_squares[move["to"]] + " value=%.2f" % game["root_values"][i]
        ##info_line = f" {game['full_search'][i]}"
        #if game["full_search"][i]:
        #    for move, score in sorted(game["train_dists"][i], key=lambda x: -x[1]):
        #        info_line += " \x1b[91m" + move_squares[move["from"]] + move_squares[move["to"]] + "\x1b[0m: %.0f%%" % round(100 * score)
        info_line = "x"
        if i >= 301 or True:
            input(info_line + " > ")
        else:
            print(info_line)
        print(e.sanity_check())
        last_state = state
