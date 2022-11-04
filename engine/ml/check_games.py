import json
import sys
from tqdm import tqdm

import show_game
import engine

move_squares = [
    b + a
    for a in "12345678"
    for b in "abcdefgh"
]
print(move_squares)

all_games = []
for game_file in tqdm(sys.argv[1:]):
    with open(game_file) as f:
        for line in f:
            game = json.loads(line)
            version = game["version"]
            assert version == "mcts-1"
            all_games.append(game)

#for i, game in tqdm(enumerate(all_games)):
for i in [37151, 57647]:
    game = all_games[i]
    e = engine.Engine(0)
    for move_index, move in enumerate(game["moves"]):
        # Find all of the legal moves.
        legal_moves = e.get_moves()
        legal_moves = [json.loads(m) for m in legal_moves]
        if move not in legal_moves:
            print("Game", i, "has illegal move", move_squares[move["from"]] + move_squares[move["to"]])
            show_game.render_state(json.loads(e.get_state()))
            print(json.loads(e.get_state())["turn"])
            print(e.get_state())
            print(move_index, len(game["moves"]))
            #print("Legal moves:", legal_moves)
            #print("Game:", game)
            break
        r = e.apply_move(json.dumps(move))
        if r is not None:
            print("Game", i, "has badly-applied move", move_squares[move["from"]] + move_squares[move["to"]])
            show_game.render_state(json.loads(e.get_state()))
            print(json.loads(e.get_state())["turn"])
            #print("Bad game at index", i)
            #print(game)
            break
        assert r is None, f"Move {move} failed: {r}"
        r = e.sanity_check()
        assert r is None, f"Move {move} failed sanity check: {r}"
