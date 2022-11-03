import json
import sys
from tqdm import tqdm

import show_game
import engine

all_games = []
for game_file in tqdm(sys.argv[1:]):
    with open(game_file) as f:
        for line in f:
            game = json.loads(line)
            version = game["version"]
            assert version == "mcts-1"
            all_games.append(game)

for i, game in tqdm(enumerate(all_games)):
    e = engine.Engine(0)
    for move in game["moves"]:
        # Find all of the legal moves.
        legal_moves = e.get_moves()
        legal_moves = json.loads(legal_moves)
        assert move in legal_moves
        r = e.apply_move(json.dumps(move))
        if r is not None:
            print("Bad game at index", i)
            print(game)
            print(json.dumps)
        assert r is None, f"Move {move} failed: {r}"
        r = e.sanity_check()
        assert r is None, f"Move {move} failed sanity check: {r}"
