import json
import pprint
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

for i, game in enumerate(tqdm(all_games)):
#for i in [37151, 57647]:
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
            print("GAME LENGTH:", len(game["moves"]))
            exit()
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
    # Check the final state.
    final_state = json.loads(e.get_state())
    if "zobrist" in final_state:
        del final_state["zobrist"]
    if "zobrist" in game["final_state"]:
        del game["final_state"]["zobrist"]
    #final_state.pop("zobrist")
    #final_state.pop("moveHistory")
    #game["finalState"].pop("zobrist")
    #game["final_state"].pop("moveHistory")
    if final_state != game["final_state"]:
        print("Game", i, "has bad final state")
        print("Expected:")
        pprint.pprint(game["final_state"])
        print("Got:")
        pprint.pprint(final_state)
        print("Moves:", game["moves"])
        print("Turn:", final_state["turn"])
        show_game.render_state(final_state)
        #exit()
