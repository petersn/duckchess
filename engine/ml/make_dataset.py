import json
import glob
import numpy as np
from tqdm import tqdm

# Load up our duck chess engine
import engine

def process_game_paths(paths):
    all_games = []
    for path in tqdm(paths[6:8]):
        with open(path) as f:
            for line in f:
                all_games.append(json.loads(line))

    total_moves = 0
    for game in all_games:
        version = game.get("version", 1)
        if version == 1:
            total_moves += len(game["moves"])
        elif version == 2:
            assert False
        elif version == 3:
            total_moves += sum(x is False for x in game["was_rand"])
        elif version == 101:
            # Version 101 is an MCTS game for RL, so we only take moves that got full searches.
            total_moves += sum(wl and mov is not None for wl, mov in zip(game["was_large"], game["train_moves"]))
        else:
            raise RuntimeError("Unknown game version: " + str(version))

    print("Total games:", len(all_games))
    print("Total moves:", total_moves)

    features_array = np.zeros((total_moves, engine.channel_count(), 8, 8), dtype=np.int8)
    policy_array = np.zeros((total_moves,), dtype=np.int64)
    value_array = np.zeros((total_moves, 1), dtype=np.float32)
    b = features_array.nbytes + policy_array.nbytes + value_array.nbytes
    print("Total storage:", b / 1024 / 1024, "MiB")

    entry = 0
    for game in tqdm(all_games):
        version = game.get("version", 1)
        value_for_white = {"1-0": +1, "0-1": -1, None: 0}[game["outcome"]]
        e = engine.Engine(0)

        moves_list = game["train_moves"] if "train_moves" in game else game["moves"]
        for i, move in enumerate(moves_list):
            if i >= len(game["moves"]):
                continue
            move_str = json.dumps(game["moves"][i])
            all_moves = [
                (m["from"], m["to"])
                for m in map(json.loads, e.get_moves())
            ]
            this_move = (move["from"], move["to"])
            assert this_move in all_moves, f"Index = {i} Move {this_move} not in {all_moves}"
            if (version == 3 and game["was_rand"][i] is True) or (version == 101 and not game["was_large"][i]):
                e.apply_move(move_str)
                continue
            # Save a triple into our arrays.
            features_slice = features_array[entry]
            e.get_state_into_array(features_slice.nbytes, features_slice.ctypes.data)
            policy_array[entry] = engine.move_to_index(move_str)
            #engine.encode_move(move_str, policy.nbytes, policy.ctypes.data)
            white_to_move = i % 4 < 2
            value_array[entry, 0] = value_for_white if white_to_move else -value_for_white
            entry += 1
            # Apply the move.
            e.apply_move(move_str)
        assert e.get_outcome() == game["outcome"]
    assert entry == total_moves

    return features_array, policy_array, value_array

if __name__ == "__main__":
    game_paths = "games/games-*.json"
    features_array, policy_array, value_array = process_game_paths(glob.glob(game_paths))
    # Save the output as three numpy arrays packed into a single .npz file.
    np.savez_compressed(
        "train.npz",
        features=features_array,
        policy=policy_array,
        value=value_array,
    )
    print("Done")
