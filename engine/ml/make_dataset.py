import json
import glob
import numpy as np
import random
from tqdm import tqdm

import show_game
import engine

# We only sparsely store the top positions in the policy for each move, to save space.
policy_truncation = 32

def process_game_paths(paths):
    all_games = []
    for path in tqdm(paths):
        with open(path) as f:
            for line in f:
                all_games.append(json.loads(line))

    total_moves = 0
    for game in all_games:
        if "version" not in game:
            continue
        version = game["version"]
        if version != 3:
            continue
        version = "pvs-1"
        if version == "pvs-1":
            total_moves += sum(not was_rand for was_rand in game["was_rand"])
        elif version == "mcts-1":
            total_moves += sum(td is not None for td in game["train_dists"])
        else:
            raise RuntimeError("Unknown game version: " + str(version))

    print("Total games:", len(all_games))
    print("Total moves:", total_moves)

    features_array = np.zeros((total_moves, engine.channel_count(), 8, 8), dtype=np.int8)
    policy_array = np.zeros((total_moves, 64, 64), dtype=np.float16)
    value_array = np.zeros((total_moves, 1), dtype=np.float32)
    byte_length = (
        features_array.nbytes
        + policy_array.nbytes
        + value_array.nbytes
    )
    print("Total storage:", byte_length / 1024 / 1024, "MiB")

    entry = 0
    for game in tqdm(all_games):
        if "version" not in game:
            continue
        version = game["version"]
        if version != 3:
            continue
        version = "pvs-1"
        value_for_white = {"1-0": +1, "0-1": -1, None: 0}[game["outcome"]]
        e = engine.Engine(0)

        for i, move in enumerate(game["moves"]):
            white_to_move = {"white": True, "black": False}[json.loads(e.get_state())["turn"]]
            this_move_flip = 0 if white_to_move else 56
            #state = json.loads(e.get_state())
            #show_game.render_state(state)
            #print(move, state["turn"], state["isDuckMove"])
            #print()

            # FIXME: Why is this necessary? Why are folks playing in terminal positions?
            if e.get_outcome() is not None:
                continue
            if move["from"] == 64:
                move["from"] = move["to"]
            move_str = json.dumps(move)
            careful_check = random.random() < 0.02
            if careful_check:
                all_moves = [
                    (m["from"], m["to"])
                    for m in map(json.loads, e.get_moves())
                ]
                this_move = (move["from"], move["to"])
                if this_move not in all_moves:
                    print(f"Index = {i} move {this_move} not in {all_moves}")
                    state = json.loads(e.get_state())
                    print(state)
                    show_game.render_state(state)
                    print(state["turn"])
                    raise RuntimeError

            # Don't train on random moves, or fast search moves.
            if (version.startswith("pvs") and game["was_rand"][i]) or (version.startswith("mcts") and not game["full_search"][i]):
                #print("Applying move", move_str)
                r = e.apply_move(move_str)
                assert r is None, f"Index = {i} Move {move_str} failed: {r}"
                r = e.sanity_check()
                assert r is None, f"Index = {i} Move {move_str} failed sanity check: {r}"
                #state = json.loads(e.get_state())
                #show_game.render_state(state)
                #input("> ")
                continue
            # Save data into our arrays.
            features_slice = features_array[entry]
            e.get_state_into_array(features_slice.nbytes, features_slice.ctypes.data)

            if version == "pvs-1":
                policy_dist = [(move_str, 1.0)]
            else:
                policy_dist = game["train_dists"][i]
            assert abs(sum(p for _, p in policy_dist) - 1.0) < 1e-6
            policy_slice = policy_array[entry]
            for move_str, p in policy_dist:
                move = json.loads(move_str)
                policy_slice[move["from"] ^ this_move_flip][move["to"] ^ this_move_flip] = p
            #policy_dist.sort(key=lambda x: x[1], reverse=True)
            #for j, (m, prob) in enumerate(policy_dist[:policy_truncation]):
            #    policy_indices_array[entry, j] = engine.move_to_index(json.dumps(m))
            #    policy_probs_array[entry, j] = prob
            #policy_array[entry] = engine.move_to_index(move_str)
            #engine.encode_move(move_str, policy.nbytes, policy.ctypes.data)

            value_array[entry, 0] = value_for_white if white_to_move else -value_for_white
            entry += 1
            # Apply the move.
            #print("Applying move", move_str)
            r = e.apply_move(move_str)
            assert r is None, f"Index = {i} Move {move_str} failed: {r}"
            r = e.sanity_check()
            assert r is None, f"Index = {i} Move {move_str} failed sanity check: {r}"
            #state = json.loads(e.get_state())
            #show_game.render_state(state)
            #input("> ")
        #assert e.get_outcome() == game["outcome"]
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
