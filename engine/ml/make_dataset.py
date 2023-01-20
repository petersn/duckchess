import os
import json
import glob
import numpy as np
import random
import multiprocessing
from dataclasses import dataclass
from tqdm import tqdm

import show_game
import engine

CHANNEL_COUNT = engine.channel_count()
POLICY_TRUNCATION = 32

@dataclass
class Dataset:
    game_offsets: np.ndarray
    features: np.ndarray
    policy_indices: np.ndarray
    policy_probs: np.ndarray
    value: np.ndarray
    wdl_index: np.ndarray
    mcts_root_value: np.ndarray

    def sanity_check(self):
        assert len(self.features) == len(self.policy_indices) == len(self.policy_probs) == len(self.value)

    def save(self, path):
        self.sanity_check()
        np.savez_compressed(
            path,
            game_count=len(self.game_offsets),
            move_count=len(self.features),
            game_offsets=self.game_offsets,
            features=self.features,
            policy_indices=self.policy_indices,
            policy_probs=self.policy_probs,
            value=self.value,
            wdl_index=self.wdl_index,
            mcts_root_value=self.mcts_root_value,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        ds = cls(
            game_offsets=data["game_offsets"],
            features=data["features"],
            policy_indices=data["policy_indices"],
            policy_probs=data["policy_probs"],
            value=data["value"],
            wdl_index=data["wdl_index"],
            mcts_root_value=data["mcts_root_value"],
        )
        ds.sanity_check()
        return ds

def process_game_path(path: str):
    cache_path = path + ".cache.npz"
    # If there's a newer cache file, skip this step.
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) > os.path.getmtime(path):
        return cache_path
    with open(path) as f:
        games = f.readlines()
    # Figure out the size of the array we're allocating, and where each game fits in.
    game_offsets = []
    total_moves = 0
    for game in games:
        game_offsets.append(total_moves)
        game = json.loads(game)
        version = game["version"]
        if version == "pvs-1":
            total_moves += sum(not was_rand for was_rand in game["was_rand"])
        elif version == "mcts-1":
            total_moves += sum(td is not None for td in game["train_dists"])
        else:
            raise RuntimeError("Unknown game version: " + str(version))
    print("Processing", path, "with", len(games), "games and", total_moves, "moves")
    features_array = np.zeros((total_moves, CHANNEL_COUNT, 8, 8), dtype=np.int8)
    policy_indices_array = np.ones((total_moves, POLICY_TRUNCATION), dtype=np.int16)
    policy_indices_array *= -1
    policy_probs_array = np.zeros((total_moves, POLICY_TRUNCATION), dtype=np.float32)
    value_array = np.zeros((total_moves, 1), dtype=np.float32)
    wdl_index_array = np.zeros((total_moves, 1), dtype=np.int8)
    mcts_root_value_array = np.zeros((total_moves, 1), dtype=np.float32)
    entry = 0
    for game_index, game in enumerate(games):
        game = json.loads(game)
        version = game["version"]
        value_for_white = {"1-0": +1, "0-1": -1, None: 0}[game["outcome"]]
        e = engine.Engine(0, False)
        for i, move in enumerate(game["moves"]):
            white_to_move = {"white": True, "black": False}[json.loads(e.get_state())["turn"]]
            this_move_flip = 0 if white_to_move else 56
            #state = json.loads(e.get_state())
            #show_game.render_state(state)
            #print(move, state["turn"], state["isDuckMove"])
            #print()

            # FIXME: Why is this necessary? Why are folks playing in terminal positions?
            if e.get_outcome() is not None:
                print("Special state:", e.get_outcome())
                print(i, len(game["moves"]))
                print("What???")
                print("PATH:", path, "GAME INDEX:", game_index)
                with open("/tmp/one-game.json", "w") as f:
                    json.dump(game, f)
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
                if r is not None:
                    print(f"Index = {i} Move {move_str} failed: {r}")
                assert r is None, f"Index = {i} Move {move_str} failed: {r}"
                r = e.sanity_check()
                if r is not None:
                    print(f"Index = {i} Sanity check failed: {r}")
                assert r is None, f"Index = {i} Move {move_str} failed sanity check: {r}"
                #state = json.loads(e.get_state())
                #show_game.render_state(state)
                #input("> ")
                continue
            # Save data into our arrays.
            features_slice = features_array[entry]
            e.get_state_into_array(features_slice.nbytes, features_slice.ctypes.data)

            if version == "pvs-1":
                policy_dist = [(json.loads(move_str), 1.0)]
            else:
                policy_dist = game["train_dists"][i]
            assert abs(sum(p for _, p in policy_dist) - 1.0) < 1e-6
            #policy_slice = policy_array[entry]
            #for move, p in policy_dist:
            #    #move = json.loads(move_str)
            #    policy_slice[move["from"] ^ this_move_flip][move["to"] ^ this_move_flip] = p
            policy_dist.sort(key=lambda x: x[1], reverse=True)
            for j, (m, prob) in enumerate(policy_dist[:POLICY_TRUNCATION]):
                # We have to take into the flip of the move.
                possibly_flipped_move = {"from": m["from"] ^ this_move_flip, "to": m["to"] ^ this_move_flip}
                policy_indices_array[entry, j] = engine.move_to_index(json.dumps(possibly_flipped_move))
                policy_probs_array[entry, j] = prob
            #policy_array[entry] = engine.move_to_index(move_str)
            #engine.encode_move(move_str, policy.nbytes, policy.ctypes.data)

            # The value array contains values from the side to move's perspective.
            value_array[entry, 0] = value_for_white if white_to_move else -value_for_white
            # The WDL array contains outcomes from the side to move's perspective.
            wdl_index_array[entry, 0] = (
                {"1-0": 0, "0-1": 2, None: 1}[game["outcome"]] if white_to_move else
                {"1-0": 2, "0-1": 0, None: 1}[game["outcome"]]
            )
            # The MCTS root value in the game JSON is from white's perspective, so we must flip it for black.
            # Also, it's from 0 (loss) to 1 (win) in the JSON, and we want -1 (loss) to +1 (win) in our array.
            mcts_root_value_array[entry, 0] = (2 * game["root_values"][i] - 1) * (
                +1 if white_to_move else -1
            )
            entry += 1
            # Apply the move.
            #print("Applying move", move_str)
            r = e.apply_move(move_str)
            if r is not None:
                print(f"Index = {i} Move {move_str} failed: {r}")
            assert r is None, f"Index = {i} Move {move_str} failed: {r}"
            r = e.sanity_check()
            if r is not None:
                print(f"Index = {i} Move {move_str} failed sanity check: {r}")
            assert r is None, f"Index = {i} Move {move_str} failed sanity check: {r}"
            #state = json.loads(e.get_state())
            #show_game.render_state(state)
            #input("> ")
    assert entry == total_moves, f"entry = {entry} total_moves = {total_moves}"
    dataset = Dataset(
        game_offsets=game_offsets,
        features=features_array,
        policy_indices=policy_indices_array,
        policy_probs=policy_probs_array,
        value=value_array,
        wdl_index=wdl_index_array,
        mcts_root_value=mcts_root_value_array,
    )
    dataset.save(cache_path)
    print("Saved dataset to", cache_path)
    return cache_path

pool = multiprocessing.Pool()

def collect_data(json_paths: list[str]) -> Dataset:
    numpy_paths = pool.map(process_game_path, json_paths)
    for path in numpy_paths:
        assert path.endswith(".cache.npz")
    total_games = 0
    total_moves = 0
    for path in numpy_paths:
        x = np.load(path)
        total_games += x["game_count"]
        total_moves += x["move_count"]
    # Make arrays to hold everything.
    game_offsets = np.zeros(total_games, dtype=np.int32)
    features_array = np.zeros((total_moves, CHANNEL_COUNT, 8, 8), dtype=np.int8)
    policy_indices_array = np.ones((total_moves, POLICY_TRUNCATION), dtype=np.int16)
    policy_indices_array *= -1
    policy_probs_array = np.zeros((total_moves, POLICY_TRUNCATION), dtype=np.float32)
    value_array = np.zeros((total_moves, 1), dtype=np.float32)
    wdl_index_array = np.zeros((total_moves, 1), dtype=np.int8)
    mcts_root_value_array = np.zeros((total_moves, 1), dtype=np.float32)
    # Now fill them in.
    entry = 0
    game = 0
    for path in tqdm(numpy_paths):
        x = np.load(path)
        game_count = x["game_count"]
        move_count = x["move_count"]
        game_offsets[game:game + game_count] = x["game_offsets"] + entry
        game += game_count
        features_array[entry:entry + move_count] = x["features"]
        policy_indices_array[entry:entry + move_count] = x["policy_indices"]
        policy_probs_array[entry:entry + move_count] = x["policy_probs"]
        value_array[entry:entry + move_count] = x["value"]
        wdl_index_array[entry:entry + move_count] = x["wdl_index"]
        mcts_root_value_array[entry:entry + move_count] = x["mcts_root_value"]
        entry += move_count
    assert entry == total_moves
    print(f"Total games = {total_games} total moves = {total_moves}")
    return Dataset(
        game_offsets=game_offsets,
        features=features_array,
        policy_indices=policy_indices_array,
        policy_probs=policy_probs_array,
        value=value_array,
        wdl_index=wdl_index_array,
        mcts_root_value=mcts_root_value_array,
    )

if __name__ == "__main__":
    #import sys, multiprocessing
    #pool = multiprocessing.Pool(30)
    #feature_counts = pool.map(process_game_path, sys.argv[1:])
    #print("Feature counts:", feature_counts)
    #print("Total features:", sum(feature_counts))

    import sys
    collect_data(sys.argv[1:])

    #game_paths = "games/games-*.json"
    #features_array, policy_indices_array, policy_probs_array, value_array = process_game_paths(glob.glob(game_paths))
    ## Save the output as three numpy arrays packed into a single .npz file.
    #np.savez_compressed(
    #    "train.npz",
    #    features=features_array,
    #    policy_indices=policy_indices_array,
    #    policy_probs=policy_probs_array,
    #    value=value_array,
    #)
    #print("Done")
