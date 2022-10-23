import json
import glob
import numpy as np
from tqdm import tqdm

# Load up our duck chess engine
import engine

channel_count = engine.channel_count()

all_games = []
for path in glob.glob("games/games-*.json"):
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

print("Total games:", len(all_games))
print("Total moves:", total_moves)

input_array = np.zeros((total_moves, channel_count, 8, 8), dtype=np.int8)
policy_array = np.zeros((total_moves,), dtype=np.int32)
value_array = np.zeros((total_moves,), dtype=np.int8)
b = input_array.nbytes + policy_array.nbytes + value_array.nbytes
print("Total storage:", b / 1024 / 1024, "MiB")

entry = 0
for game in tqdm(all_games):
    version = game.get("version", 1)
    value = {"1-0": +1, "0-1": -1, None: 0}[game["outcome"]]
    e = engine.Engine(0)
    for i, move in enumerate(game["moves"]):
        move_str = json.dumps(move)
        if version == 3 and game["was_rand"][i] is True:
            e.apply_move(move_str)
            continue
        # Save a triple into our arrays.
        features = input_array[entry]
        e.get_state_into_array(features.nbytes, features.ctypes.data)
        policy_array[entry] = (move["from"] % 64)  * 64 + move["to"]
        #engine.encode_move(move_str, policy.nbytes, policy.ctypes.data)
        value_array[entry] = value
        entry += 1
        # Apply the move.
        e.apply_move(move_str)
    assert e.get_outcome() == game["outcome"]
assert entry == total_moves

print("Completed")
# Save the output as three numpy arrays packed into a single .npz file.
np.savez_compressed(
    "train.npz",
    input=input_array,
    policy=policy_array,
    value=value_array,
)
