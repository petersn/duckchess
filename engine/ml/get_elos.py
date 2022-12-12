import re
import sys
import json
import subprocess
from tqdm import tqdm

def process_name(name):
    if name.startswith("./"):
        name = name[2:]
    if name.endswith("/"):
        name = name[:-1]
    name = name.replace("compute8.6.trt", "keras")
    return name

all_games = []
for path in tqdm(sys.argv[1:]):
    with open(path) as f:
        for line in f:
            all_games.append(json.loads(line))

print("Games:", len(all_games))

pgn = []
for game in all_games:
    #if "-1" in game["engine_white"] or "-1" in game["engine_black"]:
    #    continue
    assert game["version"].startswith("mcts-")
    result = game["outcome"]
    if result is None:
        continue
    assert result in ("1-0", "0-1", "1/2-1/2"), f"Bad result: {result}"
    pgn.append("""
[Event "mcts-compete"]
[Site "mcts-compete"]
[Date "????.??.??"]
[Round "1"]
[White "%s"]
[Black "%s"]
[Result "%s"]

1. %s
""" % (process_name(game["engine_white"]), process_name(game["engine_black"]), result, result))

print("Valid games:", len(pgn))

with open("/tmp/compete.pgn", "w") as f:
    f.write("".join(pgn))

output = subprocess.check_output(
    ["bayeselo"],
    input="readpgn /tmp/compete.pgn\nelo\nmm\nexactdist\nratings\n".encode("utf-8"),
).decode()

print(output.replace("ResultSet-EloRating>", "ResultSet-EloRating>\n"))

# Extract all of the elos.
ratings = {}
for line in output.split("\n"):
    if line.startswith(" "):
        _, name, rating, *_ = line.split()
        rating = int(rating)
        ratings[name] = rating
min_rating = min(ratings.values())
ratings = {name: rating - min_rating for name, rating in ratings.items()}
print("Ratings:", ratings)

def get_step(name):
    return int(re.search(r"step-(\d+)", name).group(1))

names = sorted(ratings.keys())
xs = [get_step(name) for name in names][-85:]
ys = [ratings[name] for name in names][-85:]
import matplotlib.pyplot as plt
plt.plot(xs, ys)
plt.scatter(xs, ys, s=6)
plt.title("Elo progression of DuckChessZero")
plt.ylabel("Elo rating")
plt.xlabel("Model number")
plt.grid(True)
plt.savefig("./web/ratings.png")

print(f"top-elo={max(ys)}")
print(f"last-elo={ys[-1]}")

