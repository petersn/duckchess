import json
import glob

all_games = []
for path in glob.glob("rl-games/games-*.json"):
    with open(path) as f:
        for line in f:
            all_games.append(json.loads(line))

print("Total games:", len(all_games))
print("Total moves:", sum(len(game["moves"]) for game in all_games))

import collections
c = collections.Counter()
for game in all_games:
    c[str(game["moves"][0])] += 1
for k, v in c.items():
    print(k, "->", v)

