import json
import sys
import collections

wins = collections.Counter()
losses = collections.Counter()
draws = collections.Counter()

for path in sys.argv[1:]:
    with open(path) as f:
        for line in f:
            game = json.loads(line)
            if game["outcome"] == "1-0":
                wins[game["engine_white"]] += 1
                losses[game["engine_black"]] += 1
            elif game["outcome"] == "0-1":
                wins[game["engine_black"]] += 1
                losses[game["engine_white"]] += 1
            else:
                assert game["outcome"] is None
                draws[game["engine_white"]] += 1
                draws[game["engine_black"]] += 1

print(wins)
print(losses)
print(draws)
