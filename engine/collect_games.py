import json
import glob

all_games = []
for path in glob.glob("games/games-*.json"):
    with open(path) as f:
        for line in f:
            all_games.append(json.loads(line))

print("Total games:", len(all_games))
print("Total moves:", sum(len(game["moves"]) for game in all_games))
