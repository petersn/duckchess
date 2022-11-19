import time
import json
import random
import argparse
import subprocess
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

import show_game
import engine as engine_lib

global_lock = Lock()

GAME_LENGTH_LIMIT = 500

parser = argparse.ArgumentParser()
parser.add_argument("--analyze", type=str, default=None)
parser.add_argument("--engine1", type=str)
parser.add_argument("--engine2", type=str)
parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--output", type=str, default="games.jsonl")

square_names = [
    letter + number
    for number in "12345678"
    for letter in "abcdefgh"
]

def uci_to_move(uci):
    departure, destination = square_names.index(uci[:2]), square_names.index(uci[2:4])
    return json.dumps({"from": departure, "to": destination})

def move_to_uci(move):
    return square_names[move["from"]] + square_names[move["to"]]

def worker(games_file, index):
    print(f"Worker {index} started")
    engine_white_command = args.engine1
    engine_black_command = args.engine2
    if random.random() < 0.5:
        engine_white_command, engine_black_command = engine_black_command, engine_white_command
    go_command = "go depth 4"

    while True:
        moves_list = []
        was_rand = []
        engine_white = subprocess.Popen(engine_white_command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        engine_black = subprocess.Popen(engine_black_command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        engine_colors = {engine_white: "\x1b[91m", engine_black: "\x1b[92m"}

        def send(engine, message):
            print(engine_colors[engine] + "< " + message.strip() + "\x1b[0m")
            engine.stdin.write(message.encode())
            engine.stdin.flush()

        def get_line(engine):
            line = engine.stdout.readline().decode()
            if not line:
                raise Exception("Engine died")
            print(engine_colors[engine] + "> " + line.strip() + "\x1b[0m")
            return line

        follow_along = engine_lib.Engine(0)

        # Ask the first engine for two moves.
        done = False
        while not done and len(moves_list) < GAME_LENGTH_LIMIT:
            for engine in [engine_white, engine_black]:
                outcome = follow_along.get_outcome()
                if outcome is not None:
                    print(f"Game over: {outcome}")
                    done = True
                    break
                for _ in range(2):
                    message = "position startpos moves" + "".join(" " + move for move in moves_list) + "\n"
                    send(engine, message)
                    send(engine, go_command + "\n")
                    while True:
                        line = get_line(engine)
                        if not line.startswith("bestmove "):
                            continue
                        move = line.split()[1]
                        if move == "0000":
                            done = True
                        break
                    if done:
                        break
                    # Sometimes randomize the move.
                    rand_move_prob = 0.8 * 0.4 ** (len(moves_list) // 4)
                    random_move = random.random() < rand_move_prob
                    #random_move = False # FIXME: Disabling this because I can't deal with it for MCTS right now
                    if random_move:
                        move = move_to_uci(json.loads(random.choice(follow_along.get_moves())))
                        print(f"Random move: {move}")

                    was_rand.append(random_move)
                    moves_list.append(move)
                    #print("Got move:", move, uci_to_move(move))
                    r = follow_along.apply_move(uci_to_move(move))
                    if r is not None:
                        print("GOT:", r)
                        raise Exception("Game over")
                    #show_game.render_state(json.loads(follow_along.get_state()))
                    #input("> ")
                if done:
                    break
        send(engine_white, "quit\n")
        send(engine_black, "quit\n")
        time.sleep(0.01)
        engine_white.kill()
        engine_black.kill()

        print("============= Game over =============")
        if games_file is not None:
            game_desc = {
                "engine_white": engine_white_command,
                "engine_black": engine_black_command,
                "go": go_command,
                "was_rand": was_rand,
                "moves": moves_list,
                "outcome": follow_along.get_outcome(),
            }
            with global_lock:
                json.dump(game_desc, games_file)
                games_file.write("\n")
                games_file.flush()
        # Swap the engines.
        engine_white_command, engine_black_command = engine_black_command, engine_white_command

    print(f"Worker {index} finished")

def analyze(path):
    pgn = []
    with open(args.analyze) as f:
        for line in f:
            game = json.loads(line)
            result = {"1-0": "1-0", "0-1": "0-1", None: "1/2-1/2"}[game["outcome"]]
            pgn.append("""
[Event "uci-compete"]
[Site "uci-compete"]
[Date "????.??.??"]
[Round "1"]
[White "%s"]
[Black "%s"]
[Result "%s"]

1. %s
""" % (game["engine_white"], game["engine_black"], result, " ".join(game["moves"])))

    print("Valid games:", len(pgn))

    with open("/tmp/uci-compete.pgn", "w") as f:
        f.write("".join(pgn))

    output = subprocess.check_output(
        ["bayeselo"],
        input="readpgn /tmp/uci-compete.pgn\nelo\nmm\nexactdist\nratings\n".encode("utf-8"),
    ).decode()

    print(output.replace("ResultSet-EloRating>", "ResultSet-EloRating>\n"))


if __name__ == "__main__":
    args = parser.parse_args()

    if args.analyze is not None:
        analyze(args.analyze)
        exit()

    games_file = open(args.output, "a") if args.output is not None else None

    tpe = ThreadPoolExecutor(max_workers=args.parallel)
    futures = [
        tpe.submit(worker, games_file, i)
        for i in range(args.parallel)
    ]
    for future in futures:
        future.result()
    print("Shutting down")
