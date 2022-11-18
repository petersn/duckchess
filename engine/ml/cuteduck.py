import time
import json
import argparse
import subprocess
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

import show_game
import engine as engine_lib

global_lock = Lock()

parser = argparse.ArgumentParser()
parser.add_argument("--engine1", type=str)
parser.add_argument("--engine2", type=str)
parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--games", type=int, default=100)
parser.add_argument("--output", type=str, default="games.jsonl")

square_names = [
    letter + number
    for number in "12345678"
    for letter in "abcdefgh"
]

def uci_to_move(uci):
    departure, destination = square_names.index(uci[:2]), square_names.index(uci[2:4])
    return json.dumps({"from": departure, "to": destination})

def worker(pgn_out, index):
    print(f"Worker {index} started")
    while True:
        moves_list = []
        engine1 = subprocess.Popen(args.engine1, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        engine2 = subprocess.Popen(args.engine2, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        engine_colors = {engine1: "\x1b[91m", engine2: "\x1b[92m"}

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
        while not done:
            for engine in [engine1, engine2]:
                outcome = follow_along.get_outcome()
                if outcome is not None:
                    print(f"Game over: {outcome}")
                    send(engine1, "quit\n")
                    send(engine2, "quit\n")
                    time.sleep(0.01)
                    engine1.kill()
                    engine2.kill()
                message = "position startpos moves" + "".join(" " + move for move in moves_list) + "\n"
                send(engine, message)
                for _ in range(2):
                    send(engine, "go depth 4\n")
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
        
        print("============= Game over =============")
        if pgn_out is not None:
            with global_lock:
                with open("games.txt", "a") as f:
                    f.write(" ".join(moves_list) + "


if __name__ == "__main__":
    args = parser.parse_args()

    tpe = ThreadPoolExecutor(max_workers=1)
    for i in range(args.parallel):
        tpe.submit(worker, i)
    tpe.shutdown(wait=True)
