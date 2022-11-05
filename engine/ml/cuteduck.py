import time
import json
import show_game
import argparse
import subprocess

import engine as engine_lib

parser = argparse.ArgumentParser()
parser.add_argument("--engine1", type=str)
parser.add_argument("--engine2", type=str)

square_names = [
    letter + number
    for number in "12345678"
    for letter in "abcdefgh"
]

def uci_to_move(uci):
    departure, destination = square_names.index(uci[:2]), square_names.index(uci[2:4])
    return json.dumps({"from": departure, "to": destination})

if __name__ == "__main__":
    args = parser.parse_args()

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
                message = "position startpos moves" + "".join(" " + move for move in moves_list) + "\n"
                send(engine, message)
                for _ in range(2):
                    time.sleep(0.5)
                    send(engine, "go depth 5\n")
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
                    print("Got move:", move, uci_to_move(move))
                    r = follow_along.apply_move(uci_to_move(move))
                    if r is not None:
                        print("GOT:", r)
                    show_game.render_state(json.loads(follow_along.get_state()))
                    input("> ")
                if done:
                    break
        
        print("============= Game over =============")
