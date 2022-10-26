#!/usr/bin/python

import os, glob, signal, subprocess, socket, atexit, time

def count_games(paths):
    total_games = 0
    for path in glob.glob(paths + "/*.json"):
        with open(path) as f:
            for line in f:
                if line.strip():
                    total_games += 1
    return total_games

def kill(proc):
    print("Killing:", proc)
    try:
        os.kill(proc.pid, signal.SIGTERM)
    except Exception as e:
        print("ERROR in kill:", e)
    proc.kill()

def generate_games(model_number):
    try:
        os.mkdir(index_to_games_dir(model_number))
    except FileExistsError:
        pass

    # Before we even launch check if we have enough games.
    if count_games(index_to_games_dir(model_number)) >= args.game_count:
        print("Enough games to start with!")
        return

    # Launch the games generation.
    games_processes = [
        subprocess.Popen([
            #"python", "accelerated_generate_games.py",
            "cargo", "run", "--release", "--bin", "mcts_generate", "--",
                "--model-dir", index_to_keras_model_path(model_number),
                "--output-dir", index_to_games_dir(model_number),
        ], close_fds=True)
        for _ in range(args.parallel_games_processes)
    ]
    # If our process dies take the games generation down with us.
    def _(games_processes):
        def handler():
            for proc in games_processes:
                kill(proc)
        atexit.register(handler)
    _(games_processes)

    # We now periodically check up on how many games we have.
    while True:
        game_count = count_games(index_to_games_dir(model_number))
        print("Game count:", game_count)
        time.sleep(10)
        if game_count >= args.game_count:
            break

    # Signal the process to die gracefully.
    for proc in games_processes:
        os.kill(proc.pid, signal.SIGTERM)
    # Wait up to two seconds, then forcefully kill it.
    time.sleep(2)
    for proc in games_processes:
        kill(proc)
    print("Exiting.")

def index_to_dir(i):
    return f"{args.prefix}/step-{i:03}"

def index_to_model_path(i):
    return f"{args.prefix}/step-{i:03}/model-{i:03}.pt"

def index_to_keras_model_path(i):
    return f"{args.prefix}/step-{i:03}/model-keras"

def index_to_games_dir(i):
    return f"{args.prefix}/step-{i:03}/games"

if __name__ == "__main__":
    import argparse
    class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter): pass
    parser = argparse.ArgumentParser(
        description="""
Performs the main loop of alternating game generation (via accelerated_generate_games.py)
with training (via train.py). You specify a prefix path which will have the structure:
  PREFIX/
    step-001/
      games/
        model-001-0.json
        model-001-1.json
        ...
      model-001.pt
      model-keras/
        ...
    step-002/
      ... 
You are expected to create the PREFIX/{games,models,models-keras} directories, and populate
the initial PREFIX/models/model-001.pt file. Then looper.py will run in a main loop of:
  1) Generate self-play games with the highest numbered present
     model until reaching the minimum game count.
  2) Train model n+1 from the current highest numbered model,
     and a pool of games from recent iterations.
It is relatively safe to interrupt and restart, as looper.py will automatically resume on
the most recent model. (However, interrupting and restarting looper.py of course
technically statistically biases the games slightly towards being shorter.)
""",
        formatter_class=Formatter,
    )
    parser.add_argument("--prefix", metavar="PATH", default=".", help="Prefix directory. Make sure this directory contains games/ and models/ subdirectories.")
    parser.add_argument("--game-count", metavar="N", type=int, default=5000, help="Minimum number of games to generate per iteration.")
    parser.add_argument("--training-steps-const", metavar="N", type=int, default=5000, help="Base number of training steps to perform per iteration.")
    parser.add_argument("--training-steps-linear", metavar="N", type=int, default=1000, help="We also apply an additional N steps for each additional iteration included in the training window.")
    parser.add_argument("--training-window", metavar="N", type=int, default=10, help="When training include games from the past N iterations.")
    parser.add_argument("--training-window-exclude", metavar="N", type=int, default=0, help="To help things get started faster we exclude games from the very first N iterations from later training game windows.")
    parser.add_argument("--parallel-games-processes", metavar="N", type=int, default=3, help="Number of games processes to run in parallel.")
    args = parser.parse_args()
    print("Arguments:", args)

    current_model_number = 1

    while True:
        start = time.time()
        old_model = index_to_model_path(current_model_number)
        new_model = index_to_model_path(current_model_number + 1)

        if os.path.exists(new_model):
            print("Model already exists, skipping:", new_model)
            current_model_number += 1
            continue

        if not os.path.exists(index_to_keras_model_path(current_model_number)):
            print("=========================== Converting pytorch -> keras")
            subprocess.check_call([
                "python", "ml/convert_model.py",
                "--input", old_model,
                "--output", index_to_keras_model_path(current_model_number),
            ])

        print("=========================== Doing data generation for:", old_model)
        print("Start time:", start)
        generate_games(current_model_number)

        try:
            os.mkdir(index_to_dir(current_model_number + 1))
        except FileExistsError:
            print("\x1b[91mWeird, should this already exist?\x1b0m", index_to_dir(current_model_number + 1))

        print("=========================== Doing training:", old_model, "->", new_model)
        # Figure out the directories of games to train on.
        low_index = min(current_model_number, max(args.training_window_exclude + 1, current_model_number - args.training_window + 1))
        high_index = current_model_number
        games_paths = [
            path
            for i in range(low_index, high_index + 1)
            for path in glob.glob(index_to_games_dir(i) + "/*.json")
        ]

        print("Game paths:", games_paths)
        steps = args.training_steps_const + args.training_steps_linear * (high_index - low_index + 1)
        assert steps > 0
        print("Steps:", steps)
        subprocess.check_call([
            "python", "ml/looper_train.py",
                "--steps", str(steps),
                "--games"] + games_paths + [
                "--old-path", old_model,
                "--new-path", new_model,
        ], close_fds=True)

        end = time.time()
        print("Total seconds for iteration:", end - start)
        current_model_number += 1