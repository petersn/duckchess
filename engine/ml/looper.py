#!/usr/bin/python

import os, glob, signal, subprocess, socket, atexit, time, math

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

game_processes = None

def generate_games(prefix, model_number):
    global game_processes
    try:
        os.mkdir(index_to_games_dir(model_number))
    except FileExistsError:
        pass

    # Before we even launch check if we have enough games.
    if count_games(index_to_games_dir(model_number)) >= args.game_count:
        print("Enough games to start with!")
        return

    model_dir = index_to_converted_model_prefix(model_number)
    output_dir = index_to_games_dir(model_number)

    def get_trt_path(index):
        compute_capability = "compute8.9" if index == 0 else "compute8.6"
        return f"{model_dir}-{compute_capability}.trt"

    # If we don't have any game processes already, then launch them.
    if game_processes is None:
        # FIXME: This is super hacky, but I just hardcode which GPUs have which compute capability.
        game_processes = [
            subprocess.Popen(
                [
                    prefix + "/mcts_generate", #"--release", "--bin", "mcts_generate", "--",
                        "--model-dir", get_trt_path(process_index),
                        "--output-dir", output_dir,
                        # FIXME: Now that I'm using TensorRT this *must* match the saved engine.
                        "--batch-size", "128" # "256" if process_index == 0 else "128"
                ],
                close_fds=True,
                env=dict(
                    os.environ,
                    TF_FORCE_GPU_ALLOW_GROWTH="true",
                    LD_LIBRARY_PATH="/usr/local/cuda/lib64", #:./run-011-duck-chess",
                    CUDA_VISIBLE_DEVICES=str(process_index),
                ),
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                #, LD_LIBRARY_PATH="/usr/lib/python3/dist-packages/tensorflow/"),
            )
            for process_index in range(args.parallel_games_processes)
        ]
        # Launch a thread to monitor stderr from each process.
        def monitor_stderr(process_index, proc):
            print("\x1b[94m===== Monitoring stderr from game process", process_index, "\x1b[0m")
            for line in proc.stderr:
                print(f"\x1b[91m[stderr {process_index}]:\xb1[0m", line.decode().strip())
                # Append this line ./ERROR_LOGS
                with open(f"./ERROR_LOGS/{process_index}.log", "ab") as f:
                    f.write(line)
            print("\x1b[94m===== Game process", process_index, "exited\x1b[0m")

        for process_index, proc in enumerate(game_processes):
            import threading
            threading.Thread(target=monitor_stderr, args=(process_index, proc)).start()

        # If our process dies take the games generation down with us.
        def _(game_processes):
            def handler():
                for proc in game_processes:
                    kill(proc)
            atexit.register(handler)
        _(game_processes)
    else:
        # Otherwise, we simply tell our existing processes to swap to a new model.
        for process_index, proc in enumerate(game_processes):
            message = f"swap:::{get_trt_path(process_index)}:::{output_dir}\n"
            print(f"\x1b[91m===== Sending message to game process {process_index}:\xb1[0m", message.strip())
            proc.stdin.write(message.encode())
            proc.stdin.flush()

    # We now periodically check up on how many games we have.
    while True:
        game_count = count_games(index_to_games_dir(model_number))
        print("Game count:", game_count)
        time.sleep(10)
        if game_count >= args.game_count:
            break

    ## Signal the process to die gracefully.
    #for proc in game_processes:
    #    os.kill(proc.pid, signal.SIGTERM)
    ## Wait up to two seconds, then forcefully kill it.
    #time.sleep(2)
    #for proc in game_processes:
    #    kill(proc)
    #print("Exiting.")

def index_to_dir(i):
    return f"{args.prefix}/step-{i:03}"

def index_to_model_path(i):
    return f"{args.prefix}/step-{i:03}/model-{i:03}.pt"

def index_to_optim_state_path(i):
    return f"{args.prefix}/step-{i:03}/optim-state-{i:03}.pt"

def index_to_converted_model_prefix(i):
    return f"{args.prefix}/step-{i:03}/model"

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
    parser.add_argument("--training-steps-const", metavar="N", type=int, default=500, help="Base number of training steps to perform per iteration.")
    parser.add_argument("--training-steps-linear", metavar="N", type=int, default=50, help="We also apply an additional N steps for each additional iteration included in the training window.")
    parser.add_argument("--training-window", metavar="N", type=int, default=20, help="When training include games from the past N iterations.")
    #parser.add_argument("--training-window-exclude", metavar="N", type=int, default=3, help="To help things get started faster we exclude games from the very first N iterations from later training game windows.")
    parser.add_argument("--parallel-games-processes", metavar="N", type=int, default=2, help="Number of games processes to run in parallel.")
    args = parser.parse_args()
    print("Arguments:", args)

    project_name = args.prefix
    for c in "./":
        if project_name.startswith(c):
            project_name = project_name[1:]
        if project_name.endswith(c):
            project_name = project_name[:-1]
    print("=== Project:", project_name)

    current_model_number = 1

    while True:
        start = time.time()
        old_model = index_to_model_path(current_model_number)
        new_model = index_to_model_path(current_model_number + 1)
        old_optim_state = index_to_optim_state_path(current_model_number)
        new_optim_state = index_to_optim_state_path(current_model_number + 1)

        if os.path.exists(new_model):
            print("Model already exists, skipping:", new_model)
            current_model_number += 1
            continue

        new_trt = index_to_converted_model_prefix(current_model_number) + "-compute8.6.trt"
        if not os.path.exists(new_trt):
            print("=========================== Converting pytorch -> keras -> onnx -> tensorrt")
            subprocess.check_call(
                [
                    "python", "ml/convert_model_BIG.py",
                    "--input", old_model,
                    "--output", index_to_converted_model_prefix(current_model_number),
                    "--big",
                ],
            )
        assert os.path.exists(new_trt)

        print("=========================== Doing data generation for:", old_model)
        print("Start time:", start)
        generate_games(args.prefix, current_model_number)

        try:
            os.mkdir(index_to_dir(current_model_number + 1))
        except FileExistsError:
            #print("\x1b[91mWeird, should this already exist?\x1b[0m", index_to_dir(current_model_number + 1))
            pass

        effective_training_window_size = min(args.training_window, max(1, current_model_number // 2))
        print("=========================== Doing training:", old_model, "->", new_model, "effective window size:", effective_training_window_size)
        # Figure out the directories of games to train on.
        #low_index = min(current_model_number, max(args.training_window_exclude + 1, current_model_number - args.training_window + 1))
        low_index = max(1, current_model_number - effective_training_window_size + 1)
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
        # We start at a learning rate of 4e-4 and decay exponentially to 3e-5 by 20 models.
        # Edit: I just adjusted it down to 5e-6.
        # Edit: I just adjusted the starting lr down to 1e-4, and the final down to 2e-6.
        lerp_coef = max(0, min(1, (current_model_number - 1) / 19))
        learning_rate = math.exp( math.log(1e-4) * (1 - lerp_coef) + math.log(5e-6) * lerp_coef )
        print("\x1b[91mLearning rate:", learning_rate, "\x1b[0m")

        optional_old_optim_state = []
        if os.path.exists(old_optim_state):
            optional_old_optim_state = ["--old-optim-state-path", old_optim_state]

        subprocess.check_call([
            "python", "ml/looper_train.py",
                "--steps", str(steps),
                "--games"] + games_paths + [
                "--old-path", old_model,
                "--new-path", new_model,
                "--new-optim-state-path", new_optim_state,
                "--learning-rate", str(learning_rate),
                "--project-name", project_name,
        ] + optional_old_optim_state, close_fds=True,
            env=dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="0",
            ),
        )

        end = time.time()
        print("Total seconds for iteration:", end - start)
        current_model_number += 1
