# Solve this dueling bandit problem
import os
import sys
import json
import glob
import random
import itertools
import subprocess
import collections
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# FIXME: This whole file is very busted right now.

# We perform a tournament.
#prefix = "run-011-duck-chess/"
#model_dir = prefix + "step-065/model-keras/"
#output_dir = "hyper-opt-games-200visits"
prefix = "run-016/"
model_dir = prefix + "step-275/model-compute8.9.trt"
output_dir = "hyper-opt-games-200visits-run016"
playouts = 200

global_lock = Lock()
gpu_allocations = 0

def allocate_gpu():
    global gpu_allocations
    with global_lock:
        gpu_allocations += 1
        return "01"[gpu_allocations % 2]

def process():
    # Allocate a GPU.
    #gpu = allocate_gpu()
    #print(f"Working on GPU {gpu}")
    subprocess.check_call(
        [
            prefix + "/compete",
            "--playouts1", str(playouts),
            "--playouts2", str(playouts),
            "--randomize-search-params",
            "--model1-dir", model_dir,
            "--model2-dir", model_dir,
            "--output-dir", output_dir,
        ],
        close_fds=True,
        env=dict(
            os.environ,
            TF_FORCE_GPU_ALLOW_GROWTH="true",
            LD_LIBRARY_PATH=prefix,
            #CUDA_VISIBLE_DEVICES=gpu,
        ),
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python search_param_opt.py --generate")
        print("  python search_param_opt.py --analyze")
        sys.exit(1)
    process_count = 1
    mode = sys.argv[1]
    if mode == "--generate":
        tpe = ThreadPoolExecutor(max_workers=2)
        futures = [
            tpe.submit(process)
            for _ in range(process_count)
        ]
        for future in futures:
            future.result()
        print("Done!")
    elif mode == "--analyze":
        # Load up all the games.
        games = []
        for path in glob.glob(output_dir + "/*.json"):
            with open(path, "r") as f:
                for line in f:
                    games.append(json.loads(line))
        print(f"Loaded {len(games)} games")
        competitor_outcomes = collections.defaultdict(list)

        def format(sp):
            alpha = sp["exploration_alpha"]
            duckalpha = sp["duck_exploration_alpha"]
            fpu = sp["first_play_urgency"]
            return f"alpha={alpha:.3f}:duckalpha={duckalpha:.3f}:fpu={fpu:.3f}"

        pgn = []
        for game in games:
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
        """ % (format(game["search_params_white"]), format(game["search_params_black"]), result, result))

        print("Valid games:", len(pgn))

        with open("/tmp/hyper-opt.pgn", "w") as f:
            f.write("".join(pgn))

        output = subprocess.check_output(
            ["bayeselo"],
            input="readpgn /tmp/hyper-opt.pgn\nelo\nmm\nexactdist\nratings\n".encode("utf-8"),
        ).decode()

        print(output.replace("ResultSet-EloRating>", "ResultSet-EloRating>\n"))

        # Extract all of the elos.
        ratings = {}
        for line in output.split("\n"):
            if line.startswith("  "):
                _, name, rating, *_ = line.split()
                rating = int(rating)
                ratings[name] = rating
        min_rating = min(ratings.values())
        ratings = {name: rating - min_rating for name, rating in ratings.items()}
        print("Ratings:", ratings)

    else:
        print("Unknown mode, try --generate or --analyze")
        sys.exit(1)

# import numpy as np
# import matplotlib.pyplot as plt
# from bayes_opt import BayesianOptimization

# xs = np.linspace(-2, 10, 10000)

# def f(x):
#     return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1/ (x ** 2 + 1)



# def plot_bo(f, bo):
#     x = np.linspace(-2, 10, 10000)
#     mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)
    
#     plt.figure(figsize=(16, 9))
#     plt.plot(x, f(x))
#     plt.plot(x, mean)
#     plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
#     plt.scatter(bo.space.params.flatten(), bo.space.target, c="red", s=50, zorder=10)
#     plt.show()

# bo = BayesianOptimization(
#     f=f,
#     pbounds={"x": (-2, 10)},
#     verbose=0,
#     random_state=987234,
# )

# bo.maximize(n_iter=10, acq="ucb", kappa=5.0)

# plot_bo(f, bo)
