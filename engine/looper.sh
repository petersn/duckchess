#!/bin/bash
set -x

RUN_DIR=$(cat CURRENT_RUN)

#time python ml/looper.py --parallel-games-processes 5 --training-window 50 --game-count 5000 --prefix $RUN_DIR
time python ml/looper.py --parallel-games-processes 6 --training-window 50 --game-count 3000 --prefix $RUN_DIR
