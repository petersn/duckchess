#!/bin/bash
set -x

RUN_DIR=$(cat CURRENT_RUN)

time python ml/looper.py --parallel-games-processes 1 --training-window 30 --game-count 5000 --prefix $RUN_DIR
