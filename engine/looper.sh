#!/bin/bash
set -x

RUN_DIR=$(cat CURRENT_RUN)

time python ml/looper.py --game-count 3000 --prefix $RUN_DIR
