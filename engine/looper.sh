#!/bin/bash
set -x

RUN_DIR=$(cat CURRENT_RUN)

time python ml/looper.py --game-count 4000 --prefix $RUN_DIR
