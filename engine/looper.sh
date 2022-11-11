#!/bin/bash
set -x

RUN_DIR=$(cat CURRENT_RUN)

time python ml/looper.py --prefix $RUN_DIR
