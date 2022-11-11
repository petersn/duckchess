#!/bin/bash
set -x

RUN_DIR=$(cat CURRENT_RUN)

time CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$RUN_DIR" python ml/check_elos.py $RUN_DIR
