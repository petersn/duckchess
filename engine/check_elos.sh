#!/bin/bash
set -x

RUN_DIR=$(cat CURRENT_RUN)

time CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH="/usr/local/cuda/lib64:$RUN_DIR" python ml/check_elos.py $RUN_DIR
