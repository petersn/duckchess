#!/bin/bash
set -e
set -x

RUN_DIR=$(cat CURRENT_RUN)

cargo build --release --bin mcts_generate
cp target/release/mcts_generate $RUN_DIR

#cargo build --bin mcts_generate
#cp target/debug/mcts_generate $RUN_DIR

