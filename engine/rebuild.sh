#!/bin/bash
set -x

RUN_DIR=$(cat CURRENT_RUN)

cargo build --release --bin mcts_generate
cp target/release/mcts_generate $RUN_DIR

cargo build --release --bin compete
cp target/release/compete $RUN_DIR

maturin build --release
pip install --force-reinstall /home/snp/proj/duckchess/engine/target/wheels/engine-0.1.0-cp310-cp310-linux_x86_64.whl

