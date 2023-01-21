#!/bin/bash
set -x

RUN_DIR=$(cat CURRENT_RUN)

maturin build --release
#pip install --force-reinstall /home/snp/proj/duckchess/engine/target/wheels/engine-0.1.0-cp310-cp310-linux_x86_64.whl
pip install --force-reinstall /home/snp/proj/duckchess/engine/target/wheels/engine-0.1.0-cp310-cp310-manylinux_2_34_x86_64.whl

