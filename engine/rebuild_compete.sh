#!/bin/bash
set -x

RUN_DIR=$(cat CURRENT_RUN)

cargo build --release --bin compete
cp target/release/compete $RUN_DIR

