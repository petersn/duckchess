#!/bin/bash

set -x

time scp -r public/model-small snpbox:duckchess/public/
time scp -r public/model-medium snpbox:duckchess/public/

