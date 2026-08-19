#!/usr/bin/env bash
# Smoke: place_empty_cup × Dual-arm Piper × three obstacle modes.
# Run inside the robot-sim container with rbtw128.
set -euo pipefail
ROOT="/storage1/fs1/sibai/Active/yuxuan/cross_embodiment/CEHJ"
export PYTHONNOUSERSITE=1
python "${ROOT}/main/run.py" --help
python "${ROOT}/main/run_all.py" --preset smoke --draw-bbox --base-seed 0 --output "${ROOT}/outputs/ihab/smoke"
