#!/usr/bin/env bash
# Smoke: place_empty_cup × Dual-arm Piper × three obstacle modes.
# Run inside the robot-sim container with rbtw128.
set -euo pipefail
ROOT="/root/autodl-tmp"
export PYTHONNOUSERSITE=1
python "${ROOT}/CEHJ_safe/main/run.py" --help
python "${ROOT}/CEHJ_safe/main/run_all.py" --preset smoke --draw-bbox --base-seed 0 --output "${ROOT}/outputs/ihab/smoke"
