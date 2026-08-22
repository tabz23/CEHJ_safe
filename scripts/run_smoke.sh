#!/usr/bin/env bash
# Smoke: place_empty_cup × Dual-arm Piper × three obstacle modes.
# Run inside the robot-sim container with rbtw128.
#
# Obstacle mesh (--obstacle-model; always static; distance uses scaled OBB from model_data):
#   086_woodenblock   cube    10.3 cm                 (default)
#   068_boxdrink      box     11.0 x 15.4 x 11.6 cm
#   105_sauce-can     can     10.0 x 11.6 x 10.0 cm
#   059_pencup        cup     9.8 x 11.7 x 9.8 cm
#   071_can           can     7.1 x 9.6 x 7.1 cm
#   101_milk-tea      cup     13.6 x 15.5 x 13.6 cm
#   023_tissue-box    box     11.6 x 6.3 x 6.8 cm
#   038_milk-box      carton  6.9 x 12.2 x 6.5 cm
#   004_fluted-block  block   9.2 x 6.5 x 9.0 cm
#   073_rubikscube    cube    6.5 x 6.8 x 7.7 cm
# Any RoboTwin-OD folder under RoboTwin/assets/objects/ also works.
set -euo pipefail
ROOT="/storage1/fs1/sibai/Active/yuxuan/cross_embodiment/CEHJ"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python "${ROOT}/main/run.py" --help
python "${ROOT}/main/run_all.py" --preset smoke --draw-bbox --base-seed 0 --output "${ROOT}/outputs/ihab/smoke"
