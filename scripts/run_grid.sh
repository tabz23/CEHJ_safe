#!/usr/bin/env bash
# Full grid we discussed: 10 tasks × 4 dual-arm embodiments × 3 obstacle modes.
# Same seed is reused across none/off_path/on_path for a given (task, embodiment, episode)
# so those three videos share a scene. The next pair gets a new seed (new scene).
#
#   bash CEHJ/scripts/run_grid.sh
#   bash CEHJ/scripts/run_grid.sh discussed 2 10     # extra replan+waypoint, 2 scenes, base seed 10
#   bash CEHJ/scripts/run_grid.sh grid 1 0 --cluttered --draw-bbox
#   bash CEHJ/scripts/run_grid.sh grid 1 0 --obstacle-model 105_sauce-can
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
PRESET="${1:-grid}"
EPISODES="${2:-1}"
BASE_SEED="${3:-0}"
shift $(( $# >= 3 ? 3 : $# )) || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python "${ROOT}/main/run_all.py" \
  --preset "${PRESET}" \
  --episodes "${EPISODES}" \
  --base-seed "${BASE_SEED}" \
  --output "${ROOT}/outputs/ihab/${PRESET}" \
  "$@"

