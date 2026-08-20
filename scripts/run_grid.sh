#!/usr/bin/env bash
# Full grid we discussed: 10 tasks × 4 dual-arm embodiments × 3 obstacle modes.
# Same seed is reused across none/off_path/on_path for a given (task, embodiment, episode)
# so those three videos share a scene. The next pair gets a new seed (new scene).
#
#   bash CEHJ/scripts/run_grid.sh
#   bash CEHJ/scripts/run_grid.sh discussed 2 10     # extra replan+waypoint, 2 scenes, base seed 10
#   bash CEHJ/scripts/run_grid.sh grid 1 0 --cluttered --draw-bbox
set -euo pipefail
ROOT="/storage1/fs1/sibai/Active/yuxuan/cross_embodiment/CEHJ"
PRESET="${1:-grid}"
EPISODES="${2:-1}"
BASE_SEED="${3:-0}"
shift $(( $# >= 3 ? 3 : $# )) || true
export PYTHONNOUSERSITE=1
python "${ROOT}/main/run_all.py" \
  --preset "${PRESET}" \
  --episodes "${EPISODES}" \
  --base-seed "${BASE_SEED}" \
  --output "${ROOT}/outputs/ihab/${PRESET}" \
  "$@"

