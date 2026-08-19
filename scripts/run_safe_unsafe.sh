#!/usr/bin/env bash
# Three episodes per task × embodiment, same seed so the scene matches:
#   none     = no wooden block (safe)
#   off_path = block beside the object→target line
#   on_path  = block on that line (unsafe; CuRobo still ignores it)
#
#   10 tasks × 4 embodiments × 3 modes = 120 episodes
#   embodiments (all): ARX-X5, franka-panda, ur5-wsg, piper
#
# Run inside the robot-sim container with rbtw128:
#   conda activate rbtw128
#   export PYTHONNOUSERSITE=1
#   bash /storage1/fs1/sibai/Active/yuxuan/cross_embodiment/CEHJ/scripts/run_safe_unsafe.sh
#
# Optional args are forwarded to run_all.py:
#   bash CEHJ/scripts/run_safe_unsafe.sh --draw-bbox
#   bash CEHJ/scripts/run_safe_unsafe.sh --embodiments piper,franka
#   bash CEHJ/scripts/run_safe_unsafe.sh --episodes 2 --base-seed 10
#   bash CEHJ/scripts/run_safe_unsafe.sh            # resumes; skips finished summary.json
#   bash CEHJ/scripts/run_safe_unsafe.sh --overwrite  # re-run everything
set -euo pipefail
ROOT="/storage1/fs1/sibai/Active/yuxuan/cross_embodiment/CEHJ"
export PYTHONNOUSERSITE=1

python "${ROOT}/main/run_all.py" \
  --preset grid \
  --tasks all \
  --embodiments all \
  --obstacle-modes none,off_path,on_path \
  --place-mode geometric \
  --plan-mode ignore_obstacle \
  --unsafe-level 3 \
  --episodes 1 \
  --base-seed 0 \
  --output "${ROOT}/outputs/ihab/safe_unsafe" \
  --draw-bbox \
  --resume \
  "$@"
