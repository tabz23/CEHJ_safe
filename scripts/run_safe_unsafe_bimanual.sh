#!/usr/bin/env bash
# Bimanual safety episodes: 8 dual-arm tasks × 4 embodiments × 3 obstacle modes.
# Same seed across none / off_path / on_path so the scene matches.
#
#   place_burger_fries, place_cans_plasticbox, stack_blocks_two, place_can_basket,
#   place_bread_basket, grab_roller, pick_dual_bottles, stack_bowls_two
#
#   8 tasks × 4 embodiments × 3 modes = 96 episodes
#   embodiments (all): ARX-X5, franka-panda, ur5-wsg, piper
#
# Run inside the robot-sim container with rbtw128:
#   conda activate rbtw128
#   export PYTHONNOUSERSITE=1
#   bash /storage1/fs1/sibai/Active/yuxuan/cross_embodiment/CEHJ/scripts/run_safe_unsafe_bimanual.sh
#
# Optional args are forwarded to run_all.py:
#   bash CEHJ/scripts/run_safe_unsafe_bimanual.sh --embodiments piper,franka
#   bash CEHJ/scripts/run_safe_unsafe_bimanual.sh --episodes 2 --base-seed 10
#   bash CEHJ/scripts/run_safe_unsafe_bimanual.sh            # resumes; skips finished summary.json
#   bash CEHJ/scripts/run_safe_unsafe_bimanual.sh --overwrite  # re-run everything
set -euo pipefail
ROOT="/storage1/fs1/sibai/Active/yuxuan/cross_embodiment/CEHJ"
export PYTHONNOUSERSITE=1

python "${ROOT}/main/run_all.py" \
  --preset grid \
  --task-set bimanual \
  --tasks all \
  --embodiments all \
  --obstacle-modes none,off_path,on_path \
  --place-mode geometric \
  --plan-mode ignore_obstacle \
  --unsafe-level 3 \
  --episodes 1 \
  --base-seed 0 \
  --output "${ROOT}/outputs/ihab/safe_unsafe_bimanual" \
  --draw-bbox \
  --resume \
  "$@"
