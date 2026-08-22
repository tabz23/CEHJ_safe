#!/usr/bin/env bash
# Bimanual safety episodes: 8 dual-arm tasks × 4 embodiments × 3 obstacle modes.
# Same seed across none / off_path / on_path so the scene matches.
#
#   place_burger_fries, place_cans_plasticbox, stack_blocks_two, place_can_basket,
#   place_bread_basket, grab_roller, pick_dual_bottles, stack_bowls_two
#
#   8 tasks × 4 embodiments × 3 modes = 96 episodes
#   Obstacle t ~ Uniform[0.3, 0.7] from the seed (same t for none/off_path/on_path).
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
#   bash CEHJ/scripts/run_safe_unsafe_bimanual.sh --obstacle-model 105_sauce-can
#   bash CEHJ/scripts/run_safe_unsafe_bimanual.sh            # resumes; skips finished summary.json
#   bash CEHJ/scripts/run_safe_unsafe_bimanual.sh --overwrite  # re-run everything
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

python "${ROOT}/main/run_all.py" \
  --preset grid \
  --task-set bimanual \
  --tasks all \
  --embodiments all \
  --obstacle-modes on_path \
  --place-mode geometric \
  --plan-mode ignore_obstacle \
  --episodes 1 \
  --base-seed 12 \
  --output "${ROOT}/outputs/ihabnew/safe_unsafe_bimanual" \
  --draw-bbox \
  --resume \
  "$@"


# python "${ROOT}/main/run_all.py" \
#   --preset grid \
#   --task-set bimanual \
#   --tasks all \
#   --embodiments all \
#   --obstacle-modes none,off_path,on_path \
#   --place-mode geometric \
#   --plan-mode ignore_obstacle \
#   --episodes 1 \
#   --base-seed 0 \
#   --output "${ROOT}/outputs/ihab/safe_unsafe_bimanual" \
#   --draw-bbox \
#   --resume \
#   "$@"
