#!/usr/bin/env bash
# Same-scene latency probe: vanilla play_once vs receding-horizon CuRobo (K=60).
#
# For each (task, embodiment, seed) we run vanilla then K=60 before the next
# scene, so the comparison shares the same layout and a hot CuRobo cache.
#
# 9 bimanual tasks (place_container_plate first), × 5 embodiments
# (aloha-agilex, piper, ARX-X5, franka-panda, ur5-wsg) × 4 seeds × 2 policies.
# Embodiment is outer: Aloha runs every task starting with place_container_plate,
# then piper, then the rest.
#
# Output (same relative episode path under each policy dir):
#   CEHJ/outputs/ihab/safe_unsafe_bimanual_plank/
#     vanilla_play_once/<task>/<emb>/on_path_geometric_ignore_obstacle_tXX_seedY/
#     plan_everyk_k60/...
#     latency_compare.csv
#
# Interactive (inside robot-sim + rbtw128):
#   bash CEHJ/scripts/run_safe_unsafe_bimanual_plank.sh
#   bash CEHJ/scripts/run_safe_unsafe_bimanual_plank.sh --embodiments piper --episodes 1
#   bash CEHJ/scripts/run_safe_unsafe_bimanual_plank.sh --obstacle-model 105_sauce-can
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
#
# LSF:
#   bsub < CEHJ/scripts/run_safe_unsafe_bimanual_plank.sh

#BSUB -J plank_bimanual
#BSUB -q general
#BSUB -G compute-sibai
#BSUB -n 8
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -R "gpuhost"
#BSUB -gpu "num=1"
#BSUB -W 168:00
#BSUB -a "docker(yangyuxuan123/robot-sim:latest)"
#BSUB -env "all,LSF_DOCKER_VOLUMES=/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active /scratch1/fs1/sibai:/scratch1/fs1/sibai,LSF_DOCKER_SHM_SIZE=8g"
#BSUB -o /storage1/fs1/sibai/Active/yuxuan/cross_embodiment/CEHJ/outputs/ihab/safe_unsafe_bimanual_plank/logs/lsf_%J.out
#BSUB -e /storage1/fs1/sibai/Active/yuxuan/cross_embodiment/CEHJ/outputs/ihab/safe_unsafe_bimanual_plank/logs/lsf_%J.err

set -euo pipefail
ROOT="/storage1/fs1/sibai/Active/yuxuan/cross_embodiment/CEHJ"
OUT_ROOT="${ROOT}/outputs/ihabnew/safe_unsafe_bimanual_drinkobstacle_testing_trajs"
mkdir -p "${OUT_ROOT}/logs"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CONDA_ENVS_DIRS="${CONDA_ENVS_DIRS:-/storage1/fs1/sibai/Active/yuxuan/conda/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/storage1/fs1/sibai/Active/yuxuan/conda/pkgs}"

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
  conda activate rbtw128
fi

python "${ROOT}/main/run_all.py" \
  --preset grid \
  --task-set bimanual \
  --tasks all \
  --embodiments aloha-agilex,piper,ARX-X5,franka-panda,ur5-wsg \
  --obstacle-modes on_path,off_path \
  --obstacle-model 068_boxdrink \
  --place-mode geometric \
  --plan-mode ignore_obstacle \
  --episodes 5 \
  --base-seed 12 \
  --draw-bbox \
  --max-steps 5000 \
  --no-mpc-windows \
  --resume \
  --controllers vanilla_play_once \
  --output "${OUT_ROOT}" \
  "$@"



python "${ROOT}/main/aggregate_plank.py" "${OUT_ROOT}"
echo "Done. Compare ${OUT_ROOT}/latency_compare.csv"
echo "Each scene ran vanilla then K=60 before the next scene."
  # --mpc-window-max 3 \
  # --mpc-window-stride 400 \