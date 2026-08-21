#!/usr/bin/env bash
# Same-scene latency probe: vanilla play_once vs receding-horizon CuRobo (K=20,50,70).
#
# For each (task, embodiment, seed) we run vanilla then K=20,50,70 before the next
# scene, so the comparison shares the same layout and a hot CuRobo cache.
#
# 8 bimanual tasks × 4 embodiments × 3 seeds × 4 policies.
# Seed formula matches run_safe_unsafe_bimanual.sh (base-seed 12, episodes 3).
#
# Output (same relative episode path under each policy dir):
#   CEHJ/outputs/ihab/safe_unsafe_bimanual_plank/
#     vanilla_play_once/<task>/<emb>/on_path_geometric_ignore_obstacle_tXX_seedY/
#     plan_everyk_k20/...
#     plan_everyk_k50/...
#     plan_everyk_k70/...
#     latency_compare.csv
#
# Interactive (inside robot-sim + rbtw128):
#   bash CEHJ/scripts/run_safe_unsafe_bimanual_plank.sh
#   bash CEHJ/scripts/run_safe_unsafe_bimanual_plank.sh --embodiments piper --episodes 1
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
OUT_ROOT="${ROOT}/outputs/ihabnew/safe_unsafe_bimanual_plank"
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
  --embodiments all \
  --obstacle-modes on_path \
  --place-mode geometric \
  --plan-mode ignore_obstacle \
  --episodes 1 \
  --base-seed 12 \
  --draw-bbox \
  --resume \
  --controllers vanilla_play_once,plan_play_once_everyk \
  --replan-ks 80,75,70,50,40 \
  --output "${OUT_ROOT}" \
  "$@"


python "${ROOT}/main/aggregate_plank.py" "${OUT_ROOT}"
echo "Done. Compare ${OUT_ROOT}/latency_compare.csv"
echo "Each scene ran vanilla then K=20,50,70 before the next scene."
