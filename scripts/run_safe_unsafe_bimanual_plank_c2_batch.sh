#!/usr/bin/env bash
# C2 / SLURM copy of run_safe_unsafe_bimanual_plank.sh (general-gpu).
#
# Same-scene latency probe: vanilla play_once vs receding-horizon CuRobo (K=60).
# Embodiment is outer: Aloha runs every task (place_container_plate first),
# then piper, then the rest.
#
# Submit on compute2:
#   sbatch CEHJ/scripts/run_safe_unsafe_bimanual_plank_c2.sh
#
# Interactive (already on a general-gpu allocation):
#   bash CEHJ/scripts/run_safe_unsafe_bimanual_plank_c2.sh --episodes 1
#
# Extra args are forwarded to run_all.py.

#SBATCH --job-name=plank_bimanual_c2
#SBATCH -A compute2-sibai
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=64000
#SBATCH -t 1-00:00:00
#SBATCH --container-image=yangyuxuan123/robot-sim:latest
#SBATCH --container-mounts="/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active,/scratch1/fs1/sibai:/scratch1/fs1/sibai"
#SBATCH --output=/storage1/fs1/sibai/Active/yuxuan/cross_embodiment/CEHJ/outputs/ihabnew/safe_unsafe_bimanual_drinkobstacle_testing_trajs/logs/slurm_%j.out
#SBATCH --error=/storage1/fs1/sibai/Active/yuxuan/cross_embodiment/CEHJ/outputs/ihabnew/safe_unsafe_bimanual_drinkobstacle_testing_trajs/logs/slurm_%j.err

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
  --obstacle-modes on_path \
  --obstacle-model 068_boxdrink \
  --place-mode geometric \
  --plan-mode ignore_obstacle \
  --episodes 4 \
  --base-seed 12 \
  --draw-bbox \
  --max-steps 4000 \
  --no-mpc-windows \
  --resume \
  --controllers vanilla_play_once \
  --output "${OUT_ROOT}" \
  "$@"

python "${ROOT}/main/aggregate_plank.py" "${OUT_ROOT}"
echo "Done. Compare ${OUT_ROOT}/latency_compare.csv"
echo "Each scene ran vanilla then K=60 before the next scene."


# python "${ROOT}/main/run_all.py" \
#   --preset grid \
#   --task-set bimanual \
#   --tasks all \
#   --embodiments aloha-agilex,piper,ARX-X5,franka-panda,ur5-wsg \
#   --obstacle-modes on_path \
#   --obstacle-model 068_boxdrink \
#   --place-mode geometric \
#   --plan-mode ignore_obstacle \
#   --episodes 4 \
#   --base-seed 12 \
#   --draw-bbox \
#   --max-steps 4000 \
#   --no-mpc-windows \
#   --resume \
#   --controllers vanilla_play_once,plan_play_once_everyk \
#   --replan-ks 60 \
#   --output "${OUT_ROOT}" \
#   "$@"
