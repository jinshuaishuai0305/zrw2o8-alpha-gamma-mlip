#!/bin/bash
#SBATCH --job-name=fps
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=improper-gpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/fps-%j.out
set -eo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT"
source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash
source /opt/envs/mace0.3.15.env
python merge_for_v5.py
python fps_select.py v5_fps_part.xyz 561 v5_rest_part.xyz
