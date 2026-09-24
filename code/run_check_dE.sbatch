#!/bin/bash
#SBATCH --account=tyut-youzhiyong
#SBATCH --partition=4V100
#SBATCH --job-name=chk_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=improper-gpu
#SBATCH --time=00:20:00
#SBATCH --output=/home/jinshuaishuai/workdir_tyut-youzhiyong/jinshuaishuai/AlZrW2O8_MACE/chk_dE-%j.out
set -eo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT"
source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash
source /etc/profile.d/lmod.sh 2>/dev/null || true
source /opt/envs/mace0.3.15.env
export DEV=cuda
python check_dE.py mace_train/AlZrW_v2_stagetwo.model mace_train/AlZrW_v3_stagetwo.model mace_train/AlZrW_v3b_stagetwo.model
