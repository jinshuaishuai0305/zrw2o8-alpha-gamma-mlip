#!/bin/bash
#SBATCH --account=tyut-youzhiyong
#SBATCH --partition=4V100
#SBATCH --job-name=conv_v3b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=improper-gpu
#SBATCH --time=00:20:00
#SBATCH --output=/home/jinshuaishuai/workdir_tyut-youzhiyong/jinshuaishuai/AlZrW2O8_MACE/conv_v3b-%j.out
set -eo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT"
source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash
source /etc/profile.d/lmod.sh 2>/dev/null || true
source /opt/envs/mace0.3.15.env
mace_create_lammps_model mace_train/AlZrW_v3b_stagetwo.model --format=mliap
ls -la mace_train/AlZrW_v3b_stagetwo.model-mliap_lammps.pt
