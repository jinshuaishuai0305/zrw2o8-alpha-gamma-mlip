#!/bin/bash
# ABACUS 批量单点：每个数组下标一个目录。1 rank / 1 GPU。
#SBATCH --job-name=zrwif
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=improper-gpu
#SBATCH --time=04:00:00
#SBATCH --output=../logs/%x-%A_%a.out
#SBATCH --error=../logs/%x-%A_%a.err
set -eo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT"
ALLOWED_ROOTS="/home/jinshuaishuai/workdir_tyut-youzhiyong/jinshuaishuai /org/tyut-youzhiyong/jinshuaishuai"
_ok=0
for _r in $ALLOWED_ROOTS; do case "$ROOT" in "$_r"|"$_r"/*) _ok=1; break ;; esac; done
if [ "$_ok" -ne 1 ]; then echo "FATAL: outside work directory: $ROOT" >&2; exit 3; fi
source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load abacus/LTSv3.10.1-sm70-auto
RUN=(mpirun -np ${SLURM_NTASKS} --map-by ${MAP_OPT} abacus)
SCANROOT=${SCANROOT:?set SCANROOT to the directory holding the numbered job dirs}
mapfile -t DIRS < <(find "$SCANROOT" -mindepth 1 -maxdepth 1 -type d | sort)
D="${DIRS[$SLURM_ARRAY_TASK_ID]}"
echo "task=${SLURM_ARRAY_TASK_ID} job=${SLURM_JOB_ID} dir=$D host=$(hostname)"
cd "$D"
"${RUN[@]}"
grep -h "FINAL_ETOT_IS" OUT.*/*.log 2>/dev/null | tail -1
if grep -q "charge density convergence is achieved" OUT.*/*.log 2>/dev/null; then
  echo "STATUS: CONVERGED $(basename "$D")"
else
  echo "STATUS: NOT_CONVERGED $(basename "$D")"; exit 1
fi
