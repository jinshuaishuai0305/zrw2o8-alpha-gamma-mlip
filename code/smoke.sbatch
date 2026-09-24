#!/bin/bash
# smoke.sbatch — pre-production acceptance test for ABACUS on sai
# Rendered from templates/smoke.sbatch.tmpl by scripts/render.sh
# Purpose: convert a SILENT failure into a visible one BEFORE spending machine time.
# Submit:  cd <job-dir> && sbatch smoke.sbatch
#SBATCH --job-name=zrwif2-test
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=improper-gpu
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$ROOT"

# ===========================================================================
# WORK-DIRECTORY GUARD — site policy, do not remove.
# Every cluster designates one writable area. Creating job directories, logs or
# checkpoints in $HOME violates that policy; on SAI it has already forced a
# manual cleanup once. This job refuses to start outside the allowed root.
# Set ALLOW_OUTSIDE_WORKDIR=1 only for a deliberate test run.
# ===========================================================================
# The same filesystem is mounted twice (login path and compute-node path), so the
# guard accepts any declared alias. Matching only one form makes every job fail with
# exit 3 on the compute node while it submits fine from the login node.
ALLOWED_ROOTS="/home/jinshuaishuai/workdir_tyut-youzhiyong/jinshuaishuai /org/tyut-youzhiyong/jinshuaishuai"
if [ "${ALLOW_OUTSIDE_WORKDIR:-0}" != "1" ]; then
  _ok=0
  for _r in $ALLOWED_ROOTS; do
    case "$ROOT" in "$_r"|"$_r"/*) _ok=1; break ;; esac
  done
  if [ "$_ok" -ne 1 ]; then
       echo "FATAL: working directory is outside the allowed work directory." >&2
       echo "  cwd     = $ROOT" >&2
       echo "  allowed = $ALLOWED_ROOTS" >&2
       echo '  policy  = write only under WORKDIR; never in the login home directory' >&2
       echo "  Do not create files in \$HOME on the cluster." >&2
       echo "  Check with: scripts/check_workdir.sh \"$ROOT\"" >&2
       exit 3
  fi
fi

source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash
# Lmod is NOT initialised in a batch shell: `module load` would exit 127 before the
# binary ever starts (measured on SAI, job 1431772). Harmless for codes activated by
# sourcing a conda prefix (MACE), essential for every `module load` code.
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load abacus/LTSv3.10.1-sm70-auto
:

RUN=(mpirun -np ${SLURM_NTASKS} --map-by ${MAP_OPT} abacus)

echo "job=${SLURM_JOB_ID} host=$(hostname) partition=${SLURM_JOB_PARTITION} dir=$PWD"
echo "=== files in the working directory ==="
ls -la
echo "=== symlink check (must print nothing) ==="
find . -maxdepth 2 -type l -print

nvidia-smi dmon -s pucvmte -o T > nvdmon_job-${SLURM_JOB_ID}.log &

"${RUN[@]}" 2>&1 | tee smoke.log

echo "=== acceptance checks ==="
rc=0
for pair in "ran:RUNNING WITH DEVICE  : GPU" "solver:DiagoElpa" \
            "converged:charge density convergence is achieved" "energy:FINAL_ETOT_IS"; do
  key="${pair%%:*}"; pat="${pair#*:}"
  # markers may be plain substrings or regexes (e.g. "CUDA version: .*CUDA device:");
  # fixed-string first, then extended regex, so both forms work and neither lies.
  if grep -qF -- "$pat" smoke.log || grep -qE -- "$pat" smoke.log; then
    echo "  PASS  $key  ($pat)"
  else
    echo "  FAIL  $key  ($pat not found)"
    rc=1
  fi
done
[ "$rc" -eq 0 ] && echo "SMOKE: PASS — safe to submit production" \
                || echo "SMOKE: FAIL — do NOT submit production yet"
exit "$rc"
