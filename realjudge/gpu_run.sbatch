#!/bin/bash
# Real-eval step 2 (GPU): ONLY the full harness run. No Claude, no polling; exits when run.sh exits.
# usage: sbatch -t <budget+slack> gpu_run.sbatch <idea_dir> <out_name>
#SBATCH -J rj-gpu
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:40:00
set -u
source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
IDEA_DIR=${1:?idea dir}
OUT_NAME=${2:-out}
cd "$IDEA_DIR" || exit 2
mkdir -p logs "$OUT_NAME"
echo "rj-gpu node $(hostname) job ${SLURM_JOB_ID:-} $(date) dir $IDEA_DIR out $OUT_NAME"
nvidia-smi dmon -s um -d 5 > "logs/dmon_${OUT_NAME}.log" 2>&1 &
DMON=$!
export AR_SEED=${AR_SEED:-0}
T0=$(date +%s)
mkdir -p .tmp
# solution code runs jailed: read-only filesystem except this idea dir (see realjudge/jail.sh)
JAIL_MODE=harness TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.tmp/cache" TRITON_CACHE_DIR="$PWD/.tmp/triton" \
  bash "$AR_REPO/realjudge/jail.sh" "$PWD" bash harness/run.sh solution "$OUT_NAME"
rc=$?
T1=$(date +%s)
kill $DMON 2>/dev/null
printf '{"rc": %d, "start": %d, "end": %d, "host": "%s", "job": "%s"}\n' $rc $T0 $T1 "$(hostname)" "${SLURM_JOB_ID:-}" > "$OUT_NAME/gpu_job.json"
echo "rj-gpu done rc=$rc wall=$((T1-T0))s $(date)"
exit $rc
