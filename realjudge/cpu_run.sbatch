#!/bin/bash
# Real-eval step 2 for CPU-only tasks (task.yaml `device: cpu`, `cpus: N`): ONLY the full harness run, no GPU.
# Mirrors gpu_run.sbatch (same jail, same outputs incl. <out>/gpu_job.json) minus the GPU and its dmon monitor.
# usage: sbatch -t <budget+slack> -c <cpus> cpu_run.sbatch <idea_dir> <out_name>
#SBATCH -J rj-cpu
#SBATCH -p compsci
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 00:50:00
set -u
source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
IDEA_DIR=${1:?idea dir}
OUT_NAME=${2:-out}
cd "$IDEA_DIR" || exit 2
mkdir -p logs "$OUT_NAME"
echo "rj-cpu node $(hostname) job ${SLURM_JOB_ID:-} cpus ${SLURM_CPUS_PER_TASK:-?} $(date) dir $IDEA_DIR out $OUT_NAME"
export AR_SEED=${AR_SEED:-0}
T0=$(date +%s)
mkdir -p .tmp
# solution code runs jailed: read-only filesystem except this idea dir (see realjudge/jail.sh)
JAIL_MODE=harness TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.tmp/cache" \
  bash "$AR_REPO/realjudge/jail.sh" "$PWD" bash harness/run.sh solution "$OUT_NAME"
rc=$?
T1=$(date +%s)
printf '{"rc": %d, "start": %d, "end": %d, "host": "%s", "job": "%s", "device": "cpu"}\n' $rc $T0 $T1 "$(hostname)" "${SLURM_JOB_ID:-}" > "$OUT_NAME/gpu_job.json"
echo "rj-cpu done rc=$rc wall=$((T1-T0))s $(date)"
exit $rc
