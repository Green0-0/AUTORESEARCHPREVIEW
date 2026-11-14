#!/bin/bash
# Harness validation inside the real harness jail (realjudge/jail.sh, JAIL_MODE=harness: read-only FS, no network),
# with nvidia-smi dmon logged outside the jail.
#   sbatch --export=ALL,MODE=smoke,SOL=<dir>,BASE=<dir> -t 00:08:00 validate_gpu.sbatch
#   sbatch --array=0-2 --export=ALL,MODE=full,SOL=<dir>,BASE=<base> -t 00:25:00 validate_gpu.sbatch   (BASE/seed<i>)
#SBATCH -J t09_val
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -o /usr/xtmp/kg349/ar/logs/t09_val_%A_%a.out
set -u
source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then export AR_SEED=$SLURM_ARRAY_TASK_ID; BASE=$BASE/seed$AR_SEED; fi
rm -rf "$BASE"; mkdir -p "$BASE"
cp -r "$AR_REPO/tasks/t09_tokenizer_transfer/harness" "$BASE/harness"; chmod -R a-w "$BASE/harness"
cp -r "$SOL" "$BASE/solution"
echo "node $(hostname) seed ${AR_SEED:-0} mode $MODE $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader
nvidia-smi dmon -s um -d 5 > "$BASE/dmon.log" 2>&1 &
DMON=$!
F=""; [ "$MODE" = smoke ] && F=--smoke
JAIL_MODE=harness bash "$AR_REPO/realjudge/jail.sh" "$BASE" bash "$BASE/harness/run.sh" "$BASE/solution" "$BASE/out" $F
kill $DMON
cat "$BASE/out/metric.json" | head -c 600; echo
