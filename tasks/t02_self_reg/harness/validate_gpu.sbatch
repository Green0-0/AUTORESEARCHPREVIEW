#!/bin/bash
# Harness validation: run.sh on a solution in MODE (smoke|full) with nvidia-smi dmon logged.
#   sbatch --export=ALL,MODE=smoke,SOL=<dir>,OUT=<dir> -t 00:06:00 validate_gpu.sbatch
#   sbatch --array=0-2 --export=ALL,MODE=full,SOL=<dir>,OUT=<base> -t 00:25:00 validate_gpu.sbatch   (OUT/seed<i>)
#SBATCH -J t02_val
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /usr/xtmp/kg349/ar/logs/t02_val_%A_%a.out
set -u
source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
H=$AR_REPO/tasks/t02_self_reg/harness
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then export AR_SEED=$SLURM_ARRAY_TASK_ID; OUT=$OUT/seed$AR_SEED; fi
mkdir -p "$OUT"
echo "node $(hostname) seed ${AR_SEED:-0} $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader
nvidia-smi dmon -s um -d 5 > "$OUT/dmon.log" 2>&1 &
DMON=$!
if [ "$MODE" = smoke ]; then bash $H/run.sh "$SOL" "$OUT" --smoke; else bash $H/run.sh "$SOL" "$OUT"; fi
kill $DMON
cat "$OUT/metric.json"
