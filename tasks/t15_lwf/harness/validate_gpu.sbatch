#!/bin/bash
# Harness-prep validation: run.sh in MODE (smoke|full) on the starter, with nvidia-smi dmon logged.
# sbatch --export=ALL,MODE=smoke,SOL=<dir>,OUT=<dir> -t 00:06:00 validate_gpu.sbatch
#SBATCH -J t15_val
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /usr/xtmp/kg349/ar/logs/t15_val_%j.out
set -u
source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
H=$AR_REPO/tasks/t15_lwf/harness
mkdir -p "$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader
nvidia-smi dmon -s um -d 5 > "$OUT/dmon.log" 2>&1 &
DMON=$!
if [ "$MODE" = smoke ]; then bash $H/run.sh "$SOL" "$OUT" --smoke; else bash $H/run.sh "$SOL" "$OUT"; fi
kill $DMON
cat "$OUT/metric.json"
