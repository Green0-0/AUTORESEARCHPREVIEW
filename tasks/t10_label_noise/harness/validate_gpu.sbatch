#!/bin/bash
# Harness validation (not used by the pipeline): starter --smoke then ONE full run, with dmon logged.
#SBATCH -J t10-val
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:25:00
#SBATCH -o /usr/xtmp/kg349/ar/logs/t10_val_%j.out
set -u
source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
T=/home/users/kg349/AUTORESEARCHPREVIEW/tasks/t10_label_noise
OUT=$AR_ROOT/tmp/t10/gpu_$SLURM_JOB_ID; mkdir -p $OUT
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
nvidia-smi dmon -s um -d 5 > $OUT/dmon.log 2>&1 & DMON=$!
date; $T/harness/run.sh $T/starter $OUT/smoke --smoke
date; $T/harness/run.sh $T/starter $OUT/full
date; kill $DMON
