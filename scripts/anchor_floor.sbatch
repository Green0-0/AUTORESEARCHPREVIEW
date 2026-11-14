#!/bin/bash
# Floor anchors + seed noise: run each task's starter, full mode, seeds 0-2.
# Array index i -> task = TASKS[i / 3], seed = i % 3. Each element does one harness run, then exits.
#   sbatch --array=0-20%4 scripts/anchor_floor.sbatch
#SBATCH -J ar-floor
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -o /usr/xtmp/kg349/ar/logs/floor_%A_%a.out

set -u
source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
TASKS=(t07_activation_tuning t10_label_noise t15_lwf t13_robust_noadv t16_synthetic_data t18_anytime t19_data_ordering t02_self_reg t09_tokenizer_transfer t11_fewshot_seg)   # index 21-23 = t02, 24-26 = t09, 27-29 = t11
i=${SLURM_ARRAY_TASK_ID}
T=${TASKS[$((i / 3))]}
SEED=$((i % 3))
BASE=$AR_ROOT/anchors/floor/$T/seed$SEED
rm -rf "$BASE"; mkdir -p "$BASE"
cp -r "$AR_REPO/tasks/$T/starter" "$BASE/solution"     # copy: harness may forbid writes into solution/
echo "task $T seed $SEED node $(hostname) $(date)"
nvidia-smi dmon -s um -d 5 > "$BASE/dmon.log" 2>&1 &
DMON=$!
AR_SEED=$SEED bash "$AR_REPO/tasks/$T/harness/run.sh" "$BASE/solution" "$BASE/out" > "$BASE/run.log" 2>&1
echo "rc=$?"
kill $DMON 2>/dev/null
cat "$BASE/out/metric.json" 2>/dev/null
echo "done $(date)"
