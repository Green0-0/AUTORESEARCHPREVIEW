#!/bin/bash
# Floor anchors + seed noise for CPU-only tasks (task.yaml device: cpu): the CPU twin of anchor_floor.sbatch.
# Array index i -> task = TASKS[i / 3], seed = i % 3. Each element does one full harness run, then exits.
#   sbatch --array=0-2 scripts/anchor_floor_cpu.sbatch
# CPU use afterwards: sacct -j <job> -o JobID,Elapsed,TotalCPU,AllocCPUS
#SBATCH -J ar-floor-cpu
#SBATCH -p compsci
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o /usr/xtmp/kg349/ar/logs/floor_cpu_%A_%a.out

set -u
source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
TASKS=(t04_tiny_walker)   # index 0-2 = t04
i=${SLURM_ARRAY_TASK_ID}
T=${TASKS[$((i / 3))]}
SEED=$((i % 3))
BASE=$AR_ROOT/anchors/floor/$T/seed$SEED
rm -rf "$BASE"; mkdir -p "$BASE"
cp -r "$AR_REPO/tasks/$T/starter" "$BASE/solution"     # copy: harness may forbid writes into solution/
echo "task $T seed $SEED node $(hostname) cpus $(nproc) $(date)"
AR_SEED=$SEED bash "$AR_REPO/tasks/$T/harness/run.sh" "$BASE/solution" "$BASE/out" > "$BASE/run.log" 2>&1
echo "rc=$?"
cat "$BASE/out/metric.json" 2>/dev/null | head -c 600; echo
echo "done $(date)"
