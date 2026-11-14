#!/bin/bash
# Harness validation inside the real harness jail (realjudge/jail.sh, JAIL_MODE=harness: read-only FS, no network).
#   sbatch --export=ALL,MODE=cpu-smoke,SOL=<dir>,BASE=<dir> -c 4 -t 00:05:00 validate_cpu.sbatch
#   sbatch --array=0-2 --export=ALL,MODE=full,SOL=<dir>,BASE=<base> -t 00:40:00 validate_cpu.sbatch   (BASE/seed<i>)
# Optional: HARNESS=<dir> (default: the repo's tasks/t04_tiny_walker/harness), JAIL=claude (decoy private data).
# CPU use: `sacct -j <job> -o JobID,Elapsed,TotalCPU,AllocCPUS`.
#SBATCH -J t04_val
#SBATCH -p compsci
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -o /usr/xtmp/kg349/ar/logs/t04_val_%A_%a.out
set -u
source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then export AR_SEED=$SLURM_ARRAY_TASK_ID; BASE=$BASE/seed$AR_SEED; fi
HARNESS=${HARNESS:-$AR_REPO/tasks/t04_tiny_walker/harness}
rm -rf "$BASE"; mkdir -p "$BASE"
cp -r "$HARNESS" "$BASE/harness"; chmod -R a-w "$BASE/harness"
cp -r "$SOL" "$BASE/solution"
echo "node $(hostname) seed ${AR_SEED:-0} mode $MODE jail ${JAIL:-harness} cpus $(nproc) $(date)"
F=""; [ "$MODE" = smoke ] && F=--smoke; [ "$MODE" = cpu-smoke ] && F=--cpu-smoke
JAIL_MODE=${JAIL:-harness} bash "$AR_REPO/realjudge/jail.sh" "$BASE" bash "$BASE/harness/run.sh" "$BASE/solution" "$BASE/out" $F
head -c 400 "$BASE/out/metric.json"; echo
python -c "import json,sys; r=json.load(open(sys.argv[1])); print({k: r.get(k) for k in ('train_wall_s','cpu_util_reaped','n_cpus')})" "$BASE/out/train_report.json"
