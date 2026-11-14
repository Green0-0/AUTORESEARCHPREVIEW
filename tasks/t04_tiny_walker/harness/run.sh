#!/bin/bash
# usage: run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]   -> <out_dir>/metric.json
# CPU-only task. full: the solution's train(ctx) call gets 10 min on 16 CPU cores (everything it does counts);
# the training process is hard-capped by `timeout`; eval runs in a separate process with its own cap.
# Runs fully offline (no network in the harness jail). Needs gymnasium[box2d] in the python env.
set -u
SOL=$(realpath "${1:?solution_dir}"); OUT=$(realpath -m "${2:?out_dir}"); FLAG=${3:-}
case "$FLAG" in
  "")          MODE=full;      CAP=690;  EVAL_CAP=300 ;;   # train() 600 s on 16 cores
  --smoke)     MODE=smoke;     CAP=105;  EVAL_CAP=120 ;;   # train() 60 s
  --cpu-smoke) MODE=cpu-smoke; CAP=75;   EVAL_CAP=90  ;;   # train() 30 s on 4 cores, 8 eval episodes
  *) echo "unknown flag $FLAG" >&2; exit 2 ;;
esac
if ! python -c "import gymnasium, Box2D" 2>/dev/null && [ -f /home/users/kg349/AUTORESEARCHPREVIEW/env.sh ]; then
  source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
fi
export AR_SEED=${AR_SEED:-0}
export PYTHONDONTWRITEBYTECODE=1   # never write __pycache__ into the solution or harness dirs
# one BLAS/OpenMP thread per process: parallelism comes from the solution's worker processes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"; rm -rf "$OUT/artifact" "$OUT/scratch" "$OUT/metric.json" "$OUT/train_report.json" "$OUT/scan.json"
mkdir -p "$OUT/scratch/tmp"
export TMPDIR=$OUT/scratch/tmp XDG_CACHE_HOME=$OUT/scratch/cache
T0=$(date +%s.%N)
python "$HERE/check.py" "$SOL" > "$OUT/scan.json"
echo "[run.sh] mode=$MODE cap=${CAP}s seed=$AR_SEED cpus=$(python -c 'import os; print(len(os.sched_getaffinity(0)))')"
timeout --signal=TERM --kill-after=10 $CAP python "$HERE/train_wrapper.py" "$SOL" "$OUT" "$MODE" > "$OUT/train.log" 2>&1
RC=$?
echo "[run.sh] train rc=$RC"; tail -n 4 "$OUT/train.log"
if [ $RC -eq 124 ] || [ $RC -eq 137 ]; then
  echo "training step killed by timeout (${CAP}s)" >&2
  python - "$OUT" <<'PY'
import json, sys, pathlib
o = pathlib.Path(sys.argv[1]); p = o / "train_report.json"
r = json.loads(p.read_text()) if p.exists() else {}
r.update(ok=False, error="hard timeout", within_budget=False); p.write_text(json.dumps(r))
PY
fi
rm -rf "$OUT/scratch"
export TMPDIR=$OUT
timeout --kill-after=10 $EVAL_CAP python "$HERE/evaluate.py" "$OUT" "$MODE" "$T0" "$SOL" 2>&1 | tee "$OUT/eval.log"
if [ ! -f "$OUT/metric.json" ]; then
  echo "{\"metric\": \"heldout_mean_return\", \"value\": -1e9, \"valid\": false, \"constraint_report\": {\"errors\": [\"eval crashed or timed out (see eval.log)\"]}, \"wall_s\": 0, \"mode\": \"$MODE\"}" > "$OUT/metric.json"
fi
python -c "import json,sys; d=json.load(open(sys.argv[1])); print('METRIC', d['value'], 'valid', d['valid'])" "$OUT/metric.json"
