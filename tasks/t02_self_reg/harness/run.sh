#!/bin/bash
# usage: run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]   -> <out_dir>/metric.json
# full: 15-min budget for the solution's train() call on 1 A5000; the training process is hard-capped by
# `timeout` (budget + startup/save margin), eval runs in a separate process with its own cap.
set -u
SOL=$(realpath "${1:?solution_dir}"); OUT=$(realpath -m "${2:?out_dir}"); FLAG=${3:-}
case "$FLAG" in
  "")          MODE=full;      CAP=960; EVAL_CAP=300 ;;   # train() 900 s + data load / save
  --smoke)     MODE=smoke;     CAP=120; EVAL_CAP=120 ;;   # train() 60 s
  --cpu-smoke) MODE=cpu-smoke; CAP=120; EVAL_CAP=120 ;;   # train() 60 s, 100 train / 500 test images
  *) echo "unknown flag $FLAG" >&2; exit 2 ;;
esac
if ! python -c "import torch" 2>/dev/null && [ -f /home/users/kg349/AUTORESEARCHPREVIEW/env.sh ]; then
  source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
fi
export AR_DATA=${AR_DATA:-/usr/xtmp/kg349/ar/data}
export AR_SEED=${AR_SEED:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONDONTWRITEBYTECODE=1   # never write __pycache__ into the solution or harness dirs
[ -f "$AR_DATA/t02_self_reg/train_x.npy" ] || { echo "missing $AR_DATA/t02_self_reg: run prepare_data.py" >&2; exit 2; }
mkdir -p "$OUT"; rm -rf "$OUT/artifact" "$OUT/metric.json" "$OUT/train_report.json" "$OUT/scan.json"
T0=$(date +%s.%N)
python "$HERE/check.py" "$SOL" > "$OUT/scan.json"
if [ "$MODE" = cpu-smoke ]; then export CUDA_VISIBLE_DEVICES=""; export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}; fi
echo "[run.sh] mode=$MODE cap=${CAP}s seed=$AR_SEED"
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
timeout --kill-after=10 $EVAL_CAP python "$HERE/evaluate.py" "$OUT" "$MODE" "$T0" 2>&1 | tee "$OUT/eval.log"
if [ ! -f "$OUT/metric.json" ]; then
  echo "{\"metric\": \"cifar100_test_top5\", \"value\": 0.0, \"valid\": false, \"constraint_report\": {\"errors\": [\"eval crashed or timed out (see eval.log)\"]}, \"wall_s\": 0, \"mode\": \"$MODE\"}" > "$OUT/metric.json"
fi
python -c "import json,sys; d=json.load(open(sys.argv[1])); print('METRIC', d['value'], 'valid', d['valid'])" "$OUT/metric.json"
