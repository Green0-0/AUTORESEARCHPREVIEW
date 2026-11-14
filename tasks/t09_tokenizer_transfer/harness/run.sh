#!/bin/bash
# usage: run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]   -> <out_dir>/metric.json
# full: 15-min budget for the solution's adapt() call on 1 A5000 (everything the solution does is inside it);
# the adaptation process is hard-capped by `timeout`; eval runs in a separate process with its own cap.
# Runs fully offline (no network in the harness jail): all data / weights come from $AR_DATA/t09_tokenizer_transfer.
set -u
SOL=$(realpath "${1:?solution_dir}"); OUT=$(realpath -m "${2:?out_dir}"); FLAG=${3:-}
case "$FLAG" in
  "")          MODE=full;      CAP=1080; EVAL_CAP=300 ;;   # adapt() 900 s + data load (3-55 s on NFS) + save
  --smoke)     MODE=smoke;     CAP=150;  EVAL_CAP=120 ;;   # adapt() 45 s
  --cpu-smoke) MODE=cpu-smoke; CAP=120;  EVAL_CAP=120 ;;   # adapt() 40 s, 2M train tokens, 6 eval windows
  *) echo "unknown flag $FLAG" >&2; exit 2 ;;
esac
if ! python -c "import torch" 2>/dev/null && [ -f /home/users/kg349/AUTORESEARCHPREVIEW/env.sh ]; then
  source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
fi
export AR_DATA=${AR_DATA:-/usr/xtmp/kg349/ar/data}
export AR_SEED=${AR_SEED:-0}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1   # never write __pycache__ into the solution or harness dirs
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$AR_DATA/t09_tokenizer_transfer/train_ids.npy" ] || { echo "missing $AR_DATA/t09_tokenizer_transfer: run prepare_data.py" >&2; exit 2; }
mkdir -p "$OUT"; rm -rf "$OUT/artifact" "$OUT/scratch" "$OUT/metric.json" "$OUT/train_report.json" "$OUT/scan.json"
# writable caches inside the out dir (the rest of the filesystem is read-only in the jail)
mkdir -p "$OUT/scratch/tmp"
export TMPDIR=$OUT/scratch/tmp TRITON_CACHE_DIR=$OUT/scratch/triton TORCHINDUCTOR_CACHE_DIR=$OUT/scratch/inductor
export XDG_CACHE_HOME=$OUT/scratch/cache
T0=$(date +%s.%N)
python "$HERE/check.py" "$SOL" > "$OUT/scan.json"
if [ "$MODE" = cpu-smoke ]; then export CUDA_VISIBLE_DEVICES=""; export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}; fi
echo "[run.sh] mode=$MODE cap=${CAP}s seed=$AR_SEED"
timeout --signal=TERM --kill-after=10 $CAP python "$HERE/train_wrapper.py" "$SOL" "$OUT" "$MODE" > "$OUT/train.log" 2>&1
RC=$?
echo "[run.sh] adapt rc=$RC"; tail -n 4 "$OUT/train.log"
if [ $RC -eq 124 ] || [ $RC -eq 137 ]; then
  echo "adaptation step killed by timeout (${CAP}s)" >&2
  python - "$OUT" <<'PY'
import json, sys, pathlib
o = pathlib.Path(sys.argv[1]); p = o / "train_report.json"
r = json.loads(p.read_text()) if p.exists() else {}
r.update(ok=False, error="hard timeout", within_budget=False); p.write_text(json.dumps(r))
PY
fi
rm -rf "$OUT/scratch"
timeout --kill-after=10 $EVAL_CAP python "$HERE/evaluate.py" "$OUT" "$MODE" "$T0" 2>&1 | tee "$OUT/eval.log"
if [ ! -f "$OUT/metric.json" ]; then
  echo "{\"metric\": \"heldout_ppl_per_token\", \"value\": 1e9, \"valid\": false, \"constraint_report\": {\"errors\": [\"eval crashed or timed out (see eval.log)\"]}, \"wall_s\": 0, \"mode\": \"$MODE\"}" > "$OUT/metric.json"
fi
python -c "import json,sys; d=json.load(open(sys.argv[1])); print('METRIC', d['value'], 'valid', d['valid'])" "$OUT/metric.json"
