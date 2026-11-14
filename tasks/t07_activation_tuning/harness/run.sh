#!/bin/bash
# usage: run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]   -> <out_dir>/metric.json
# full: 15-min training budget on 1 A5000 (train() call), whole run hard-capped by `timeout`.
set -u
SOL=${1:?solution_dir}; OUT=${2:?out_dir}; FLAG=${3:-}
case "$FLAG" in
  "") MODE=full; CAP=1080 ;;          # 900 s train + data load + eval
  --smoke) MODE=smoke; CAP=150 ;;
  --cpu-smoke) MODE=cpu-smoke; CAP=180 ;;
  *) echo "unknown flag $FLAG" >&2; exit 2 ;;
esac
if [ -z "${AR_DATA:-}" ] && [ -f /home/users/kg349/AUTORESEARCHPREVIEW/env.sh ]; then
  source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
fi
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"; rm -f "$OUT/metric.json" "$OUT/train_report.json"
T0=$(date +%s.%N)
if [ "$MODE" = cpu-smoke ]; then export CUDA_VISIBLE_DEVICES=""; fi
timeout --signal=TERM --kill-after=10 $CAP python "$HERE/train_wrapper.py" "$SOL" "$OUT" "$MODE"
RC=$?
if [ $RC -eq 124 ] || [ $RC -eq 137 ]; then
  echo "training step killed by timeout (${CAP}s)" >&2
  python - "$OUT" "$MODE" <<'EOF'
import json, sys, pathlib
o = pathlib.Path(sys.argv[1]); p = o / "train_report.json"
r = json.loads(p.read_text()) if p.exists() else {}
r.update(ok=False, error="hard timeout", within_budget=False); p.write_text(json.dumps(r))
EOF
fi
timeout 300 python "$HERE/evaluate.py" "$OUT" "$MODE" "$T0"
if [ ! -f "$OUT/metric.json" ]; then
  echo "{\"metric\": \"cifar100_test_top1\", \"value\": 0.0, \"valid\": false, \"constraint_report\": {\"error\": \"eval failed\"}, \"wall_s\": 0, \"mode\": \"$MODE\"}" > "$OUT/metric.json"
fi
cat "$OUT/metric.json" | python -c "import json,sys; d=json.load(sys.stdin); print('METRIC', d['value'], 'valid', d['valid'])"
