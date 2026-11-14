#!/bin/bash
# run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]  ->  <out_dir>/metric.json
# Step 1 (train.py): harness-owned task stream, under a hard timeout.  Step 2 (eval.py): separate process.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOL="$(cd "$1" && pwd)"; OUT="$2"; MODE=full
case "${3:-}" in --smoke) MODE=smoke;; --cpu-smoke) MODE=cpu-smoke;; "") ;; *) echo "unknown flag $3"; exit 2;; esac
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
rm -rf "$OUT/artifact" "$OUT/metric.json" "$OUT/train_report.json"
case $MODE in full) TO=1080;; smoke) TO=150;; cpu-smoke) TO=150;; esac   # stream budget + startup/audit slack
if [ "$MODE" = cpu-smoke ]; then export CUDA_VISIBLE_DEVICES=""; fi
START=$(date +%s.%N)
cd "$OUT"
timeout -k 10 $TO python "$HERE/train.py" --solution "$SOL" --out "$OUT" --mode $MODE 2>&1 | tee "$OUT/train.log"
echo "train exit ${PIPESTATUS[0]}" | tee -a "$OUT/train.log"
timeout -k 10 300 python "$HERE/eval.py" --solution "$SOL" --out "$OUT" --mode $MODE --wall-start $START 2>&1 | tee "$OUT/eval.log"
test -f "$OUT/metric.json"
