#!/bin/bash
# Usage: run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]   -> <out_dir>/metric.json
set -u
if [ $# -lt 2 ]; then echo "usage: $0 <solution_dir> <out_dir> [--smoke|--cpu-smoke]" >&2; exit 2; fi
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOL=$(cd "$1" && pwd) || exit 2
mkdir -p "$2"; OUT=$(cd "$2" && pwd)
MODE=full
case "${3:-}" in --smoke) MODE=smoke;; --cpu-smoke) MODE=cpu-smoke;; "") ;; *) echo "bad flag $3" >&2; exit 2;; esac
[ -z "${VIRTUAL_ENV:-}" ] && source "${AR_REPO:-/home/users/kg349/AUTORESEARCHPREVIEW}/env.sh"
export AR_SEED=${AR_SEED:-0}
DATA=${AR_DATA:-/usr/xtmp/kg349/ar/data}/t18_anytime
case $MODE in
  full)      TRAIN_BUDGET=900; EVAL_BUDGET=900; TRAIN=$DATA/train.npz;;
  smoke)     TRAIN_BUDGET=60;  EVAL_BUDGET=120; TRAIN=$DATA/smoke_train.npz;;
  cpu-smoke) TRAIN_BUDGET=90;  EVAL_BUDGET=80;  TRAIN=$DATA/smoke_train.npz; export CUDA_VISIBLE_DEVICES="";;
esac
rm -rf "$OUT/artifact" "$OUT/metric.json" "$OUT/eval.json"; mkdir -p "$OUT/artifact"
DMON=""
if [ $MODE != cpu-smoke ] && command -v nvidia-smi >/dev/null; then
  nvidia-smi dmon -s um -d 5 > "$OUT/dmon.log" 2>&1 & DMON=$!
  trap '[ -n "$DMON" ] && kill $DMON 2>/dev/null' EXIT
fi
T0=$(date +%s%N)
( cd "$SOL" && exec timeout -k 10 $TRAIN_BUDGET python train.py --train "$TRAIN" --out "$OUT/artifact" \
    --mode $MODE --budget-s $TRAIN_BUDGET ) > "$OUT/train.log" 2>&1
TRAIN_RC=$?
T1=$(date +%s%N)
( cd "$SOL" && exec timeout -k 10 $EVAL_BUDGET python "$HERE/evaluate.py" --solution "$SOL" \
    --artifact "$OUT/artifact" --out "$OUT" --mode $MODE ) > "$OUT/eval.log" 2>&1
EVAL_RC=$?
T2=$(date +%s%N)
python "$HERE/finalize.py" --out "$OUT" --mode $MODE --train-rc $TRAIN_RC --eval-rc $EVAL_RC \
  --train-s $(( (T1 - T0) / 1000000 )) --eval-s $(( (T2 - T1) / 1000000 )) --train-budget $TRAIN_BUDGET
