#!/usr/bin/env bash
# run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]  -> <out_dir>/metric.json
# Step 1 (timed, budgeted): fixed training loop driven by solution/scheduler.py. Step 2: separate test eval.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOL="$(cd "$1" && pwd)"; OUT="$2"; MODE=full
case "${3:-}" in --smoke) MODE=smoke;; --cpu-smoke) MODE=cpu-smoke;; "") ;; *) echo "bad flag $3"; exit 2;; esac
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
[ -f /home/users/kg349/AUTORESEARCHPREVIEW/env.sh ] && [ -z "${VIRTUAL_ENV:-}" ] && . /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
export AR_SEED="${AR_SEED:-0}"
if [ "$MODE" = cpu-smoke ]; then export CUDA_VISIBLE_DEVICES=""; fi
BUDGET=$(python -c "import sys; sys.path.insert(0,'$HERE'); import common; print(int(common.MODES['$MODE']['budget_s']))")
T0=$(date +%s.%N)
timeout -k 10 $((BUDGET + 15)) python "$HERE/train.py" --solution "$SOL" --out "$OUT" --mode "$MODE" --t0 "$T0" 2>&1 | tee "$OUT/train.log"
RC=${PIPESTATUS[0]}
WALL=$(python -c "print($(date +%s.%N) - $T0)")
python "$HERE/eval.py" --solution "$SOL" --out "$OUT" --mode "$MODE" --train-rc "$RC" --wall "$WALL" 2>&1 | tee "$OUT/eval.log"
