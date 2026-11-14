#!/bin/bash
# usage: run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]  -> <out_dir>/metric.json
set -u
SOL=$(realpath "$1"); OUT=$(realpath -m "$2"); FLAG=${3:-}
H=$(dirname "$(realpath "$0")")
case "$FLAG" in
  "")          MODE=full;      BUDGET=1200; EVAL_T=900; DEV=cuda ;;
  --smoke)     MODE=smoke;     BUDGET=75;   EVAL_T=60;  DEV=cuda ;;
  --cpu-smoke) MODE=cpu-smoke; BUDGET=90;   EVAL_T=80;  DEV=cpu  ;;
  *) echo "unknown flag $FLAG"; exit 2 ;;
esac
if ! python -c "import torch" 2>/dev/null && [ -f /home/users/kg349/AUTORESEARCHPREVIEW/env.sh ]; then
  source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
fi
export AR_DATA=${AR_DATA:-/usr/xtmp/kg349/ar/data}
export AR_SEED=${AR_SEED:-0}
TD=$AR_DATA/t13_robust_noadv
TRAIN=$TD/train.npz; [ $MODE = cpu-smoke ] && TRAIN=$TD/train_tiny.npz
[ -f "$TRAIN" ] || { echo "missing $TRAIN: run prepare_data.py"; exit 2; }
mkdir -p "$OUT"; rm -rf "$OUT/artifact" "$OUT/metric.json"; mkdir -p "$OUT/artifact"
T_START=$(date +%s.%N)

python "$H/check.py" "$SOL" > "$OUT/scan_pre.json"

echo "[run.sh] mode=$MODE budget=${BUDGET}s train=$TRAIN"
T0=$(date +%s.%N)
if [ $MODE = cpu-smoke ]; then
  ( cd "$SOL" && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 timeout --kill-after=10 $BUDGET \
      python "$SOL/train.py" --train "$TRAIN" --out "$OUT/artifact" --mode $MODE --budget-s $BUDGET --device $DEV --seed $AR_SEED ) \
      > "$OUT/train.log" 2>&1
else
  ( cd "$SOL" && timeout --kill-after=10 $BUDGET \
      python "$SOL/train.py" --train "$TRAIN" --out "$OUT/artifact" --mode $MODE --budget-s $BUDGET --device $DEV --seed $AR_SEED ) \
      > "$OUT/train.log" 2>&1
fi
RC=$?
TW=$(python -c "import time;print(time.time()-$T0)")
echo "[run.sh] train rc=$RC wall=${TW}s"; tail -n 3 "$OUT/train.log"

# rescan including the artifact (model.py is executed at eval time)
python "$H/check.py" "$SOL" "$OUT/artifact" > "$OUT/scan.json"
[ $MODE = cpu-smoke ] && export CUDA_VISIBLE_DEVICES=""
timeout --kill-after=10 $EVAL_T python "$H/evaluate.py" --artifact "$OUT/artifact" --out "$OUT" --mode $MODE \
  --train-rc $RC --train-wall $TW --budget-s $BUDGET --scan "$OUT/scan.json" --t-start $T_START 2>&1 | tee "$OUT/eval.log"
if [ ! -f "$OUT/metric.json" ]; then
  python - "$OUT" "$MODE" "$RC" "$TW" <<'PY'
import json, sys
out, mode, rc, tw = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
json.dump({"metric": "pgd20_robust_acc", "value": 0.0, "valid": False, "wall_s": tw, "mode": mode,
           "constraint_report": {"errors": ["evaluation crashed or timed out (see eval.log)"], "train_rc": rc}},
          open(f"{out}/metric.json", "w"), indent=1)
PY
fi
