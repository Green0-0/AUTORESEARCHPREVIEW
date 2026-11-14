#!/bin/bash
# Offline judge-strategy study (plan §9.1-9.2): two dedicated vLLM replicas (PP=2) on one 4x A5000 node, then
# analysis/judge_study.py. Both replicas pull from ONE shared work queue (longest prompts first, KV-token budget per
# replica), so both stay busy until the last calls; the thinking phase for the best 1-2 strategies follows the
# no-thinking phase in the same process with no gap. The job ends when the script exits; the watchdog stops it if a
# replica idles IDLE_MIN minutes or a server dies. Resumable: every call attempt is appended to $OUT_DIR/calls.jsonl,
# so a resubmit only makes the missing calls (and re-asks J4r/J5 prompts that changed because the pool grew).
#   sbatch slurm/judge_study.sbatch                                  (default: train pool, all strategies)
#   STUDY_ARGS="--pool test --think-variants J3_facts" OUT_DIR=$AR_ROOT/analysis/judge_study_test sbatch slurm/judge_study.sbatch
# Expected at 60 ideas: ~1000 no-thinking calls (~35-45 min) + thinking for 2 strategies (~20-80 min) + ~3 min start.
#SBATCH -J ar-judgestudy
#SBATCH -p compsci-gpu
#SBATCH -N 1
#SBATCH --gres=gpu:a5000:4
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /usr/xtmp/kg349/ar/logs/judge_study_%j.out
set -u
source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
JOB=${SLURM_JOB_ID:-local$$}
OUT=$AR_ROOT/logs/judge_study_$JOB
OUT_DIR=${OUT_DIR:-$AR_ROOT/analysis/judge_study}
STUDY_ARGS=${STUDY_ARGS:-}
mkdir -p "$OUT" "$OUT_DIR"
IDLE_MIN=${IDLE_MIN:-15}

echo "node $(hostname)  job $JOB  $(date)  out $OUT_DIR  args: $STUDY_ARGS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee "$OUT/gpus.csv"
nvidia-smi dmon -s um -d 5 > "$OUT/dmon.log" 2>&1 &
DMON=$!

VLLM_PIDS=()
PORTS=()
serve() {  # $1 = GPUs, $2 = port   (flags exactly as probes/probe_gpu.sbatch, plan §1.0)
  CUDA_VISIBLE_DEVICES=$1 vllm serve $AR_MODEL --served-model-name qwen \
    --pipeline-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 --enable-prefix-caching \
    --kv-cache-dtype bfloat16 --attention-backend TRITON_ATTN \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --port $2 > "$OUT/vllm_$2.log" 2>&1 &
  VLLM_PIDS+=($!)
  PORTS+=($2)
}
serve 0,1 8000
serve 2,3 8001

ORCH=""
WD=""
cleanup() {
  trap - TERM INT EXIT
  [ -n "$WD" ] && kill $WD 2>/dev/null
  if [ -n "$ORCH" ] && kill -0 $ORCH 2>/dev/null; then
    kill -TERM $ORCH 2>/dev/null
    for i in $(seq 1 60); do kill -0 $ORCH 2>/dev/null || break; sleep 2; done
    kill -KILL $ORCH 2>/dev/null
  fi
  for p in "${VLLM_PIDS[@]}"; do kill -TERM $p 2>/dev/null; done
  for i in $(seq 1 15); do
    alive=0; for p in "${VLLM_PIDS[@]}"; do kill -0 $p 2>/dev/null && alive=1; done
    [ $alive -eq 0 ] && break; sleep 2
  done
  for p in "${VLLM_PIDS[@]}"; do kill -KILL $p 2>/dev/null; done
  kill $DMON 2>/dev/null
  echo "teardown done $(date)"
}
trap 'echo "SIGTERM/INT received $(date)"; cleanup; exit 143' TERM INT
trap cleanup EXIT

# ---- wait for health; abort the moment any server dies (startup ~160 s)
healthy=0
for t in $(seq 1 90); do
  for pid in "${VLLM_PIDS[@]}"; do
    kill -0 $pid 2>/dev/null || { echo "vllm pid $pid died after $((t*10))s; see $OUT/vllm_*.log"; exit 1; }
  done
  ok=1; for p in "${PORTS[@]}"; do curl -sf localhost:$p/health >/dev/null || ok=0; done
  if [ $ok -eq 1 ]; then healthy=1; echo "all replicas healthy after $((t*10))s"; break; fi
  sleep 10
done
[ $healthy -eq 1 ] || { echo "replicas not healthy after 15 min; aborting"; exit 1; }

# ---- study + watchdog
cd "$AR_REPO"
python analysis/judge_study.py --out "$OUT_DIR" --ports "${PORTS[@]}" $STUDY_ARGS > "$OUT/judge_study.log" 2>&1 &
ORCH=$!
python "$AR_REPO/slurm/watchdog.py" --ports "${PORTS[@]}" --orch-pid $ORCH --vllm-pids "${VLLM_PIDS[@]}" \
  --idle-minutes $IDLE_MIN --interval 30 --log "$OUT/watchdog.jsonl" > "$OUT/watchdog.log" 2>&1 &
WD=$!
wait $ORCH
RC=$?
echo "judge_study exited rc=$RC $(date)"
cp "$OUT/judge_study.log" "$OUT_DIR/judge_study_$JOB.log" 2>/dev/null
ORCH=""
cleanup
exit $RC
