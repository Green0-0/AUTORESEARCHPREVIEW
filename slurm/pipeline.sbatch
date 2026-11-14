#!/bin/bash
# Pipeline job (plan §2): one fitz node, 4x A5000. Two vLLM replicas (PP=2) on :8000 (GPUs 0,1) and
# :8001 (GPUs 2,3), then the orchestrator for every config given. Which replica a run uses is set by
# its config's llm.base_url (arm A -> :8000, arm B -> :8001).
#
#   sbatch slurm/pipeline.sbatch [orchestrator flags] configs/a.yaml configs/b.yaml ...
#   e.g. sbatch slurm/pipeline.sbatch --resume configs/t19_optimized_s1.yaml configs/t19_naive_s1.yaml
#   one replica only (2 GPUs):  sbatch --gres=gpu:a5000:2 --export=ALL,REPLICAS=1 slurm/pipeline.sbatch cfg.yaml
#
# No idle GPUs (§2.1): aborts at once if a vLLM server dies during startup; slurm/watchdog.py stops
# the job if a replica has no running requests for 10 min, a server dies, or the orchestrator exits;
# the job ends as soon as the orchestrator exits (max_hours / max_ideas / signal).
# Set -t to the runs' stop.max_hours + ~1 h (startup + drain). --signal gives the orchestrator a
# graceful SIGTERM 10 min before the wall-clock limit.
#SBATCH -J ar-pipeline
#SBATCH -p compsci-gpu
#SBATCH -N 1
#SBATCH --gres=gpu:a5000:4
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 26:00:00
#SBATCH --signal=B:TERM@600
#SBATCH -o /usr/xtmp/kg349/ar/logs/pipeline_%j.out

set -u
source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
JOB=${SLURM_JOB_ID:-local$$}
OUT=$AR_ROOT/logs/pipeline_$JOB
mkdir -p "$OUT"
REPLICAS=${REPLICAS:-2}
IDLE_MIN=${IDLE_MIN:-10}

# ---- orchestrator args: *.yaml/*.yml -> --config X; everything else passed through
ORCH_ARGS=()
NCFG=0
for a in "$@"; do
  case "$a" in
    *.yaml|*.yml) ORCH_ARGS+=(--config "$a"); NCFG=$((NCFG+1));;
    *) ORCH_ARGS+=("$a");;
  esac
done
if [ $NCFG -eq 0 ]; then echo "no config given"; exit 2; fi

echo "node $(hostname)  job $JOB  $(date)  replicas $REPLICAS  args: ${ORCH_ARGS[*]}"
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
[ "$REPLICAS" -ge 2 ] && serve 2,3 8001

ORCH=""
WD=""
cleanup() {
  trap - TERM INT EXIT
  [ -n "$WD" ] && kill $WD 2>/dev/null
  if [ -n "$ORCH" ] && kill -0 $ORCH 2>/dev/null; then
    kill -TERM $ORCH 2>/dev/null
    for i in $(seq 1 120); do kill -0 $ORCH 2>/dev/null || break; sleep 2; done
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

# ---- orchestrator + watchdog
cd "$AR_REPO"
python -m ar.orchestrator "${ORCH_ARGS[@]}" > "$OUT/orchestrator.log" 2>&1 &
ORCH=$!
python "$AR_REPO/slurm/watchdog.py" --ports "${PORTS[@]}" --orch-pid $ORCH --vllm-pids "${VLLM_PIDS[@]}" \
  --idle-minutes $IDLE_MIN --interval 30 --log "$OUT/watchdog.jsonl" > "$OUT/watchdog.log" 2>&1 &
WD=$!
echo "orchestrator pid $ORCH  watchdog pid $WD  logs $OUT"

# `wait` returns early when a trapped signal arrives; the trap handles that path.
wait $ORCH
RC=$?
echo "orchestrator exited rc=$RC $(date)"
ORCH=""
cleanup
exit $RC
