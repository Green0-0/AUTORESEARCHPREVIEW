#!/bin/bash
# Day-0 probe: 2 vLLM replicas (4-bit Qwen3.8-27B, PP=2 each) + throughput benchmark
# + claude -p check from a compute node. Exits as soon as the benchmark finishes,
# so the GPUs never sit idle.
#SBATCH -J ar-probe
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a5000:4
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 01:15:00
#SBATCH -o /usr/xtmp/kg349/ar/logs/probe_%j.out

set -u
source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh
OUT=$AR_ROOT/logs/probe_$SLURM_JOB_ID
mkdir -p $OUT
echo "node $(hostname)  job $SLURM_JOB_ID  $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee $OUT/gpus.csv
nvidia-smi dmon -s um -d 10 > $OUT/dmon.log 2>&1 &
DMON=$!

serve() {  # $1 = GPUs, $2 = port
  CUDA_VISIBLE_DEVICES=$1 vllm serve $AR_MODEL --served-model-name qwen \
    --pipeline-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 --enable-prefix-caching \
    --kv-cache-dtype bfloat16 --attention-backend TRITON_ATTN \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --port $2 > $OUT/vllm_$2.log 2>&1 &
  PIDS="${PIDS:-} $!"
}
serve 0,1 8000
serve 2,3 8001

# Claude Code check already passed on a CPU node; kept here cheap (1 session) to confirm from a GPU node.
{
  echo "--- anthropic reachability"; curl -sS -o /dev/null -w "%{http_code}\n" https://api.anthropic.com/ 2>&1
  echo "--- 3 concurrent claude -p sessions"
  for i in 1; do
    ( cd $AR_ROOT/tmp && timeout 300 claude -p "Reply with exactly: ok-$i" --model sonnet \
        --max-turns 1 --output-format json > $OUT/claude_$i.json 2> $OUT/claude_$i.err; echo "claude $i exit $?" ) &
  done; wait
} > $OUT/claude_probe.log 2>&1 &
CL=$!

# Wait for health, but bail out the moment any server process dies (no idle GPUs).
for t in $(seq 1 90); do
  for pid in $PIDS; do kill -0 $pid 2>/dev/null || { echo "vllm pid $pid died after $((t*10))s"; break 3; }; done
  curl -sf localhost:8000/health >/dev/null && curl -sf localhost:8001/health >/dev/null && { echo "both healthy after $((t*10))s"; break; }
  sleep 10
done
wait $CL

if curl -sf localhost:8000/health && curl -sf localhost:8001/health; then
  python $AR_REPO/probes/bench_vllm.py --ports 8000 8001 --model qwen --out $OUT/bench.json
else
  echo "a replica failed to come up; see $OUT/vllm_*.log"
fi
kill $DMON; pkill -f "vllm serve" ; sleep 5
echo "done $(date)"
