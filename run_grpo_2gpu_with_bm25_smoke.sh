#!/usr/bin/env bash
set -eo pipefail
cd /g22538014wej/11x22f/Search-R1
source /opt/conda/etc/profile.d/conda.sh
conda activate /g22538014wej/11x22f/envs/search1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export SEARCH_R1_MEMORY_DEBUG=1
export SEARCH_R1_MEMORY_DIR=/g22538014wej/11x22f/Search-R1/logs/smoke_b8_5u_20261111
export SEARCH_R1_MEMORY_LIMIT_GB=96
gpu_count=$(nvidia-smi -L | wc -l)
if [[ "$gpu_count" -lt 2 ]]; then
  echo "需要至少 2 张可见 GPU，当前检测到 $gpu_count 张。"
  exit 1
fi
retriever_pid=""
cleanup() {
  if [[ -n "$retriever_pid" ]]; then
    kill "$retriever_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
if ! curl -fsS --max-time 2 http://127.0.0.1:8000/docs >/dev/null 2>/dev/null; then
  bash start_bm25_local.sh > logs/smoke_b8_5u_20261111/retriever.log 2>&1 &
  retriever_pid=$!
  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS --max-time 2 http://127.0.0.1:8000/docs >/dev/null 2>/dev/null; then ready=1; break; fi
    if ! kill -0 "$retriever_pid" 2>/dev/null; then
      echo "BM25 启动失败"; tail -n 40 logs/smoke_b8_5u_20261111/retriever.log; exit 1
    fi
    sleep 1
  done
  if [[ "$ready" -ne 1 ]]; then echo "BM25 180 秒内未就绪"; exit 1; fi
fi
curl -fsS --max-time 10 -H 'Content-Type: application/json' -d '{"queries":["Who is the author of The Hobbit?"],"topk":3,"return_scores":true}' http://127.0.0.1:8000/retrieve > logs/smoke_b8_5u_20261111/retrieval_probe.json
bash train_grpo_2gpu_smoke.sh
