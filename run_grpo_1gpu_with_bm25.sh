#!/usr/bin/env bash
set -eo pipefail
cd /g22538014wej/11x22f/Search-R1
source /opt/conda/etc/profile.d/conda.sh
conda activate /g22538014wej/11x22f/envs/search1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
gpu_count=$(nvidia-smi -L | wc -l)
if [[ "$gpu_count" -lt 1 ]]; then
  echo "需要至少 1 张可见 GPU，当前检测到 $gpu_count 张。"
  exit 1
fi
mkdir -p logs
retriever_pid=""
cleanup() {
  if [[ -n "$retriever_pid" ]]; then
    kill "$retriever_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
if ! curl -fsS --max-time 2 http://127.0.0.1:8000/docs >/dev/null 2>/dev/null; then
  bash start_bm25_local.sh > logs/retrieval_bm25_combined.log 2>&1 &
  retriever_pid=$!
  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS --max-time 2 http://127.0.0.1:8000/docs >/dev/null 2>/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "$retriever_pid" 2>/dev/null; then
      echo "BM25 进程已退出，启动日志："
      tail -n 30 logs/retrieval_bm25_combined.log
      exit 1
    fi
    sleep 1
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "BM25 服务未能在 180 秒内就绪，查看 logs/retrieval_bm25_combined.log"
    exit 1
  fi
fi
bash train_grpo_1gpu.sh
