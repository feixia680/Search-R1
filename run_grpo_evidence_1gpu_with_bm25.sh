#!/usr/bin/env bash
# Start/reuse local BM25, then run the evidence-reward 1-GPU experiment.
# Usage: bash run_grpo_evidence_1gpu_with_bm25.sh [Hydra overrides...]
# Config only: bash run_grpo_evidence_1gpu_with_bm25.sh --print-config
set -eo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONDA_ENV=${CONDA_ENV:-/g22538014wej/cityins1122/envs/search1}
source /opt/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$ROOT"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

if [[ "${1:-}" == --print-config ]]; then
  shift
  exec bash "$ROOT/train_grpo_evidence_1gpu.sh" --print-config "$@"
fi

expected_gpus=1
selected_gpus=${CUDA_VISIBLE_DEVICES:-0}
gpu_count=$(nvidia-smi -L 2>/dev/null | grep -Ec '^GPU [0-9]+:' || true)
if [[ "$gpu_count" -lt "$expected_gpus" ]]; then
  echo "需要至少 1 张可见 GPU，当前检测到 $gpu_count 张。" >&2
  exit 1
fi
python3 - "$selected_gpus" "$expected_gpus" <<'PREFLIGHT'
import subprocess, sys
ids, count = sys.argv[1:]
parts = ids.split(',')
if len(parts) != int(count) or len(set(parts)) != int(count):
    raise SystemExit(f'需要 {count} 个不同 GPU ID，当前 CUDA_VISIBLE_DEVICES={ids}')
result = subprocess.check_output(['nvidia-smi', '-i', ids, '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
rows = [line for line in result.splitlines() if line.strip()]
if len(rows) != int(count):
    raise SystemExit(f'预期 {count} 张 GPU，实际查询到 {len(rows)} 张')
for row in rows:
    gpu, used = row.split(',')
    if int(used.strip()) >= 500:
        raise SystemExit(f'GPU {gpu.strip()} 已占用 {used.strip()} MiB；停止启动')
PREFLIGHT

mkdir -p logs
retriever_pid=""
cleanup() {
  if [[ -n "$retriever_pid" ]]; then
    kill "$retriever_pid" 2>/dev/null || true
    wait "$retriever_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if ! curl -fsS --max-time 2 http://127.0.0.1:8000/docs >/dev/null 2>&1; then
  bm25_log="logs/retrieval_bm25_evidence_1gpu_$(date +%Y%m%d_%H%M%S).log"
  bash start_bm25_local.sh > "$bm25_log" 2>&1 &
  retriever_pid=$!
  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS --max-time 2 http://127.0.0.1:8000/docs >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$retriever_pid" 2>/dev/null; then
      echo "BM25 进程已退出，启动日志：$bm25_log" >&2
      tail -n 30 "$bm25_log" >&2
      exit 1
    fi
    sleep 1
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "BM25 服务未能在 180 秒内就绪，查看 $bm25_log" >&2
    exit 1
  fi
fi
bash "$ROOT/train_grpo_evidence_1gpu.sh" "$@"
