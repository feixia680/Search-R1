#!/usr/bin/env bash
set -eo pipefail
cd /g22538014wej/11x22f/Search-R1
source /opt/conda/etc/profile.d/conda.sh
conda activate /g22538014wej/11x22f/envs/search1
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export HF_HOME="$PWD/data/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$PWD/data/cache/xdg"
export TMPDIR=/g22538014wej/11x22f/tmp
export RAY_TMPDIR=/g22538014wej/11x22f/ray
export VLLM_ATTENTION_BACKEND=XFORMERS
export SEARCH_R1_MEMORY_DEBUG=1
export SEARCH_R1_MEMORY_LIMIT_GB=96
export SEARCH_R1_MEMORY_DIR="$PWD/logs/memory_diag_20261111"
mkdir -p "$SEARCH_R1_MEMORY_DIR" "$TMPDIR" "$RAY_TMPDIR"
retriever_pid=""
cleanup() {
  if [[ -n "$retriever_pid" ]]; then
    kill "$retriever_pid" 2>/dev/null || true
  fi
  ray stop --force > "$SEARCH_R1_MEMORY_DIR/ray_cleanup.log" 2>&1 || true
}
trap cleanup EXIT
if ! curl -fsS --noproxy '*' --max-time 2 http://127.0.0.1:8000/docs >/dev/null; then
  bash start_bm25_local.sh > "$SEARCH_R1_MEMORY_DIR/bm25.log" 2>&1 &
  retriever_pid=$!
  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS --noproxy '*' --max-time 2 http://127.0.0.1:8000/docs >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$retriever_pid" || exit 1
    sleep 1
  done
  [[ "$ready" == 1 ]] || exit 1
fi
# The existing script supplies the exact model and algorithm settings.
# Append diagnostic overrides before its logging redirection; formal files are not edited.
sed -e 's/data.train_data_num=null/data.train_data_num=40/' \
    -e 's/data.val_data_num=200/data.val_data_num=4/' \
    -e 's/data.train_batch_size=8/data.train_batch_size=4/' \
    -e 's/data.val_batch_size=8/data.val_batch_size=4/' \
    -e 's/ppo_mini_batch_size=8/ppo_mini_batch_size=4/' \
    -e 's/ppo_micro_batch_size=1/ppo_micro_batch_size=2/' \
    -e 's/log_prob_micro_batch_size=1/log_prob_micro_batch_size=2/g' \
    -e 's/trainer.total_training_steps=625/trainer.total_training_steps=7/' \
    -e 's/trainer.save_freq=125/trainer.save_freq=-1/' \
    -e 's/trainer.test_freq=125/trainer.test_freq=-1/' \
    -e 's/hotpotqa-qwen25-3b-grpo-2gpu/hotpotqa-qwen25-3b-memory-diag/g' \
    train_grpo_2gpu.sh | bash
