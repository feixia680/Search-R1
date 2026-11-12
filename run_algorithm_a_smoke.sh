#!/usr/bin/env bash
# One bounded update, never a full training run.
set -eo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_ROOT=${BASE_ROOT:-/g22538014wej/11x22f/Search-R1}
source /opt/conda/etc/profile.d/conda.sh
conda activate /g22538014wej/11x22f/envs/search1
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${ALGORITHM_A_GPU:-0}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export RAY_ADDRESS=local
export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export JAVA_HOME="$CONDA_PREFIX/opt/java21"
export PATH="$JAVA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$JAVA_HOME/lib/server:${LD_LIBRARY_PATH:-}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-local-bm25-no-api}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$BASE_ROOT/data/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export RUN_DIR=${RUN_DIR:-"$ROOT/algorithm_a_runs/smoke_$(date +%Y%m%d_%H%M%S)"}
export RAY_TMPDIR=$(mktemp -d /tmp/sra.XXXXXX)
export TMPDIR="$RUN_DIR/tmp"
mkdir -p "$RUN_DIR" "$RAY_TMPDIR" "$TMPDIR"
ln -s "$RAY_TMPDIR" "$RUN_DIR/ray_runtime"
used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used >= 500 )); then echo "GPU $CUDA_VISIBLE_DEVICES busy (${used} MiB); refusing to start."; exit 2; fi
PORT=${ALGORITHM_A_PORT:-8011}
python - "$PORT" <<'PY'
import socket,sys
with socket.socket() as s:
    s.bind(('127.0.0.1',int(sys.argv[1])))
PY
retriever_pid=''
cleanup() { if [[ -n "$retriever_pid" ]]; then kill "$retriever_pid" 2>/dev/null || true; wait "$retriever_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python -u search_r1/search/retrieval_server.py --index_path "$BASE_ROOT/data/index/bm25" \
 --corpus_path "$BASE_ROOT/data/corpus/hotpot_contexts.jsonl" --retriever_name bm25 --topk 3 --port "$PORT" \
 > "$RUN_DIR/retrieval.log" 2>&1 &
retriever_pid=$!
ready=0
for ((i=0;i<120;i++)); do
 if ! kill -0 "$retriever_pid" 2>/dev/null; then tail -30 "$RUN_DIR/retrieval.log"; exit 1; fi
 if python -c 'import sys,requests; requests.get("http://127.0.0.1:"+sys.argv[1]+"/docs", timeout=2).raise_for_status()' "$PORT" >/dev/null 2>&1; then ready=1; break; fi
 sleep 1
done
if (( ready == 0 )); then echo 'Retriever startup timed out'; exit 1; fi
git rev-parse HEAD > "$RUN_DIR/base_commit.txt"
git diff > "$RUN_DIR/source.patch"
tar -czf "$RUN_DIR/source_files.tar.gz" search_r1/llm_agent/trajectory.py \
 search_r1/llm_agent/generation.py verl/trainer/main_ppo.py verl/trainer/ppo/ray_trainer.py \
 verl/trainer/config/ppo_trainer.yaml verl/utils/torch_functional.py run_algorithm_a_smoke.sh
python -m verl.trainer.main_ppo \
 data.train_files="$BASE_ROOT/data/processed/hotpotqa/train.parquet" data.train_data_num=2 \
 data.val_files="$BASE_ROOT/data/processed/hotpotqa/test.parquet" data.val_data_num=1 \
 data.train_batch_size=2 data.val_batch_size=1 \
 data.max_prompt_length=1792 data.max_start_length=512 data.max_response_length=128 data.max_obs_length=512 \
 algorithm.adv_estimator=grpo \
 actor_rollout_ref.model.path="$BASE_ROOT/models/Qwen2.5-3B" \
 actor_rollout_ref.model.enable_gradient_checkpointing=true actor_rollout_ref.model.use_remove_padding=true \
 actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
 actor_rollout_ref.actor.ppo_mini_batch_size=4 actor_rollout_ref.actor.ppo_micro_batch_size=1 \
 actor_rollout_ref.actor.fsdp_config.param_offload=true \
 actor_rollout_ref.actor.fsdp_config.grad_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
 actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.state_masking=true \
 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n_agent=2 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size=1 actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=true \
 trainer.logger='[console]' +trainer.val_before_train=false \
 trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.total_epochs=1 trainer.total_training_steps=2 \
 trainer.save_freq=1 trainer.test_freq=-1 trainer.default_hdfs_dir=null \
 trainer.default_local_dir="$RUN_DIR/checkpoints" trainer.experiment_name=algorithm-a-smoke \
 max_turns=2 retriever.topk=3 retriever.url="http://127.0.0.1:$PORT/retrieve" \
 search_reward.beta="${SEARCH_BETA:-0.02}" search_reward.gamma="${SEARCH_GAMMA:-0.05}" \
 search_reward.trace_dir="$RUN_DIR/traces" hydra.run.dir="$RUN_DIR/hydra" \
 2>&1 | tee "$RUN_DIR/train.log"
# Upstream increments global_steps before checking the stop condition;
# total_training_steps=2 executes step 1 once. Mini-batch=4 covers all 2x2 rollouts.
echo "Smoke finished: $RUN_DIR"
