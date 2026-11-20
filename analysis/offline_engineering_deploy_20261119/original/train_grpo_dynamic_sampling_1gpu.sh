#!/usr/bin/env bash
# Bounded dynamic-sampling.
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/search_environment.sh"
BASE_ROOT=${BASE_ROOT:-$PWD}
export PROJECT_CACHE_DIR="$BASE_ROOT/data/cache"
export HF_HOME="$PROJECT_CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$PROJECT_CACHE_DIR/xdg"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=true RAY_ADDRESS=local
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

export DATA_DIR=${EVIDENCE_DATA_DIR:-$PWD/algorithm_a_data/evidence}
export TRAIN_DATA_DIR="$DATA_DIR"
export TEST_DATA_DIR="$DATA_DIR"
WAND_PROJECT='Search-R1'
export BASE_MODEL=${BASE_MODEL:-$BASE_ROOT/models/Qwen2.5-3B}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-hotpotqa-qwen25-3b-grpo-dynamic-sampling-1gpu}
EVIDENCE_ALPHA=${EVIDENCE_ALPHA:-0.1}
TRAIN_SEED=${TRAIN_SEED:-42}
SAMPLING_ENABLED=${SAMPLING_ENABLED:-true}
MAX_CANDIDATE_BATCHES=${MAX_CANDIDATE_BATCHES:-3}
BEHAVIOR_SAMPLING_ENABLED=${BEHAVIOR_SAMPLING_ENABLED:-true}
BEHAVIOR_MONITOR_ENABLED=${BEHAVIOR_MONITOR_ENABLED:-true}
BEHAVIOR_STRATA_PATH=${BEHAVIOR_STRATA_PATH:-$DATA_DIR/behavior_strata.json}
RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}
RUN_DIR=${RUN_DIR:-$PWD/algorithm_a_runs/dynamic_sampling_1gpu_a${EVIDENCE_ALPHA}_s${TRAIN_SEED}_$(date +%Y%m%d_%H%M%S)}

LOG_FILE=/dev/null
if [[ "${1:-}" == --print-config ]]; then
    shift
    set -- "$@" --cfg job
else
    python3 evaluation/check_evidence_training_inputs.py \
        --base-root "$BASE_ROOT" --evidence-dir "$DATA_DIR" \
        --retriever-url "$RETRIEVER_URL" --model "$BASE_MODEL" \
        --gpu-ids "$CUDA_VISIBLE_DEVICES" --gpus 1 --alpha "$EVIDENCE_ALPHA"
    if [[ -e "$RUN_DIR" ]]; then
        echo "Run directory already exists; choose a new RUN_DIR: $RUN_DIR" >&2
        exit 1
    fi
    export DYNAMIC_DIAGNOSTIC_DIR="$RUN_DIR/numerical_diagnostics"
    export TMPDIR="$RUN_DIR/tmp"
    mkdir -p "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"
    export RAY_TMPDIR=$(mktemp -d /tmp/sra.XXXXXX)
    ln -s "$RAY_TMPDIR" "$RUN_DIR/ray_runtime"
    cp "${BASH_SOURCE[0]}" "$RUN_DIR/launcher.sh"
    cp "$DATA_DIR/annotation_audit.json" "$RUN_DIR/annotation_audit.json"
    if [[ -f "$BEHAVIOR_STRATA_PATH" ]]; then cp "$BEHAVIOR_STRATA_PATH" "$RUN_DIR/behavior_strata.json"; fi
    git rev-parse HEAD > "$RUN_DIR/base_commit.txt"
    git diff --stat > "$RUN_DIR/source_diff_stat.txt"
    for f in evaluation/build_behavior_strata.py evaluation/check_behavior_inputs.py scripts/search_environment.sh scripts/run_dynamic_sampling.sh evaluation/manage_dynamic_sampling_launch.py start_bm25_local.sh render_mc_aerial_c36_argmax.sh verl/single_controller/ray/base.py search_r1/dynamic_sampling/*.py verl/trainer/ppo/dynamic_sampling_trainer.py verl/trainer/main_dynamic_sampling.py verl/workers/dynamic_sampling_worker.py verl/workers/actor/dynamic_sampling_actor.py verl/trainer/config/dynamic_sampling.yaml verl/trainer/main_ppo.py verl/trainer/ppo/ray_trainer.py verl/trainer/ppo/core_algos.py verl/workers/actor/dp_actor.py search_r1/llm_agent/{generation,trajectory,evidence,visibility}.py; do
        mkdir -p "$RUN_DIR/source/$(dirname "$f")"
        cp "$f" "$RUN_DIR/source/$f"
    done
    LOG_FILE="$RUN_DIR/train.log"
    echo "Run: $RUN_DIR | GPUs: $CUDA_VISIBLE_DEVICES | alpha=$EVIDENCE_ALPHA"
fi

# Both evidence launchers retain global batch 8, obs 1024 and the 625-step
# scheduler (178 warmup steps). Compare the saved step-500 checkpoint.
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_dynamic_sampling \
    "data.train_files=$TRAIN_DATA_DIR/train.parquet" \
    "data.val_files=$TEST_DATA_DIR/test.parquet" \
    data.train_data_num=null \
    data.val_data_num=200 \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=2816 \
    data.max_response_length=256 \
    data.max_start_length=512 \
    data.max_obs_length=1024 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    "actor_rollout_ref.model.path=$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=2 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    'trainer.logger=[console]' \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    "+trainer.seed=$TRAIN_SEED" \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=125 \
    trainer.test_freq=125 \
    "trainer.project_name=$WAND_PROJECT" \
    "trainer.experiment_name=$EXPERIMENT_NAME" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=625 \
    "trainer.default_local_dir=$RUN_DIR/checkpoints" \
    max_turns=2 \
    "retriever.url=$RETRIEVER_URL" \
    retriever.topk=3 \
    "dynamic_sampling.enabled=$SAMPLING_ENABLED" \
    "dynamic_sampling.max_candidate_batches=$MAX_CANDIDATE_BATCHES" \
    "behavior_sampling.enabled=$BEHAVIOR_SAMPLING_ENABLED" \
    "behavior_sampling.manifest_path=$BEHAVIOR_STRATA_PATH" \
    "behavior_monitor.enabled=$BEHAVIOR_MONITOR_ENABLED" \
    "search_reward.alpha=$EVIDENCE_ALPHA" \
    search_reward.beta=0.0 \
    search_reward.gamma=0.0 \
    search_reward.delta=0.0 \
    search_reward.require_final_answer=false \
    search_reward.invalid_evidence_policy=skip \
    "search_reward.trace_dir=$RUN_DIR/traces" \
    "hydra.run.dir=$RUN_DIR/hydra" \
    "$@" \
    2>&1 | tee "$LOG_FILE"
