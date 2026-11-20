#!/usr/bin/env bash
# Two-step Answer + Evidence / DAPO token-loss smoke test on two GPUs.
set -eo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export EXPERIMENT_NAME=hotpotqa-qwen25-3b-evidence-dapo-token-2gpu-smoke
export RUN_DIR=${RUN_DIR:-$ROOT/algorithm_a_runs/evidence_dapo_token_2gpu_smoke_$(date +%Y%m%d_%H%M%S)}

# This trainer stops after incrementing its counter, so 3 executes two steps.
exec bash "$ROOT/run_grpo_evidence_2gpu_with_bm25.sh" "$@" \
    +actor_rollout_ref.actor.loss_agg_mode=token-mean \
    max_turns=5 \
    data.max_prompt_length=8192 \
    data.train_data_num=16 \
    data.val_data_num=4 \
    data.val_batch_size=4 \
    trainer.total_training_steps=3 \
    trainer.save_freq=2 \
    trainer.test_freq=-1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0
