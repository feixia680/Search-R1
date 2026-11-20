#!/usr/bin/env bash
# Tested on 2 x A800 80GB. Keep the legacy per-rank loss groups at size 1.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export RUN_DIR=${RUN_DIR:-$PWD/algorithm_a_runs/dynamic_sampling_fast_2gpu_$(date +%Y%m%d_%H%M%S)}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-hotpotqa-qwen25-3b-grpo-dynamic-sampling-fast-2gpu}
ENGINEERING_ACTOR_MICRO_BATCH_SIZE=${ENGINEERING_ACTOR_MICRO_BATCH_SIZE:-4}
ENGINEERING_LOGPROB_MICRO_BATCH_SIZE=${ENGINEERING_LOGPROB_MICRO_BATCH_SIZE:-8}
preview=()
if [[ "${1:-}" == --print-config ]]; then preview=(--print-config); shift; fi
exec bash run_grpo_dynamic_sampling_2gpu_with_bm25.sh "${preview[@]}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.grad_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  "actor_rollout_ref.actor.ppo_micro_batch_size=$ENGINEERING_ACTOR_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.engineering_loss_group_size=1 \
  "actor_rollout_ref.rollout.log_prob_micro_batch_size=$ENGINEERING_LOGPROB_MICRO_BATCH_SIZE" \
  "actor_rollout_ref.ref.log_prob_micro_batch_size=$ENGINEERING_LOGPROB_MICRO_BATCH_SIZE" \
  "$@"
