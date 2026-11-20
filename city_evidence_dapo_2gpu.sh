#!/usr/bin/env bash
# Two-GPU Answer + Evidence training with DAPO token-level loss.
set -eo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-hotpotqa-evidence-dapo-token-2gpu}
export RUN_DIR=${RUN_DIR:-$ROOT/algorithm_a_runs/evidence_dapo_token_2gpu_$(date +%Y%m%d_%H%M%S)}

preview=()
if [[ "${1:-}" == --print-config ]]; then
    preview=(--print-config)
    shift
fi

exec bash "$ROOT/run_grpo_evidence_2gpu_with_bm25.sh" "${preview[@]}" \
    +actor_rollout_ref.actor.loss_agg_mode=token-mean \
    max_turns=5 \
    data.max_prompt_length=8192 \
    "$@"
