#!/usr/bin/env bash
# Same generated batch, real forward/backward, lr=0; no checkpoints.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
preview=()
if [[ "${1:-}" == --print-config ]]; then preview=(--print-config); shift; fi
exec bash run_grpo_dynamic_sampling_2gpu_with_bm25.sh "${preview[@]}" \
  engineering.profile_only=true actor_rollout_ref.actor.optim.lr=0 \
  trainer.total_training_steps=2 trainer.test_freq=-1 trainer.save_freq=-1 \
  +trainer.skip_final_validation=true dynamic_sampling.enabled=false \
  behavior_sampling.enabled=false behavior_monitor.enabled=false "$@"
