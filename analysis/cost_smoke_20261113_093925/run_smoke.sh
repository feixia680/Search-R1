#!/usr/bin/env bash
set -uo pipefail
cd /g22538014wej/11x22f/Search-R1
export CUDA_VISIBLE_DEVICES=0 COST_ARM=state
export RUN_DIR="/g22538014wej/11x22f/Search-R1/analysis/cost_smoke_20261113_093925/run"
export EXPERIMENT_NAME=cost-state-1gpu-smoke
bash run_grpo_cost_1gpu_with_bm25.sh \
 data.train_data_num=16 \
 "data.val_files=/g22538014wej/11x22f/Search-R1/analysis/cost_smoke_20261113_093925/val_original200_first8.parquet" \
 data.val_data_num=null \
 data.val_batch_size=8 \
 trainer.total_training_steps=3 \
 trainer.save_freq=2 \
 trainer.test_freq=-1 \
 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
status=$?
echo "$status" > "/g22538014wej/11x22f/Search-R1/analysis/cost_smoke_20261113_093925/exit_code"
exit "$status"
