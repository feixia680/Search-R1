#!/usr/bin/env bash
set -eo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=run
case "${1:-}" in --check) MODE=check;; --smoke) MODE=smoke;; --eval-only) MODE=eval_only;; "") ;; *) echo '用法：bash 训练.sh [--check|--smoke|--eval-only]'; exit 2;; esac
[[ $# -le 1 ]] || exit 2
JOB=train
source "$ROOT/scripts/musique_launch.sh"
musique_check "$EVIDENCE"
steps=${GRPO_STEPS:-200}
[[ "$steps" =~ ^[1-9][0-9]*$ ]] || { echo "GRPO_STEPS must be a positive integer"; exit 2; }
interval=25
train_count=5000
batch=25
val_count=null
val_batch=8
final_split=dev_holdout
eval_budgets=(1 2 4)
eval_extra=()
sft_extra=()
if [[ "$MODE" == smoke ]]; then
    steps=2
    interval=1
    batch=4
    val_count=4
    val_batch=4
    final_split=dev200
    eval_budgets=(2)
    eval_extra=(--limit 4)
    sft_extra=(--max-updates 2)
fi
final_evaluation() {
    local trained_run=$1
    local sft_checkpoint=$2
    local grpo_checkpoint="$trained_run/checkpoints/actor/global_step_$steps"
    test -s "$grpo_checkpoint/config.json"
    local labels=(evidence sft grpo)
    local paths=("$EVIDENCE" "$sft_checkpoint" "$grpo_checkpoint")
    local comparison=()
    for i in "${!labels[@]}"; do
        local label=${labels[i]}
        python -u evaluation/musique_policy.py evaluate --split "$final_split" --data "$MUSIQUE_DATA_DIR" \
            --url "$URL" --model "${paths[i]}" --out "$RUN/eval/$label" --batch-size "${EVAL_BATCH:-12}" \
            --budgets "${eval_budgets[@]}" "${eval_extra[@]}" 2>&1 | tee "$RUN/eval_$label.log"
        comparison+=(--run "$label" "$RUN/eval/$label")
    done
    python evaluation/compare_musique_runs.py "${comparison[@]}" --out "$RUN/eval"
}
if [[ "$MODE" == eval_only ]]; then
    : "${TRAIN_RUN:?Set TRAIN_RUN to the completed full training directory}"
    steps=$(python - "$TRAIN_RUN/grpo.yaml" <<'PYCFG'
import sys,yaml
print(int(yaml.safe_load(open(sys.argv[1]))["trainer"]["total_training_steps"])-1)
PYCFG
)
    final_evaluation "$TRAIN_RUN" "${SFT_MODEL:-$TRAIN_RUN/sft}"
    exit 0
fi
count=${COLLECT_N:-512}
[[ "$count" =~ ^[1-9][0-9]*$ && "$count" -le 19938 ]] || { echo 'COLLECT_N must be 1..19938'; exit 2; }
sft="$RUN/sft"
if [[ -n "${SFT_MODEL:-}" ]]; then
    sft="$SFT_MODEL"
    musique_check "$sft"
elif [[ "$MODE" != check ]]; then
    branches=${BRANCH_DIR:-$RUN/branches}
    if [[ -z "${BRANCH_DIR:-}" ]]; then
        python -u evaluation/musique_policy.py collect --model "$EVIDENCE" --split train --count "$count" \
            --batch-size "${COLLECT_BATCH:-16}" --data "$MUSIQUE_DATA_DIR" --url "$URL" --out "$branches" 2>&1 | tee "$RUN/collect.log"
    fi
    test -s "$branches/collection.json"
    python -u evaluation/musique_sft.py --model "$EVIDENCE" --collection "$branches" \
        --data "$MUSIQUE_DATA_DIR" --out "$sft" "${sft_extra[@]}" 2>&1 | tee "$RUN/sft.log"
    musique_check "$sft"
fi
# This trainer starts at step 1 and tests the stopping condition after incrementing.
# The default run uses 25 distinct questions per step for 200 steps.
save_interval=$interval
if (( steps % save_interval != 0 )); then save_interval=$steps; fi
epochs=$(python - "$POLICY/train.parquet" "$batch" "$steps" "$train_count" <<'PYROWS'
import sys,pyarrow.parquet as pq
count=int(sys.argv[4])
assert pq.ParquetFile(sys.argv[1]).metadata.num_rows >= count
batches=count//int(sys.argv[2])
print((int(sys.argv[3])+batches-1)//batches)
PYROWS
)
config=(
    "data.train_files=$POLICY/train.parquet" "data.val_files=$POLICY/dev200.parquet"
    "data.train_data_num=$train_count" "data.val_data_num=$val_count" "data.train_batch_size=$batch" "data.val_batch_size=$val_batch"
    data.max_start_length=512 data.max_prompt_length=5632 data.max_response_length=256 data.max_obs_length=1024
    algorithm.adv_estimator=grpo "actor_rollout_ref.model.path=$sft"
    actor_rollout_ref.model.enable_gradient_checkpointing=true actor_rollout_ref.model.use_remove_padding=true
    actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1
    "actor_rollout_ref.actor.ppo_mini_batch_size=$batch" actor_rollout_ref.actor.ppo_micro_batch_size=1
    actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.actor.fsdp_config.grad_offload=true
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true actor_rollout_ref.ref.fsdp_config.param_offload=true
    actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.state_masking=true
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 actor_rollout_ref.rollout.n_agent=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 actor_rollout_ref.ref.log_prob_micro_batch_size=1
    max_turns=4 "retriever.url=$URL" retriever.topk=3 search_reward.alpha=0.1
    search_reward.invalid_evidence_policy=error "search_reward.trace_dir=$RUN/traces"
    trainer.nnodes=1 trainer.n_gpus_per_node=1 "trainer.total_epochs=$epochs" "trainer.total_training_steps=$((steps+1))"
    "trainer.test_freq=$interval" "trainer.save_freq=$save_interval" +trainer.val_before_train=false +trainer.seed=42
    trainer.default_hdfs_dir=null "trainer.default_local_dir=$RUN/checkpoints" "hydra.run.dir=$RUN/hydra"
    trainer.experiment_name=musique_sft_grpo 'trainer.logger=[console]'
)
python -m verl.trainer.main_ppo "${config[@]}" --cfg job --resolve > "$RUN/grpo.yaml"
if [[ "$MODE" == check ]]; then
    python evaluation/musique_policy.py --help >/dev/null
    python evaluation/musique_sft.py --help >/dev/null
    python - "$RUN/grpo.yaml" "$POLICY" "$sft" "$steps" <<'PY'
import sys,yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text())
assert cfg["data"]["train_files"]==sys.argv[2]+"/train.parquet"
assert cfg["data"]["val_files"]==sys.argv[2]+"/dev200.parquet"
assert cfg["actor_rollout_ref"]["model"]["path"]==sys.argv[3]
assert cfg["data"]["train_data_num"]==5000 and cfg["data"]["train_batch_size"]==25
assert cfg["trainer"]["total_training_steps"]==int(sys.argv[4])+1
assert cfg["max_turns"]==4 and cfg["algorithm"]["adv_estimator"]=="grpo" and cfg["retriever"]["topk"]==3
print("训练配置、训练/验证划分、模型和本地检索 CPU 检查通过。")
PY
    exit 0
fi
export RAY_ADDRESS=local RAY_TMPDIR=$(mktemp -d /tmp/musique-ray.XXXXXX)
python -u -m verl.trainer.main_ppo "${config[@]}" 2>&1 | tee "$RUN/grpo.log"
test -s "$RUN/checkpoints/actor/global_step_$steps/config.json"
echo "SFT 模型 $sft"
echo "GRPO 模型 $RUN/checkpoints/actor/global_step_$steps"


python evaluation/verify_musique_training.py "$RUN" "$sft" "$steps"
final_evaluation "$RUN" "$sft"
