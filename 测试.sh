#!/usr/bin/env bash
set -eo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODE=run
case "${1:-}" in --check) MODE=check;; "") ;; *) echo '用法：bash 测试.sh [--check]'; exit 2;; esac
[[ $# -le 1 ]] || exit 2
JOB=eval
source "$ROOT/scripts/musique_launch.sh"
split=${EVAL_SPLIT:-dev200}
[[ "$split" == dev200 || "$split" == dev_holdout ]] || { echo "Invalid EVAL_SPLIT"; exit 2; }
models=("$EVIDENCE" "$DYNAMIC")
labels=(evidence dynamic)
if [[ -n "${MODEL_PATH:-}" ]]; then models=("$MODEL_PATH"); labels=(candidate); fi
musique_check "${models[@]}"
if [[ "$MODE" == check ]]; then
    python evaluation/musique_policy.py --help >/dev/null
    echo "CPU 检查通过。评测集 $split；搜索预算 1、2、4；每次 top-3。"
    exit 0
fi
read -r -a budgets <<< "${BUDGETS:-1 2 4}"
options=(--budgets "${budgets[@]}")
if [[ "${ANSWER_AT_CAP:-0}" == 1 ]]; then options+=(--answer-at-cap); fi
if [[ -n "${EVAL_N:-}" ]]; then options+=(--limit "$EVAL_N"); fi
for i in "${!models[@]}"; do
    python -u evaluation/musique_policy.py evaluate --model "${models[i]}" --split "$split" \
        --batch-size "${EVAL_BATCH:-4}" --data "$MUSIQUE_DATA_DIR" --url "$URL" --out "$RUN/${labels[i]}" "${options[@]}" \
        2>&1 | tee "$RUN/${labels[i]}.log"
done
echo "逐题轨迹和指标位于 $RUN"

