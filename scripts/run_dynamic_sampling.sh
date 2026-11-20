#!/usr/bin/env bash
set -eo pipefail
GPUS=$1
shift
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/search_environment.sh"
if [[ "${1:-}" == --print-config ]]; then
    exec bash "$SEARCH_PROJECT_ROOT/train_grpo_dynamic_sampling_${GPUS}gpu.sh" "$@"
fi
if [[ "$GPUS" == 1 ]]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
else
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
fi
exec "$CONDA_ENV/bin/python" "$SEARCH_PROJECT_ROOT/evaluation/manage_dynamic_sampling_launch.py" --gpus "$GPUS" "$@"
