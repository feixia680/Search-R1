#!/usr/bin/env bash
# Run all CPU checks and, when GPUs are available, GPU runtime checks plus a short
# dynamic-sampling training smoke test.
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

export http_proxy="${http_proxy:-http://127.0.0.1:17897}"
export https_proxy="${https_proxy:-http://127.0.0.1:17897}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:17897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:17897}"

CONDA_ENV=${CONDA_ENV:-/g22538014wej/11x22f/envs/search1}
if [[ -x "$CONDA_ENV/bin/python" ]]; then
  # Some servers expose the environment as a prefix without registering it in
  # `conda env list`; using its bin directory is reliable in that layout.
  export PATH="$CONDA_ENV/bin:/opt/conda/bin:$PATH"
  export CONDA_PREFIX="$CONDA_ENV"
  export CONDA_DEFAULT_ENV="$CONDA_ENV"
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  source /opt/conda/etc/profile.d/conda.sh
  if ! conda activate "$CONDA_ENV"; then
    echo "无法激活环境：$CONDA_ENV" >&2
    exit 1
  fi
else
  echo "找不到 Conda 环境：$CONDA_ENV" >&2
  exit 1
fi
PYTHON_BIN=${PYTHON_BIN:-python}
REPORT_DIR=${ENGINEERING_REPORT_DIR:-$ROOT/engineering_runs/$(date +%Y%m%d_%H%M%S)}
mkdir -p "$REPORT_DIR"

run_cpu_checks() {
  "$PYTHON_BIN" -m pytest -q tests/runtime tests/dynamic_sampling | tee "$REPORT_DIR/cpu_tests.log"
  "$PYTHON_BIN" -m py_compile \
    search_r1/runtime/*.py \
    search_r1/dynamic_sampling/*.py \
    evaluation/gpu_engineering_smoke.py \
    verl/trainer/main_dynamic_sampling.py \
    verl/trainer/ppo/dynamic_sampling_trainer.py \
    verl/workers/dynamic_sampling_worker.py \
    verl/workers/actor/dynamic_sampling_actor.py \
    verl/single_controller/ray/base.py
  for file in \
    train_grpo_dynamic_sampling_1gpu.sh \
    train_grpo_dynamic_sampling_2gpu.sh \
    run_grpo_dynamic_sampling_1gpu_with_bm25.sh \
    run_grpo_dynamic_sampling_2gpu_with_bm25.sh; do
    bash -n "$file"
  done
  "$PYTHON_BIN" evaluation/check_dynamic_sampling_configs.py | tee "$REPORT_DIR/config_checks.log"
  git rev-parse HEAD > "$REPORT_DIR/commit.txt"
}

run_cpu_checks
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep -Ec '^GPU [0-9]+:' || true)
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" == 0 ]]; then
  cat >"$REPORT_DIR/summary.json" <<JSON
{"gpu_available": false, "cpu_checks": "passed", "message": "GPU checks skipped; rerun this script on a GPU server."}
JSON
  echo "GPU unavailable. CPU checks passed; GPU checks were skipped. Report: $REPORT_DIR"
  exit 0
fi

export ENGINEERING_REAL_OOM=${ENGINEERING_REAL_OOM:-0}
GPU_REPORT="$REPORT_DIR/gpu_${GPU_COUNT}gpu"
"$PYTHON_BIN" evaluation/gpu_engineering_smoke.py --report-dir "$GPU_REPORT" | tee "$REPORT_DIR/gpu_smoke.log"
if [[ "$GPU_COUNT" -ge 2 ]]; then
  TWO_GPU_REPORT="$REPORT_DIR/gpu_2gpu"
  torchrun --standalone --nproc_per_node=2 evaluation/gpu_engineering_smoke.py \
    --report-dir "$TWO_GPU_REPORT" | tee "$REPORT_DIR/gpu_2gpu_smoke.log"
fi

if [[ "${ENGINEERING_TRAINING_SMOKE:-1}" == 1 ]]; then
  TRAIN_GPUS=${ENGINEERING_TRAINING_GPUS:-1}
  if [[ "$TRAIN_GPUS" -gt "$GPU_COUNT" ]]; then
    echo "跳过训练 smoke：要求 $TRAIN_GPUS 张 GPU，当前只有 $GPU_COUNT 张。" >&2
  else
    TRAIN_RUN_DIR="$REPORT_DIR/training_smoke_${TRAIN_GPUS}gpu"
    export RUN_DIR="$TRAIN_RUN_DIR"
    export EXPERIMENT_NAME=engineering-gpu-smoke
    export EVIDENCE_ALPHA=0.1
    export TRAIN_SEED=42
    export MAX_CANDIDATE_BATCHES=1
    export SAMPLING_ENABLED=true
    bash "run_grpo_dynamic_sampling_${TRAIN_GPUS}gpu_with_bm25.sh" \
      trainer.total_training_steps=2 \
      trainer.save_freq=0 \
      trainer.test_freq=0 \
      +trainer.skip_final_validation=true \
      2>&1 | tee "$REPORT_DIR/training_smoke.log"
  fi
fi
cat >"$REPORT_DIR/summary.json" <<JSON
{"gpu_available": true, "gpu_count": $GPU_COUNT, "cpu_checks": "passed", "gpu_checks": "passed", "training_smoke": "${ENGINEERING_TRAINING_SMOKE:-1}"}
JSON
echo "Engineering validation completed. Report: $REPORT_DIR"
