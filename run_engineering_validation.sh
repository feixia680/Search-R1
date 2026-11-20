#!/usr/bin/env bash
# Complete engineering validation suite. This script never starts model training.
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

export http_proxy="${http_proxy:-http://127.0.0.1:17897}"
export https_proxy="${https_proxy:-http://127.0.0.1:17897}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:17897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:17897}"

set +u
source "$ROOT/scripts/search_environment.sh"
set -u
PYTHON_BIN=${PYTHON_BIN:-python}
REPORT_DIR=${ENGINEERING_REPORT_DIR:-$ROOT/engineering_runs/$(date +%Y%m%d_%H%M%S)}
mkdir -p "$REPORT_DIR"

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
  city_dynamic_1gpu.sh \
  run_grpo_dynamic_sampling_2gpu_with_bm25.sh; do
  bash -n "$file"
done
"$PYTHON_BIN" evaluation/check_dynamic_sampling_configs.py | tee "$REPORT_DIR/config_checks.log"
git rev-parse HEAD > "$REPORT_DIR/commit.txt"

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep -Ec '^GPU [0-9]+:' || true)
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" == 0 ]]; then
  cat >"$REPORT_DIR/summary.json" <<JSON
{"gpu_available": false, "cpu_checks": "passed", "gpu_checks": "skipped", "multi_gpu": "unavailable", "training_smoke": "not_run"}
JSON
  cat >"$REPORT_DIR/resume_metrics.md" <<'MD'
# Engineering validation metrics

- CPU tests and configuration checks passed.
- GPU metrics are pending because this server has no visible CUDA device.
- Training smoke was intentionally not run by this validation script.
MD
  echo "GPU unavailable. CPU checks passed; GPU checks skipped. Report: $REPORT_DIR"
  exit 0
fi

export ENGINEERING_REAL_OOM=${ENGINEERING_REAL_OOM:-0}
GPU_REPORT="$REPORT_DIR/gpu_${GPU_COUNT}gpu"
"$PYTHON_BIN" evaluation/gpu_engineering_smoke.py --report-dir "$GPU_REPORT" | tee "$REPORT_DIR/gpu_smoke.log"
if [[ "$GPU_COUNT" -ge 2 ]]; then
  torchrun --standalone --nproc_per_node=2 evaluation/gpu_engineering_smoke.py \
    --report-dir "$REPORT_DIR/gpu_2gpu" | tee "$REPORT_DIR/gpu_2gpu_smoke.log"
  MULTI_GPU_STATUS=passed
else
  MULTI_GPU_STATUS=unavailable
fi
cat >"$REPORT_DIR/summary.json" <<JSON
{"gpu_available": true, "gpu_count": $GPU_COUNT, "cpu_checks": "passed", "gpu_checks": "passed", "multi_gpu": "$MULTI_GPU_STATUS", "training_smoke": "not_run"}
JSON
"$PYTHON_BIN" - "$REPORT_DIR" "$GPU_COUNT" <<'PY'
import json
import sys
from pathlib import Path
report = Path(sys.argv[1])
gpu_count = int(sys.argv[2])
lines = ["# Engineering validation metrics", "", "Training smoke: not run by this script.", ""]
path = report / ("gpu_2gpu" if gpu_count >= 2 else "gpu_%dgpu" % gpu_count) / "gpu_smoke.json"
if not path.exists():
    lines.append("GPU metrics: unavailable")
else:
    data = json.loads(path.read_text())
    trace = data.get("trace") or {}
    recovery = data.get("recovery") or {}
    balance = data.get("token_balance") or {}
    real_oom = data.get("real_oom") or {}
    lines.extend([
        "- Concurrent Rollout Trace: %s events from %s trajectories; JSONL integrity passed." % (trace.get("events", 0), trace.get("trajectories", 0)),
        "- Token packing: local padding ratio %.2f%%; CUDA packing latency %.2f ms." % (100 * balance.get("padding_ratio_local", 0.0), balance.get("cuda_pack_ms_local", 0.0)),
        "- Token load balance: rank loads %s; imbalance ratio %.3f." % (balance.get("rank_loads", []), balance.get("load_imbalance_ratio", 0.0)),
        "- OOM recovery policy: %s after attempts %s." % ("recovered" if recovery.get("recovered") else "failed", recovery.get("attempts", [])),
        "- Real CUDA OOM probe: %s." % ("triggered and caught" if real_oom.get("triggered") else "not triggered"),
    ])
(report / "resume_metrics.md").write_text("\n".join(lines) + "\n")
PY
echo "Engineering validation completed. Report: $REPORT_DIR"
