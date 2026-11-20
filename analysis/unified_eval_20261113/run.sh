#!/usr/bin/env bash
set -eo pipefail
cd /g22538014wej/11x22f/Search-R1
source /opt/conda/etc/profile.d/conda.sh
conda activate /g22538014wej/11x22f/envs/search1
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_ATTENTION_BACKEND=XFORMERS
export JAVA_HOME="$CONDA_PREFIX/opt/java21"
export PATH="$JAVA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$JAVA_HOME/lib/server:${LD_LIBRARY_PATH:-}"
export OPENAI_API_KEY=local-bm25-no-api
export HF_DATASETS_CACHE="$PWD/data/cache/huggingface/datasets"
OUT="$PWD/analysis/unified_eval_20261113"
retriever_pid=""
trap 'if [[ -n "$retriever_pid" ]]; then kill "$retriever_pid" 2>/dev/null || true; wait "$retriever_pid" 2>/dev/null || true; fi' EXIT
if [[ $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits) -ge 500 ]]; then
    echo 'GPU 0 busy'; exit 1
fi
python "$OUT/evaluate.py" --prepare-only
if ! env -u LD_LIBRARY_PATH curl -fsS --max-time 2 http://127.0.0.1:8013/docs >/dev/null 2>&1; then
    python -u analysis/grpo_20261112/retrieval_eval.py --index_path data/index/bm25 --corpus_path data/corpus/hotpot_contexts.jsonl --retriever_name bm25 --topk 3 > "$OUT/retriever.log" 2>&1 &
    retriever_pid=$!
    ready=0
    for _ in $(seq 1 180); do
        if env -u LD_LIBRARY_PATH curl -fsS --max-time 2 http://127.0.0.1:8013/docs >/dev/null 2>&1; then ready=1; break; fi
        kill -0 "$retriever_pid"
        sleep 1
    done
    [[ "$ready" == 1 ]]
fi
python -u "$OUT/evaluate.py" --model "$PWD/verl_checkpoints/hotpotqa-qwen25-3b-grpo-2gpu/actor/global_step_500" --label baseline 2>&1 | tee "$OUT/baseline.log"
python -u "$OUT/evaluate.py" --model "$PWD/algorithm_a_runs/evidence_2gpu_a0.1_s42_20261112_174015/checkpoints/actor/global_step_500" --label evidence 2>&1 | tee "$OUT/evidence.log"
date -u > "$OUT/COMPLETE"
