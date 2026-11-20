#!/usr/bin/env bash
set -eo pipefail
cd /g22538014wej/11x22f/Search-R1
source /opt/conda/etc/profile.d/conda.sh
conda activate /g22538014wej/11x22f/envs/search1
export PYTHONPATH="$PWD" NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_ATTENTION_BACKEND=XFORMERS
export JAVA_HOME="$CONDA_PREFIX/opt/java21"
export PATH="$JAVA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$JAVA_HOME/lib/server:${LD_LIBRARY_PATH:-}"
export OPENAI_API_KEY=local-bm25-no-api
export HF_DATASETS_CACHE="$PWD/data/cache/huggingface/datasets"
OUT="$PWD/analysis/followup_20261112"
mkdir -p "$OUT"
python -u analysis/grpo_20261112/retrieval_eval.py --index_path data/index/bm25 --corpus_path data/corpus/hotpot_contexts.jsonl --retriever_name bm25 --topk 3 > "$OUT/retriever.log" 2>&1 &
RETRIEVER_PID=$!
trap 'kill "$RETRIEVER_PID" 2>/dev/null || true' EXIT
for n in $(seq 1 120); do
 if env -u LD_LIBRARY_PATH curl -fsS --max-time 2 http://127.0.0.1:8013/docs >/dev/null 2>&1; then break; fi
 kill -0 "$RETRIEVER_PID"
 sleep 1
done
python -u "$OUT/policy_eval.py" --model "$PWD/verl_checkpoints/hotpotqa-qwen25-3b-grpo-2gpu/actor/global_step_500" --label step500 --out "$OUT/policies" > "$OUT/policy_eval.log" 2>&1
for VARIANT in em evidence evidence_cost; do
 export EVIDENCE_VARIANT="$VARIANT" EVIDENCE_ALPHA=0.5 SEARCH_BETA=0.05
 bash "$OUT/train_variant.sh" > "$OUT/train_${VARIANT}.log" 2>&1
 MODEL="$PWD/verl_checkpoints/followup-20261112-${VARIANT}/actor/global_step_125"
 [ -f "$MODEL/config.json" ]
 python -u analysis/grpo_20261112/paired_eval.py --model "$MODEL" --label "$VARIANT" --out "$OUT/training_eval" > "$OUT/eval_${VARIANT}.log" 2>&1
done
python "$OUT/summarize.py"
date -u > "$OUT/COMPLETE"
echo FOLLOWUP_ALL_COMPLETE
