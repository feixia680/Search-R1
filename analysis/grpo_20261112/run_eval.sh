#!/usr/bin/env bash
set -eo pipefail
cd /g22538014wej/11x22f/Search-R1
source /opt/conda/etc/profile.d/conda.sh
conda activate /g22538014wej/11x22f/envs/search1
export PYTHONPATH="$PWD" NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export JAVA_HOME="$CONDA_PREFIX/opt/java21"
export PATH="$JAVA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$JAVA_HOME/lib/server:${LD_LIBRARY_PATH:-}"
export OPENAI_API_KEY=local-bm25-no-api
export HF_DATASETS_CACHE="$PWD/data/cache/huggingface/datasets"
OUT="$PWD/analysis/grpo_20261112"
python -u "$OUT/retrieval_eval.py" --index_path data/index/bm25 --corpus_path data/corpus/hotpot_contexts.jsonl --retriever_name bm25 --topk 3 > "$OUT/retriever.log" 2>&1 &
RETRIEVER_PID=$!
trap 'kill "$RETRIEVER_PID" 2>/dev/null || true' EXIT
READY=0
for n in $(seq 1 90); do
 if env -u LD_LIBRARY_PATH curl -fsS --max-time 2 http://127.0.0.1:8013/docs >/dev/null 2>&1; then READY=1; break; fi
 if ! kill -0 "$RETRIEVER_PID" 2>/dev/null; then tail -n 30 "$OUT/retriever.log"; exit 1; fi
 sleep 1
done
[ "$READY" -eq 1 ]
for LABEL in base step375 step500; do
 MODEL="$PWD/models/Qwen2.5-3B"
 [ "$LABEL" != step375 ] || MODEL="$PWD/verl_checkpoints/hotpotqa-qwen25-3b-grpo-2gpu/actor/global_step_375"
 [ "$LABEL" != step500 ] || MODEL="$PWD/verl_checkpoints/hotpotqa-qwen25-3b-grpo-2gpu/actor/global_step_500"
 python -u "$OUT/paired_eval.py" --model "$MODEL" --label "$LABEL" --out "$OUT" > "$OUT/$LABEL.log" 2>&1
done
echo ALL_EVALUATIONS_COMPLETE
