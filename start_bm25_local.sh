#!/usr/bin/env bash
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/search_environment.sh"
export JAVA_HOME="${BM25_JAVA_HOME:-$CONDA_ENV/opt/java21}"
export PATH="$JAVA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$JAVA_HOME/lib/server:${LD_LIBRARY_PATH:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-bm25-no-api}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SEARCH_PROJECT_ROOT/data/cache/huggingface/datasets}"
INDEX_PATH=${BM25_INDEX_PATH:-$SEARCH_PROJECT_ROOT/data/index/bm25}
CORPUS_PATH=${BM25_CORPUS_PATH:-$SEARCH_PROJECT_ROOT/data/corpus/hotpot_contexts.jsonl}
[[ -d "$INDEX_PATH" ]] || { echo "Missing BM25 index: $INDEX_PATH" >&2; exit 1; }
[[ -f "$CORPUS_PATH" ]] || { echo "Missing BM25 corpus: $CORPUS_PATH" >&2; exit 1; }
[[ -x "$JAVA_HOME/bin/java" ]] || { echo "Missing Java: $JAVA_HOME/bin/java" >&2; exit 1; }
mkdir -p "$HF_DATASETS_CACHE"
exec "$CONDA_ENV/bin/python" -u "$SEARCH_PROJECT_ROOT/search_r1/search/retrieval_server.py" \
    --index_path "$INDEX_PATH" --corpus_path "$CORPUS_PATH" \
    --retriever_name bm25 --topk 3 --port "${BM25_PORT:-8000}"
