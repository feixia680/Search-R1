#!/bin/bash
set -eo pipefail
cd /g22538014wej/11x22f/Search-R1
source /opt/conda/etc/profile.d/conda.sh
conda activate /g22538014wej/11x22f/envs/search1
export JAVA_HOME="$CONDA_PREFIX/opt/java21"
export PATH="$JAVA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$JAVA_HOME/lib/server:${LD_LIBRARY_PATH:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-bm25-no-api}"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/g22538014wej/11x22f/Search-R1/data/cache/huggingface/datasets}"
mkdir -p "$HF_DATASETS_CACHE"
exec python -u search_r1/search/retrieval_server.py --index_path data/index/bm25 --corpus_path data/corpus/hotpot_contexts.jsonl --retriever_name bm25 --topk 3
