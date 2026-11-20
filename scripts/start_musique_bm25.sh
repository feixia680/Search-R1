#!/usr/bin/env bash
set -eo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
data_dir=${MUSIQUE_DATA_DIR:-"$root/data/musique_cpu_v1"}
export BM25_INDEX_PATH="$data_dir/index"
export BM25_CORPUS_PATH="$data_dir/prepared/collection/passages.jsonl"
export BM25_PORT=${BM25_PORT:-18081}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec bash "$root/start_bm25_local.sh"
