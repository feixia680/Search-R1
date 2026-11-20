#!/usr/bin/env bash
# Compatibility entrypoint: start/reuse BM25 before the single-GPU trainer.
set -eo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$ROOT/city_dynamic_1gpu.sh" "$@"
