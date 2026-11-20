#!/usr/bin/env bash
set -eo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BRANCH_DIR="$root/analysis/musique_70m_20261119_2252/branches_ready"
for file in collection.json demonstrations.jsonl; do
    [[ -s "$BRANCH_DIR/$file" ]] || { echo "缺少已核验的训练数据 $BRANCH_DIR/$file" >&2; exit 1; }
done
exec bash "$root/训练.sh" "$@"
