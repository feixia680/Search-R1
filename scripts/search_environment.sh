# Sourced by launchers; resolve project and environment from this file's location.
SEARCH_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CONDA_ENV="${CONDA_ENV:-$(dirname "$SEARCH_PROJECT_ROOT")/envs/search1}"
if [[ ! -x "$CONDA_ENV/bin/python" ]]; then
    echo "Missing Python environment: $CONDA_ENV/bin/python" >&2
    return 1
fi
# Launchers use errexit/pipefail without nounset: Conda activation hooks
# reference unset compiler variables on this server.
if [[ "${CONDA_PREFIX:-}" != "$CONDA_ENV" ]]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
fi
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$SEARCH_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
cd "$SEARCH_PROJECT_ROOT"
