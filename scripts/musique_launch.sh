# Shared setup for the two project-root launchers. The caller sets ROOT and MODE.
source "$ROOT/scripts/search_environment.sh"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PROJECT_CACHE_DIR="$ROOT/data/cache" HF_HOME="$ROOT/data/cache/huggingface"
export XDG_CACHE_HOME="$ROOT/data/cache/xdg" VLLM_ATTENTION_BACKEND=XFORMERS
export MUSIQUE_DATA_DIR="$ROOT/data/musique_cpu_v1"
POLICY="$MUSIQUE_DATA_DIR/prepared/policy"
EVIDENCE="$ROOT/algorithm_a_runs/evidence_2gpu_a0.1_s42_20261112_174015/checkpoints/actor/global_step_500"
DYNAMIC="$ROOT/algorithm_a_runs/dynamic_sampling_1gpu_a0.1_s42_20261118_085420/checkpoints/actor/global_step_500"
gpu_list=${CUDA_VISIBLE_DEVICES-0}
export CUDA_VISIBLE_DEVICES=${gpu_list%%,*}
if [[ "$MODE" != check ]]; then
    python -c 'import torch; assert torch.cuda.is_available(), "GPU unavailable; run --check on CPU"'
fi
# Each invocation owns its service, so one job stopping cannot interrupt another job.
export BM25_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
URL="http://127.0.0.1:$BM25_PORT/retrieve"
if [[ "$MODE" == check ]]; then
    RUN=$(mktemp -d)
else
    mkdir -p "$ROOT/algorithm_a_runs"
    RUN=$(mktemp -d "$ROOT/algorithm_a_runs/musique_${JOB}_$(date +%Y%m%d_%H%M%S)_XXXXXX")
    echo "本次输出 $RUN"
fi
bash "$ROOT/scripts/start_musique_bm25.sh" >"$RUN/bm25.log" 2>&1 &
search_pid=$!
finish_musique() {
    kill "$search_pid" 2>/dev/null || true
    wait "$search_pid" 2>/dev/null || true
    if [[ "$MODE" == check ]]; then rm -r -- "$RUN"; fi
}
trap finish_musique EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python - "$BM25_PORT" "$search_pid" <<'PYREADY'
import os,sys,time,urllib.request
port,pid=sys.argv[1:]
deadline=time.monotonic()+120
while time.monotonic()<deadline:
    os.kill(int(pid),0)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/openapi.json",timeout=2) as response:
            if response.status==200:
                break
    except OSError:
        time.sleep(0.5)
else:
    raise SystemExit("Timed out waiting for MuSiQue BM25")
PYREADY
musique_check() {
    python - "$MUSIQUE_DATA_DIR" "$URL" "$@" <<'PY'
import hashlib,json,sys
from pathlib import Path
import pyarrow.parquet as pq
import requests
from transformers import AutoConfig,AutoTokenizer
root=Path(sys.argv[1])
trace=json.loads((root/"diagnostics/dev200_trace.jsonl").open().readline())
resp=requests.post(sys.argv[2],json={"queries":[trace["question"]],"topk":3,"return_scores":True},timeout=10)
resp.raise_for_status()
assert [x["document"]["id"] for x in resp.json()["result"][0]]==[x["id"] for x in trace["trace"][0]["hits"]]
checks={v:k for k,v in [line.rstrip().split("  ",1) for line in (root/"SHA256SUMS").open()]}
for name in ("train","dev200","dev_holdout"):
    rel=f"prepared/policy/{name}.parquet"
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==checks[rel]
for arg in sys.argv[3:]:
    p=Path(arg)
    cfg=AutoConfig.from_pretrained(p,local_files_only=True)
    assert cfg.max_position_embeddings>=5888
    assert AutoTokenizer.from_pretrained(p,local_files_only=True).is_fast
    idx=p/"model.safetensors.index.json"
    shards=set(json.loads(idx.read_text())["weight_map"].values()) if idx.exists() else {"model.safetensors"}
    assert all((p/s).is_file() and (p/s).stat().st_size>0 for s in shards),p
    print("Checkpoint OK",p)
print("Frozen parquet and MuSiQue HTTP top-3 OK",flush=True)
PY
}

