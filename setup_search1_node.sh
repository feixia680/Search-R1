#!/bin/bash
set -eo pipefail
export http_proxy=http://127.0.0.1:17897 https_proxy=http://127.0.0.1:17897 HTTP_PROXY=http://127.0.0.1:17897 HTTPS_PROXY=http://127.0.0.1:17897
source /opt/conda/etc/profile.d/conda.sh
E=/g22538014wej/11x22f/envs/search1
if [ ! -f "$E/conda-meta/history" ]; then conda create -p "$E" --clone edtformer2 -y; fi
ln -sfn "$E" /opt/conda/envs/search1
conda activate "$E"
python -m pip uninstall -y prodigy-prot torchaudio
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install vllm==0.6.3 transformers==4.47.1 datasets==2.21.0 numpy==1.26.4 tensordict==0.5.0 accelerate codetiming dill hydra-core pandas pybind11 ray wandb ipython matplotlib
python -m pip install --no-deps numpy==1.26.4 pyserini==0.43.0 faiss-cpu==1.8.0.post1
python -m pip install cython flask onnxruntime pyjnius
python -m pip install -e /g22538014wej/11x22f/Search-R1 --no-deps
mkdir -p "$E/opt/java21"
if [ ! -x "$E/opt/java21/bin/java" ]; then
 curl -fsSL --retry 3 https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jre/hotspot/normal/eclipse -o "$E/opt/java21.tar.gz"
 tar -xzf "$E/opt/java21.tar.gz" -C "$E/opt/java21" --strip-components=1
fi
python -m pip check
printf '\nSETUP_COMPLETE\n'
