#!/bin/bash
set -euo pipefail
cd /workspace/coder
mkdir -p evidence llama
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
apt-get update -qq
apt-get install -y -qq python3-venv curl libgomp1 libcurl4 build-essential python3-dev > evidence/apt.log 2>&1
python3 -m venv venv
venv/bin/pip install -q 'huggingface-hub==0.36.0' 'tokenizers==0.22.1'
for asset in llama-b12432-bin-ubuntu-cuda-12.8-x64.tar.gz cudart-llama-b12432-bin-ubuntu-cuda-12.8-x64.tar.gz; do
 curl -fLsS --retry 2 "https://github.com/ggml-org/llama.cpp/releases/download/b12432/$asset" -o "$asset"
 case "$asset" in
  llama-*) expected=5adb98e3c44fe090006b6986964ebe4640373b71684014d9010484dca9cc7d65 ;;
  cudart-*) expected=07a94f63bee00b37f5d22c88e8f8af14f2edcb4608f77dde7dce160f328c21bc ;;
 esac
 echo "$expected  $asset" | sha256sum -c -
 sha256sum "$asset" >> evidence/runtime-archives.sha256
 tar -xf "$asset" -C llama
done
find llama -name llama-server -type f > evidence/binary-path.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > evidence/gpu.csv
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > evidence/gpu-samples.csv &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import hashlib,json,time
start=time.monotonic()
p=hf_hub_download('ggml-org/Qwen3-Coder-30B-A3B-Instruct-GGUF',filename='Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf',revision='ef85c7f175f00be8af4e1de0c14e59ce7e5d0db3')
h=hashlib.sha256()
with open(p,'rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
assert h.hexdigest()=='f6050b4da22869aa3fb9d94bb3314e9fc5a027d4741c1f97761c387d6b2b2335'
Path('model.gguf').symlink_to(p)
p=hf_hub_download('Qwen/Qwen3-Coder-30B-A3B-Instruct',filename='tokenizer.json',revision='b2cff646eb4bb1d68355c01b18ae02e7cf42d120');Path('tokenizer.json').symlink_to(p)
Path('evidence/downloads.json').write_text(json.dumps({'gguf_sha256':h.hexdigest(),'download_seconds':time.monotonic()-start}))
PY
export LLAMA_BIN="$(find /workspace/coder/llama -name llama-server -type f | head -1)"
export LD_LIBRARY_PATH="$(dirname "$LLAMA_BIN"):/workspace/coder/llama/cudart-llama-b12432-bin-ubuntu-cuda-12.8-x64:${LD_LIBRARY_PATH:-}"
"$LLAMA_BIN" --version > evidence/llama-version.txt 2>&1
"$LLAMA_BIN" --help > evidence/llama-help.txt 2>&1
if [ "${1:-llama}" = baseline ]; then
 # Complete downloads/install before measurements to avoid background IO interference.
 venv/bin/python - <<'PY' > evidence/baseline-download.log 2>&1 &
from huggingface_hub import snapshot_download
from pathlib import Path
p=snapshot_download('Qwen/Qwen3-Coder-30B-A3B-Instruct',revision='b2cff646eb4bb1d68355c01b18ae02e7cf42d120',allow_patterns=['*.json','*.safetensors','*.jinja','*.txt'])
Path('bf16').symlink_to(p,target_is_directory=True)
PY
 download_pid=$!
 venv/bin/pip install 'vllm==0.33.0' > evidence/vllm-install.log 2>&1 &
 install_pid=$!
 wait "$download_pid" "$install_pid"
 export VLLM_USE_FLASHINFER_SAMPLER=0
 printf 'VLLM_USE_FLASHINFER_SAMPLER=0\n' > evidence/vllm-environment.txt
 timeout 1500 venv/bin/python benchmark_coder_backends.py --engine llama --seconds 1400
 venv/bin/python - <<'PY' > evidence/torch-probe.json
import torch,json
x=torch.arange(32,device='cuda',dtype=torch.float32)
assert (x*2).sum().item()==992
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'probe':'pass'}))
PY
 timeout 2100 venv/bin/python benchmark_coder_backends.py --engine vllm --seconds 2000
else
 timeout 2700 venv/bin/python benchmark_coder_backends.py --engine llama --seconds 2600
fi
venv/bin/pip freeze > evidence/requirements.txt
printf 'completed\n' > evidence/DONE
