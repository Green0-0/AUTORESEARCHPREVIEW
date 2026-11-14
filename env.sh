# Source in every job / shell: keeps caches and big files off the near-full home quota.
# Modules load FIRST: the uv module sets UV_CACHE_DIR to ~/.cache/uv, which the exports below override.
[ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
module load uv/0.11.26 gcc/13.3 2>/dev/null   # no system CUDA: torch/vLLM wheels bundle CUDA 13; system nvcc 12.9 breaks JIT
export AR_ROOT=/usr/xtmp/kg349/ar
export AR_REPO=/home/users/kg349/AUTORESEARCHPREVIEW
export HF_HOME=$AR_ROOT/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_CACHE_DIR=$AR_ROOT/uv-cache
export UV_PYTHON_INSTALL_DIR=$AR_ROOT/uv-python
export XDG_CACHE_HOME=$AR_ROOT/cache
export VLLM_CACHE_ROOT=$AR_ROOT/cache/vllm
export TORCH_HOME=$AR_ROOT/cache/torch
export TRITON_CACHE_DIR=$AR_ROOT/cache/triton
export FLASHINFER_WORKSPACE_BASE=$AR_ROOT/cache
export FLASHINFER_JIT_DIR=$AR_ROOT/cache/flashinfer
export VLLM_USE_FLASHINFER_SAMPLER=0   # avoid FlashInfer JIT (no matching nvcc)
export PIP_CACHE_DIR=$AR_ROOT/cache/pip
export TMPDIR=$AR_ROOT/tmp
export AR_MODEL=RedHatAI/Qwen3.8-27B-INT4
export AR_DATA=$AR_ROOT/data
export PATH=$HOME/.local/bin:$PATH
unset ANTHROPIC_API_KEY   # claude -p must bill the subscription
[ -f $AR_ROOT/venv/bin/activate ] && . $AR_ROOT/venv/bin/activate
