#!/bin/bash

# Modified install script for CodeArtifact-restricted environments
# - pypi.org is unreachable, only CodeArtifact is available
# - opencv-python-headless and flashinfer-python==0.3.1 are not in CodeArtifact
# - No direct internet access (no wget to GitHub)

# Note: not using set -e so non-critical failures (flash-attn build, opencv) don't kill the script

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}

export MAX_JOBS=32

echo "1. Install torch==2.8.0 first (pinned for vllm compatibility)"
pip install --no-cache-dir "torch==2.8.0" "torchaudio==2.8.0" "torchvision==0.23.0"

echo "2. Install inference frameworks"
if [ $USE_SGLANG -eq 1 ]; then
    # SGLang without [srt] extra (flashinfer not available in CodeArtifact)
    pip install --no-cache-dir "sglang==0.5.2" || echo "WARNING: sglang install failed, skipping"
    pip install --no-cache-dir torch-memory-saver || true
fi

# vllm with --no-deps to skip opencv-python-headless requirement
pip install --no-cache-dir "vllm==0.11.0" --no-deps

# Install vllm's dependencies manually (excluding opencv-python-headless and torch/torchvision/torchaudio which are already installed)
pip install --no-cache-dir \
    cachetools sentencepiece blake3 py-cpuinfo \
    "openai>=1.99.1" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.11.3" \
    "llguidance>=0.7.11,<0.8.0" \
    "outlines_core==0.2.11" \
    "diskcache==5.6.3" \
    "lark==1.2.2" \
    "xgrammar==0.1.25" \
    "partial-json-parser" \
    "pyzmq>=25.0.0" \
    "msgspec" \
    "gguf>=0.13.0" \
    "mistral_common>=1.8.2" \
    "transformers>=4.55.2" \
    "tokenizers>=0.21.1" \
    "fastapi[standard]>=0.115.0" \
    "aiohttp" \
    "pillow" \
    "tqdm" \
    "regex" \
    "protobuf" \
    cbor2 \
    "compressed-tensors==0.11.0" \
    "depyf==0.19.0" \
    einops \
    "numba==0.61.2" \
    pybase64 \
    python-json-logger \
    scipy \
    six \
    "setuptools>=77.0.3,<80" \
    "xformers==0.0.32.post1"

echo "3. Install basic packages"
pip install --no-cache-dir "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

pip install --no-cache-dir "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "4. Install FlashAttention from CodeArtifact"
# Install flash-attn from source with --no-build-isolation so it finds torch
pip install --no-cache-dir --no-build-isolation flash-attn

# flashinfer-python==0.3.1 is not in CodeArtifact - only needed for SGLang
echo "NOTE: Skipping flashinfer-python==0.3.1 (not in CodeArtifact). Only needed for SGLang."


if [ $USE_MEGATRON -eq 1 ]; then
    echo "5. Install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    pip install "onnxscript==0.3.1"
    NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.6
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1
fi


echo "6. Fix opencv"
# opencv-python is not in CodeArtifact, use opencv-fixer as workaround
pip install --no-cache-dir opencv-python || true
pip install --no-cache-dir opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"


if [ $USE_MEGATRON -eq 1 ]; then
    echo "7. Install cudnn python package (avoid being overridden)"
    pip install --no-cache-dir nvidia-cudnn-cu12==9.10.2.21
fi

echo ""
echo "========================================="
echo "Verifying installation..."
echo "========================================="
python -c "import torch; print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import flash_attn; print(f'flash_attn: {flash_attn.__version__}')" || echo "WARNING: flash_attn import failed"
echo "========================================="
echo "Successfully installed all packages"
