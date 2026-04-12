# syntax=docker/dockerfile:1
#
# Bakes model weights directly into the image at build time.
# Model is selected via build args — defaults to Nemotron Nano 12B VL BF16.
#
# Build locally:
#   # Default model (12B):
#   docker build \
#     --secret id=HF_TOKEN,env=HF_TOKEN \
#     -t ghcr.io/easedai/nemotron:12b-vl-bf16 \
#     .
#
#   # 30B model:
#   docker build \
#     --secret id=HF_TOKEN,env=HF_TOKEN \
#     --build-arg MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
#     -t ghcr.io/easedai/nemotron:30b-a3b-bf16 \
#     .
#

FROM vllm/vllm-openai:latest

# ── Model selection ────────────────────────────────────────────────────────────
# Override at build time via --build-arg MODEL_ID=...
# The value is baked into the image ENV so entrypoint.sh knows which model to load.
ARG MODEL_ID=nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
ENV MODEL_ID=${MODEL_ID}

# ── Cache directories ──────────────────────────────────────────────────────────
# HF_HOME   -- HuggingFace model/tokenizer cache (standard HF convention)
# VLLM_CACHE_ROOT -- vLLM compiled Triton kernels and torch.compile artifacts
ENV HF_HOME=/hf
ENV VLLM_CACHE_ROOT=/vllm-cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# OpenCV is required for VLLM_VIDEO_LOADER_BACKEND=opencv (VL models only)
RUN pip install --no-cache-dir opencv-python-headless

# ── Download model weights at build time ───────────────────────────────────────
# HF_TOKEN is mounted as a BuildKit secret — never written into any image layer.
# snapshot_download respects HF_HOME and stores the model at
#   $HF_HOME/hub/models--<org>--<name>/snapshots/<hash>/
# vLLM resolves MODEL_ID to that path automatically at runtime.
RUN --mount=type=secret,id=HF_TOKEN \
    python3 -c "\
import os; \
from huggingface_hub import snapshot_download; \
model_id = os.environ['MODEL_ID']; \
token = open('/run/secrets/HF_TOKEN').read().strip() if os.path.exists('/run/secrets/HF_TOKEN') else None; \
print(f'Downloading {model_id} (token={bool(token)})'); \
snapshot_download(model_id, token=token, ignore_patterns=['*.pt', 'original/']); \
print('Download complete')"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
