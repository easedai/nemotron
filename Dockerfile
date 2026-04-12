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

# ── vLLM 0.19.0 patches for NanoNemotronVLProcessor compat ──────────────────
# Five crash sites in the multimodal dummy-input path; applied at build time.
COPY patch_vllm.py /tmp/patch_vllm.py
RUN python3 /tmp/patch_vllm.py && rm /tmp/patch_vllm.py

# ── vast.ai onstart hook ──────────────────────────────────────────────────────
# vast.ai's ssh_direc/ssh_proxy runtype bypasses the Docker ENTRYPOINT and
# runs /root/onstart.sh instead.  The base vllm/vllm-openai image ships
# /root/onstart.sh as a symlink to /vllm-workspace/onstart.sh, and
# /vllm-workspace is declared as a Docker VOLUME — so any write through the
# symlink lands inside the volume path and is discarded at container start.
# Remove the symlink first so the COPY creates a real file outside the volume.
RUN rm -f /root/onstart.sh
COPY onstart.sh /root/onstart.sh
RUN chmod +x /root/onstart.sh

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
