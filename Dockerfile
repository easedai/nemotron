# syntax=docker/dockerfile:1
#
# Build:
#   export HF_TOKEN=hf_...
#   docker build --secret id=HF_TOKEN,env=HF_TOKEN \
#     -t ghcr.io/easedai/nemotron:latest .
#
#   # 30B model:
#   docker build --secret id=HF_TOKEN,env=HF_TOKEN \
#     --build-arg MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
#     -t ghcr.io/easedai/nemotron:30b-a3b-bf16 .
#

FROM vllm/vllm-openai:latest

# ── Model selection ────────────────────────────────────────────────────────────
ARG MODEL_ID=nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
ENV MODEL_ID=${MODEL_ID}

# ── Cache directories ──────────────────────────────────────────────────────────
ENV HF_HOME=/hf
ENV VLLM_CACHE_ROOT=/vllm-cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# OpenCV is required for VLLM_VIDEO_LOADER_BACKEND=opencv (VL models only).
# hf_transfer enables fast multi-part HF downloads at build time.
RUN pip install --no-cache-dir opencv-python-headless "huggingface_hub[hf_transfer]"

# ── Download model weights at build time ───────────────────────────────────────
# HF_TOKEN is mounted as a BuildKit secret — never written into any image layer.
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
COPY patch_vllm.py /tmp/patch_vllm.py
RUN python3 /tmp/patch_vllm.py && rm /tmp/patch_vllm.py

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
