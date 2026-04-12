# syntax=docker/dockerfile:1
#
# Bakes model weights directly into the image at build time.
# Model is selected via build args — defaults to Nemotron Nano 12B VL BF16.
#
# Build locally:
#   # Default model (12B):
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=hf_token,env=HF_TOKEN \
#     -t ghcr.io/easedai/nemotron:12b-vl-bf16 \
#     .
#
#   # Different model:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=hf_token,env=HF_TOKEN \
#     --build-arg MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
#     --build-arg MODEL_DIR=nemotron-30b-a3b-bf16 \
#     -t ghcr.io/easedai/nemotron:30b-a3b-bf16 \
#     .
#
FROM vllm/vllm-openai:latest

# ── Model selection ────────────────────────────────────────────────────────────
# Override at build time via --build-arg. MODEL_DIR becomes the subdirectory
# under /root/.cache/huggingface/ and is exposed as MODEL_PATH at runtime so
# entrypoint.sh knows where to load the model from.
ARG MODEL_ID=nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
ARG MODEL_DIR=nemotron-nano-12b-vl-bf16

# Fast HuggingFace downloads via the Rust hf_transfer library
RUN pip install --no-cache-dir "huggingface_hub[hf_transfer]>=0.24"

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME=/root/.cache/huggingface
# Bake the resolved path into the image so the runtime entrypoint can find it
ENV MODEL_PATH=/root/.cache/huggingface/${MODEL_DIR}

# Download model weights during build.
# HF_TOKEN is passed as a BuildKit secret — never written into any image layer
# and will not appear in `docker history` or image manifests.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) \
    huggingface-cli download \
        ${MODEL_ID} \
        --local-dir "${MODEL_PATH}" \
        --token     "$HF_TOKEN"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
