# syntax=docker/dockerfile:1
#
# Qwen3-VL inference server — Unsloth + FastAPI.
# Model weights are baked in at build time; the container starts with no
# network download needed.  Pass --secret id=HF_TOKEN for gated/private repos.
#
# ── Build (model only, no adapter) ─────────────────────────────────────────
#   docker build \
#     --secret id=HF_TOKEN,env=HF_TOKEN \
#     -t ghcr.io/<org>/qwen3-vl:latest .
#
# ── Build with a baked-in adapter ──────────────────────────────────────────
#   docker build \
#     --secret id=HF_TOKEN,env=HF_TOKEN \
#     --build-arg ADAPTER_URL=username/my-lora \   # HF Hub ID  …or…
#     --build-arg ADAPTER_URL=https://…/adapter.zip \  # archive URL
#     --build-arg ADAPTER_NAME=my-ocr-model \
#     -t ghcr.io/<org>/qwen3-vl:my-ocr-model .
#
# ── Run ────────────────────────────────────────────────────────────────────
#   docker run --gpus all \
#     -e API_KEY=sk-secret \
#     -p 8001:8001 \
#     ghcr.io/<org>/qwen3-vl:my-ocr-model
#
#   Mount a runtime adapter (if none was baked in):
#     -v /path/to/adapter:/adapters/checkpoint:ro \
#     -e ADAPTER_PATH=/adapters/checkpoint
#
# ── Runtime env vars ───────────────────────────────────────────────────────
#   API_KEY            Bearer token for auth (warns but starts without it)
#   BASE_MODEL         Override the baked-in model ID
#   ADAPTER_PATH       Override adapter path (default: /adapters/checkpoint if baked in)
#   ADAPTER_NAME       Model ID to expose for the adapter
#   LOAD_IN_4BIT       Set to "1" to enable 4-bit quantization
#   PORT               Listening port (default: 8001)

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models/hf \
    HF_XET_HIGH_PERFORMANCE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY qwen3_server.py .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /models/hf /adapters/checkpoint

# ── Build-time model download ───────────────────────────────────────────────
# Model weights are downloaded into HF_HOME (/models/hf) so the container
# starts without re-downloading.  Pass --build-arg BASE_MODEL=… to override.
ARG BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"

RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN="$(cat /run/secrets/HF_TOKEN 2>/dev/null || true)" \
    hf download "$BASE_MODEL"

# Bake the chosen model as the container default so entrypoint.sh picks it up.
ENV BASE_MODEL=${BASE_MODEL}

# ── Optional build-time adapter bake-in ────────────────────────────────────
# Pass --build-arg ADAPTER_URL=https://…/adapter.zip to embed an adapter into
# the image.  ADAPTER_NAME sets the default model ID for /v1/models (can still
# be overridden at runtime via the ADAPTER_NAME env var).
ARG ADAPTER_URL=""
ARG ADAPTER_NAME=""

RUN if [ -n "$ADAPTER_URL" ]; then \
        echo "Downloading adapter from $ADAPTER_URL …"; \
        mkdir -p /tmp/adapter_extract; \
        curl -fsSL "$ADAPTER_URL" -o /tmp/adapter.zip; \
        unzip -q /tmp/adapter.zip -d /tmp/adapter_extract; \
        rm /tmp/adapter.zip; \
        ADAPTER_DIR=$(find /tmp/adapter_extract -name "adapter_config.json" | head -1 | xargs dirname); \
        echo "Installing adapter from $ADAPTER_DIR"; \
        cp -r "$ADAPTER_DIR"/. /adapters/checkpoint/; \
        rm -rf /tmp/adapter_extract; \
    fi

# Persist the build-arg ADAPTER_NAME as an env default so entrypoint.sh picks
# it up without requiring the user to re-pass it at runtime.
ENV ADAPTER_NAME_DEFAULT=${ADAPTER_NAME}
# Mark that an adapter was baked in so entrypoint.sh can set ADAPTER_PATH.
ENV ADAPTER_BAKED=${ADAPTER_URL:+1}

EXPOSE 8001

ENTRYPOINT ["/entrypoint.sh"]
