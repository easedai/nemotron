#!/bin/bash
# vLLM supervisor wrapper — managed by Supervisor in the vastai/base-image.
# Logs go to stdout so they appear in Vast.ai's logging system.

# Source vastai supervisor utilities when present (vastai base image only).
utils=/opt/supervisor-scripts/utils
# shellcheck disable=SC1090,SC1091
[ -f "${utils}/logging.sh" ]     && . "${utils}/logging.sh"
[ -f "${utils}/environment.sh" ] && . "${utils}/environment.sh"

# Activate the venv if it exists (vastai base image), otherwise use system Python.
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate

# ── Cache directories ──────────────────────────────────────────────────────────
# HF_HOME and VLLM_CACHE_ROOT are baked into the image via ENV in the Dockerfile.
# They can be overridden at runtime but must match the build-time values so vLLM
# resolves MODEL_ID to the baked-in weights without a network download.
export HF_HOME="${HF_HOME:-/models/hf}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/models/vllm-cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export VLLM_VIDEO_LOADER_BACKEND="${VLLM_VIDEO_LOADER_BACKEND:-opencv}"

MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16}"
VLLM_PORT="${VLLM_PORT:-8080}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"

# Auto-generate CUDA_VISIBLE_DEVICES from total GPU count when not explicitly set.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  TOTAL_GPUS=$(( TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE ))
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(( TOTAL_GPUS - 1 )))
fi
export CUDA_VISIBLE_DEVICES

if [ -z "${VLLM_API_KEY:-}" ]; then
  echo "[vllm] WARNING: VLLM_API_KEY is not set — API will be unauthenticated"
fi

echo "[vllm] HF_HOME=${HF_HOME}  VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
echo "[vllm] model=${MODEL_ID}  port=${VLLM_PORT}  max_model_len=${VLLM_MAX_MODEL_LEN}"
echo "[vllm] tensor_parallel=${TENSOR_PARALLEL_SIZE}  data_parallel=${DATA_PARALLEL_SIZE}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

set -- \
  --model                   "${MODEL_ID}" \
  --trust-remote-code \
  --port                    "${VLLM_PORT}" \
  --host                    0.0.0.0 \
  --gpu-memory-utilization  "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-model-len           "${VLLM_MAX_MODEL_LEN}" \
  --served-model-name       "${MODEL_ID}" \
  --data-parallel-size      "${DATA_PARALLEL_SIZE}"

[ "${TENSOR_PARALLEL_SIZE}" -gt 1 ] && set -- "$@" --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"

# Vision-Language models: add video/multimodal flags
# Matches any model ID containing "-VL-" or ending in "-VL" (case-sensitive).
if [[ "${MODEL_ID}" == *"-VL-"* || "${MODEL_ID}" == *"-VL" ]]; then
  set -- "$@" \
    --media-io-kwargs        '{"video": {"fps": 2, "num_frames": 128}}' \
    --allowed-local-media-path / \
    --video-pruning-rate     0.75
fi

[ -n "${VLLM_API_KEY:-}" ] && set -- "$@" --api-key "${VLLM_API_KEY}"

exec python3 -m vllm.entrypoints.openai.api_server "$@"
