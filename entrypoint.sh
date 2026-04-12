#!/bin/bash
# Worker entrypoint — VLLM_API_KEY is injected by the eased orchestrator at
# launch time via vast.ai environment variables. A unique key is generated per
# instance so every worker is independently authenticated.
set -euo pipefail

: "${VLLM_API_KEY:?VLLM_API_KEY must be set by the orchestrator}"
: "${MODEL_PATH:=/root/.cache/huggingface/nemotron-nano-12b-vl-bf16}"
: "${VLLM_MAX_MODEL_LEN:=4096}"
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.95}"

echo "[entrypoint] Starting vLLM — model: ${MODEL_PATH}"
echo "[entrypoint] Max model len: ${VLLM_MAX_MODEL_LEN}"
echo "[entrypoint] GPU memory utilization: ${VLLM_GPU_MEMORY_UTILIZATION}"

exec python -m vllm.entrypoints.openai.api_server \
  --model              "${MODEL_PATH}" \
  --dtype              bfloat16 \
  --port               8000 \
  --host               0.0.0.0 \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-model-len      "${VLLM_MAX_MODEL_LEN}" \
  --enforce-eager \
  --trust-remote-code \
  --disable-log-requests \
  --api-key            "${VLLM_API_KEY}"
