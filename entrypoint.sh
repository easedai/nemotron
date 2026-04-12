#!/bin/bash
# Worker entrypoint — VLLM_API_KEY is injected by the eased orchestrator at
# launch time via vast.ai environment variables. A unique key is generated per
# instance so every worker is independently authenticated.
set -eo pipefail

MODEL_PATH="${MODEL_PATH:-/root/.cache/huggingface/nemotron-nano-12b-vl-bf16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_PORT="${VLLM_PORT:-8080}"

if [ -z "${VLLM_API_KEY:-}" ]; then
  echo "[entrypoint] WARNING: VLLM_API_KEY is not set — API will be unauthenticated"
fi

echo "[entrypoint] Starting vLLM — model: ${MODEL_PATH}"
echo "[entrypoint] Max model len: ${VLLM_MAX_MODEL_LEN}"
echo "[entrypoint] GPU memory utilization: ${VLLM_GPU_MEMORY_UTILIZATION}"
echo "[entrypoint] Port: ${VLLM_PORT}"

exec python -m vllm.entrypoints.openai.api_server \
  --model              "${MODEL_PATH}" \
  --dtype              bfloat16 \
  --port               "${VLLM_PORT}" \
  --host               0.0.0.0 \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-model-len      "${VLLM_MAX_MODEL_LEN}" \
  --enforce-eager \
  --trust-remote-code \
  --no-enable-log-requests \
  ${VLLM_API_KEY:+--api-key "${VLLM_API_KEY}"}
