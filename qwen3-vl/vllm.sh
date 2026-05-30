#!/bin/bash
# vLLM supervisor wrapper for Vast.ai — managed by Supervisor.
# Logs go to stdout so they appear in Vast.ai's logging system.

utils=/opt/supervisor-scripts/utils
# shellcheck disable=SC1090,SC1091
[ -f "${utils}/logging.sh" ]     && . "${utils}/logging.sh"
[ -f "${utils}/environment.sh" ] && . "${utils}/environment.sh"

# ── Env discovery (same resilient multi-path approach as nemotron) ────────────
_env_log=""

if [ -f /etc/vllm-api-key ] && [ -z "${API_KEY:-}" ]; then
  API_KEY="$(tr -d '\n\r' < /etc/vllm-api-key)"
  [ -n "${API_KEY}" ] && export API_KEY && _env_log="${_env_log}api-key<-/etc/vllm-api-key "
fi

if [ -f /etc/vllm-env.sh ]; then
  # shellcheck disable=SC1091
  . /etc/vllm-env.sh
  _env_log="${_env_log}vllm-env.sh(loaded) "
else
  _env_log="${_env_log}vllm-env.sh(MISSING) "
fi

for _pid_env in /proc/[0-9]*/environ; do
  [ -r "${_pid_env}" ] || continue
  while IFS= read -r -d '' _kv; do
    _varname="${_kv%%=*}"
    case "${_varname}" in
      API_KEY|BASE_MODEL|LORA_NAME|LORA_PATH|PORT|MAX_MODEL_LEN|GPU_MEMORY_UTILIZATION|TENSOR_PARALLEL_SIZE|MAX_LORA_RANK|HF_TOKEN|HF_HOME|VLLM_CACHE_ROOT)
        if [ -z "${!_varname+x}" ] || [ -z "${!_varname}" ]; then
          export "${_varname}=${_kv#*=}"
          _env_log="${_env_log}${_varname} "
        fi
        ;;
    esac
  done < "${_pid_env}" 2>/dev/null
done
unset _kv _varname _pid_env

echo "[vllm] env discovery: ${_env_log}"
unset _env_log

# ── Activate venv ─────────────────────────────────────────────────────────────
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate

# ── Defaults ──────────────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/models/hf}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/models/vllm-cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3-VL-8B-Instruct}"
LORA_NAME="${LORA_NAME:-caption}"
LORA_PATH="${LORA_PATH:-/adapters/checkpoint}"
PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-image=5,video=2}"

if [ -z "${API_KEY:-}" ]; then
  echo "[vllm] WARNING: API_KEY is not set — API will be unauthenticated"
fi

# ── Adapter check ─────────────────────────────────────────────────────────────
if [ ! -f "${LORA_PATH}/adapter_config.json" ]; then
  echo "[vllm] ERROR: adapter_config.json not found at ${LORA_PATH}"
  echo "[vllm] Mount the checkpoint directory to ${LORA_PATH} and restart."
  exit 1
fi

# ── HuggingFace login ─────────────────────────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
  python3 -c "from huggingface_hub import login; login('${HF_TOKEN}')" 2>/dev/null || true
fi

echo "[vllm] base_model=${BASE_MODEL}"
echo "[vllm] lora=${LORA_NAME}@${LORA_PATH}"
echo "[vllm] port=${PORT}  max_model_len=${MAX_MODEL_LEN}  tp=${TENSOR_PARALLEL_SIZE}"

set -- \
  "${BASE_MODEL}" \
  --enable-lora \
  --lora-modules         "${LORA_NAME}=${LORA_PATH}" \
  --max-lora-rank        "${MAX_LORA_RANK}" \
  --port                 "${PORT}" \
  --host                 0.0.0.0 \
  --trust-remote-code \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len        "${MAX_MODEL_LEN}" \
  --limit-mm-per-prompt  "${LIMIT_MM_PER_PROMPT}"

[ "${TENSOR_PARALLEL_SIZE}" -gt 1 ] && set -- "$@" --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
[ -n "${API_KEY:-}" ]               && set -- "$@" --api-key "${API_KEY}"

exec python3 -m vllm.entrypoints.openai.api_server "$@"
