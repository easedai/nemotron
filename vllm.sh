#!/bin/bash
# vLLM supervisor wrapper — managed by Supervisor in the vastai/base-image.
# Logs go to stdout so they appear in Vast.ai's logging system.

# Source vastai supervisor utilities when present (vastai base image only).
utils=/opt/supervisor-scripts/utils
# shellcheck disable=SC1090,SC1091
[ -f "${utils}/logging.sh" ]     && . "${utils}/logging.sh"
[ -f "${utils}/environment.sh" ] && . "${utils}/environment.sh"

# ── Orchestrator env discovery ───────────────────────────────────────────────
# The orchestrator injects config (incl. VLLM_API_KEY) via two paths:
#   1. Docker -e KEY=VAL flags → container process env
#   2. EXTRA_COMMANDS writes /etc/vllm-env.sh (sourced below)
#
# vast.ai's ssh_direc/ssh_proxy runtype + its base-image boot chain can strip
# or overwrite either source, so we try every path, log what we find, and only
# fall through to the insecure "no API key" warning when genuinely nothing
# is available.

_env_log=""

# (1) Dedicated key file — simplest fallback, written by EXTRA_COMMANDS.
if [ -f /etc/vllm-api-key ] && [ -z "${VLLM_API_KEY:-}" ]; then
  VLLM_API_KEY="$(tr -d '\n\r' < /etc/vllm-api-key)"
  [ -n "${VLLM_API_KEY}" ] && export VLLM_API_KEY && _env_log="${_env_log}api-key<-/etc/vllm-api-key "
fi

# (2) /etc/vllm-env.sh — full env override file.
if [ -f /etc/vllm-env.sh ]; then
  # shellcheck disable=SC1091
  . /etc/vllm-env.sh
  _env_log="${_env_log}vllm-env.sh(loaded) "
else
  _env_log="${_env_log}vllm-env.sh(MISSING) "
fi

# (3) Scan every process's environ for Docker -e vars.
# Process env lives in kernel memory and can't be overwritten by file writes,
# so this works even when vast.ai's boot chain clobbers our EXTRA_COMMANDS.
_scan_found=""
for _pid_env in /proc/[0-9]*/environ; do
  [ -r "${_pid_env}" ] || continue
  while IFS= read -r -d '' _kv; do
    _varname="${_kv%%=*}"
    case "${_varname}" in
      VLLM_API_KEY|VLLM_PORT|VLLM_MAX_MODEL_LEN|VLLM_GPU_MEMORY_UTILIZATION|VLLM_VIDEO_LOADER_BACKEND|VLLM_CACHE_ROOT|MODEL_ID|HF_HOME|HF_HUB_ENABLE_HF_TRANSFER|TENSOR_PARALLEL_SIZE|DATA_PARALLEL_SIZE|CUDA_VISIBLE_DEVICES)
        if [ -z "${!_varname+x}" ] || [ -z "${!_varname}" ]; then
          export "${_varname}=${_kv#*=}"
          _scan_found="${_scan_found}${_varname} "
        fi
        ;;
    esac
  done < "${_pid_env}" 2>/dev/null
done
unset _kv _varname _pid_env
[ -n "${_scan_found}" ] && _env_log="${_env_log}scan(${_scan_found% })"

echo "[vllm] env discovery: ${_env_log}"
unset _env_log _scan_found

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
