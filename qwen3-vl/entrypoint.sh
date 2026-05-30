#!/usr/bin/env bash
set -euo pipefail
#
# Translate env vars to qwen3_server.py CLI flags.
#
# Build-time adapter bake-in:
#   If ADAPTER_BAKED=1 (set by Dockerfile when ADAPTER_URL was provided),
#   default ADAPTER_PATH to /adapters/checkpoint and ADAPTER_NAME to the
#   build-arg value (ADAPTER_NAME_DEFAULT), both of which can still be
#   overridden at runtime.
#
# Runtime env vars:
#   Required:  API_KEY (warns but starts without it)
#   Common:    BASE_MODEL, ADAPTER_PATH, ADAPTER_NAME, PORT
#   Optional:  LOAD_IN_4BIT, LOAD_IN_8BIT, NO_MERGE, DEVICE, HOST, HF_TOKEN

# HuggingFace login for gated models.
if [[ -n "${HF_TOKEN:-}" ]]; then
    python3 -c "from huggingface_hub import login; login('${HF_TOKEN}')" 2>/dev/null || true
fi

# Apply baked-in adapter defaults (can be overridden by runtime env vars).
if [[ "${ADAPTER_BAKED:-}" == "1" ]]; then
    ADAPTER_PATH="${ADAPTER_PATH:-/adapters/checkpoint}"
    ADAPTER_NAME="${ADAPTER_NAME:-${ADAPTER_NAME_DEFAULT:-}}"
fi

ARGS=()
[[ -n "${BASE_MODEL:-}"   ]] && ARGS+=(--base-model   "$BASE_MODEL")
[[ -n "${ADAPTER_PATH:-}" ]] && ARGS+=(--adapter      "$ADAPTER_PATH")
[[ -n "${ADAPTER_NAME:-}" ]] && ARGS+=(--adapter-name "$ADAPTER_NAME")
[[ -n "${API_KEY:-}"      ]] && ARGS+=(--api-key       "$API_KEY")
[[ -n "${DEVICE:-}"       ]] && ARGS+=(--device        "$DEVICE")
[[ -n "${HOST:-}"         ]] && ARGS+=(--host          "$HOST")
[[ -n "${PORT:-}"         ]] && ARGS+=(--port          "$PORT")

[[ "${LOAD_IN_4BIT:-}" == "1" || "${LOAD_IN_4BIT:-}" == "true" ]] && ARGS+=(--load-in-4bit)
[[ "${LOAD_IN_8BIT:-}" == "1" || "${LOAD_IN_8BIT:-}" == "true" ]] && ARGS+=(--load-in-8bit)
[[ "${NO_MERGE:-}"     == "1" || "${NO_MERGE:-}"     == "true" ]] && ARGS+=(--no-merge)

exec python3 /app/qwen3_server.py "${ARGS[@]}" "$@"
