#!/bin/bash
# Entrypoint for Dockerfile.vastai.
#
# The vastai/vllm base image's own boot chain (entrypoint.sh → boot_default.sh
# → /etc/vast_boot.d/*.sh) handles CUDA compat, SSH key propagation, and other
# host setup — but that chain is an opaque black box we can't version or modify.
#
# Instead we start supervisord directly.  The orchestrator already injects SSH
# keys and applies vLLM patches via EXTRA_COMMANDS before onstart.sh runs, so
# we don't need the base image's provisioning machinery.
#
# supervisord reads /etc/supervisor/supervisord.conf (from the base image) and
# /etc/supervisor/conf.d/vllm.conf (copied by our Dockerfile), which together
# launch /opt/supervisor-scripts/vllm.sh with auto-restart.
exec supervisord -n -u root -c /etc/supervisor/supervisord.conf
