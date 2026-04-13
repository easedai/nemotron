#!/bin/bash
# vast.ai runs this file on container start via its .launch script.
# Output is captured to /var/log/onstart.log.
# The Docker ENTRYPOINT is bypassed when using ssh_direc/ssh_proxy runtype,
# so we exec our own /entrypoint.sh instead.
#
# Both image variants bake /entrypoint.sh at build time:
#   Dockerfile        → entrypoint.sh (direct vLLM launch)
#   Dockerfile.vastai → vastai_entrypoint.sh (starts supervisord → vllm.sh)
exec /entrypoint.sh "$@"
