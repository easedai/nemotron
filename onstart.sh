#!/bin/bash
# vast.ai runs this file on container start via its .launch script.
# Output is captured to /var/log/onstart.log.
# The Docker ENTRYPOINT is bypassed when using ssh_direc/ssh_proxy runtype,
# so vLLM must be started here instead.
exec /opt/instance-tools/bin/entrypoint.sh
