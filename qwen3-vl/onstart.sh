#!/bin/bash
# Vast.ai runs this on container start via its .launch script.
# Delegates to the base image's entrypoint which starts supervisord.
exec /entrypoint.sh "$@"
