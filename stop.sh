#!/bin/bash
# Tear down the Hy3 cluster on both nodes. Edit WORKER_IP to match start.sh.
set -uo pipefail
WORKER_IP="10.100.112.1"
CONTAINER="vllm_node"
docker rm -f "$CONTAINER" 2>/dev/null && echo "head: $CONTAINER removed" || echo "head: no container"
ssh "$WORKER_IP" "docker rm -f $CONTAINER" 2>/dev/null && echo "worker: $CONTAINER removed" || echo "worker: no container"
