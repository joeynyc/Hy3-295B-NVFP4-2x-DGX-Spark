#!/bin/bash
# Hy3-295B NVFP4 on 2x DGX Spark — full cluster bring-up.
# Run on the HEAD node. Edit the config block, then: ./start.sh
set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────
HEAD_IP="10.100.112.2"            # this node's cluster-fabric IP
WORKER_IP="10.100.112.1"          # remote node's cluster-fabric IP
CLUSTER_IF="enp1s0f1np1"          # NIC carrying the cluster IPs
SECOND_RAIL="enP2p1s0f1np1"       # second RoCE rail (leave set even if uncabled)
NCCL_HCAS="rocep1s0f1,roceP2p1s0f1"
IMAGE="vllm-node-minimax-m3:latest"   # your GB10 vLLM image (see README)
MODEL_HOST_PATH="$HOME/models/Hy3-NVFP4"   # same path on BOTH nodes
PROFILE="profiles/hy3-nvfp4-tp2-256k-tq.sh"
CONTAINER="vllm_node"
RAY_PORT=29501
API_PORT=8000
# ─────────────────────────────────────────────────────────────────────────

here="$(cd "$(dirname "$0")" && pwd)"
say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

say "Preflight"
[ -c /dev/null ] && [ -w /dev/null ] || {
  echo "FATAL: /dev/null is broken (known Spark reboot bug)."
  echo "Fix:   sudo sh -c 'rm -f /dev/null && mknod -m 666 /dev/null c 1 3'"; exit 1; }
if systemctl is-active --quiet earlyoom; then
  echo "WARNING: earlyoom is active — stock Spark config prefers killing vllm/ray."
  echo "         Consider: sudo systemctl stop earlyoom"
fi
[ -d "$MODEL_HOST_PATH" ] || { echo "FATAL: model not found at $MODEL_HOST_PATH"; exit 1; }
ssh -o BatchMode=yes "$WORKER_IP" "test -d $MODEL_HOST_PATH" \
  || { echo "FATAL: model not found on worker at $MODEL_HOST_PATH"; exit 1; }
mtu=$(cat /sys/class/net/$CLUSTER_IF/mtu)
[ "$mtu" -ge 9000 ] || echo "NOTE: cluster MTU is $mtu; jumbo frames recommended (scripts/set-cluster-mtu-9000.sh)"

say "Waiting for memory to be free on both nodes"
for i in $(seq 1 40); do
  h=$(free -g | awk 'NR==2{print $3}')
  w=$(ssh -o BatchMode=yes "$WORKER_IP" "free -g | awk 'NR==2{print \$3}'")
  [ "$h" -lt 30 ] && [ "$w" -lt 30 ] && break
  echo "  head=${h}G worker=${w}G — waiting for previous run to release..."
  sleep 10
done

say "Removing stale containers"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
ssh "$WORKER_IP" "docker rm -f $CONTAINER" >/dev/null 2>&1 || true

docker_env() {  # $1 = node ip
  echo "-e VLLM_HOST_IP=$1 -e RAY_OVERRIDE_NODE_IP_ADDRESS=$1 -e RAY_NODE_IP_ADDRESS=$1 \
   -e NCCL_SOCKET_IFNAME=$CLUSTER_IF -e GLOO_SOCKET_IFNAME=$CLUSTER_IF \
   -e TP_SOCKET_IFNAME=$CLUSTER_IF -e UCX_NET_DEVICES=$CLUSTER_IF \
   -e MN_IF_NAME=$CLUSTER_IF -e OMPI_MCA_btl_tcp_if_include=$CLUSTER_IF \
   -e NCCL_IB_HCA=$NCCL_HCAS -e NCCL_IB_DISABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
   -e RAY_memory_monitor_refresh_ms=0 -e RAY_num_prestart_python_workers=0 \
   -e RAY_object_store_memory=1073741824"
}

say "Starting containers"
docker run -d --rm --name "$CONTAINER" --gpus all --network host --privileged --ipc host \
  $(docker_env "$HEAD_IP") -v "$MODEL_HOST_PATH":/models/Hy3-NVFP4 \
  "$IMAGE" sleep infinity >/dev/null
ssh "$WORKER_IP" "docker run -d --rm --name $CONTAINER --gpus all --network host --privileged --ipc host \
  $(docker_env "$WORKER_IP") -v $MODEL_HOST_PATH:/models/Hy3-NVFP4 \
  $IMAGE sleep infinity" >/dev/null

say "Applying mods (expert_bias fp32 — vllm#47777)"
for target in "docker exec -i $CONTAINER" "ssh $WORKER_IP docker exec -i $CONTAINER"; do
  $target bash < "$here/mods/fix-hy3-expert-bias-fp32.sh"
done

say "Starting Ray"
docker exec -d "$CONTAINER" bash -c "ray start --block --head --port $RAY_PORT \
  --object-store-memory 1073741824 --num-cpus 2 --node-ip-address $HEAD_IP \
  --include-dashboard=false --disable-usage-stats >> /proc/1/fd/1 2>&1"
sleep 8
ssh "$WORKER_IP" "docker exec -d $CONTAINER bash -c 'ray start --block \
  --object-store-memory 1073741824 --num-cpus 2 --disable-usage-stats \
  --address=$HEAD_IP:$RAY_PORT --node-ip-address $WORKER_IP >> /proc/1/fd/1 2>&1'"
for i in $(seq 1 12); do
  n=$(docker exec "$CONTAINER" ray status 2>/dev/null | grep -c "^ 1 node_" || true)
  [ "$n" -ge 2 ] && { echo "  Ray: $n nodes active"; break; }
  sleep 5
done

say "Launching vLLM ($PROFILE)"
docker cp "$here/$PROFILE" "$CONTAINER":/workspace/serve-profile.sh
docker exec -d "$CONTAINER" bash -c "bash /workspace/serve-profile.sh > /proc/1/fd/1 2>&1"

say "Waiting for API (model load takes ~10-15 min)"
for i in $(seq 1 90); do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$API_PORT/v1/models" || true)
  [ "$code" = 200 ] && break
  docker ps --format '{{.Names}}' | grep -q "$CONTAINER" || { echo "FATAL: container died — docker logs $CONTAINER"; exit 1; }
  sleep 20
done
[ "$code" = 200 ] || { echo "FATAL: API did not come up"; exit 1; }

say "API up — warmup + benchmark"
curl -s "http://127.0.0.1:$API_PORT/v1/models" | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"][0]; print(f"  model={d[\"id\"]}  max_model_len={d[\"max_model_len\"]}")'
"$here/scripts/hy3ctl" bench >/dev/null 2>&1 || true   # warmup
"$here/scripts/hy3ctl" bench

say "Ready:  http://$HEAD_IP:$API_PORT/v1   (chat: scripts/hy3chat · status: scripts/hy3ctl status)"
