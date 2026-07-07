#!/bin/bash
# Set MTU 9000 on both 200G RoCE rails, on BOTH Sparks, and persist via netplan drop-in.
# WHY: netdev MTU 1500 caps the RoCE path MTU at 1024 bytes; MTU >= 4200 raises it
# to 4096, cutting per-packet overhead ~4x on the NCCL allreduce path.
# WARNING: flaps the cluster links — stop vllm first (hy3ctl stop). Run with sudo.
set -euo pipefail
NODE2=10.100.112.1
RAILS="enp1s0f1np1 enP2p1s0f1np1"

apply_local() {
  for ifc in $RAILS; do ip link set "$ifc" mtu 9000; done
  cat > /etc/netplan/98-cluster-mtu.yaml << "YAML"
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      mtu: 9000
    enP2p1s0f1np1:
      mtu: 9000
YAML
  chmod 600 /etc/netplan/98-cluster-mtu.yaml
  echo "$(hostname): MTU set + persisted"
}

if [ "${1:-}" = "--local" ]; then apply_local; exit 0; fi
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
apply_local
echo "Now run on node2:  ssh $NODE2 sudo $(realpath "$0") --local"
echo "(or copy this script over and run it there with sudo)"
echo "Verify: ping -M do -s 8972 -c 2 10.100.112.1"
