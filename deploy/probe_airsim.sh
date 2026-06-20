#!/usr/bin/env bash
# Verify AirSim is reachable and the depth camera works.
# Run this after setup_airsim.sh and waiting ~60s for UE5 to load.
#
# Usage:
#   bash deploy/probe_airsim.sh <HOST> <PORT>
set -euo pipefail

HOST="${1:?Usage: probe_airsim.sh <HOST> <PORT>}"
PORT="${2:?Usage: probe_airsim.sh <HOST> <PORT>}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/hetzner_agents}"
SSH_OPTS="-i ${SSH_KEY} -p ${PORT} -o StrictHostKeyChecking=no -o LogLevel=ERROR"

echo "==> Running AirSim connection probe on remote..."
ssh ${SSH_OPTS} "root@${HOST}" \
  "cd /workspace/dreamer && venv/bin/python scripts/airsim_probe.py --ip 127.0.0.1"
