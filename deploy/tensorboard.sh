#!/usr/bin/env bash
# Open a TensorBoard SSH tunnel to a vast.ai instance.
# Blocks until you Ctrl+C. While running, open http://localhost:6006
#
# Usage:
#   bash deploy/tensorboard.sh <HOST> <PORT>
set -euo pipefail

HOST="${1:?Usage: tensorboard.sh <HOST> <PORT>}"
PORT="${2:?Usage: tensorboard.sh <HOST> <PORT>}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/hetzner_agents}"

echo "TensorBoard tunnel open -> http://localhost:6006"
echo "Ctrl+C to close."
ssh -i "$SSH_KEY" -p "$PORT" -o StrictHostKeyChecking=no \
  -N -L 6006:localhost:16006 "root@${HOST}"
