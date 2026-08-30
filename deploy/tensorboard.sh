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
# Forwards to remote :6006 (the port create_instance.sh maps and the remote
# tensorboard binds) and starts tensorboard in the same invocation, so the
# tunnel never points at a port nothing is listening on.
ssh -i "$SSH_KEY" -p "$PORT" -o StrictHostKeyChecking=no \
  -L 6006:localhost:6006 "root@${HOST}" \
  'cd /workspace/dreamer && (venv/bin/tensorboard --logdir logs/runs/dreamer_v3 --port 6006 2>/dev/null || tensorboard --logdir logs/runs/dreamer_v3 --port 6006)'
