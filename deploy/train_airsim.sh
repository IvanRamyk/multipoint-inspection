#!/usr/bin/env bash
# Push code and launch DreamerV3 training with the AirSim backend.
# Assumes AirSim is already running on the remote (deploy/setup_airsim.sh done).
#
# Usage:
#   bash deploy/train_airsim.sh <HOST> <PORT> [dreamer overrides...]
#
# Example:
#   bash deploy/train_airsim.sh 74.48.78.46 36249 \
#     algo.total_steps=15000 metric.log_every=200 checkpoint.every=2000
set -euo pipefail

HOST="${1:?Usage: train_airsim.sh <HOST> <PORT> [overrides...]}"; shift
PORT="${1:?Usage: train_airsim.sh <HOST> <PORT> [overrides...]}"; shift
TRAIN_ARGS="${*:-algo.total_steps=15000 metric.log_every=200 checkpoint.every=2000}"

SSH_KEY="${SSH_KEY:-${HOME}/.ssh/hetzner_agents}"
SSH_OPTS="-i ${SSH_KEY} -p ${PORT} -o StrictHostKeyChecking=no -o LogLevel=ERROR"
REMOTE="root@${HOST}"
REMOTE_DIR="/workspace/dreamer"

echo "==> Syncing code to ${REMOTE}:${REMOTE_DIR}/"
rsync -az --progress \
  --exclude venv --exclude logs --exclude results \
  --exclude .git --exclude __pycache__ \
  --exclude '*.pyc' --exclude '.DS_Store' --exclude '*.egg-info' \
  -e "ssh ${SSH_OPTS}" \
  ./ "${REMOTE}:${REMOTE_DIR}/"

echo ""
echo "==> Switching env config to sanity_airsim.yaml..."
ssh ${SSH_OPTS} "${REMOTE}" \
  "sed -i 's|config_path:.*|config_path: configs/sanity_airsim.yaml|' /workspace/dreamer/configs/sheeprl/env/drone_inspection.yaml"

echo ""
echo "==> Verifying AirSim is reachable..."
ssh ${SSH_OPTS} "${REMOTE}" \
  "cd ${REMOTE_DIR} && venv/bin/python scripts/airsim_probe.py --ip 127.0.0.1" || {
  echo ""
  echo "ERROR: AirSim probe failed. Is AirSim running?"
  echo "  Start it: ssh ${SSH_OPTS} ${REMOTE} 'bash /workspace/dreamer/deploy/start_airsim.sh'"
  echo "  Wait 60s, then retry this script."
  exit 1
}

echo ""
echo "==> Launching training (AirSim backend) in tmux session 'train'..."
FULL_CMD="venv/bin/python scripts/train_dreamer.py fabric.accelerator=gpu fabric.precision=16-mixed ${TRAIN_ARGS}"

ssh ${SSH_OPTS} "${REMOTE}" "
  cd ${REMOTE_DIR}
  tmux kill-session -t train 2>/dev/null || true
  tmux new-session -d -s train -x 220 -y 50
  tmux send-keys -t train 'cd ${REMOTE_DIR} && ${FULL_CMD} 2>&1 | tee train_airsim.log' Enter
  echo 'Training started.'
"

echo ""
echo "AirSim training is running. Useful commands:"
echo ""
echo "  # Attach to training output:"
echo "  ssh ${SSH_OPTS} ${REMOTE} -t 'tmux attach -t train'"
echo ""
echo "  # Watch AirSim UE5 log:"
echo "  ssh ${SSH_OPTS} ${REMOTE} 'tail -f ~/airsim/airsim.log'"
echo ""
echo "  # TensorBoard tunnel:"
echo "  bash deploy/tensorboard.sh ${HOST} ${PORT}"
echo ""
echo "  # Fetch checkpoints:"
echo "  bash deploy/fetch_results.sh ${HOST} ${PORT}"
