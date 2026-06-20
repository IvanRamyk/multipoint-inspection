#!/usr/bin/env bash
# Pull training artifacts from a vast.ai instance to your local machine.
# Fetches: checkpoints, TensorBoard events, route plots, and train.log.
#
# Usage:
#   bash deploy/fetch_results.sh <HOST> <PORT>
#
# Outputs land in ./logs/ and ./results/ (same paths the local eval scripts expect).
set -euo pipefail

HOST="${1:?Usage: fetch_results.sh <HOST> <PORT>}"; shift
PORT="${1:?Usage: fetch_results.sh <HOST> <PORT>}"

REMOTE="root@${HOST}"
REMOTE_DIR="/workspace/dreamer"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/hetzner_agents}"
SSH_OPTS="-i ${SSH_KEY} -p ${PORT} -o StrictHostKeyChecking=no -o LogLevel=ERROR"

echo "==> Fetching logs/ (checkpoints + TensorBoard events) from ${REMOTE}..."
rsync -az --progress \
  -e "ssh ${SSH_OPTS}" \
  "${REMOTE}:${REMOTE_DIR}/logs/" ./logs/

echo ""
echo "==> Fetching results/ (route plots) from ${REMOTE}..."
rsync -az --progress \
  -e "ssh ${SSH_OPTS}" \
  "${REMOTE}:${REMOTE_DIR}/results/" ./results/ 2>/dev/null || echo "  (no results/ yet — run eval_dreamer.py first)"

echo ""
echo "==> Fetching train.log..."
scp ${SSH_OPTS} "${REMOTE}:${REMOTE_DIR}/train.log" ./train_remote.log 2>/dev/null \
  || echo "  (no train.log yet)"

echo ""
echo "Done. Local paths:"
echo "  Checkpoints: logs/runs/dreamer_v3/DroneInspection-sheeprl-v0/*/version_0/checkpoint/"
echo "  TensorBoard: ./venv/bin/tensorboard --logdir logs/runs/dreamer_v3 --port 6006"
echo ""
echo "To eval the latest fetched checkpoint:"
echo "  CKPT=\$(ls -td logs/runs/dreamer_v3/DroneInspection-sheeprl-v0/*/version_0/checkpoint 2>/dev/null | head -1 | xargs -I{} ls -t {}/ckpt_*.ckpt 2>/dev/null | head -1)"
echo "  ./venv/bin/python scripts/eval_dreamer.py \"\$CKPT\" --config configs/easy.yaml --episodes 5 --render --greedy"
