#!/usr/bin/env bash
# Pull training artifacts from a vast.ai instance to your local machine.
# Fetches the whole logs/ tree — per-run subfolders containing checkpoints,
# TensorBoard events, the merged config.yaml, AND the training videos
# (<run>/version_0/videos/) — plus results/ and train.log.
#
# Each run lives in its own timestamped folder:
#   logs/runs/dreamer_v3/<Env-id>/<timestamp>_..._<seed>/version_0/
#     checkpoint/ckpt_*.ckpt      model checkpoints
#     events.out.tfevents.*       TensorBoard scalars
#     config.yaml                 full merged Hydra config for that run
#     videos/ep<NNNNNN>_*.mp4      per-episode training videos (if recording on)
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
echo "Done. Each run is under logs/runs/dreamer_v3/<Env-id>/<timestamp>_..._<seed>/version_0/"
echo "  Checkpoints:      .../version_0/checkpoint/ckpt_*.ckpt"
echo "  Training videos:  .../version_0/videos/ep*.mp4"
echo "  TensorBoard:      ./venv/bin/tensorboard --logdir logs/runs/dreamer_v3 --port 6006"
echo ""
echo "Newest run's training videos:"
echo "  ls -t logs/runs/dreamer_v3/*/*/version_0/videos/*.mp4 2>/dev/null | head"
