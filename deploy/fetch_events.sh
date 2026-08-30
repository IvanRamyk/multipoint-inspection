#!/usr/bin/env bash
# Pull ONLY the lightweight training telemetry from a vast.ai instance:
# TensorBoard event files, the merged config.yaml, and train.log.
#
# This is the mid-run health-check counterpart to fetch_results.sh. Checkpoints
# are ~105 MB each, so fetching the full logs/ tree every half hour just to read
# a reward curve is wasteful; this transfers kilobytes instead.
#
# Usage:
#   bash deploy/fetch_events.sh <HOST> <PORT>
#
# Artifacts land in ./logs/ at the same paths fetch_results.sh uses, so
# tools/run_health.py sees an ordinary (checkpoint-less) run directory.
set -euo pipefail

HOST="${1:?Usage: fetch_events.sh <HOST> <PORT>}"; shift
PORT="${1:?Usage: fetch_events.sh <HOST> <PORT>}"

REMOTE="root@${HOST}"
REMOTE_DIR="/workspace/dreamer"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/hetzner_agents}"
SSH_OPTS="-i ${SSH_KEY} -p ${PORT} -o StrictHostKeyChecking=no -o LogLevel=ERROR"

echo "==> Fetching TensorBoard events + config from ${REMOTE} (no checkpoints)..."
# --prune-empty-dirs keeps the local tree from filling with empty checkpoint/ dirs.
rsync -az --prune-empty-dirs \
  --include '*/' \
  --include 'events.out.tfevents.*' \
  --include 'config.yaml' \
  --exclude '*' \
  -e "ssh ${SSH_OPTS}" \
  "${REMOTE}:${REMOTE_DIR}/logs/" ./logs/

echo "==> Fetching train.log..."
scp ${SSH_OPTS} "${REMOTE}:${REMOTE_DIR}/train.log" ./train_remote.log 2>/dev/null \
  || echo "  (no train.log yet)"

echo "Done. Inspect with: tools/run_health.py <version_0 dir>"
