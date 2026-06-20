#!/usr/bin/env bash
# Start (or restart) AirSim headlessly in a tmux session.
# Assumes setup_airsim.sh has already been run on this machine.
#
# Usage (run on the remote, or via SSH):
#   bash deploy/start_airsim.sh
set -euo pipefail

AIRSIM_DIR="${HOME}/airsim"
LOG="${AIRSIM_DIR}/airsim.log"

# Find the launch script — Cosys-AirSim prebuilts use *.sh
LAUNCH=$(find "$AIRSIM_DIR" -maxdepth 1 -name "*.sh" | head -1)
if [ -z "$LAUNCH" ]; then
  echo "ERROR: No .sh launcher found in ${AIRSIM_DIR}."
  echo "       Run deploy/setup_airsim.sh first."
  exit 1
fi

echo "==> Killing any existing AirSim session..."
tmux kill-session -t airsim 2>/dev/null || true

echo "==> Starting AirSim: ${LAUNCH}"
echo "    Flags: -RenderOffscreen (headless GPU rendering)"
echo "    Log:   ${LOG}"

tmux new-session -d -s airsim -x 220 -y 50
tmux send-keys -t airsim \
  "\"${LAUNCH}\" -RenderOffscreen 2>&1 | tee \"${LOG}\"" \
  Enter

echo ""
echo "AirSim is starting. UE5 takes ~30-60s to load."
echo "Watch the log: tmux attach -t airsim"
echo "Or tail it:    tail -f ${LOG}"
