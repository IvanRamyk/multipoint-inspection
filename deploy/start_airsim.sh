#!/usr/bin/env bash
# Start (or restart) AirSim headlessly as the 'airsim' non-root user.
# Assumes setup_airsim.sh has already been run on this machine.
#
# Usage (run on the remote, or via SSH):
#   bash deploy/start_airsim.sh
set -euo pipefail

AIRSIM_DIR="${HOME}/airsim"
# Log must be writable by the airsim user, not /root
LOG="/home/airsim/airsim.log"

# Find the launch script — Cosys-AirSim prebuilts nest it under Linux/
LAUNCH=$(find "$AIRSIM_DIR" -name "Blocks.sh" | head -1)
if [ -z "$LAUNCH" ]; then
  echo "ERROR: No .sh launcher found in ${AIRSIM_DIR}."
  echo "       Run deploy/setup_airsim.sh first."
  exit 1
fi

echo "==> Killing any existing AirSim processes..."
pkill -f "Blocks" 2>/dev/null || true
sleep 1

# UE5 refuses to run as root. Create a dedicated user if needed.
if ! id airsim >/dev/null 2>&1; then
  echo "==> Creating non-root 'airsim' user for UE5..."
  useradd -m -s /bin/bash airsim
fi

# Share AirSim settings and binary ownership
mkdir -p /home/airsim/Documents/AirSim
cp ~/Documents/AirSim/settings.json /home/airsim/Documents/AirSim/settings.json 2>/dev/null || true
chown -R airsim:airsim /home/airsim
chown -R airsim:airsim "$AIRSIM_DIR"
# /root itself is mode 700 — airsim needs execute (traverse) permission to reach the binary
chmod o+x /root

echo "==> Starting AirSim as 'airsim' user: ${LAUNCH}"
echo "    Log: ${LOG}"

# Write wrapper into airsim's home so they own it and can write the log
WRAPPER=/home/airsim/run_airsim.sh
cat > "$WRAPPER" << SCRIPT
#!/bin/bash
export HOME=/home/airsim
export USER=airsim
export DISPLAY=:0
exec "${LAUNCH}" -RenderOffscreen >> "${LOG}" 2>&1
SCRIPT
chmod +x "$WRAPPER"
chown airsim:airsim "$WRAPPER"

# Touch the log so airsim can write to it
touch "$LOG"
chown airsim:airsim "$LOG"

echo "==> Launching via runuser (detached)..."
# Run in background, disown so it survives SSH exit
nohup runuser -u airsim -- bash "$WRAPPER" </dev/null &
AIRSIM_PID=$!
disown $AIRSIM_PID
echo "$AIRSIM_PID" > /tmp/airsim.pid

echo ""
echo "AirSim started (PID ${AIRSIM_PID}). UE5 takes ~30-60s to load."
echo "Check log:  tail -f ${LOG}"
echo "Stop it:    kill \$(cat /tmp/airsim.pid)"
