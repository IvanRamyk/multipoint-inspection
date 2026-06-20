#!/usr/bin/env bash
# One-shot setup of Cosys-AirSim (headless) on a vast.ai GPU instance.
#
# What this does:
#   1. Installs Vulkan runtime (needed by UE5 for offscreen rendering)
#   2. Downloads the Cosys-AirSim Blocks prebuilt Linux environment
#   3. Writes ~/Documents/AirSim/settings.json for our drone + depth camera
#   4. Starts AirSim headlessly in a tmux session called 'airsim'
#
# Usage:
#   bash deploy/setup_airsim.sh [AIRSIM_ZIP_URL]
#
# The default URL points to a known-good Cosys-AirSim Blocks release.
# Override with your own prebuilt env if needed.
#
# After this script: run deploy/probe_airsim.sh to verify the connection,
# then train with backend=airsim.
set -euo pipefail

AIRSIM_DIR="${HOME}/airsim"
SETTINGS_DIR="${HOME}/Documents/AirSim"

# Cosys-AirSim Blocks prebuilt Linux binary.
# Check https://github.com/Cosys-Lab/Cosys-AirSim/releases for latest.
AIRSIM_URL="${1:-https://github.com/Cosys-Lab/Cosys-AirSim/releases/download/v2.0.0-UE5.3/Blocks_Ubuntu.zip}"

echo "==> [1/4] Installing Vulkan runtime and UE5 deps..."
apt-get update -qq
apt-get install -y --no-install-recommends \
  libvulkan1 \
  vulkan-tools \
  libgl1 \
  libglib2.0-0 \
  libegl1 \
  libgles2 \
  xdg-user-dirs \
  unzip \
  wget \
  tmux >/dev/null

# Verify Vulkan sees the GPU
echo "Vulkan GPU check:"
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverVersion" || echo "  (vulkaninfo not fully available — this is OK in containers, UE uses EGL)"

echo ""
echo "==> [2/4] Downloading Cosys-AirSim Blocks environment..."
mkdir -p "$AIRSIM_DIR"
cd "$AIRSIM_DIR"

if [ -f "Blocks.sh" ]; then
  echo "  Already downloaded, skipping."
else
  echo "  Downloading from: ${AIRSIM_URL}"
  wget -q --show-progress -O airsim_env.zip "$AIRSIM_URL"
  unzip -q airsim_env.zip
  rm airsim_env.zip
  chmod +x ./*.sh 2>/dev/null || true
  echo "  Download complete."
fi

echo ""
echo "==> [3/4] Writing settings.json..."
mkdir -p "$SETTINGS_DIR"
cat > "$SETTINGS_DIR/settings.json" << 'EOF'
{
  "SettingsVersion": 2.0,
  "SimMode": "Multirotor",
  "ClockSpeed": 5.0,
  "ViewMode": "NoDisplay",
  "Vehicles": {
    "drone": {
      "VehicleType": "SimpleFlight",
      "Cameras": {
        "front": {
          "CaptureSettings": [
            {"ImageType": 1, "Width": 64, "Height": 64, "FOV_Degrees": 90}
          ],
          "X": 0.30, "Y": 0.0, "Z": 0.0,
          "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
        },
        "chase": {
          "CaptureSettings": [
            {"ImageType": 0, "Width": 640, "Height": 480, "FOV_Degrees": 90}
          ],
          "X": -3.0, "Y": 0.0, "Z": -1.5,
          "Pitch": 15.0, "Roll": 0.0, "Yaw": 0.0
        }
      }
    }
  }
}
EOF
echo "  Written to ${SETTINGS_DIR}/settings.json"

echo ""
echo "==> [4/4] Starting AirSim headlessly in tmux session 'airsim'..."
bash "$(dirname "$0")/start_airsim.sh"

echo ""
echo "Done. AirSim is starting up (takes ~30-60s for UE5 to load)."
echo ""
echo "Next steps:"
echo "  1. Wait ~60s for UE5 to load, then verify:"
echo "     bash deploy/probe_airsim.sh <HOST> <PORT>"
echo ""
echo "  2. Run training with AirSim backend:"
echo "     bash deploy/train_airsim.sh <HOST> <PORT>"
