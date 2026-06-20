#!/usr/bin/env bash
# One-shot setup for a fresh vast.ai / RunPod CUDA instance that was started
# from a PyTorch base image (NOT from our Dockerfile). Run it once after SSH.
#
#   bash setup_vast.sh <git-clone-url> [branch]
#
# If you didn't push the repo to git, skip the clone and rsync the folder in
# instead (see deploy/README.md), then run this from inside the repo with no
# args to just install deps.
set -euo pipefail

REPO_URL="${1:-}"
BRANCH="${2:-main}"

echo "==> apt deps for headless PyBullet/PyFlyt"
apt-get update -qq
apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libgomp1 >/dev/null

if [ -n "$REPO_URL" ]; then
  echo "==> cloning $REPO_URL ($BRANCH)"
  git clone --branch "$BRANCH" "$REPO_URL" dreamer
  cd dreamer
fi

echo "==> installing project (base + dreamer extra; torch from base image)"
pip install --no-cache-dir -e ".[dreamer]"

echo "==> sanity: torch sees the GPU?"
python -c "import torch; print('cuda available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "==> done. Launch training with deploy/README.md commands."
