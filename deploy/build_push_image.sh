#!/usr/bin/env bash
# Build the training image and push it to Docker Hub.
# Only needed when pyproject.toml dependencies change — not on code edits.
#
# Usage:
#   bash deploy/build_push_image.sh <dockerhub-username>
#
# Example:
#   bash deploy/build_push_image.sh ivanramyk
#
# After pushing, set IMAGE=<user>/dreamer-train:latest in train_remote.sh
# (or pass it as DREAMER_IMAGE env var).
set -euo pipefail

DOCKER_USER="${1:?Usage: build_push_image.sh <dockerhub-username>}"
IMAGE="${DOCKER_USER}/dreamer-train:latest"

echo "==> Building ${IMAGE} ..."
docker build \
  --platform linux/amd64 \
  -f deploy/Dockerfile.train \
  -t "$IMAGE" \
  .

echo ""
echo "==> Pushing ${IMAGE} to Docker Hub ..."
docker push "$IMAGE"

echo ""
echo "Done. Use this image on vast.ai:"
echo "  DREAMER_IMAGE=${IMAGE} bash deploy/train_remote.sh <HOST> <PORT> ..."
echo ""
echo "Or when creating a new instance:"
echo "  vastai create instance <OFFER_ID> --image ${IMAGE} --ssh --direct --disk 20 --env '-p 6006:6006'"
