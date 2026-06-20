#!/usr/bin/env bash
# Create a vast.ai instance from an offer ID and wait until it's SSH-ready.
#
# Usage:
#   bash deploy/create_instance.sh <OFFER_ID>
#
# Example:
#   bash deploy/create_instance.sh 37678984
#
# Requires: vastai CLI authenticated (vastai set api-key <KEY>)
set -euo pipefail

VASTAI="/Users/ivan.ramyk/dev/personal/dreamer/venv/bin/vastai"
OFFER_ID="${1:?Usage: create_instance.sh <OFFER_ID>}"

# pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime is on DockerHub even if vast UI
# doesn't list it. Falls back cleanly if the tag is unavailable — vastai will
# report an image pull error on the instance, and you can destroy + retry with
# FALLBACK below.
IMAGE="pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
FALLBACK_IMAGE="pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"

echo "==> Creating instance from offer ${OFFER_ID}"
echo "    image : ${IMAGE}"
echo "    disk  : 20 GB"
echo "    mode  : ssh + direct (no Jupyter)"
echo ""

# --env exposes TensorBoard port (6006) so you can also open it directly
# without an SSH tunnel, though SSH tunnel is also fine.
# -p 6006:6006 maps container:6006 -> host:6006
RAW=$("$VASTAI" create instance "$OFFER_ID" \
  --image "$IMAGE" \
  --disk 20 \
  --ssh \
  --direct \
  --env '-p 6006:6006' \
  --onstart-cmd "env | grep _ >> /etc/environment; echo instance ready" \
  --raw)

echo "Raw response: $RAW"

# Extract the new instance ID from {"success": true, "new_contract": 12345678}
INSTANCE_ID=$(echo "$RAW" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['new_contract'])" 2>/dev/null || echo "")

if [ -z "$INSTANCE_ID" ]; then
  echo ""
  echo "ERROR: Could not parse instance ID from response above."
  echo "If the image failed, retry with the fallback:"
  echo "  Edit this script: IMAGE=\"${FALLBACK_IMAGE}\""
  exit 1
fi

echo ""
echo "Instance created: ID = ${INSTANCE_ID}"
echo "==> Waiting for it to become SSH-ready (this takes 1–3 min)..."

for i in $(seq 1 40); do
  sleep 10
  SSH_URL=$("$VASTAI" ssh-url "$INSTANCE_ID" --raw 2>/dev/null | tr -d '"' || echo "")

  if [[ "$SSH_URL" == ssh://* ]]; then
    # Parse ssh://root@1.2.3.4:12345
    HOST=$(echo "$SSH_URL" | sed 's|ssh://[^@]*@||' | cut -d: -f1)
    PORT=$(echo "$SSH_URL" | sed 's|ssh://[^@]*@||' | cut -d: -f2)
    echo ""
    echo "==> Instance is ready!"
    echo "    Instance ID : ${INSTANCE_ID}"
    echo "    SSH URL     : ${SSH_URL}"
    echo "    Host        : ${HOST}"
    echo "    Port        : ${PORT}"
    echo ""
    echo "Save these for later:"
    echo "  INSTANCE_ID=${INSTANCE_ID}"
    echo "  HOST=${HOST}"
    echo "  PORT=${PORT}"
    echo ""
    echo "Next — launch training (Phase-3 easy run):"
    echo "  bash deploy/train_remote.sh ${HOST} ${PORT} \\"
    echo "    algo.total_steps=2000000 env.num_envs=4 \\"
    echo "    metric.log_every=1000 checkpoint.every=50000"
    echo ""
    echo "Or quick GPU sanity check (minutes):"
    echo "  bash deploy/train_remote.sh ${HOST} ${PORT} \\"
    echo "    algo.total_steps=15000 metric.log_every=200 checkpoint.every=2000"
    echo ""
    echo "TensorBoard (SSH tunnel, run locally):"
    echo "  ssh -p ${PORT} -L 6006:localhost:6006 root@${HOST} \\"
    echo "    'tensorboard --logdir /workspace/dreamer/logs/runs/dreamer_v3 --port 6006 --bind_all'"
    echo "  # then open http://localhost:6006"
    echo ""
    echo "Destroy when done (stop billing):"
    echo "  ${VASTAI} destroy instance ${INSTANCE_ID}"
    exit 0
  fi

  echo "  (${i}/40) not ready yet, waiting 10s..."
done

echo ""
echo "Timed out waiting. The instance may still be provisioning."
echo "Check status: ${VASTAI} show instance ${INSTANCE_ID}"
echo "Get SSH URL:  ${VASTAI} ssh-url ${INSTANCE_ID}"
