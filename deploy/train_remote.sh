#!/usr/bin/env bash
# Push code to a vast.ai instance and launch DreamerV3 training.
#
# Usage:
#   bash deploy/train_remote.sh <HOST> <PORT> [dreamer overrides...]
#
# Examples:
#   # Phase-3 easy run (recommended first GPU run)
#   bash deploy/train_remote.sh 123.45.67.89 12345 \
#     algo.total_steps=2000000 env.num_envs=4
#
#   # Quick sanity check on GPU (minutes)
#   bash deploy/train_remote.sh 123.45.67.89 12345 \
#     algo.total_steps=15000 metric.log_every=200 checkpoint.every=2000
#
# After training starts you can:
#   - Monitor:  bash deploy/tail_remote.sh <HOST> <PORT>
#   - Fetch:    bash deploy/fetch_results.sh <HOST> <PORT>
set -euo pipefail

HOST="${1:?Usage: train_remote.sh <HOST> <PORT> [dreamer overrides...]}"; shift
PORT="${1:?Usage: train_remote.sh <HOST> <PORT> [dreamer overrides...]}"; shift
TRAIN_ARGS="${*:-algo.total_steps=2000000 env.num_envs=4}"

REMOTE="root@${HOST}"
REMOTE_DIR="/workspace/dreamer"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/hetzner_agents}"
SSH_OPTS="-i ${SSH_KEY} -p ${PORT} -o StrictHostKeyChecking=no -o LogLevel=ERROR"

echo "==> Syncing code to ${REMOTE}:${REMOTE_DIR}/"
rsync -az --progress \
  --exclude venv \
  --exclude logs \
  --exclude results \
  --exclude .git \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude '*.egg-info' \
  -e "ssh ${SSH_OPTS}" \
  ./ "${REMOTE}:${REMOTE_DIR}/"

echo ""
# If a pre-built image is in use (DREAMER_IMAGE set), deps are already baked in.
# We only need to do a fast editable install of the local package metadata.
# On a bare PyTorch image (no DREAMER_IMAGE), do the full Python 3.11 + venv setup.
if [ -n "${DREAMER_IMAGE:-}" ]; then
  echo "==> Custom image detected (${DREAMER_IMAGE}), doing fast editable re-install..."
  # shellcheck disable=SC2029
  ssh ${SSH_OPTS} "${REMOTE}" "
    set -e
    cd ${REMOTE_DIR}
    # Deps are baked in; just re-register the local package so imports work.
    pip install -q --no-cache-dir --no-deps -e '.'
    python -c \"import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')\"
  "
else
  echo "==> Installing deps on remote (bare image — first run is slow, subsequent runs skip pip)..."
  # shellcheck disable=SC2029
  ssh ${SSH_OPTS} "${REMOTE}" "
    set -e
    cd ${REMOTE_DIR}

    # sheeprl requires Python <3.12. Install 3.11 via deadsnakes if needed.
    if python3.11 --version >/dev/null 2>&1; then
      echo 'Python 3.11 already available.'
    else
      echo 'Installing Python 3.11 via deadsnakes PPA...'
      apt-get update -qq
      apt-get install -y --no-install-recommends software-properties-common >/dev/null
      add-apt-repository -y ppa:deadsnakes/ppa >/dev/null
      apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev >/dev/null
    fi

    dpkg -l libgl1 >/dev/null 2>&1 || apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 >/dev/null

    if [ ! -f venv/bin/python ]; then
      echo 'Creating Python 3.11 venv...'
      python3.11 -m venv venv
    fi

    PYPROJECT_HASH=\$(md5sum pyproject.toml | cut -d' ' -f1)
    HASH_FILE=\"venv/.pyproject_hash\"
    if [ ! -f \"\$HASH_FILE\" ] || [ \"\$(cat \$HASH_FILE)\" != \"\$PYPROJECT_HASH\" ]; then
      echo 'Installing project deps into venv...'
      venv/bin/pip install --no-cache-dir -e '.[dreamer]'
      echo \"\$PYPROJECT_HASH\" > \"\$HASH_FILE\"
    else
      echo 'Deps up to date, skipping pip install.'
    fi
    venv/bin/python -c \"import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')\"
  "
fi

echo ""
echo "==> Launching training in tmux session 'train'..."
# Custom image has python in PATH; bare image uses the local venv.
PYTHON="${DREAMER_IMAGE:+python}"
PYTHON="${PYTHON:-venv/bin/python}"
FULL_CMD="${PYTHON} scripts/train_dreamer.py fabric.accelerator=gpu fabric.precision=16-mixed ${TRAIN_ARGS}"

# shellcheck disable=SC2029
ssh ${SSH_OPTS} "${REMOTE}" "
  cd ${REMOTE_DIR}
  # Kill any existing session to avoid stale state
  tmux kill-session -t train 2>/dev/null || true
  # Start new detached session, log stdout+stderr to a file
  tmux new-session -d -s train -x 220 -y 50
  tmux send-keys -t train 'cd ${REMOTE_DIR} && ${FULL_CMD} 2>&1 | tee train.log' Enter
  echo 'Training started.'
"

echo ""
echo "Training is running remotely. Useful commands:"
echo ""
echo "  # Attach to live output:"
echo "  ssh ${SSH_OPTS} ${REMOTE} -t 'tmux attach -t train'"
echo ""
echo "  # Tail the log file:"
echo "  ssh ${SSH_OPTS} ${REMOTE} 'tail -f ${REMOTE_DIR}/train.log'"
echo ""
echo "  # Fetch checkpoints + TB events:"
echo "  bash deploy/fetch_results.sh ${HOST} ${PORT}"
echo ""
echo "  # TensorBoard tunnel (open http://localhost:6006 while it runs):"
echo "  bash deploy/tensorboard.sh ${HOST} ${PORT}"
