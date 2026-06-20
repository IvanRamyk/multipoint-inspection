#!/usr/bin/env bash
# Find the cheapest suitable GPU on vast.ai for DreamerV3 training.
# Requires: pip install vastai  &&  vastai set api-key <YOUR_KEY>
#
# Usage: bash deploy/vast_search.sh [--limit N]
set -euo pipefail

LIMIT="${2:-10}"

echo "==> Searching vast.ai for cheapest GPU (>=16GB VRAM, CUDA>=12, reliable)..."
echo ""

vastai search offers \
  'reliability > 0.98 num_gpus=1 gpu_ram >= 16 cuda_vers >= 12.0 inet_down > 100 disk_space >= 20' \
  --order 'dph_total asc' \
  --limit "$LIMIT"

echo ""
echo "Pick an offer ID from above, then create an instance:"
echo "  vastai create instance <OFFER_ID> \\"
echo "    --image pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \\"
echo "    --disk 20 \\"
echo "    --onstart-cmd 'apt-get update -qq && apt-get install -y git libgl1 libglib2.0-0 libgomp1 -qq'"
echo ""
echo "Then get SSH details with:"
echo "  vastai show instance <INSTANCE_ID>"
echo "  vastai ssh-url <INSTANCE_ID>"
