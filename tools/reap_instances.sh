#!/usr/bin/env bash
# Destroy every vast.ai instance on the account, then verify none remain.
#
# Used by the orchestrator's conclude step and by a human who wants billing to
# stop right now. Deliberately account-wide rather than campaign-scoped: the
# failure mode worth protecting against is an instance nobody is tracking, and a
# campaign-scoped reaper would skip exactly those.
#
# Usage:
#   bash tools/reap_instances.sh [--dry-run]
set -euo pipefail

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

VASTAI="${VASTAI:-$(command -v vastai || echo ./venv/bin/vastai)}"

ids_now() {
  "$VASTAI" show instances --raw 2>/dev/null \
    | python3 -c "
import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(3)          # unreadable: signal the caller, do not print 'none'
if isinstance(d, dict):
    d = d.get('instances', [])
print(' '.join(str(i['id']) for i in d if isinstance(i, dict) and 'id' in i))
"
}

if ! IDS=$(ids_now); then
  echo "ERROR: could not read the instance list — cannot confirm anything was reaped." >&2
  exit 3
fi

if [ -z "$IDS" ]; then
  echo "No instances running."
  exit 0
fi

echo "Instances to destroy: $IDS"
if [ "$DRY_RUN" = "1" ]; then
  echo "(dry run — nothing destroyed)"
  exit 0
fi

for id in $IDS; do
  echo "==> destroying $id"
  "$VASTAI" destroy instance "$id" || echo "  (destroy failed for $id)"
done

sleep 10
if ! REMAINING=$(ids_now); then
  echo "WARNING: destroys issued but the instance list is unreadable — verify manually." >&2
  exit 3
fi

if [ -n "$REMAINING" ]; then
  echo "WARNING: still alive and probably still billing: $REMAINING" >&2
  exit 1
fi

echo "All instances destroyed."
