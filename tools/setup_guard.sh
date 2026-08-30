#!/usr/bin/env bash
# Start an autonomous session's enforcement layer. RUN THIS YOURSELF, not via an
# agent — starting it is how you authorize the spend.
#
# It copies budget_guard.py OUT of the repository to ~/.dreamer-guard/ and runs it
# from there. That relocation is the whole point: the guard process holds its own
# loaded copy of the code, outside every agent's working directory, so no edit an
# agent makes to this repo can change what the guard does for the rest of the
# session. Editing tools/budget_guard.py afterwards affects only the NEXT session.
#
# It also snapshots checksums of tools/ and deploy/ and makes them read-only, so
# the adversarial reviewer can detect tampering at every gate.
#
# Usage:
#   bash tools/setup_guard.sh --cap 15 --ttl 12
#   bash tools/setup_guard.sh --status
#   bash tools/setup_guard.sh --stop
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GUARD_HOME="${GUARD_HOME:-${HOME}/.dreamer-guard}"
STATE_DIR="${REPO_ROOT}/experiments/state"
PID_FILE="${GUARD_HOME}/guard.pid"
LOG_FILE="${GUARD_HOME}/guard.log"

CAP=""
TTL="12"
POLL="300"
ACTION="start"

while [ $# -gt 0 ]; do
  case "$1" in
    --cap)    CAP="$2"; shift 2 ;;
    --ttl)    TTL="$2"; shift 2 ;;
    --poll)   POLL="$2"; shift 2 ;;
    --status) ACTION="status"; shift ;;
    --stop)   ACTION="stop"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

guard_pid() {
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  kill -0 "$pid" 2>/dev/null || return 1
  echo "$pid"
}

case "$ACTION" in
  status)
    if pid=$(guard_pid); then
      echo "guard running (pid ${pid})"
      echo "log:    ${LOG_FILE}"
      [ -f "${STATE_DIR}/budget.json" ] && cat "${STATE_DIR}/budget.json"
      [ -f "${STATE_DIR}/KILLED" ] && { echo "--- KILLED ---"; cat "${STATE_DIR}/KILLED"; }
    else
      echo "guard NOT running"
      exit 1
    fi
    exit 0
    ;;
  stop)
    if pid=$(guard_pid); then
      kill "$pid"
      rm -f "$PID_FILE"
      echo "guard stopped (pid ${pid})."
      echo "NOTE: stopping the guard does not destroy instances."
      echo "      Run tools/reap_instances.sh if you are done."
    else
      echo "guard NOT running"
    fi
    exit 0
    ;;
esac

[ -n "$CAP" ] || { echo "ERROR: --cap is required (USD hard cap)." >&2; exit 2; }

if pid=$(guard_pid); then
  echo "ERROR: a guard is already running (pid ${pid}). Stop it first with --stop." >&2
  exit 1
fi

VASTAI="${VASTAI:-$(command -v vastai || echo "${REPO_ROOT}/venv/bin/vastai")}"
if [ ! -x "$VASTAI" ] && ! command -v "$VASTAI" >/dev/null 2>&1; then
  echo "ERROR: vastai CLI not found (looked for '${VASTAI}'). pip install vastai." >&2
  exit 2
fi

mkdir -p "$GUARD_HOME" "$STATE_DIR"

echo "==> Copying the guard out of the repo to ${GUARD_HOME}"
cp "${REPO_ROOT}/tools/budget_guard.py" "${GUARD_HOME}/budget_guard.py"
chmod 500 "${GUARD_HOME}/budget_guard.py"

echo "==> Snapshotting checksums of tools/ and deploy/"
( cd "$REPO_ROOT" && shasum -a 256 tools/*.py tools/*.sh deploy/*.sh > tools/checksums.sha256 ) || true
# The manifest must not list itself, or verifying it becomes self-referential.
( cd "$REPO_ROOT" && grep -v 'tools/checksums.sha256' tools/checksums.sha256 > tools/.cks.tmp \
    && mv tools/.cks.tmp tools/checksums.sha256 )
cp "${REPO_ROOT}/tools/checksums.sha256" "${GUARD_HOME}/checksums.sha256"

echo "==> Making tools/ and deploy/ read-only"
# A speed bump, not a wall: agents run as this user and could chmod back. Its
# value is turning silent drift into a detectable, deliberate act — which the
# reviewer catches via the checksum manifest above.
chmod 444 "${REPO_ROOT}"/tools/*.py "${REPO_ROOT}"/tools/*.sh \
          "${REPO_ROOT}"/deploy/*.sh 2>/dev/null || true

PYTHON="$(command -v python3)"
echo "==> Starting guard: cap=\$${CAP} ttl=${TTL}h poll=${POLL}s"
nohup "$PYTHON" "${GUARD_HOME}/budget_guard.py" \
  --state-dir "$STATE_DIR" \
  --cap "$CAP" --ttl "$TTL" --poll "$POLL" \
  --vastai "$VASTAI" --reset \
  >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
sleep 2

if pid=$(guard_pid); then
  echo ""
  echo "Guard running (pid ${pid})."
  echo "  log:     ${LOG_FILE}"
  echo "  budget:  ${STATE_DIR}/budget.json  (first snapshot in ${POLL}s)"
  echo ""
  echo "The session is now authorized to spend up to \$${CAP} over ${TTL}h."
  echo "Next: /loop 30m /loop-experiments"
  echo "Stop: bash tools/setup_guard.sh --stop  &&  bash tools/reap_instances.sh"
else
  echo "ERROR: guard failed to start. Last log lines:" >&2
  tail -20 "$LOG_FILE" >&2
  exit 1
fi
