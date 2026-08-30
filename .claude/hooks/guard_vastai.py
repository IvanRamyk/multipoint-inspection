#!/usr/bin/env python3
"""PreToolUse hook: refuse Bash calls that spend money outside the sanctioned
tooling, or that tamper with the measurement and enforcement layer.

Reads the hook payload on stdin. Exit 2 blocks the call with the message on
stderr; exit 0 allows it. Deliberately narrow — a hook that fires on innocent
commands is a hook someone disables.

This is policy, not enforcement. The load-bearing layer is the external guard
started by tools/setup_guard.sh, which runs from a copy outside this repository.
"""

from __future__ import annotations

import json
import re
import sys

PROTECTED = r"(?:\./)?(?:tools|deploy)/"
STATE_FILES = r"experiments/state/(?:KILLED|budget\.json)"

SANCTIONED = re.compile(r"tools/(?:launch_run\.py|reap_instances\.sh|setup_guard\.sh)")
SPENDING = re.compile(r"vastai\s+(?:create|destroy|copy|execute)")
REDIRECT_INTO_PROTECTED = re.compile(r">>?\s*['\"]?" + PROTECTED)
REDIRECT_INTO_STATE = re.compile(r">>?\s*['\"]?" + STATE_FILES)
MUTATORS = re.compile(r"\b(?:sed\s+-i|tee|mv|cp|rm|truncate|install|dd|ln)\b")
CHMOD = re.compile(r"\bchmod\b")


def block(message: str) -> None:
    print(f"BLOCKED by .claude/hooks/guard_vastai.py: {message}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        # A payload we cannot parse is not grounds to block real work.
        return 0

    if payload.get("tool_name") not in (None, "Bash"):
        return 0
    command = (payload.get("tool_input") or {}).get("command") or ""
    if not command.strip():
        return 0

    # Analyse each pipeline segment on its own. Splitting first is what keeps an
    # innocent `foo > /tmp/x && python3 tools/run_health.py ...` from looking like
    # a write into tools/.
    segments = [s for s in re.split(r"(?:&&|\|\||[;\n|])", command) if s.strip()]

    for segment in segments:
        if SPENDING.search(segment) and not SANCTIONED.search(segment):
            block(
                "direct 'vastai create/destroy' is not allowed. Use tools/launch_run.py to start "
                "a run — it registers the run before spending — or tools/reap_instances.sh to "
                "tear everything down. An instance created outside the registry is spend nobody "
                "is tracking."
            )

        touches_protected = re.search(PROTECTED, segment) is not None
        if REDIRECT_INTO_PROTECTED.search(segment) or (touches_protected and MUTATORS.search(segment)):
            block(
                "writing to tools/ or deploy/ from the shell is not allowed — that is the "
                "measurement and enforcement layer. If a tool is genuinely wrong, report it and "
                "stop; a human changes it between sessions."
            )

        touches_state = re.search(STATE_FILES, segment) is not None
        if touches_state and (MUTATORS.search(segment) or REDIRECT_INTO_STATE.search(segment)):
            block(
                "the kill switch and the budget snapshot are written by the external guard only. "
                "If KILLED exists the session is over: finish evaluating what is already on disk, "
                "write the report, and stop."
            )

        if CHMOD.search(segment) and touches_protected:
            block("re-enabling writes on tools/ or deploy/ is not allowed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
