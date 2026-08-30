#!/usr/bin/env python3
"""Test matrix for the PreToolUse guard hook.

Run it by path:  python3 .claude/hooks/guard_vastai_test.py

It must be a FILE, never pasted inline into a shell command. The hook inspects
the whole Bash command string, so a test whose fixtures contain blocked
substrings would be blocked by its own fixtures.
"""

import json
import pathlib
import subprocess
import sys

HOOK = str(pathlib.Path(__file__).with_name("guard_vastai.py"))

BLOCK, ALLOW = 2, 0

CASES = [
    # spending must go through the registry
    ("vastai destroy instance 123", BLOCK),
    ("vastai create instance 999 --image foo", BLOCK),
    ("python3 tools/launch_run.py --campaign x --seed 1", ALLOW),
    ("bash tools/reap_instances.sh", ALLOW),
    ("vastai show instances --raw", ALLOW),
    ("vastai search offers 'gpu_ram >= 16' --raw", ALLOW),

    # the measurement layer is not writable from the shell
    ("echo hacked > tools/run_health.py", BLOCK),
    ("sed -i s/45/5/ tools/aggregate_seeds.py", BLOCK),
    ("cp /tmp/evil.py deploy/train_remote.sh", BLOCK),
    ("mv /tmp/evil.py tools/run_health.py", BLOCK),
    ("rm tools/checksums.sha256", BLOCK),
    ("rsync -a /tmp/evil/ deploy/", BLOCK),
    ("chmod 644 tools/budget_guard.py", BLOCK),

    # ...but reading out of it is ordinary work
    ("cp tools/run_health.py /tmp/backup.py", ALLOW),
    ("mv tools/run_health.py /tmp/gone.py", ALLOW),
    ("shasum -a 256 tools/*.py > /tmp/cks.test", ALLOW),
    ("diff tools/run_health.py /tmp/backup.py", ALLOW),
    ("python3 tools/run_health.py logs/x/version_0 --json out.json", ALLOW),
    ("python3 tools/aggregate_seeds.py --campaign c > /tmp/agg.json", ALLOW),
    ("bash deploy/fetch_events.sh 1.2.3.4 555", ALLOW),

    # the kill switch and budget snapshot belong to the guard
    ("rm experiments/state/KILLED", BLOCK),
    ("echo '{}' > experiments/state/budget.json", BLOCK),
    ("cat experiments/state/budget.json", ALLOW),

    # a redirect in one segment must not taint a later mention of tools/
    ("cat a > /tmp/x && python3 tools/aggregate_seeds.py --campaign c", ALLOW),
    ("ls logs/runs", ALLOW),
]


def main() -> int:
    failures = 0
    for command, want in CASES:
        proc = subprocess.run(
            [sys.executable, HOOK],
            input=json.dumps({"tool_name": "Bash", "tool_input": {"command": command}}),
            capture_output=True, text=True,
        )
        if proc.returncode != want:
            failures += 1
            verb = "should have blocked" if want == BLOCK else "should have allowed"
            print(f"FAIL ({verb}): {command}")
    print(f"{len(CASES)} cases, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
