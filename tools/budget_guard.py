#!/usr/bin/env python3
"""External spend and wall-clock guard for autonomous experiment sessions.

This is the only load-bearing safety layer. Everything else — file permissions,
settings deny-rules, the reviewer's checksum verification — is policy that a
determined agent could work around. This is a separate OS process, launched by
the human from a copy of this file that lives OUTSIDE the repository, holding its
own loaded copy of the code. No edit an agent makes to the repo can change what
this process does for the rest of the session.

It does two things forever:
  1. integrates cost across all running vast.ai instances and publishes a
     snapshot to <state-dir>/budget.json, which the launcher reads before
     spending anything;
  2. when the cap or the TTL is breached, destroys every instance, verifies the
     list is empty, and drops a <state-dir>/KILLED flag that stops the launcher.

Deliberately stdlib-only, so it runs under any Python 3.9+ without the project's
venv. Usage:

    python3 budget_guard.py --state-dir /path/to/repo/experiments/state \\
        --cap 15 --ttl 12 --reset

Run it via tools/setup_guard.sh, which copies it out of the repo first.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

# Cost accrual is integrated across polls rather than computed from per-instance
# start times: an instance that appears and disappears between two polls still
# contributes the time it was visible, and no clock skew between us and vast.ai
# can inflate or hide a charge.
DEFAULT_POLL = 300.0
DESTROY_ATTEMPTS = 5


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[guard {utcnow()}] {msg}", flush=True)


def write_json_atomic(path: str, payload: object) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=os.path.basename(path), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def vastai(binary: str, *args: str, timeout: float = 60.0) -> str:
    """Run the vastai CLI, returning stdout. Empty string on any failure."""
    try:
        out = subprocess.run(
            [binary, *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"vastai {' '.join(args)} failed: {exc}")
        return ""
    if out.returncode != 0:
        log(f"vastai {' '.join(args)} exited {out.returncode}: {out.stderr.strip()[:200]}")
        return ""
    return out.stdout


def list_instances(binary: str) -> list[dict] | None:
    """Currently-billing instances, as reported by vast.ai.

    Returns None when the query itself failed, which is distinct from an empty
    list meaning "nothing is running". The caller must not read a failed poll as
    zero cost, or an hour of API flakiness would hide an hour of billing.
    """
    raw = vastai(binary, "show", "instances", "--raw")
    if not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log("could not parse instance list as JSON")
        return None
    if isinstance(data, dict):
        data = data.get("instances", [])
    if not isinstance(data, list):
        return None
    return [i for i in data if isinstance(i, dict)]


def credit(binary: str) -> float | None:
    raw = vastai(binary, "show", "user", "--raw")
    if not raw.strip():
        return None
    try:
        return float(json.loads(raw).get("credit"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def destroy_all(binary: str, instances: list[dict]) -> list[str]:
    """Destroy every given instance. Returns the ids that are still alive after.

    A verification query that itself fails yields ``["<unverified>"]`` rather than
    an empty list, so the caller retries instead of reporting a clean teardown it
    never confirmed.
    """
    for inst in instances:
        iid = str(inst.get("id"))
        log(f"destroying instance {iid}")
        vastai(binary, "destroy", "instance", iid)
    time.sleep(10)
    after = list_instances(binary)
    if after is None:
        return ["<unverified>"]
    return [str(i.get("id")) for i in after]


def enforce(binary: str, state_dir: str, reason: str) -> None:
    """Kill everything, then latch the KILLED flag.

    The flag is written even if a destroy fails, because the launcher refusing to
    start anything new matters more than a clean teardown, and a stuck instance
    still needs the human's attention either way.
    """
    log(f"BREACH: {reason} — destroying all instances")
    remaining: list[str] = ["<unverified>"]
    for attempt in range(1, DESTROY_ATTEMPTS + 1):
        instances = list_instances(binary)
        if instances is None:
            log(f"attempt {attempt}: cannot list instances, retrying")
            time.sleep(15)
            continue
        if not instances:
            remaining = []
            log("no instances left to destroy")
            break
        remaining = destroy_all(binary, instances)
        if not remaining:
            log(f"all instances destroyed (attempt {attempt})")
            break
        log(f"attempt {attempt}: still alive: {', '.join(remaining)}")
        time.sleep(15)

    note = f"{reason}\nkilled_at: {utcnow()}\n"
    if remaining:
        note += (
            "WARNING: these instances survived every destroy attempt and are "
            f"probably still billing: {', '.join(remaining)}\n"
        )
    with open(os.path.join(state_dir, "KILLED"), "w") as fh:
        fh.write(note)
    log("KILLED flag written" + (" WITH SURVIVORS" if remaining else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="External spend/TTL guard for autonomous RL sessions.")
    ap.add_argument("--state-dir", required=True,
                    help="Absolute path to experiments/state in the repo being guarded.")
    ap.add_argument("--cap", type=float, required=True, help="Hard spend cap in USD.")
    ap.add_argument("--ttl", type=float, default=12.0, help="Hard wall-clock cap in hours.")
    ap.add_argument("--poll", type=float, default=DEFAULT_POLL, help="Seconds between polls.")
    ap.add_argument("--vastai", default=None, help="Path to the vastai CLI (default: PATH lookup).")
    ap.add_argument("--reset", action="store_true",
                    help="Start a fresh session. Without this, an existing budget.json is resumed "
                         "and its accrued spend carried forward, which is the safe default.")
    args = ap.parse_args()

    state_dir = os.path.abspath(args.state_dir)
    os.makedirs(state_dir, exist_ok=True)
    budget_path = os.path.join(state_dir, "budget.json")
    killed_path = os.path.join(state_dir, "KILLED")

    binary = args.vastai or shutil.which("vastai")
    if not binary:
        log("FATAL: vastai CLI not found. Pass --vastai /path/to/vastai.")
        return 2

    spent = 0.0
    session_started = time.time()
    credit_start = credit(binary)

    if args.reset:
        for stale in (killed_path,):
            if os.path.exists(stale):
                os.unlink(stale)
                log(f"cleared stale {os.path.basename(stale)}")
    else:
        prior = {}
        try:
            with open(budget_path) as fh:
                prior = json.load(fh)
        except (OSError, json.JSONDecodeError):
            prior = {}
        if prior.get("spent") is not None:
            spent = float(prior["spent"])
            session_started = float(prior.get("session_started_epoch", session_started))
            if prior.get("credit_start") is not None:
                credit_start = float(prior["credit_start"])
            log(f"resumed prior session: spent=${spent:.2f}")

    log(f"guarding {state_dir}  cap=${args.cap:.2f}  ttl={args.ttl}h  poll={args.poll:.0f}s")
    if credit_start is not None:
        log(f"account credit at start: ${credit_start:.2f}")

    last_poll = time.time()
    last_dph = 0.0
    while True:
        time.sleep(args.poll)
        now = time.time()
        elapsed_hours = (now - session_started) / 3600.0

        polled = list_instances(binary)
        poll_ok = polled is not None
        instances = polled if poll_ok else []
        if poll_ok:
            dph = sum(float(i.get("dph_total") or 0.0) for i in instances)
            last_dph = dph
        else:
            # Bill the interval at the last rate we actually observed, so an API
            # outage cannot make a running instance look free.
            dph = last_dph
            log(f"instance poll failed — accruing at last known rate ${dph:.3f}/h")
        interval_hours = (now - last_poll) / 3600.0
        spent += dph * interval_hours
        last_poll = now

        credit_now = credit(binary)
        credit_spent = None
        if credit_start is not None and credit_now is not None:
            credit_spent = credit_start - credit_now
            # The credit delta catches charges the hourly rate misses (storage,
            # bandwidth). Take whichever number is larger — never the friendlier one.
            spent = max(spent, credit_spent)

        snapshot = {
            "spent": round(spent, 4),
            "cap": args.cap,
            "headroom": round(args.cap - spent, 4),
            "ttl_hours": args.ttl,
            "elapsed_hours": round(elapsed_hours, 3),
            "session_started": datetime.fromtimestamp(session_started, timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "session_started_epoch": session_started,
            "credit_start": credit_start,
            "credit_now": credit_now,
            "credit_spent": None if credit_spent is None else round(credit_spent, 4),
            "instance_count": len(instances),
            "hourly_rate": round(dph, 4),
            "instances": [
                {
                    "id": i.get("id"),
                    "dph_total": i.get("dph_total"),
                    "gpu_name": i.get("gpu_name"),
                    "status": i.get("actual_status"),
                }
                for i in instances
            ],
            "updated_at": utcnow(),
        }
        write_json_atomic(budget_path, snapshot)
        log(f"spent=${spent:.2f}/{args.cap:.2f}  rate=${dph:.3f}/h  "
            f"instances={len(instances)}  elapsed={elapsed_hours:.2f}h")

        if os.path.exists(killed_path):
            # Already latched (by us on a previous breach, or by the human). Keep
            # publishing the snapshot but make sure nothing is still billing.
            if instances:
                log("KILLED is set but instances are alive — re-reaping")
                enforce(binary, state_dir, "re-reap after KILLED flag was already set")
            continue

        if spent >= args.cap:
            enforce(binary, state_dir, f"spend cap reached: ${spent:.2f} >= ${args.cap:.2f}")
        elif elapsed_hours >= args.ttl:
            enforce(binary, state_dir, f"TTL reached: {elapsed_hours:.2f}h >= {args.ttl}h")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("interrupted — NOTE: instances are NOT destroyed on guard exit")
        sys.exit(130)
