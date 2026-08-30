#!/usr/bin/env python3
"""Record a deliberate exception to a failed gate, so a judgment call is visible.

Precommitted thresholds are set before a single run exists, so some of them are
guesses. A gate that is guessed slightly wrong produces a false negative, and
throwing away a genuinely good result because a number landed at 4.8 instead of
5.0 is real waste. This gives the orchestrator a way to continue anyway.

It is deliberately not a way to change the bar. The spec is untouched and the
aggregate keeps reporting FAIL; what this writes is an exception logged *against*
that failure, carrying the measured value, the threshold, how far off it was, and
the argument for continuing. The morning report leads with it. So the record shows
exactly which bar was missed, by how much, and who decided it did not matter —
which is the part that makes this honest rather than a loophole.

Three things bound it:

  * a budget per campaign, because a gate that can always be overridden is not a
    gate;
  * integrity failures cannot be overridden at all — a checksum mismatch, an
    altered config, a synthetic eval or too few seeds is not an edge case, it is
    a broken measurement, and no argument makes the number mean something;
  * the blind reviewer's own verdict is copied in, so the human can see whether
    the independent check agreed.

    tools/request_override.py --campaign c --rung R1 --kind advance-rung \\
        --category near-miss --justification "..."
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _state as st  # noqa: E402

DEFAULT_BUDGET = 2

# Failures that describe a broken or insufficient measurement rather than a
# borderline result. No justification can make these overridable: the number
# itself is not trustworthy, so there is nothing to exercise judgment about.
NON_OVERRIDABLE = [
    (r"checksum|shasum|tamper", "the measurement code was altered"),
    (r"config_hash|mismatched eval|not the rung's", "the eval used a different config than the rung"),
    (r"\bfake\b|synthetic", "the eval was synthetic, not a real rollout"),
    (r"only \d+ seeds|no evaluations found|without an eval",
     "too few seeds — this is missing evidence, not a borderline result"),
    (r"could not be measured", "a criterion was set but never measured"),
    (r"tree_clean|uncommitted", "the working tree was dirty at measurement time"),
]

# Maps a failure message to the aggregate fields that quantify the miss, so the
# record carries the magnitude instead of only the agent's characterisation.
MISS_PATTERNS = [
    (r"mean ([\d.]+) below required ([\d.]+)", "mean success rate"),
    (r"worst seed \S+ at ([\d.]+) is below the per-seed floor ([\d.]+)", "worst seed"),
    (r"pursuit alignment[^)]*\) ([\d.-]+) is below the required ([\d.]+)", "pursuit alignment"),
    (r"mean initial separation ([\d.-]+) is below the required ([\d.]+)", "initial separation"),
    (r"mean steps to interception ([\d.-]+) is below the required ([\d.]+)", "steps to interception"),
    (r"idle fraction ([\d.-]+) is above the required ([\d.]+)", "idle fraction"),
    (r"fraction of the initial gap closed[^)]*? ([\d.-]+) is below the required ([\d.]+)",
     "separation closed"),
]


def overrides_dir(campaign: str) -> Path:
    return st.campaign_dir(campaign) / "overrides"


def repo_relative(path: Path) -> str:
    """Path as written in the record: repo-relative when possible, absolute otherwise.

    Callers pass ``--from-json`` both ways, and ``relative_to`` raises on a path
    that is relative or outside the repo — which must not take down an override.
    """
    resolved = path if path.is_absolute() else (Path.cwd() / path)
    try:
        return str(resolved.resolve().relative_to(st.REPO_ROOT))
    except ValueError:
        return str(resolved)


def granted(campaign: str) -> list[dict]:
    """Overrides already granted for this campaign, oldest first."""
    d = overrides_dir(campaign)
    if not d.is_dir():
        return []
    out = []
    for path in sorted(d.glob("*.json")):
        data = st.read_json(path, default=None)
        if data and data.get("status") == "granted":
            out.append(data)
    return out


def classify(failures: list[str]) -> tuple[list[dict], list[dict]]:
    """Split failures into overridable and blocked, with the reason for blocking."""
    overridable, blocked = [], []
    for failure in failures:
        reason = None
        for pattern, why in NON_OVERRIDABLE:
            if re.search(pattern, failure, re.IGNORECASE):
                reason = why
                break
        if reason:
            blocked.append({"failure": failure, "blocked_because": reason})
        else:
            overridable.append({"failure": failure})
    return overridable, blocked


def quantify(failure: str) -> dict | None:
    """Measured value, threshold and relative shortfall, when the text carries them."""
    for pattern, label in MISS_PATTERNS:
        m = re.search(pattern, failure)
        if not m:
            continue
        try:
            value, threshold = float(m.group(1)), float(m.group(2))
        except ValueError:
            return None
        gap = abs(threshold - value)
        relative = gap / abs(threshold) if threshold else None
        return {
            "metric": label,
            "measured": value,
            "threshold": threshold,
            "shortfall": round(gap, 4),
            "relative_shortfall": None if relative is None else round(relative, 4),
        }
    return None


def reviewer_verdict(campaign: str, rung: str) -> dict | None:
    """The blind reviewer's own conclusion, copied in for the human to weigh."""
    path = st.campaign_dir(campaign) / "verdicts" / f"{rung}.json"
    data = st.read_json(path, default=None)
    if not data:
        return None
    return {
        "verdict": data.get("verdict"),
        "recomputed_success": data.get("recomputed_success"),
        "notes": data.get("notes"),
        "path": repo_relative(path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Log a deliberate exception to a failed gate.")
    ap.add_argument("--campaign", required=True)
    # Not required, because --list is about the campaign's budget and has no rung.
    ap.add_argument("--rung", default=None)
    ap.add_argument("--kind", choices=("advance-rung", "keep-run"), default="advance-rung",
                    help="advance-rung: continue the ladder despite a failed aggregate. "
                         "keep-run: continue a run the health check wanted killed.")
    ap.add_argument("--category", required=False,
                    choices=("near-miss", "edge-case", "measurement-artefact"),
                    help="How you characterise the failure. Required unless --list.")
    ap.add_argument("--justification", required=False,
                    help="Why this failure does not invalidate the result. Be specific: name the "
                         "metric, the measured value, and what makes this an exception rather "
                         "than a miss.")
    ap.add_argument("--from-json", default=None,
                    help="Aggregate or health JSON holding the failures (default: the rung's "
                         "AGG_<rung>.json).")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help=f"Overrides allowed per campaign (default {DEFAULT_BUDGET}).")
    ap.add_argument("--list", action="store_true", help="Show overrides used and remaining.")
    args = ap.parse_args()

    used = granted(args.campaign)
    if args.list:
        print(json.dumps({
            "campaign": args.campaign,
            "budget": args.budget,
            "used": len(used),
            "remaining": max(0, args.budget - len(used)),
            "granted": [{"rung": o["rung"], "kind": o["kind"], "category": o["category"],
                         "granted_at": o["granted_at"]} for o in used],
        }, indent=2))
        return 0

    missing = [name for name, value in
               (("--rung", args.rung), ("--category", args.category),
                ("--justification", args.justification)) if not value]
    if missing:
        raise SystemExit(f"{', '.join(missing)} required to grant an override")

    if st.is_killed():
        raise SystemExit(
            f"REFUSING: the kill switch is set, so the session is over.\n{st.kill_reason()}"
        )

    if len(used) >= args.budget:
        prior = ", ".join(f"{o['rung']} ({o['category']})" for o in used)
        raise SystemExit(
            f"REFUSING: this campaign's {args.budget} overrides are spent on {prior}.\n"
            "Two gates missed is not a run of bad luck — either the thresholds were wrong or the "
            "task is not doing what the spec assumed. Both are the human's call. Conclude the "
            "campaign and report what you found."
        )

    source = Path(args.from_json) if args.from_json else (
        st.campaign_dir(args.campaign) / "evals" / f"AGG_{args.rung}.json"
    )
    report = st.read_json(source, default=None)
    if report is None:
        raise SystemExit(f"cannot read the failing report at {source}")
    failures = report.get("failures") or []
    if not failures:
        raise SystemExit(
            f"{source.name} reports no failures — there is nothing to override. If the gate "
            "passed, just advance."
        )

    overridable, blocked = classify(failures)
    if blocked:
        lines = "\n".join(f"  - {b['failure']}\n    ({b['blocked_because']})" for b in blocked)
        raise SystemExit(
            "REFUSING: these failures cannot be overridden by any argument, because they mean "
            f"the measurement itself is not trustworthy:\n{lines}\n"
            "Fix the measurement and re-evaluate. A number that was produced wrongly does not "
            "become right by being explained."
        )

    for item in overridable:
        item["quantified"] = quantify(item["failure"])

    n = len(used) + 1
    record = {
        "campaign": args.campaign,
        "rung": args.rung,
        "kind": args.kind,
        "category": args.category,
        "justification": args.justification,
        "overridden_failures": overridable,
        "source_report": repo_relative(source),
        "claimed_success": {
            "mean": report.get("mean"), "stderr": report.get("stderr"),
            "min": report.get("min"), "n_seeds": report.get("n_seeds"),
        },
        "criteria_as_specified": report.get("criteria"),
        # The spec is NOT edited. This records an exception against it, so the
        # report still shows which bar was missed and by how much.
        "spec_unchanged": True,
        "reviewer_verdict": reviewer_verdict(args.campaign, args.rung),
        "granted_by": "orchestrator",
        "granted_at": st.utcnow(),
        "status": "granted",
        "override_number": n,
        "budget": args.budget,
        "remaining_after": max(0, args.budget - n),
    }

    out = overrides_dir(args.campaign) / f"{args.rung}-{n}.json"
    st.write_json_atomic(out, record)

    quantified = [i["quantified"] for i in overridable if i.get("quantified")]
    magnitude = ""
    if quantified:
        worst = max(quantified, key=lambda q: q.get("relative_shortfall") or 0.0)
        rel = worst.get("relative_shortfall")
        magnitude = (f" worst miss: {worst['metric']} {worst['measured']} vs {worst['threshold']}"
                     + (f" ({rel:.0%} short)" if rel is not None else ""))

    st.journal(
        args.campaign,
        f"OVERRIDE {n}/{args.budget} granted for {args.rung} ({args.kind}, {args.category}): "
        f"{args.justification[:160]}.{magnitude} Gate still reports FAIL; spec unchanged. "
        f"-> {out.name}"
    )

    print(json.dumps(record, indent=2))
    print(f"\nOVERRIDE {n}/{args.budget} granted for {args.rung}. "
          f"{record['remaining_after']} left in this campaign.")
    print(f"  The gate still reports FAIL and the spec is unchanged — this is an exception on "
          f"the record, not a lowered bar.")
    if magnitude:
        print(f" {magnitude.strip()}")
    rv = record["reviewer_verdict"]
    if rv:
        print(f"  blind reviewer said: {rv['verdict']}")
        if rv["verdict"] in ("fail", "suspicious"):
            print("  NOTE: the independent reviewer did not agree. Say so in the report.")
    print(f"  path={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
