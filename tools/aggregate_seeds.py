#!/usr/bin/env python3
"""Aggregate a rung's per-seed evaluations into one gate-ready verdict.

Deep RL is brittle with respect to random seed — roughly a quarter to a third of
runs can fail on tasks the same code solves reliably on other seeds. So a single
seed says almost nothing, and a *mean* over seeds can hide one total failure. This
reports mean, standard error, and the per-seed minimum, and it fails a gate when
any single seed falls below the floor even if the mean clears the bar.

    tools/aggregate_seeds.py --campaign c --rung R1 \\
        --min-mean 0.60 --min-seed 0.45 --min-seeds 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _state as st  # noqa: E402


def stderr_of(values: list[float]) -> float:
    """Standard error of the mean. Zero for a single sample, by convention."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var / n)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate per-seed evals for one rung.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--config-hash", default=None,
                    help="Restrict to one experiment cell. Defaults to the single hash "
                         "found in the rung, and errors if the rung mixes several.")
    ap.add_argument("--tag", default=None,
                    help="Aggregate the tagged eval variant (e.g. 'review') instead of the primary.")
    ap.add_argument("--min-seeds", type=int, default=3,
                    help="Refuse to declare a pass with fewer seeds than this.")
    ap.add_argument("--min-mean", type=float, default=None, help="Required mean success rate.")
    ap.add_argument("--min-seed", type=float, default=None,
                    help="Required success rate for the WORST seed.")
    ap.add_argument("--out", default=None, help="Output JSON path (default: campaign evals/).")
    args = ap.parse_args()

    runs = [r for r in st.read_runs(args.campaign) if r.get("rung") == args.rung]
    if not runs:
        raise SystemExit(f"no runs registered for rung {args.rung} in campaign {args.campaign}")

    hashes = sorted({r["config_hash"] for r in runs})
    cfg_hash = args.config_hash
    if cfg_hash is None:
        if len(hashes) > 1:
            raise SystemExit(
                f"rung {args.rung} spans {len(hashes)} config hashes: {hashes}. "
                "Pass --config-hash to pick the cell — averaging across configs would be meaningless."
            )
        cfg_hash = hashes[0]

    evals_dir = st.campaign_dir(args.campaign) / "evals"
    suffix = f"_{args.tag}" if args.tag else ""
    per_seed = []
    missing = []
    for rec in runs:
        if rec["config_hash"] != cfg_hash:
            continue
        path = evals_dir / f"{rec['run_id']}{suffix}.json"
        if not path.exists():
            missing.append(rec["run_id"])
            continue
        data = json.loads(path.read_text())
        if data.get("config_hash") and data["config_hash"] != cfg_hash:
            raise SystemExit(
                f"{path.name} was produced against config_hash {data['config_hash']}, "
                f"not the rung's {cfg_hash}. Refusing to aggregate a mismatched eval."
            )
        per_seed.append({
            "run_id": rec["run_id"],
            "seed": rec.get("seed"),
            "success_rate": data["success_rate"],
            "episodes": data["episodes"],
            "eval_seed_base": data.get("eval_seed_base"),
            "mean_min_dist": data.get("mean_min_dist"),
            "mean_steps": data.get("mean_steps"),
            "eval_path": str(path.relative_to(st.REPO_ROOT)),
        })

    per_seed.sort(key=lambda s: (s["seed"] is None, s["seed"]))
    rates = [s["success_rate"] for s in per_seed]

    report: dict = {
        "campaign": args.campaign,
        "rung": args.rung,
        "config_hash": cfg_hash,
        "tag": args.tag,
        "n_seeds": len(rates),
        "per_seed": per_seed,
        "missing_evals": missing,
        "criteria": {
            "min_seeds": args.min_seeds,
            "min_mean": args.min_mean,
            "min_seed": args.min_seed,
        },
        "aggregated_at": st.utcnow(),
    }

    if rates:
        report.update({
            "mean": round(sum(rates) / len(rates), 4),
            "stderr": round(stderr_of(rates), 4),
            "min": round(min(rates), 4),
            "max": round(max(rates), 4),
            "total_episodes": sum(s["episodes"] for s in per_seed),
        })

    failures = []
    if not rates:
        failures.append("no evaluations found")
    else:
        if len(rates) < args.min_seeds:
            failures.append(f"only {len(rates)} seeds evaluated, {args.min_seeds} required")
        if missing:
            failures.append(f"runs without an eval: {', '.join(missing)}")
        if args.min_mean is not None and report["mean"] < args.min_mean:
            failures.append(f"mean {report['mean']:.3f} below required {args.min_mean:.3f}")
        if args.min_seed is not None and report["min"] < args.min_seed:
            worst = min(per_seed, key=lambda s: s["success_rate"])
            failures.append(
                f"worst seed {worst['run_id']} at {worst['success_rate']:.3f} is below the "
                f"per-seed floor {args.min_seed:.3f}"
            )

    report["pass"] = not failures
    report["failures"] = failures

    out_path = Path(args.out) if args.out else evals_dir / f"AGG_{args.rung}{suffix}.json"
    st.write_json_atomic(out_path, report)

    print(json.dumps(report, indent=2))
    if rates:
        summary = (f"aggregate {args.rung}: mean={report['mean']:.0%} "
                   f"±{report['stderr']:.1%} (n={len(rates)}, worst={report['min']:.0%}) "
                   f"{'PASS' if report['pass'] else 'FAIL'}")
    else:
        summary = f"aggregate {args.rung}: no evaluations found — FAIL"
    st.journal(args.campaign, summary)
    print(f"\n{summary}\npath={out_path}")
    # Exit 1 on a failed gate so a shell caller can branch without parsing JSON.
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
