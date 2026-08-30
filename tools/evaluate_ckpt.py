#!/usr/bin/env python3
"""Evaluate a finished run's newest checkpoint and record the result.

This is the acceptance-gate primitive. `Rewards/rew_avg` from TensorBoard is only
a proxy for success — sheeprl never logs `info["is_success"]` — so a real number
requires rolling out the policy. This wraps scripts/eval_target_ckpt.py, pins the
evaluation to the run's registered env config (never a different, easier one),
and writes the JSON the aggregator and the reviewer read.

    tools/evaluate_ckpt.py --campaign c --run-id R1-chase_easy2-s1 --episodes 30
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _state as st  # noqa: E402
from eval.behaviour_metrics import aggregate as aggregate_behaviour  # noqa: E402


def newest_checkpoint(run_dir: Path) -> Path | None:
    """Highest-step checkpoint in a run directory.

    Sorted by the numeric step parsed from ckpt_<step>_0.ckpt rather than by
    mtime: a re-fetched older checkpoint would otherwise look newest.
    """
    cks = list((run_dir / "checkpoint").glob("ckpt_*.ckpt"))
    if not cks:
        return None
    def step_of(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1
    return max(cks, key=step_of)


def resolve_run_dir(rec: dict) -> Path:
    """Local run directory for a record, falling back to a newest-run guess."""
    if rec.get("local_run_dir"):
        candidate = Path(rec["local_run_dir"])
        if not candidate.is_absolute():
            candidate = st.REPO_ROOT / candidate
        if candidate.is_dir():
            return candidate
    raise SystemExit(
        f"run {rec['run_id']} has no usable local_run_dir. Fetch it first with "
        "deploy/fetch_results.sh, then set local_run_dir on the record."
    )


def fake_eval(rec: dict, args: argparse.Namespace) -> dict:
    """Synthetic episodes for pipeline verification. Never touches torch.

    ``--fake-behaviour`` picks which kind of policy to imitate, so the behaviour
    gate can be exercised without training anything: ``pursuit`` looks like a real
    chase, ``hover`` looks like a policy that waits for the target to arrive, and
    ``freebie`` looks like catches handed over by the spawn geometry.
    """
    rng = random.Random(f"{rec['run_id']}:{args.seed}")
    profiles = {
        "pursuit": dict(align=(0.72, 0.92), travel=(0.9, 1.4), station=(0.0, 0.06),
                        sep=(9.0, 14.0), steps=(90, 260)),
        "hover": dict(align=(-0.10, 0.18), travel=(0.0, 0.12), station=(0.85, 1.0),
                      sep=(6.0, 11.0), steps=(180, 480)),
        "freebie": dict(align=(0.85, 1.0), travel=(0.6, 1.0), station=(0.0, 0.05),
                        sep=(1.0, 2.4), steps=(8, 30)),
    }
    p = profiles[args.fake_behaviour]

    def between(lo_hi):
        return rng.uniform(*lo_hi)

    results = []
    for i in range(args.episodes):
        caught = rng.random() < args.fake_success_rate
        steps = int(between(p["steps"])) if caught else 600
        sep0 = between(p["sep"])
        results.append({
            "episode": i + 1,
            "seed": args.seed + i,
            "caught": caught,
            "steps": steps,
            "min_dist": round(rng.uniform(0.4, 1.8) if caught else rng.uniform(3.0, 14.0), 3),
            "reward": round(rng.uniform(35.0, 60.0) if caught else rng.uniform(-25.0, -2.0), 3),
            "behaviour": {
                "steps": steps,
                "caught": caught,
                "initial_separation": round(sep0, 3),
                "min_separation": round(rng.uniform(0.3, 1.6) if caught else rng.uniform(2.0, 9.0), 3),
                "separation_closed_frac": round(between((0.8, 1.0)) if caught else between((0.1, 0.6)), 3),
                "mean_pursuit_alignment": round(between(p["align"]), 3),
                "early_pursuit_alignment": round(between(p["align"]), 3),
                "travel_ratio": round(between(p["travel"]), 3),
                "displacement_ratio": round(between((0.55, 0.8)), 3),
                "station_keeping_fraction": round(between(p["station"]), 3),
                "idle_fraction": round(between((0.0, 0.1)), 3),
                "path_efficiency": round(between((0.5, 0.9)), 3),
                "closing_fraction": round(between((0.5, 0.8)), 3),
                "heading_reversals_per_100_steps": round(between((2.0, 12.0)), 2),
                "altitude_std": round(between((0.2, 1.2)), 3),
                "steps_to_catch": steps if caught else None,
                "time_to_catch_s": round(steps / 30.0, 3) if caught else None,
            },
        })
    caught_flags = [r["caught"] for r in results]
    return {
        "checkpoint": "<fake>",
        "ckpt_sha256": "<fake>",
        "step_tag": "fake",
        "task": args.task,
        "env_config": rec["env_config"],
        "episodes": len(results),
        "eval_seed_base": args.seed,
        "greedy": True,
        "success_rate": sum(caught_flags) / len(caught_flags),
        "successes": sum(caught_flags),
        "mean_reward": sum(r["reward"] for r in results) / len(results),
        "mean_steps": sum(r["steps"] for r in results) / len(results),
        "mean_min_dist": sum(r["min_dist"] for r in results) / len(results),
        "behaviour": aggregate_behaviour([r["behaviour"] for r in results]),
        "results": results,
        "fake": True,
        "fake_behaviour": args.fake_behaviour,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a run's newest checkpoint.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--task", default="chase", choices=("chase", "target"))
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1000,
                    help="Eval seed base. The reviewer must use a DIFFERENT base.")
    ap.add_argument("--checkpoint", default=None,
                    help="Explicit checkpoint path (default: the run's newest).")
    ap.add_argument("--out", default=None, help="Output JSON path (default: campaign evals/).")
    ap.add_argument("--tag", default=None,
                    help="Filename suffix, e.g. 'review', so a re-eval does not overwrite.")
    ap.add_argument("--trajectories", default=None,
                    help="Directory for per-episode .npz trajectories (default: the campaign's "
                         "trajectories/<run_id><tag>/). The contact sheet the reviewer reads is "
                         "built from these, so they are written by default, not on request.")
    ap.add_argument("--no-trajectories", action="store_true",
                    help="Skip trajectory recording. Then no contact sheet can be built.")
    ap.add_argument("--plots", action="store_true", help="Also render plots (slow).")
    ap.add_argument("--force", action="store_true", help="Re-evaluate even if the JSON exists.")
    ap.add_argument("--fake", action="store_true",
                    help="Synthesise episodes instead of running a policy. Verification only.")
    ap.add_argument("--fake-success-rate", type=float, default=0.7)
    ap.add_argument("--fake-behaviour", default="pursuit", choices=("pursuit", "hover", "freebie"),
                    help="Which kind of policy the synthetic behaviour metrics should imitate, "
                         "so the behaviour gate can be tested without training.")
    args = ap.parse_args()

    rec = st.find_run(args.campaign, args.run_id)
    if rec is None:
        raise SystemExit(f"no run {args.run_id} registered in campaign {args.campaign}")

    evals_dir = st.campaign_dir(args.campaign) / "evals"
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = Path(args.out) if args.out else evals_dir / f"{args.run_id}{suffix}.json"
    if out_path.exists() and not args.force:
        print(f"eval already exists, skipping: {out_path}  (use --force to redo)")
        print(out_path.read_text())
        return 0

    if args.fake:
        payload = fake_eval(rec, args)
        payload.update({
            "run_id": args.run_id,
            "config_hash": rec["config_hash"],
            "rung": rec.get("rung"),
            "seed": rec.get("seed"),
        })
        st.write_json_atomic(out_path, payload)
    else:
        run_dir = resolve_run_dir(rec)
        ckpt = Path(args.checkpoint) if args.checkpoint else newest_checkpoint(run_dir)
        if ckpt is None:
            raise SystemExit(f"no checkpoints under {run_dir}/checkpoint")

        # Trajectories are the reviewer's contact sheet. Record them by default, to
        # a deterministic per-eval dir, so the visual check is never silently blank.
        traj_dir = None
        if not args.no_trajectories:
            if args.trajectories:
                traj_dir = Path(args.trajectories)
                if not traj_dir.is_absolute():
                    traj_dir = st.REPO_ROOT / traj_dir
            else:
                traj_dir = st.campaign_dir(args.campaign) / "trajectories" / f"{args.run_id}{suffix}"

        cmd = [
            str(st.REPO_ROOT / "venv/bin/python"),
            str(st.REPO_ROOT / "scripts/eval_target_ckpt.py"), str(ckpt),
            "--task", args.task,
            # Pinned to the REGISTERED config, not a caller-supplied one. This is
            # what stops a rung from being quietly graded on an easier task.
            "--config", rec["env_config"],
            "--episodes", str(args.episodes),
            "--seed", str(args.seed),
            "--json", str(out_path),
        ]
        if traj_dir is not None:
            cmd += ["--trajectories", str(traj_dir)]
        if not Path(cmd[0]).exists():
            cmd[0] = sys.executable
        if not args.plots:
            cmd.append("--no-plots")
        print(f"==> {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(st.REPO_ROOT), check=False)
        if proc.returncode != 0:
            raise SystemExit(f"eval_target_ckpt.py exited {proc.returncode}")

        payload = json.loads(out_path.read_text())
        payload.update({
            "run_id": args.run_id,
            "config_hash": rec["config_hash"],
            "rung": rec.get("rung"),
            "seed": rec.get("seed"),
        })
        if traj_dir is not None:
            payload["trajectories"] = str(traj_dir.relative_to(st.REPO_ROOT))
        st.write_json_atomic(out_path, payload)

    st.update_run(args.campaign, args.run_id, status="evaluated", eval={
        "path": str(out_path.relative_to(st.REPO_ROOT)),
        "success_rate": payload["success_rate"],
        "episodes": payload["episodes"],
        "eval_seed_base": payload["eval_seed_base"],
    })
    st.journal(args.campaign,
               f"eval {args.run_id}: success={payload['success_rate']:.0%} "
               f"({payload['successes']}/{payload['episodes']} eps, seed base {args.seed}) "
               f"-> {out_path.name}")
    print(f"\nsuccess_rate={payload['success_rate']:.3f}  path={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
