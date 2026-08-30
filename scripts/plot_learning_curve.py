#!/usr/bin/env python3
"""Plot the learning curve (reward + episode length) from a run's tfevents.

Reads TensorBoard scalars and writes a PNG. Defaults to the newest run under
logs/runs. Safe to call repeatedly during training (e.g. from the watcher).

    ./venv/bin/python scripts/plot_learning_curve.py
    ./venv/bin/python scripts/plot_learning_curve.py --run-dir logs/runs/.../version_0 --out results/curve.png
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _newest_run_dir() -> str | None:
    dirs = glob.glob("logs/runs/dreamer_v3/*/*/version_0")
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def _read(run_dir: str, tag: str):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not files:
        return [], []
    ea = EventAccumulator(max(files, key=os.path.getmtime), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    sc = ea.Scalars(tag)
    return [e.step for e in sc], [e.value for e in sc]


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot learning curve from tfevents")
    ap.add_argument("--run-dir", type=str, default=None, help="version_0 dir (default: newest)")
    ap.add_argument("--out", type=str, default="results/learning_curve.png")
    ap.add_argument("--threshold", type=float, default=45.0, help="reference line (catch threshold)")
    args = ap.parse_args()

    run_dir = args.run_dir or _newest_run_dir()
    if not run_dir:
        print("No run dir found under logs/runs.")
        sys.exit(1)

    steps_r, rew = _read(run_dir, "Rewards/rew_avg")
    steps_l, eplen = _read(run_dir, "Game/ep_len_avg")
    if not steps_r:
        print(f"No Rewards/rew_avg logged yet in {run_dir}")
        sys.exit(0)

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(steps_r, rew, "o-", color="tab:blue", lw=1.8, ms=3, label="mean reward")
    ax1.axhline(args.threshold, ls=":", color="orange", alpha=0.8, label=f"catch threshold ({args.threshold:.0f})")
    ax1.set_xlabel("env steps"); ax1.set_ylabel("mean episode reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue"); ax1.grid(alpha=0.3)
    if steps_l:
        ax2 = ax1.twinx()
        ax2.plot(steps_l, eplen, "-", color="tab:green", alpha=0.5, lw=1.2, label="ep length")
        ax2.set_ylabel("mean episode length", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
    ax1.set_title(f"Learning curve — {Path(run_dir).parent.name}\nlatest: step {steps_r[-1]}, reward {rew[-1]:.1f}")
    ax1.legend(loc="lower right", fontsize=8)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"Saved {out}  (step {steps_r[-1]}, reward {rew[-1]:.1f}, n={len(steps_r)})")


if __name__ == "__main__":
    main()
