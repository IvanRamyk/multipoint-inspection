#!/usr/bin/env python3
"""Watch a training run: periodically render a 3D video of the newest checkpoint
and refresh the learning-curve chart — so you can see the policy improve.

Runs as a separate process alongside training (does NOT touch the training loop;
it only reads checkpoints + tfevents). Every ``--interval`` seconds it:
  1. finds the newest fully-written checkpoint in the run dir,
  2. evals it greedily and renders a step-tagged 3D video (fixed seeds, so the
     same target routes recur and improvement is directly comparable),
  3. regenerates results/learning_curve.png.

Exits when no new checkpoint appears for ``--idle-exit`` consecutive cycles
(training finished) — or when killed.

    ./venv/bin/python scripts/watch_train.py --task chase --interval 180
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _newest_run_dir() -> str | None:
    dirs = glob.glob("logs/runs/dreamer_v3/*/*/version_0")
    return max(dirs, key=os.path.getmtime) if dirs else None


def _newest_ready_ckpt(run_dir: str, min_age: float = 5.0) -> str | None:
    """Newest checkpoint that finished writing (mtime older than min_age)."""
    cks = glob.glob(os.path.join(run_dir, "checkpoint", "ckpt_*.ckpt"))
    ready = [c for c in cks if time.time() - os.path.getmtime(c) >= min_age]
    if not ready:
        return None
    # ckpt_<step>_0.ckpt — sort by the numeric step.
    return max(ready, key=lambda c: int(Path(c).stem.split("_")[1]))


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(_ROOT), env={**os.environ, "MPLBACKEND": "Agg"},
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    ap = argparse.ArgumentParser(description="Periodic checkpoint video + chart watcher")
    ap.add_argument("--task", choices=["target", "chase"], default="chase")
    ap.add_argument("--config", type=str, default=None, help="env config (default: task default)")
    ap.add_argument("--run-dir", type=str, default=None, help="version_0 dir (default: newest)")
    ap.add_argument("--interval", type=float, default=180.0, help="seconds between renders")
    ap.add_argument("--episodes", type=int, default=2, help="eval episodes per checkpoint")
    ap.add_argument("--seed", type=int, default=100, help="fixed base seed (comparable routes)")
    ap.add_argument("--out-dir", type=str, default="results/watch")
    ap.add_argument("--idle-exit", type=int, default=4, help="exit after this many cycles with no new checkpoint")
    args = ap.parse_args()

    py = sys.executable
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = None
    idle = 0
    print(f"[watch] task={args.task} interval={args.interval:.0f}s → {out_dir}")

    while True:
        run_dir = args.run_dir or _newest_run_dir()
        ckpt = _newest_ready_ckpt(run_dir) if run_dir else None

        if ckpt and ckpt != last_ckpt:
            step = Path(ckpt).stem.split("_")[1]
            print(f"[watch] rendering checkpoint step {step}", flush=True)
            eval_cmd = [
                py, "scripts/eval_target_ckpt.py", ckpt,
                "--task", args.task, "--episodes", str(args.episodes),
                "--seed", str(args.seed), "--video3d",
                "--out", str(out_dir / f"step_{int(step):07d}"),
            ]
            if args.config:
                eval_cmd += ["--config", args.config]
            _run(eval_cmd)
            # refresh chart
            chart_cmd = [py, "scripts/plot_learning_curve.py", "--out", str(out_dir / "learning_curve.png")]
            if run_dir:
                chart_cmd += ["--run-dir", run_dir]
            _run(chart_cmd)
            print(f"[watch] step {step}: video + chart updated", flush=True)
            last_ckpt = ckpt
            idle = 0
        else:
            idle += 1
            # Stop once training is clearly done: no new checkpoint AND no trainer running.
            trainer_alive = subprocess.run(
                ["pgrep", "-f", "python -m sheeprl"], stdout=subprocess.DEVNULL
            ).returncode == 0
            if idle >= args.idle_exit and not trainer_alive:
                print("[watch] no new checkpoints and trainer gone — exiting.", flush=True)
                break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
