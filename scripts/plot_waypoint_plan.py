#!/usr/bin/env python3
"""Visualize non-self-intersecting random waypoint plans (no RL, no physics).

Generates several random routes and plots their ground tracks, asserting none
self-intersects. Use to eyeball that 2-opt ordering gives natural, non-crossing
routes before wiring the target into an environment.

    ./venv/bin/python scripts/plot_waypoint_plan.py --n 8 --method 2opt --plans 6
    ./venv/bin/python scripts/plot_waypoint_plan.py --n 8 --method angular
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.targets.waypoint_plan import generate_points, order_angular, order_simple_2opt, path_is_simple


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot non-self-intersecting waypoint plans")
    ap.add_argument("--n", type=int, default=8, help="waypoints per plan")
    ap.add_argument("--dome", type=float, default=20.0)
    ap.add_argument("--min-sep", type=float, default=3.0)
    ap.add_argument("--method", choices=["2opt", "angular"], default="2opt")
    ap.add_argument("--plans", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/waypoint_plans.png")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cols = 3
    rows = (args.plans + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    r = args.dome / 2.0
    th = np.linspace(0, 2 * np.pi, 200)
    all_simple = True

    for k in range(args.plans):
        ax = axes[k]
        pts = generate_points(args.n, args.dome, rng, args.min_sep)
        order = order_angular(pts) if args.method == "angular" else order_simple_2opt(pts, rng)
        route = pts[order]
        simple = path_is_simple(route)
        all_simple = all_simple and simple

        ax.plot(r * np.cos(th), r * np.sin(th), "k--", lw=0.8, alpha=0.4)
        ax.plot(route[:, 0], route[:, 1], "-o", lw=1.4, ms=4, color="tab:blue")
        ax.plot(route[0, 0], route[0, 1], "s", ms=9, color="green")   # start
        ax.plot(route[-1, 0], route[-1, 1], "*", ms=13, color="red")  # end
        for idx, (x, y) in enumerate(route[:, :2]):
            ax.annotate(str(idx), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_aspect("equal")
        ax.set_title(f"plan {k+1}: {'SIMPLE' if simple else 'CROSSES!'}", fontsize=9,
                     color="green" if simple else "red")
        ax.grid(alpha=0.3)

    for k in range(args.plans, len(axes)):
        axes[k].axis("off")

    fig.suptitle(f"Waypoint plans (n={args.n}, method={args.method}, dome={args.dome}m)")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"Saved {out}")
    print(f"All {args.plans} plans non-self-intersecting: {all_simple}")
    if not all_simple:
        sys.exit(1)


if __name__ == "__main__":
    main()
