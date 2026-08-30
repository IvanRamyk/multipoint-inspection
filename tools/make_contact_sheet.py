#!/usr/bin/env python3
"""Tile episode trajectories into one PNG a reviewer can actually look at.

Numbers can say a policy pursued the target. They are worse at saying the flight
looks competent, which is what a human notices immediately and what decides
whether a result is showable. One image of N episodes side by side is the cheapest
way to put that judgment in front of an agent: a single Read call, a few thousand
tokens, no video decoding.

Each panel is one episode: the drone's track, the target's track, both start
points, the catch radius at the interception, and a caption carrying the numbers
that matter for that episode. Deliberately top-down and to scale, because the
question being asked is about geometry — did it fly at the target, or did it
wander until the target arrived.

    tools/make_contact_sheet.py --trajectories <dir> --out sheet.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.behaviour_metrics import episode_metrics  # noqa: E402


def load_episodes(traj_dir: Path, limit: int) -> list[dict]:
    """Load up to ``limit`` episodes, preferring a mix of catches and misses.

    A sheet of nothing but successes hides the failure mode; a sheet of nothing
    but failures hides what success looks like. Interleaving both is what makes
    the picture diagnostic.
    """
    files = sorted(traj_dir.glob("*.npz"))
    if not files:
        return []
    episodes = []
    for path in files:
        with np.load(path) as data:
            episodes.append({
                "name": path.stem,
                "drone": data["drone"],
                "target": data["target"],
                "caught": bool(data["caught"]),
                "reach_distance": float(data["reach_distance"]),
                "dome_size": float(data["dome_size"]),
                "dt": float(data["dt"]),
            })
    catches = [e for e in episodes if e["caught"]]
    misses = [e for e in episodes if not e["caught"]]
    mixed: list[dict] = []
    while len(mixed) < limit and (catches or misses):
        if catches:
            mixed.append(catches.pop(0))
        if len(mixed) < limit and misses:
            mixed.append(misses.pop(0))
    return mixed[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description="Tile episode trajectories into one reviewable PNG.")
    ap.add_argument("--trajectories", required=True, help="Directory of .npz files from eval.")
    ap.add_argument("--out", required=True, help="Output PNG path.")
    ap.add_argument("--max-episodes", type=int, default=9)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj_dir = Path(args.trajectories)
    episodes = load_episodes(traj_dir, args.max_episodes)
    if not episodes:
        print(f"no .npz trajectories under {traj_dir}", file=sys.stderr)
        return 2

    cols = min(args.cols, len(episodes))
    rows = (len(episodes) + cols - 1) // cols
    # Panels are tall enough to carry a four-line caption without the next row's
    # title landing on the previous row's axis labels.
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.6 * cols, 5.6 * rows), squeeze=False,
        gridspec_kw={"hspace": 0.30, "wspace": 0.25},
    )

    for idx, ax in enumerate([a for row in axes for a in row]):
        if idx >= len(episodes):
            ax.axis("off")
            continue
        ep = episodes[idx]
        drone, target = ep["drone"], ep["target"]
        m = episode_metrics(
            drone, target, caught=ep["caught"], dt=ep["dt"],
            reach_distance=ep["reach_distance"], dome_size=ep["dome_size"],
        )

        ax.plot(target[:, 0], target[:, 1], "-", color="#d1495b", lw=1.6, label="target", alpha=0.9)
        ax.plot(drone[:, 0], drone[:, 1], "-", color="#3d5a80", lw=1.6, label="drone", alpha=0.9)
        ax.plot(*drone[0, :2], "o", color="#3d5a80", ms=7, mec="white", mew=1.2)
        ax.plot(*target[0, :2], "s", color="#d1495b", ms=7, mec="white", mew=1.2)

        # Mark the interception and draw the catch radius to scale, so "caught
        # from 2 m away at step 20" is visible rather than inferred.
        if ep["caught"]:
            ax.add_patch(plt.Circle(
                (target[-1, 0], target[-1, 1]), ep["reach_distance"],
                fill=False, ec="#2a9d8f", lw=1.8, ls="--"))
            ax.plot(*drone[-1, :2], "*", color="#2a9d8f", ms=15, mec="white", mew=0.8)

        radius = ep["dome_size"] / 2.0
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, ec="#adb5bd", lw=0.9, ls=":"))
        ax.set_xlim(-radius * 1.25, radius * 1.25)
        ax.set_ylim(-radius * 1.25, radius * 1.25)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

        status = "CAUGHT" if ep["caught"] else "missed"

        def fmt(key: str, spec: str) -> str:
            v = m.get(key)
            return "n/a" if v is None else format(v, spec)

        ax.set_title(
            f"{ep['name']} — {status}\n"
            f"align={fmt('mean_pursuit_alignment', '+.2f')}  "
            f"travel={fmt('travel_ratio', '.2f')}  "
            f"direct={fmt('displacement_ratio', '.2f')}\n"
            f"station={fmt('station_keeping_fraction', '.0%')}  "
            f"rev/100={fmt('heading_reversals_per_100_steps', '.0f')}  "
            f"steps={m['steps']}\n"
            f"sep {m['initial_separation']:.1f}→{m['min_separation']:.1f} m",
            fontsize=8.5,
        )
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    header = args.title or f"{traj_dir.name} — {len(episodes)} episodes (top-down, to scale)"
    fig_height = 5.6 * rows
    # Pinned to the very top with va="top" so it cannot drift down onto the
    # first row's own four-line panel titles.
    fig.suptitle(header, fontsize=12, y=1.0 - 0.08 / fig_height, va="top")
    # The metric legend goes at the FOOT, not in the suptitle: as a multi-line
    # suptitle it lands on the first row's own multi-line panel titles.
    fig.text(
        0.5, 0.012,
        "align = cosine(velocity, bearing to target): ~1 is real pursuit, ~0 means it wandered into the target.    "
        "travel = net displacement / starting gap: ~0 means it never went anywhere.\n"
        "direct = displacement / distance flown: real pursuit curves, so ~0.6-0.8 is normal and ~0.2 is looping.    "
        "station = fraction of the episode spent holding position.",
        ha="center", va="bottom", fontsize=9.5,
    )
    # No tight_layout: it overrides the explicit hspace and re-crowds the
    # four-line panel titles. Reserve fixed margins in figure fractions instead.
    fig.subplots_adjust(
        top=1.0 - 1.05 / fig_height,
        bottom=0.55 / fig_height,
        left=0.06, right=0.97,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out} ({len(episodes)} panels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
