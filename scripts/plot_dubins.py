#!/usr/bin/env python3
"""Visualize Dubins moving-target trajectories (no RL, no physics).

Rolls out the target motion models for a number of episodes and plots the
paths, so you can eyeball whether the motion looks natural and stays inside the
flight dome before wiring it into the environment.

Usage:
    ./venv/bin/python scripts/plot_dubins.py --mode car2d --dome 20 --episodes 6
    ./venv/bin/python scripts/plot_dubins.py --mode airplane3d --dome 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.targets import DubinsAirplane3D, DubinsCar2D, StaticTarget


def _build(mode: str, dome: float, speed: float, dt: float):
    if mode == "static":
        return StaticTarget(dome_size=dome, dt=dt)
    if mode == "car2d":
        return DubinsCar2D(dome_size=dome, speed=speed, dt=dt)
    if mode == "airplane3d":
        return DubinsAirplane3D(dome_size=dome, speed=speed, dt=dt)
    raise ValueError(f"Unknown mode {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Dubins target trajectories")
    parser.add_argument("--mode", choices=["static", "car2d", "airplane3d"], default="car2d")
    parser.add_argument("--dome", type=float, default=20.0, help="Flight dome size (m)")
    parser.add_argument("--speed", type=float, default=1.0, help="Target speed (m/s)")
    parser.add_argument("--steps", type=int, default=600, help="Steps per episode")
    parser.add_argument("--episodes", type=int, default=6, help="Number of trajectories")
    parser.add_argument("--agent-hz", type=int, default=30, help="Control rate (dt = 1/hz)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/dubins_trajectories.png")
    args = parser.parse_args()

    dt = 1.0 / args.agent_hz
    rng = np.random.default_rng(args.seed)
    dome_radius = args.dome / 2.0

    is_3d = args.mode == "airplane3d"
    fig = plt.figure(figsize=(12, 5.5))
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_side = fig.add_subplot(1, 2, 2, projection="3d" if is_3d else None)

    max_r = 0.0
    for ep in range(args.episodes):
        target = _build(args.mode, args.dome, args.speed, dt)
        target.reset(rng)
        pts = [target.position.copy()]
        for _ in range(args.steps):
            pts.append(target.step().copy())
        pts = np.array(pts)
        max_r = max(max_r, float(np.max(np.hypot(pts[:, 0], pts[:, 1]))))

        ax_xy.plot(pts[:, 0], pts[:, 1], lw=1.0, alpha=0.8)
        ax_xy.plot(pts[0, 0], pts[0, 1], "o", ms=5, color="green")
        ax_xy.plot(pts[-1, 0], pts[-1, 1], "s", ms=5, color="red")

        if is_3d:
            ax_side.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=1.0, alpha=0.8)
        else:
            t = np.arange(len(pts)) * dt
            ax_side.plot(t, pts[:, 2], lw=1.0, alpha=0.8)

    # Dome boundary circle on the top-down plot.
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_xy.plot(dome_radius * np.cos(theta), dome_radius * np.sin(theta), "k--", lw=1.0, label="dome edge")
    ax_xy.set_aspect("equal")
    ax_xy.set_title(f"Top-down ({args.mode}, dome={args.dome}m, v={args.speed}m/s)")
    ax_xy.set_xlabel("x (m)"); ax_xy.set_ylabel("y (m)")
    ax_xy.legend(loc="upper right", fontsize=8)

    if is_3d:
        ax_side.set_title("3D path")
        ax_side.set_xlabel("x"); ax_side.set_ylabel("y"); ax_side.set_zlabel("z (m)")
    else:
        ax_side.set_title("Altitude vs time")
        ax_side.set_xlabel("t (s)"); ax_side.set_ylabel("z (m)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")
    print(f"Max radius reached: {max_r:.2f} m  (dome radius = {dome_radius:.2f} m)")
    if max_r > dome_radius * 1.02:
        print("WARNING: a trajectory left the dome — tighten boundary_frac / turn_rate_max.")
    else:
        print("OK: all trajectories stayed within the dome.")


if __name__ == "__main__":
    main()
