#!/usr/bin/env python3
"""Render a recorded chase episode (.npz with drone+target position tracks) to MP4.

Unlike results/render_chase_video.py (which does a fresh PyBullet rollout and needs
a checkpoint), this animates the ACTUAL episode that was recorded during evaluation
— the real "last episodes". One-off presentation tool (lives in gitignored results/).

Usage:
  venv/bin/python results/render_traj_npz.py <ep.npz> --out out.mp4 [--title "..."]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, FFMpegWriter
import imageio_ffmpeg
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=3, help="subsample steps for speed")
    ap.add_argument("--trail", type=int, default=120, help="trailing points drawn")
    args = ap.parse_args()

    z = np.load(args.npz)
    drone = z["drone"].astype(float)      # (T,3)
    target = z["target"].astype(float)
    reach = float(z["reach_distance"])
    dt = float(z["dt"]) if "dt" in z.files else 1 / 30
    dist = np.linalg.norm(drone - target, axis=1)
    T = len(drone)
    idx = list(range(0, T, args.stride))
    if idx[-1] != T - 1:
        idx.append(T - 1)
    mind = float(dist.min()); mind_i = int(dist.argmin())

    # bounds
    allp = np.concatenate([drone, target], 0)
    lo = allp.min(0) - 3; hi = allp.max(0) + 3
    rng = (hi - lo).max()
    ctr = (hi + lo) / 2
    lo = ctr - rng / 2; hi = ctr + rng / 2

    fig = plt.figure(figsize=(9, 7), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(max(0, lo[2]), hi[2])
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("alt (m)")

    (d_line,) = ax.plot([], [], [], "-", color="#1f77b4", lw=2.0, label="pursuer (DreamerV3)")
    (t_line,) = ax.plot([], [], [], "-", color="#d62728", lw=2.0, label="target")
    (d_dot,) = ax.plot([], [], [], "o", color="#1f77b4", ms=8)
    (t_dot,) = ax.plot([], [], [], "o", color="#d62728", ms=8)
    (link,) = ax.plot([], [], [], ":", color="gray", lw=1.0)
    ax.legend(loc="upper right", fontsize=9)
    txt = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, fontsize=10,
                    va="top", family="monospace",
                    bbox=dict(facecolor="white", alpha=0.7, pad=4))

    base_title = args.title or Path(args.npz).stem

    def upd(fi):
        i = idx[fi]
        s = max(0, i - args.trail * args.stride)
        d_line.set_data(drone[s:i + 1, 0], drone[s:i + 1, 1]); d_line.set_3d_properties(drone[s:i + 1, 2])
        t_line.set_data(target[s:i + 1, 0], target[s:i + 1, 1]); t_line.set_3d_properties(target[s:i + 1, 2])
        d_dot.set_data(drone[i:i + 1, 0], drone[i:i + 1, 1]); d_dot.set_3d_properties(drone[i:i + 1, 2])
        t_dot.set_data(target[i:i + 1, 0], target[i:i + 1, 1]); t_dot.set_3d_properties(target[i:i + 1, 2])
        link.set_data([drone[i, 0], target[i, 0]], [drone[i, 1], target[i, 1]])
        link.set_3d_properties([drone[i, 2], target[i, 2]])
        ax.view_init(elev=24, azim=(fi * 0.4) % 360)
        near = "  <-- closest" if abs(i - mind_i) < args.stride else ""
        txt.set_text(f"{base_title}\nt={i*dt:5.1f}s  step {i:4d}/{T}\n"
                     f"separation = {dist[i]:6.2f} m{near}\n"
                     f"episode min = {mind:.2f} m   (catch < {reach:.2f} m)")
        return d_line, t_line, d_dot, t_dot, link, txt

    anim = FuncAnimation(fig, upd, frames=len(idx), interval=1000 / args.fps, blit=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=2600))
    plt.close(fig)
    print(f"wrote {args.out}  (min_sep {mind:.2f} m @ step {mind_i}, len {T})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
