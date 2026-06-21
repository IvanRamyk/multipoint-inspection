#!/usr/bin/env python3
"""Generate supervisor-ready plots from a DreamerV3 TensorBoard run directory.

Usage:
    python scripts/plot_results.py <version_0_dir> [--out results/]

Produces:
    training_overview.png  — reward + episode length + key losses (4-panel)
    reward_curve.png       — reward only, large, presentation-ready
    summary.txt            — key numbers in plain text
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_scalars(events_dir: str) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(events_dir, size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in ea.Tags()["scalars"]:
        out[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return out


def smooth(vals: list[float], w: int = 5) -> list[float]:
    if len(vals) < w:
        return vals
    kernel = np.ones(w) / w
    padded = np.pad(vals, (w // 2, w // 2), mode="edge")
    return list(np.convolve(padded, kernel, mode="valid")[: len(vals)])


def plot_series(ax, steps, vals, label, color, smooth_w=3, alpha_raw=0.25):
    steps = np.array(steps)
    vals = np.array(vals)
    s = smooth(list(vals), smooth_w)
    ax.plot(steps, vals, color=color, alpha=alpha_raw, linewidth=0.8)
    ax.plot(steps, s, color=color, linewidth=2.0, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Path to version_0 directory")
    ap.add_argument("--out", default="results", help="Output directory")
    ap.add_argument("--backend", default="AirSim", help="Backend label for titles")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading events from {run_dir} ...")
    data = load_scalars(str(run_dir))

    available = sorted(data.keys())
    print(f"Available tags ({len(available)}):")
    for t in available:
        n = len(data[t])
        if n:
            print(f"  {t}  ({n} pts, last={data[t][-1][1]:.4f})")

    def get(tag):
        if tag not in data or not data[tag]:
            return [], []
        pts = data[tag]
        return [p[0] for p in pts], [p[1] for p in pts]

    # ── 4-panel overview ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"DreamerV3 Training — Drone Inspection ({args.backend} backend, N=1 waypoint)",
        fontsize=13, fontweight="bold"
    )

    panels = [
        ("Rewards/rew_avg",         "Avg Episode Reward",    axes[0, 0], "#2196F3"),
        ("Game/ep_len_avg",          "Avg Episode Length (steps)", axes[0, 1], "#4CAF50"),
        ("Loss/world_model_loss",    "World Model Loss",      axes[1, 0], "#FF5722"),
        ("Loss/policy_loss",         "Policy Loss",           axes[1, 1], "#9C27B0"),
    ]

    for tag, ylabel, ax, color in panels:
        steps, vals = get(tag)
        if not steps:
            ax.text(0.5, 0.5, f"No data\n({tag})", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(ylabel)
            continue
        plot_series(ax, steps, vals, ylabel, color)
        ax.set_xlabel("Env steps")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    overview_path = out_dir / "training_overview.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {overview_path}")

    # ── Large reward curve ────────────────────────────────────────────────────
    steps_r, vals_r = get("Rewards/rew_avg")
    if steps_r:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        plot_series(ax2, steps_r, vals_r, "Avg reward (smoothed)", "#2196F3", smooth_w=5)
        ax2.axhline(60, color="gray", linestyle="--", linewidth=1.2, label="Target (+60)")
        ax2.set_xlabel("Environment steps", fontsize=12)
        ax2.set_ylabel("Episode reward", fontsize=12)
        ax2.set_title(
            f"DreamerV3 Reward Curve — {args.backend} Backend (Sanity: N=1 waypoint)",
            fontsize=13
        )
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)
        reward_path = out_dir / "reward_curve.png"
        fig2.savefig(reward_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {reward_path}")

    # ── Summary text ──────────────────────────────────────────────────────────
    lines = [
        "DreamerV3 Training Summary",
        "=" * 40,
        f"Backend:       {args.backend}",
        f"Task:          N=1 waypoint, dome=5m, reach=3m",
        f"Config:        sanity_airsim.yaml",
        f"Algorithm:     DreamerV3-XS (sheeprl 0.5.7)",
        "",
    ]

    for tag, label in [
        ("Rewards/rew_avg",       "Final avg reward"),
        ("Game/ep_len_avg",       "Final avg ep length"),
        ("Loss/world_model_loss", "World model loss"),
        ("Loss/policy_loss",      "Policy loss"),
        ("Loss/value_loss",       "Value loss"),
        ("Time/sps_train",        "Train steps/sec"),
        ("Time/sps_env_interaction", "Env steps/sec"),
    ]:
        steps, vals = get(tag)
        if vals:
            last_step = steps[-1]
            last_val = vals[-1]
            peak = max(vals) if "rew" in tag else None
            s = f"{label:<28} {last_val:>10.3f}  (step {last_step}"
            if peak is not None:
                s += f", peak {peak:.3f}"
            s += ")"
            lines.append(s)

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {summary_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
