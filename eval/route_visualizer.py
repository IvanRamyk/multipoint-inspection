"""Route visualization utilities using Matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_route_topdown(
    positions: np.ndarray,
    waypoints: np.ndarray,
    visited_mask: np.ndarray,
    base: np.ndarray,
    obstacles: list[dict] | None = None,
    save_path: str | Path | None = None,
    reach_radius: float | None = None,
    title: str | None = None,
) -> None:
    """Plot a top-down (XY) view of the drone route.

    Args:
        positions: (T, 3) array of drone positions over time.
        waypoints: (N, 3) waypoint positions.
        visited_mask: (N,) boolean mask — True if visited.
        base: (3,) base position.
        obstacles: Optional list of obstacle dicts with 'position' and 'size'.
        save_path: If provided, save the figure to this path.
        reach_radius: If provided, draw a filled disk of this radius around each
            waypoint and around the base (the "must enter" zones).
        title: If provided, override the default plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Drone path.
    positions = np.asarray(positions)
    ax.plot(positions[:, 0], positions[:, 1], "b-", linewidth=0.8, alpha=0.7, label="Path")
    ax.plot(positions[0, 0], positions[0, 1], "bs", markersize=6)
    ax.plot(positions[-1, 0], positions[-1, 1], "b^", markersize=6)

    # Waypoints (with reach radius disk if provided).
    for i, wp in enumerate(waypoints):
        color = "green" if visited_mask[i] else "red"
        if reach_radius is not None:
            disk = plt.Circle((wp[0], wp[1]), reach_radius, color=color, alpha=0.18)
            ax.add_patch(disk)
        ax.plot(wp[0], wp[1], "o", color=color, markersize=10)
        ax.annotate(f"WP{i}", (wp[0], wp[1]), textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Base return zone (same radius as waypoint reach).
    if reach_radius is not None:
        base_disk = plt.Circle((base[0], base[1]), reach_radius, color="black", alpha=0.08, linestyle="--")
        ax.add_patch(base_disk)

    # Base.
    ax.plot(base[0], base[1], "k*", markersize=15, label="Base")

    # Obstacles.
    if obstacles:
        for obs in obstacles:
            pos = obs["position"]
            size = obs["size"]
            rect = plt.Rectangle(
                (pos[0] - size / 2, pos[1] - size / 2),
                size, size, color="gray", alpha=0.5,
            )
            ax.add_patch(rect)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title or "Drone Route — Top Down")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved top-down plot to {save_path}")
    plt.close(fig)


def plot_route_3d(
    positions: np.ndarray,
    waypoints: np.ndarray,
    visited_mask: np.ndarray,
    base: np.ndarray,
    obstacles: list[dict] | None = None,
    save_path: str | Path | None = None,
    reach_radius: float | None = None,
    title: str | None = None,
) -> None:
    """Plot a 3D view of the drone route.

    Args:
        positions: (T, 3) array of drone positions over time.
        waypoints: (N, 3) waypoint positions.
        visited_mask: (N,) boolean mask — True if visited.
        base: (3,) base position.
        obstacles: Optional list of obstacle dicts with 'position' and 'size'.
        save_path: If provided, save the figure to this path.
        reach_radius: If provided, draw a translucent sphere of this radius around
            each waypoint and the base.
        title: If provided, override the default plot title.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Drone path.
    positions = np.asarray(positions)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b-", linewidth=0.8, alpha=0.7, label="Path")

    # Pre-compute unit sphere mesh once if needed.
    if reach_radius is not None:
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 16)
        sx = np.outer(np.cos(u), np.sin(v))
        sy = np.outer(np.sin(u), np.sin(v))
        sz = np.outer(np.ones_like(u), np.cos(v))

    # Waypoints (with reach radius sphere if provided).
    for i, wp in enumerate(waypoints):
        color = "green" if visited_mask[i] else "red"
        if reach_radius is not None:
            ax.plot_surface(
                wp[0] + reach_radius * sx,
                wp[1] + reach_radius * sy,
                wp[2] + reach_radius * sz,
                color=color, alpha=0.15, linewidth=0,
            )
        ax.scatter(wp[0], wp[1], wp[2], c=color, s=80, marker="o")
        ax.text(wp[0], wp[1], wp[2], f" WP{i}", fontsize=8)

    # Base + base-return sphere.
    if reach_radius is not None:
        ax.plot_surface(
            base[0] + reach_radius * sx,
            base[1] + reach_radius * sy,
            base[2] + reach_radius * sz,
            color="black", alpha=0.07, linewidth=0,
        )
    ax.scatter(base[0], base[1], base[2], c="black", s=200, marker="*", label="Base")

    # Obstacles.
    if obstacles:
        for obs in obstacles:
            pos = obs["position"]
            ax.scatter(pos[0], pos[1], pos[2], c="gray", s=100, marker="s", alpha=0.5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title or "Drone Route — 3D")
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved 3D plot to {save_path}")
    plt.close(fig)
