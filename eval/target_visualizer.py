"""Visualization for the moving-target interception task.

Plots the drone path against the (moving) target path: a top-down XY view, a 3D
view, and distance-to-target over time with the catch threshold marked.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: safe inside the training process and CLI eval.

import matplotlib.pyplot as plt
import numpy as np


def _ffmpeg_writer(fps: int):
    """Return a matplotlib ffmpeg writer, using imageio's bundled binary.

    System ffmpeg is often absent; ``imageio_ffmpeg`` ships one. Returns None
    if no ffmpeg is available (caller falls back to GIF).
    """
    import matplotlib.animation as animation

    try:
        import imageio_ffmpeg

        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    if animation.writers.is_available("ffmpeg"):
        return animation.FFMpegWriter(fps=fps, bitrate=2400)
    return None


def animate_target_episode_3d(
    drone: np.ndarray,
    target: np.ndarray,
    base: np.ndarray,
    reach_distance: float,
    dome_size: float,
    caught: bool,
    dt: float = 1.0 / 30.0,
    save_path: str | Path = "results/target_eval_3d.mp4",
    fps: int = 30,
    max_frames: int = 300,
    orbit_deg_per_frame: float = 0.6,
    title_suffix: str = "",
) -> str:
    """Render the episode as a 3D video with a slowly orbiting camera.

    Drone (blue) and target (red) fly with growing trails; the current target
    position carries a translucent catch sphere. The camera azimuth rotates so
    the 3D geometry of the chase reads clearly. MP4 if ffmpeg is available, else
    an animated GIF.

    Args mirror :func:`animate_target_episode`, plus:
        orbit_deg_per_frame: Camera azimuth increment per rendered frame.

    Returns:
        The actual path written.
    """
    import matplotlib.animation as animation

    drone = np.asarray(drone)
    target = np.asarray(target)
    T = len(drone)
    stride = max(1, T // max_frames)
    idx = list(range(0, T, stride))
    if idx[-1] != T - 1:
        idx.append(T - 1)
    status = "CAUGHT" if caught else "missed"

    # Fixed axis bounds spanning both paths (equal aspect-ish).
    allpts = np.vstack([drone, target])
    lo = allpts.min(axis=0)
    hi = allpts.max(axis=0)
    ctr = (lo + hi) / 2.0
    span = max((hi - lo).max(), 2.0) * 0.6

    # Precompute a unit sphere mesh for the catch radius.
    u = np.linspace(0, 2 * np.pi, 16)
    v = np.linspace(0, np.pi, 10)
    sx = reach_distance * np.outer(np.cos(u), np.sin(v))
    sy = reach_distance * np.outer(np.sin(u), np.sin(v))
    sz = reach_distance * np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(max(0, ctr[2] - span), ctr[2] + span)
    ax.set_title(f"3D chase — {status} {title_suffix}".strip())
    ax.scatter(*base, c="black", s=120, marker="*")

    drone_trail, = ax.plot([], [], [], "b-", lw=1.6, alpha=0.85, label="Drone")
    target_trail, = ax.plot([], [], [], "r-", lw=1.6, alpha=0.85, label="Target")
    drone_dot, = ax.plot([], [], [], "bo", ms=8)
    target_dot, = ax.plot([], [], [], "ro", ms=8)
    ax.legend(loc="upper right", fontsize=9)
    sphere_holder = {"surf": None}

    def update(frame_i):
        k = idx[frame_i]
        drone_trail.set_data(drone[:k + 1, 0], drone[:k + 1, 1])
        drone_trail.set_3d_properties(drone[:k + 1, 2])
        target_trail.set_data(target[:k + 1, 0], target[:k + 1, 1])
        target_trail.set_3d_properties(target[:k + 1, 2])
        drone_dot.set_data([drone[k, 0]], [drone[k, 1]]); drone_dot.set_3d_properties([drone[k, 2]])
        target_dot.set_data([target[k, 0]], [target[k, 1]]); target_dot.set_3d_properties([target[k, 2]])
        if sphere_holder["surf"] is not None:
            sphere_holder["surf"].remove()
        col = "green" if (caught and k == T - 1) else "red"
        sphere_holder["surf"] = ax.plot_surface(
            target[k, 0] + sx, target[k, 1] + sy, target[k, 2] + sz,
            color=col, alpha=0.15, linewidth=0,
        )
        ax.view_init(elev=22, azim=(-60 + frame_i * orbit_deg_per_frame))
        return drone_trail, target_trail, drone_dot, target_dot

    ani = animation.FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=False)

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = _ffmpeg_writer(fps)
    if writer is None:
        out = out.with_suffix(".gif")
        ani.save(str(out), writer=animation.PillowWriter(fps=fps))
    else:
        ani.save(str(out), writer=writer)
    plt.close(fig)
    print(f"Saved {out}")
    return str(out)


def animate_target_episode(
    drone: np.ndarray,
    target: np.ndarray,
    base: np.ndarray,
    reach_distance: float,
    dome_size: float,
    caught: bool,
    dt: float = 1.0 / 30.0,
    save_path: str | Path = "results/target_eval.mp4",
    fps: int = 30,
    max_frames: int = 300,
    title_suffix: str = "",
) -> str:
    """Render an episode as a video: top-down chase + distance-to-target.

    Left panel animates the drone and target moving with growing trails and the
    catch radius; right panel animates distance-to-target with a playhead and
    the catch threshold. Writes MP4 if ffmpeg is available, else falls back to
    an animated GIF (``.gif``).

    Args:
        drone: (T, 3) drone positions.
        target: (T, 3) target positions.
        base: (3,) base position.
        reach_distance: Catch radius in meters.
        dome_size: Flight dome size (boundary circle).
        caught: Whether the target was intercepted.
        dt: Seconds per step.
        save_path: Output path (``.mp4``; switched to ``.gif`` if no ffmpeg).
        fps: Frames per second.
        max_frames: Cap on rendered frames; the episode is strided to fit.
        title_suffix: Extra text appended to the title.

    Returns:
        The actual path written.
    """
    import matplotlib.animation as animation

    drone = np.asarray(drone)
    target = np.asarray(target)
    T = len(drone)
    stride = max(1, T // max_frames)
    idx = list(range(0, T, stride))
    if idx[-1] != T - 1:
        idx.append(T - 1)

    dist = np.linalg.norm(drone - target, axis=1)
    t = np.arange(T) * dt
    r = dome_size / 2.0
    status = "CAUGHT" if caught else "missed"

    fig, (axm, axd) = plt.subplots(1, 2, figsize=(14, 7))

    # Static top-down scaffolding.
    th = np.linspace(0, 2 * np.pi, 200)
    axm.plot(r * np.cos(th), r * np.sin(th), "k--", lw=1.0, alpha=0.5)
    axm.plot(base[0], base[1], "k*", ms=15)
    axm.plot(drone[0, 0], drone[0, 1], "bs", ms=8, alpha=0.6)
    axm.set_aspect("equal")
    axm.set_xlim(-r * 1.1, r * 1.1); axm.set_ylim(-r * 1.1, r * 1.1)
    axm.set_xlabel("X (m)"); axm.set_ylabel("Y (m)")
    axm.set_title(f"Top-down — {status} {title_suffix}".strip())
    axm.grid(True, alpha=0.3)

    drone_trail, = axm.plot([], [], "b-", lw=1.4, alpha=0.8, label="Drone")
    target_trail, = axm.plot([], [], "r-", lw=1.4, alpha=0.8, label="Target")
    drone_dot, = axm.plot([], [], "bo", ms=9)
    target_dot, = axm.plot([], [], "ro", ms=9)
    catch_ring = plt.Circle((0, 0), reach_distance, color="red", alpha=0.15)
    axm.add_patch(catch_ring)
    axm.legend(loc="upper right", fontsize=9)

    # Static distance panel.
    axd.plot(t, dist, color="0.8", lw=1.0)
    axd.axhline(reach_distance, color="red", ls="--", lw=1.0, label=f"Catch radius ({reach_distance} m)")
    axd.set_xlim(0, t[-1]); axd.set_ylim(0, dist.max() * 1.1)
    axd.set_xlabel("Time (s)"); axd.set_ylabel("Distance to target (m)")
    axd.set_title("Distance to target")
    axd.grid(True, alpha=0.3); axd.legend(fontsize=9)
    dist_line, = axd.plot([], [], "b-", lw=1.6)
    playhead, = axd.plot([], [], "bo", ms=7)

    def update(frame_i):
        k = idx[frame_i]
        drone_trail.set_data(drone[:k + 1, 0], drone[:k + 1, 1])
        target_trail.set_data(target[:k + 1, 0], target[:k + 1, 1])
        drone_dot.set_data([drone[k, 0]], [drone[k, 1]])
        target_dot.set_data([target[k, 0]], [target[k, 1]])
        catch_ring.center = (target[k, 0], target[k, 1])
        dist_line.set_data(t[:k + 1], dist[:k + 1])
        playhead.set_data([t[k]], [dist[k]])
        # Turn the catch ring green at the end if caught.
        catch_ring.set_color("green" if (caught and k == T - 1) else "red")
        return drone_trail, target_trail, drone_dot, target_dot, catch_ring, dist_line, playhead

    ani = animation.FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=False)

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = _ffmpeg_writer(fps)
    if writer is None:
        out = out.with_suffix(".gif")
        ani.save(str(out), writer=animation.PillowWriter(fps=fps))
    else:
        ani.save(str(out), writer=writer)
    plt.close(fig)
    print(f"Saved {out}")
    return str(out)


def plot_target_episode(
    drone: np.ndarray,
    target: np.ndarray,
    base: np.ndarray,
    reach_distance: float,
    dome_size: float,
    caught: bool,
    dt: float = 1.0 / 30.0,
    save_prefix: str | Path = "results/target_eval",
    title_suffix: str = "",
) -> None:
    """Render top-down, 3D, and distance-vs-time plots for one episode.

    Args:
        drone: (T, 3) drone positions over the episode.
        target: (T, 3) target positions over the episode (same length as drone).
        base: (3,) drone start / base position.
        reach_distance: Catch radius in meters.
        dome_size: Flight dome size (for the boundary circle).
        caught: Whether the target was intercepted.
        dt: Seconds per step (for the time axis).
        save_prefix: Path prefix; files get ``_topdown.png``, ``_3d.png``,
            ``_distance.png`` suffixes.
        title_suffix: Extra text appended to plot titles.
    """
    drone = np.asarray(drone)
    target = np.asarray(target)
    prefix = Path(save_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    status = "CAUGHT" if caught else "missed"

    # -- Top-down -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(drone[:, 0], drone[:, 1], "b-", lw=1.2, alpha=0.8, label="Drone")
    ax.plot(target[:, 0], target[:, 1], "r-", lw=1.2, alpha=0.8, label="Target")
    ax.plot(drone[0, 0], drone[0, 1], "bs", ms=9, label="Drone start")
    ax.plot(target[0, 0], target[0, 1], "ro", ms=9, label="Target start")
    ax.plot(target[-1, 0], target[-1, 1], "r*", ms=15, label="Target end")
    # Catch radius around the final target position.
    ax.add_patch(plt.Circle((target[-1, 0], target[-1, 1]), reach_distance,
                            color="red", alpha=0.15))
    # Dome boundary.
    th = np.linspace(0, 2 * np.pi, 200)
    r = dome_size / 2.0
    ax.plot(r * np.cos(th), r * np.sin(th), "k--", lw=1.0, alpha=0.5, label="Dome edge")
    ax.plot(base[0], base[1], "k*", ms=15)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"Top-down — {status} {title_suffix}".strip())
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_topdown.png", dpi=150)
    plt.close(fig)
    print(f"Saved {prefix}_topdown.png")

    # -- 3D -------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(drone[:, 0], drone[:, 1], drone[:, 2], "b-", lw=1.2, alpha=0.8, label="Drone")
    ax.plot(target[:, 0], target[:, 1], target[:, 2], "r-", lw=1.2, alpha=0.8, label="Target")
    ax.scatter(*drone[0], c="blue", s=60, marker="s")
    ax.scatter(*target[-1], c="red", s=120, marker="*")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"3D path — {status} {title_suffix}".strip())
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{prefix}_3d.png", dpi=150)
    plt.close(fig)
    print(f"Saved {prefix}_3d.png")

    # -- Distance vs time -----------------------------------------------------
    dist = np.linalg.norm(drone - target, axis=1)
    t = np.arange(len(dist)) * dt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, dist, "b-", lw=1.2, label="Distance to target")
    ax.axhline(reach_distance, color="red", ls="--", lw=1.0, label=f"Catch radius ({reach_distance} m)")
    if caught:
        ax.plot(t[-1], dist[-1], "g*", ms=15, label="Catch")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (m)")
    ax.set_title(f"Distance to target — {status} {title_suffix}".strip())
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_distance.png", dpi=150)
    plt.close(fig)
    print(f"Saved {prefix}_distance.png")
