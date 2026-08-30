"""Quantify whether a policy's behaviour looks like the task it was given.

A success rate says the target was reached. It does not say the policy pursued
anything: a drone that hovers until a target wanders into its catch radius scores
the same as one that flies it down. These metrics separate the two, so the gate
can reject a number that came from the wrong behaviour.

The two failure modes they are built to catch:

**"The target is caught too easily."** ``initial_separation``, ``steps_to_catch``,
and ``target_path_length`` say whether there was a chase to win at all. A catch at
step 20 from 3 m away, against a target that barely moved, is a spawn artefact.

**"It doesn't look like pursuit."** ``mean_pursuit_alignment`` is the load-bearing
one: the cosine between the drone's velocity and its bearing to the target,
averaged over the episode. A genuine pursuer holds this near 1. A wanderer that
collides with the target by luck averages near 0, however good its success rate.
``path_efficiency``, ``closing_fraction``, ``heading_reversals`` and
``idle_fraction`` describe the same thing from other angles.

Everything is computed from recorded trajectories, so it is deterministic and
free — no rendering, no vision, no extra simulation. That is what makes these
numbers safe to precommit as acceptance criteria in a campaign spec.
"""

from __future__ import annotations

import numpy as np

# Below this speed the drone is treated as not flying anywhere, so its velocity
# direction is noise and must not be fed into an alignment average. Set well above
# station-keeping jitter: a hovering quadcopter oscillates continuously, and a
# threshold near zero reads that noise as purposeful motion.
_MOVING_SPEED_EPS = 0.25
# Bearing is undefined once the drone is essentially on top of the target.
_BEARING_EPS = 1e-6
# Window and radius for station-keeping: a drone that has not left this radius
# after this many steps was holding position, whatever its instantaneous speed.
_STATION_WINDOW = 15
_STATION_RADIUS = 0.5


def _unit(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise unit vectors and their original norms."""
    norms = np.linalg.norm(vectors, axis=1)
    safe = np.where(norms > _BEARING_EPS, norms, 1.0)
    return vectors / safe[:, None], norms


def episode_metrics(
    drone: np.ndarray,
    target: np.ndarray,
    *,
    caught: bool,
    dt: float,
    reach_distance: float,
    dome_size: float | None = None,
    velocities: np.ndarray | None = None,
) -> dict:
    """Behaviour metrics for one episode.

    Args:
        drone: (T, 3) drone positions.
        target: (T, 3) target positions, same length.
        caught: Whether the episode ended in an interception.
        dt: Seconds per step (1 / agent_hz).
        reach_distance: Catch radius, for context on the separation numbers.
        dome_size: Spawn dome size, used for the out-of-dome fraction.
        velocities: (T, 3) recorded drone velocities. When absent, velocity is
            finite-differenced from positions, which is noisier but workable.

    Returns:
        A flat dict of floats and ints. Missing inputs yield None for the
        metrics that need them rather than a silent zero.
    """
    drone = np.asarray(drone, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    steps = len(drone)
    if steps < 2 or len(target) < 2:
        return {"steps": steps, "insufficient_data": True}

    n = min(len(drone), len(target))
    drone, target = drone[:n], target[:n]

    separation = np.linalg.norm(drone - target, axis=1)
    to_target = target - drone
    bearing, _ = _unit(to_target)

    if velocities is not None and len(velocities) >= n:
        vel = np.asarray(velocities, dtype=np.float64)[:n]
    else:
        # Finite difference; repeat the last row so lengths line up.
        vel = np.vstack([np.diff(drone, axis=0) / dt, (drone[-1] - drone[-2])[None, :] / dt])
    vel_dir, speed = _unit(vel)

    # Pursuit alignment: only over steps where the drone was actually moving,
    # because the direction of a near-zero velocity is meaningless.
    moving = speed > _MOVING_SPEED_EPS
    alignment = np.sum(vel_dir * bearing, axis=1)
    aligned = alignment[moving]

    drone_path = float(np.sum(np.linalg.norm(np.diff(drone, axis=0), axis=1)))
    target_path = float(np.sum(np.linalg.norm(np.diff(target, axis=0), axis=1)))
    straight_line = float(np.linalg.norm(target[-1] - drone[0]))
    # Ratio of straight-line distance to distance actually flown. Below 1 means
    # wasted travel. NOTE a value ABOVE 1 is not excellence — it means the drone
    # flew less than the straight-line distance to where the target ended, which
    # happens when the target came to the drone. Read it with travel_ratio.
    efficiency = straight_line / drone_path if drone_path > _BEARING_EPS else None
    net_displacement = float(np.linalg.norm(drone[-1] - drone[0]))
    # How far the drone actually GOT, relative to the gap it had to close. Keyed on
    # net displacement rather than path length on purpose: path length accumulates
    # station-keeping jitter, so a hovering drone can rack up more "travel" than
    # the gap it never crossed. A hoverer sits near 0 however lucky it gets.
    travel_ratio = (
        net_displacement / separation[0] if separation[0] > _BEARING_EPS else None
    )
    # Net displacement against distance flown: 1 is a straight line, near 0 is
    # oscillating in place. Legitimate pursuit of a MOVING target curves, so expect
    # roughly 0.6-0.8 here rather than 1 — its discriminating power is against
    # looping and circling, which land near 0.2.
    displacement_ratio = (
        net_displacement / drone_path if drone_path > _BEARING_EPS else None
    )

    # Station keeping: fraction of steps after which the drone had still not left a
    # small radius _STATION_WINDOW steps later. This is the honest "was it hovering"
    # measure, since a quadcopter holding position never has zero speed.
    if n > _STATION_WINDOW:
        moved_away = np.linalg.norm(drone[_STATION_WINDOW:] - drone[:-_STATION_WINDOW], axis=1)
        station_keeping = float(np.mean(moved_away < _STATION_RADIUS))
    else:
        station_keeping = None

    d_sep = np.diff(separation)
    closing_fraction = float(np.mean(d_sep < 0.0))

    # A reversal is a step where the direction of travel flips by more than 90
    # degrees. Smooth pursuit has almost none; oscillation has many.
    reversals = 0
    if moving.sum() >= 2:
        dirs = vel_dir[moving]
        dots = np.sum(dirs[:-1] * dirs[1:], axis=1)
        reversals = int(np.sum(dots < 0.0))

    metrics: dict = {
        "steps": int(n),
        "caught": bool(caught),
        "initial_separation": round(float(separation[0]), 4),
        "final_separation": round(float(separation[-1]), 4),
        "min_separation": round(float(separation.min()), 4),
        "reach_distance": round(float(reach_distance), 4),
        # How much of the initial gap the policy actually closed. Near 0 means
        # the catch was handed to it by the spawn.
        "separation_closed_frac": round(
            float((separation[0] - separation.min()) / separation[0]) if separation[0] > _BEARING_EPS else 0.0, 4
        ),
        "steps_to_catch": int(n) if caught else None,
        "time_to_catch_s": round(float(n * dt), 3) if caught else None,
        "drone_path_length": round(drone_path, 4),
        "target_path_length": round(target_path, 4),
        "straight_line_distance": round(straight_line, 4),
        "net_displacement": round(net_displacement, 4),
        "path_efficiency": None if efficiency is None else round(float(efficiency), 4),
        "travel_ratio": None if travel_ratio is None else round(float(travel_ratio), 4),
        "displacement_ratio": None if displacement_ratio is None else round(float(displacement_ratio), 4),
        "mean_speed": round(float(np.mean(speed)), 4),
        "max_speed": round(float(np.max(speed)), 4),
        "idle_fraction": round(float(np.mean(~moving)), 4),
        "station_keeping_fraction": None if station_keeping is None else round(station_keeping, 4),
        "mean_pursuit_alignment": None if aligned.size == 0 else round(float(np.mean(aligned)), 4),
        "closing_fraction": round(closing_fraction, 4),
        "heading_reversals": reversals,
        "heading_reversals_per_100_steps": round(100.0 * reversals / n, 3),
        "altitude_mean": round(float(np.mean(drone[:, 2])), 4),
        "altitude_std": round(float(np.std(drone[:, 2])), 4),
    }

    # Alignment early in the episode separates "flew at it from the start" from
    # "eventually stumbled into it".
    first_third = max(2, n // 3)
    early = alignment[:first_third][moving[:first_third]]
    metrics["early_pursuit_alignment"] = (
        None if early.size == 0 else round(float(np.mean(early)), 4)
    )

    if dome_size:
        radius = float(dome_size) / 2.0
        outside = np.linalg.norm(drone[:, :2], axis=1) > radius
        metrics["outside_dome_fraction"] = round(float(np.mean(outside)), 4)

    return metrics


def aggregate(per_episode: list[dict]) -> dict:
    """Mean of each numeric metric across episodes, plus catch-only variants.

    Catch-only means matter because interception timing is only defined for
    episodes that intercepted; mixing misses in would wash the signal out.
    """
    usable = [m for m in per_episode if not m.get("insufficient_data")]
    if not usable:
        return {"episodes": 0, "insufficient_data": True}

    numeric_keys = [
        "initial_separation", "min_separation", "separation_closed_frac",
        "drone_path_length", "target_path_length", "path_efficiency",
        "travel_ratio", "displacement_ratio",
        "mean_speed", "idle_fraction", "station_keeping_fraction",
        "mean_pursuit_alignment", "early_pursuit_alignment", "closing_fraction",
        "heading_reversals_per_100_steps", "altitude_std", "outside_dome_fraction",
    ]

    def mean_of(key: str, rows: list[dict]) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    out: dict = {"episodes": len(usable)}
    for key in numeric_keys:
        out[key] = mean_of(key, usable)

    catches = [m for m in usable if m.get("caught")]
    out["catches"] = len(catches)
    out["steps_to_catch_mean"] = mean_of("steps_to_catch", catches)
    out["time_to_catch_s_mean"] = mean_of("time_to_catch_s", catches)
    out["initial_separation_on_catch"] = mean_of("initial_separation", catches)
    out["pursuit_alignment_on_catch"] = mean_of("mean_pursuit_alignment", catches)
    # The fastest catch in the set — a single spawn-adjacent freebie is easier to
    # see here than in a mean.
    catch_steps = [m["steps_to_catch"] for m in catches if m.get("steps_to_catch")]
    out["fastest_catch_steps"] = int(min(catch_steps)) if catch_steps else None
    return out
