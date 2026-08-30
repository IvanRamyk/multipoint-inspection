#!/usr/bin/env python3
"""Does the behaviour metric set actually separate pursuit from luck?

Builds trajectories whose behaviour is known by construction and checks that the
metrics label them the way a human would. If a hovering drone that gets handed a
catch scores like a real pursuer, the metric is worthless and the gate is theatre
— so this is the check that matters most.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from eval.behaviour_metrics import aggregate, episode_metrics

DT = 1.0 / 30.0
REACH = 2.0
DOME = 16.0


def pursuer(steps=200):
    """Flies straight at a target moving on a steady heading. True pursuit.

    Note this is PURE pursuit, aiming at where the target is rather than leading
    it, so alignment sits near 0.85 not 1.0: chasing a mover carries an inherent
    lag angle. Real learned policies should look like this or better.
    """
    target = np.stack([
        np.linspace(6.0, 2.0, steps),
        np.linspace(-5.0, 5.0, steps),
        np.full(steps, 3.0),
    ], axis=1)
    drone = [np.array([-6.0, -6.0, 3.0])]
    speed = 3.0 * DT
    for t in range(1, steps):
        d = drone[-1]
        direction = target[t] - d
        n = np.linalg.norm(direction)
        drone.append(d if n < 1e-9 else d + direction / n * speed)
    return np.array(drone), target


def hoverer(steps=200):
    """Holds position while the target's route happens to pass through it.

    This is the policy the user described: numerically it catches the target, but
    nobody watching would call it pursuit. The jitter is deliberate — a real
    quadcopter never has zero velocity, so a metric keyed on instantaneous speed
    would read this as purposeful motion.
    """
    drone = np.tile(np.array([0.0, 0.0, 3.0]), (steps, 1))
    drone = drone + np.random.default_rng(0).normal(0, 0.02, drone.shape)
    target = np.stack([
        np.linspace(7.0, -7.0, steps),
        np.linspace(0.3, -0.3, steps),
        np.full(steps, 3.0),
    ], axis=1)
    return drone, target


def wanderer(steps=200):
    """Flies energetically in loops and happens to end up near the target."""
    rng = np.random.default_rng(1)
    angles = np.cumsum(rng.normal(0, 0.9, steps))
    step = 3.0 * DT
    drone = [np.array([0.0, 0.0, 3.0])]
    for t in range(1, steps):
        d = drone[-1]
        drone.append(d + np.array([np.cos(angles[t]), np.sin(angles[t]), 0.0]) * step)
    drone = np.array(drone)
    target = np.tile(drone[-1] + np.array([0.5, 0.5, 0.0]), (steps, 1))
    return drone, target


def spawn_freebie(steps=15):
    """Target spawns inside the catch radius. Caught in a handful of steps."""
    target = np.tile(np.array([1.0, 0.0, 3.0]), (steps, 1))
    drone = np.stack([
        np.linspace(0.0, 0.9, steps),
        np.zeros(steps),
        np.full(steps, 3.0),
    ], axis=1)
    return drone, target


COLUMNS = [
    ("align", "mean_pursuit_alignment", "{:+7.2f}"),
    ("early", "early_pursuit_alignment", "{:+7.2f}"),
    ("travel", "travel_ratio", "{:7.2f}"),
    ("disp", "displacement_ratio", "{:6.2f}"),
    ("statn", "station_keeping_fraction", "{:6.2f}"),
    ("sep0", "initial_separation", "{:6.1f}"),
    ("steps", "steps", "{:6d}"),
]


def main():
    cases = [
        ("true pursuer", pursuer()),
        ("hoverer (lucky)", hoverer()),
        ("wanderer (lucky)", wanderer()),
        ("spawn freebie", spawn_freebie()),
    ]

    header = f"{'case':18s}" + "".join(f"{name:>8s}" for name, _, _ in COLUMNS)
    print(header)
    rows = {}
    for label, (drone, target) in cases:
        m = episode_metrics(drone, target, caught=True, dt=DT,
                            reach_distance=REACH, dome_size=DOME)
        rows[label] = m
        line = f"{label:18s}"
        for _, key, fmt in COLUMNS:
            v = m.get(key)
            line += " " + ("     n/a" if v is None else fmt.format(v))
        print(line)

    pursue = rows["true pursuer"]
    hover = rows["hoverer (lucky)"]
    wander = rows["wanderer (lucky)"]
    freebie = rows["spawn freebie"]

    checks = [
        ("pursuer alignment is high",
         pursue["mean_pursuit_alignment"] > 0.8),
        ("pursuer actually travelled the gap",
         pursue["travel_ratio"] > 0.8),
        # Pursuit of a MOVING target curves by necessity, so a legitimate chase
        # sits well below 1 here. The metric earns its keep against looping.
        ("pursuer path is reasonably direct for a moving target",
         pursue["displacement_ratio"] > 0.55),

        ("hoverer alignment is near zero",
         abs(hover["mean_pursuit_alignment"] or 0.0) < 0.3),
        ("hoverer is caught by station keeping",
         hover["station_keeping_fraction"] > 0.9),
        ("hoverer barely travelled",
         hover["travel_ratio"] < 0.3),

        ("wanderer alignment far below pursuer's",
         wander["mean_pursuit_alignment"] < pursue["mean_pursuit_alignment"] - 0.4),
        ("wanderer path is not direct",
         wander["displacement_ratio"] < 0.5),
        ("wanderer is NOT mistaken for hovering",
         wander["station_keeping_fraction"] < 0.3),

        ("freebie starts inside a trivial distance",
         freebie["initial_separation"] < 2.0),
        ("freebie is caught in very few steps",
         freebie["steps"] < 30),
    ]

    print()
    failures = 0
    for label, ok in checks:
        if not ok:
            failures += 1
        print(f"{'ok  ' if ok else 'FAIL'} {label}")

    agg = aggregate(list(rows.values()))
    print(f"\naggregate: alignment={agg['mean_pursuit_alignment']} "
          f"station={agg['station_keeping_fraction']} "
          f"fastest_catch={agg['fastest_catch_steps']}")
    print(f"\n{len(checks)} checks, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
