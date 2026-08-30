#!/usr/bin/env python3
"""Run and visualize an episode of the moving-target interception task.

Policies (no trained checkpoint needed):
  * ``pursuit`` — proportional pursuit: fly straight at the observed target.
    Catches the Dubins target and validates the env / reward / catch logic.
  * ``random``  — random actions, as a baseline for contrast.

Usage:
    ./venv/bin/python scripts/eval_target.py --config configs/target/l2_moving.yaml --policy pursuit
    ./venv/bin/python scripts/eval_target.py --config configs/target/l3_aerial.yaml --policy pursuit --episodes 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.core.config import EnvConfig
from envs.tasks.drone_target_env import DroneTargetEnv
from eval.target_visualizer import (
    animate_target_episode,
    animate_target_episode_3d,
    plot_target_episode,
)

# State layout: [pos(3), vel(3), wind(3), target_rel(3), target_vel(3)].
_TARGET_REL = slice(9, 12)
_TARGET_VEL = slice(12, 15)


def pursuit_action(obs: dict, gain: float = 0.4, clip: float = 0.5, lead: float = 1.0) -> np.ndarray:
    """Proportional lead-pursuit toward the target.

    Commands a velocity proportional to the target-relative position (plus a
    little lead on the target's velocity so we aim ahead of it), clipped to a
    moderate magnitude. The clip matters: commanding near-max velocity in all
    axes at once destabilises PyFlyt's velocity controller, so we stay in the
    well-behaved regime (~2.5 m/s), which is still faster than the target.
    """
    state = obs["state"]
    rel = state[_TARGET_REL].astype(np.float64)
    vel = state[_TARGET_VEL].astype(np.float64)
    aim = rel + lead * vel
    return np.clip(gain * aim, -clip, clip).astype(np.float32)


def run_episode(env: DroneTargetEnv, policy: str, seed: int) -> dict:
    """Run one episode; return trajectories and summary stats."""
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        if policy == "pursuit":
            action = pursuit_action(obs)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    drone = np.array(env.positions)
    target = np.array(env.target_positions)
    min_dist = float(np.min(np.linalg.norm(drone - target, axis=1))) if len(drone) else float("nan")
    return {
        "drone": drone,
        "target": target,
        "reward": total_reward,
        "steps": len(drone),
        "caught": bool(info["is_success"]),
        "min_dist": min_dist,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run + visualize a moving-target episode")
    parser.add_argument("--config", type=str, default="configs/target/l2_moving.yaml")
    parser.add_argument("--policy", choices=["pursuit", "random"], default="pursuit")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/target_eval")
    parser.add_argument("--video", action="store_true", help="Render a 2D (top-down + distance) MP4 per episode")
    parser.add_argument("--video3d", action="store_true", help="Render a 3D orbiting MP4 per episode")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    args = parser.parse_args()

    config = EnvConfig.from_yaml(args.config)
    print(f"Config: {args.config}  | target_mode={config.target_mode} "
          f"speed={config.target_speed} reach={config.target_reach_distance} dome={config.flight_dome_size}")
    print(f"Policy: {args.policy}\n")

    env = DroneTargetEnv(config=config)
    results = []
    for ep in range(args.episodes):
        r = run_episode(env, args.policy, seed=args.seed + ep)
        results.append(r)
        status = "CAUGHT" if r["caught"] else "missed"
        print(f"Episode {ep + 1}: reward={r['reward']:7.2f}  steps={r['steps']:4d}  "
              f"min_dist={r['min_dist']:.2f}m  {status}")
        suffix = f"({args.policy}, {Path(args.config).stem}, ep{ep + 1})"
        prefix = args.out if args.episodes == 1 else f"{args.out}_ep{ep + 1}"
        plot_target_episode(
            drone=r["drone"], target=r["target"],
            base=np.array(config.base_position),
            reach_distance=config.target_reach_distance,
            dome_size=config.flight_dome_size,
            caught=r["caught"], dt=1.0 / config.agent_hz,
            save_prefix=prefix, title_suffix=suffix,
        )
        if args.video:
            animate_target_episode(
                drone=r["drone"], target=r["target"],
                base=np.array(config.base_position),
                reach_distance=config.target_reach_distance,
                dome_size=config.flight_dome_size,
                caught=r["caught"], dt=1.0 / config.agent_hz,
                save_path=f"{prefix}.mp4", fps=args.fps, title_suffix=suffix,
            )
        if args.video3d:
            animate_target_episode_3d(
                drone=r["drone"], target=r["target"],
                base=np.array(config.base_position),
                reach_distance=config.target_reach_distance,
                dome_size=config.flight_dome_size,
                caught=r["caught"], dt=1.0 / config.agent_hz,
                save_path=f"{prefix}_3d.mp4", fps=args.fps, title_suffix=suffix,
            )
    env.close()

    n = len(results)
    print(f"\n--- Summary ({n} episode{'s' if n != 1 else ''}) ---")
    print(f"Success rate: {np.mean([r['caught'] for r in results]) * 100:.0f}%")
    print(f"Mean reward:  {np.mean([r['reward'] for r in results]):.2f}")
    print(f"Mean steps:   {np.mean([r['steps'] for r in results]):.0f}")


if __name__ == "__main__":
    main()
