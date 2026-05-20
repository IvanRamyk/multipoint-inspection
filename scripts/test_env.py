"""Test script: run random agent episodes and visualize results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.config import EnvConfig
from envs.drone_inspection_env import DroneInspectionEnv
from eval.route_visualizer import plot_route_3d, plot_route_topdown


def run_episode(env: DroneInspectionEnv, seed: int | None = None) -> dict:
    """Run a single random-agent episode.

    Args:
        env: The drone inspection environment.
        seed: Optional seed for the episode reset.

    Returns:
        Dict with episode statistics.
    """
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    return {
        "reward": total_reward,
        "steps": steps,
        "waypoints_visited": info["waypoints_visited"],
        "total_waypoints": info["total_waypoints"],
        "success": terminated and info["waypoints_visited"] == info["total_waypoints"],
        "terminated": terminated,
        "truncated": truncated,
    }


def main() -> None:
    """Run test episodes with a random agent."""
    parser = argparse.ArgumentParser(description="Test drone inspection environment")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render simulation")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    # Load config.
    if args.config:
        config = EnvConfig.from_yaml(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = EnvConfig()
        print("Using default config")

    print(f"  Waypoints: {config.num_waypoints}")
    print(f"  Obstacles: {config.num_obstacles}")
    print(f"  Wind: {'ON' if config.wind_enabled else 'OFF'}")
    print(f"  Max steps: {config.max_episode_steps}")
    print()

    # Create environment.
    render_mode = "human" if args.render else None
    env = DroneInspectionEnv(config=config, render_mode=render_mode)

    # Run check_env.
    print("Running gymnasium check_env...")
    from gymnasium.utils.env_checker import check_env
    check_env(env.unwrapped, skip_render_check=True)
    print("ENV CHECK PASSED\n")

    # Run episodes.
    results = []
    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        result = run_episode(env, seed=ep_seed)
        results.append(result)
        status = "SUCCESS" if result["success"] else ("COLLISION" if result["terminated"] else "TIMEOUT")
        print(
            f"Episode {ep + 1}: reward={result['reward']:7.2f}  "
            f"steps={result['steps']:4d}  "
            f"WP={result['waypoints_visited']}/{result['total_waypoints']}  "
            f"{status}"
        )

    # Summary.
    rewards = [r["reward"] for r in results]
    successes = [r["success"] for r in results]
    print(f"\n--- Summary ({args.episodes} episodes) ---")
    print(f"Mean reward:  {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Success rate: {np.mean(successes) * 100:.1f}%")

    # Save route plot of last episode.
    if env.positions and env.waypoints is not None:
        positions = np.array(env.positions)
        base = np.array(config.base_position)
        save_dir = Path("results")

        plot_route_topdown(
            positions, env.waypoints, env.visited, base,
            save_path=save_dir / "test_route_topdown.png",
        )
        plot_route_3d(
            positions, env.waypoints, env.visited, base,
            save_path=save_dir / "test_route_3d.png",
        )

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
