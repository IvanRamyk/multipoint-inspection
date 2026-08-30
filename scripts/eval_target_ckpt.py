#!/usr/bin/env python3
"""Evaluate a TRAINED DreamerV3 checkpoint on the moving-target task.

Loads a sheeprl checkpoint (world model + actor), runs greedy episodes of
DroneTargetEnv, and renders plots / videos of the trained agent. This is the
decoupled "generate videos on demand" tool: training just checkpoints
periodically, and you call this on any checkpoint (or several, to see
progression) without recording anything during training.

Usage:
    ./venv/bin/python scripts/eval_target_ckpt.py <ckpt.ckpt> \
        --config configs/target/l2_moving.yaml --episodes 3 --video3d

    # progression: eval several checkpoints
    for c in .../ckpt_2000_0.ckpt .../ckpt_4000_0.ckpt; do
      ./venv/bin/python scripts/eval_target_ckpt.py "$c" --config configs/target/l2_moving.yaml --video3d
    done
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from lightning import Fabric
from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import envs  # noqa: F401 — triggers gymnasium registration
from envs.core.config import EnvConfig
from envs.tasks.drone_target_env import DroneTargetEnv
from envs.tasks.drone_chase_env import DroneChaseEnv
from envs.sheeprl_wrapper import SheepRLCompatWrapper
from eval.behaviour_metrics import aggregate as aggregate_behaviour
from eval.behaviour_metrics import episode_metrics
from eval.target_visualizer import animate_target_episode, animate_target_episode_3d, plot_target_episode

# Task registry: name -> (env class, default config).
_TASKS = {
    "target": (DroneTargetEnv, "configs/target/l2_moving.yaml"),
    "chase": (DroneChaseEnv, "configs/target/chase.yaml"),
}


def _sha256(path: Path) -> str:
    """Digest a file in chunks — checkpoints are ~105 MB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval a trained DreamerV3 checkpoint on DroneTargetEnv")
    parser.add_argument("checkpoint", type=str, help="Path to ckpt_*.ckpt")
    parser.add_argument("--task", choices=list(_TASKS), default="target",
                        help="Which drone task the checkpoint was trained on")
    parser.add_argument("--config", type=str, default=None,
                        help="Env config YAML (defaults to the task's default config)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of greedy argmax")
    parser.add_argument("--video", action="store_true", help="Render 2D (top-down + distance) MP4 per episode")
    parser.add_argument("--video3d", action="store_true", help="Render orbiting 3D MP4 per episode")
    parser.add_argument("--catches-only", action="store_true", help="Only render plots/videos for episodes that caught the target")
    parser.add_argument("--out", type=str, default="results/ckpt_eval",
                        help="Output prefix; the checkpoint step is appended")
    parser.add_argument("--json", type=str, default=None,
                        help="Write per-episode results and the aggregate success rate to this "
                             "JSON path. This is the machine-readable acceptance-gate output.")
    parser.add_argument("--trajectories", type=str, default=None,
                        help="Directory to save per-episode trajectory .npz files. Needed to "
                             "build a contact sheet later without re-running the policy.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip all plotting. Use for automated evaluation where only the "
                             "numbers matter — matplotlib rendering dominates the runtime.")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    step_tag = ckpt_path.stem  # e.g. ckpt_4000_0

    run_dir = ckpt_path.parent.parent
    train_config_path = run_dir / "config.yaml"
    if not train_config_path.exists():
        print(f"Training config not found at {train_config_path}")
        sys.exit(1)
    cfg = OmegaConf.load(train_config_path)

    fabric = Fabric(devices=1, accelerator="cpu", num_nodes=1)
    state = fabric.load(str(ckpt_path))
    print(f"Loaded checkpoint {ckpt_path} (step tag {step_tag})")

    env_cls, default_cfg = _TASKS[args.task]
    env_config = EnvConfig.from_yaml(args.config or default_cfg)
    raw_env = env_cls(config=env_config)
    env = SheepRLCompatWrapper(raw_env)

    import gymnasium as gym
    from sheeprl.algos.dreamer_v3.agent import build_agent

    screen_size = cfg.env.screen_size
    cnn_channels = 1 if cfg.env.grayscale else 3
    obs_space = gym.spaces.Dict({
        "depth": gym.spaces.Box(0, 255, (cnn_channels, screen_size, screen_size), dtype=np.uint8),
        "state": env.observation_space["state"],
    })
    actions_dim = [env.action_space.shape[0]]

    world_model, _, _, _, player = build_agent(
        fabric, actions_dim, True, cfg, obs_space,
        world_model_state=state["world_model"], actor_state=state["actor"],
    )
    player.num_envs = 1
    player.eval()
    world_model.eval()

    cnn_keys = list(cfg.algo.cnn_keys.encoder)

    def obs_to_torch(obs: dict) -> dict:
        out = {}
        for k, v in obs.items():
            t = torch.from_numpy(v.copy()).to(fabric.device).float()
            if k in cnn_keys:
                t = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.0 - 0.5
            else:
                t = t.view(1, 1, -1)
            out[k] = t
        return out

    results = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        player.init_states()
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                real_actions = player.get_actions(obs_to_torch(obs), greedy=not args.stochastic)
            action = torch.stack(real_actions, -1).cpu().numpy().reshape(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        drone = np.array(raw_env.positions)
        target = np.array(raw_env.target_positions)
        caught = bool(info["is_success"])
        min_dist = float(np.min(np.linalg.norm(drone - target, axis=1)))
        behaviour = episode_metrics(
            drone, target, caught=caught, dt=1.0 / env_config.agent_hz,
            reach_distance=env_config.target_reach_distance,
            dome_size=env_config.flight_dome_size,
            velocities=np.array(raw_env.velocities) if getattr(raw_env, "velocities", None) else None,
        )
        results.append({
            "episode": ep + 1,
            "seed": args.seed + ep,
            "caught": caught,
            "steps": len(drone),
            "min_dist": min_dist,
            "reward": float(total_reward),
            "behaviour": behaviour,
        })
        if args.trajectories:
            traj_dir = Path(args.trajectories)
            traj_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                traj_dir / f"{step_tag}_ep{ep + 1}.npz",
                drone=drone, target=target, caught=caught,
                reach_distance=env_config.target_reach_distance,
                dome_size=env_config.flight_dome_size,
                dt=1.0 / env_config.agent_hz,
            )
        status = "CAUGHT" if caught else "missed"
        print(f"Episode {ep + 1}: reward={total_reward:7.2f}  steps={len(drone):4d}  "
              f"min_dist={min_dist:.2f}m  {status}")

        if args.no_plots:
            continue
        if args.catches_only and not caught:
            continue  # skip rendering for missed episodes

        suffix = f"({step_tag}, ep{ep + 1})"
        prefix = f"{args.out}_{step_tag}_ep{ep + 1}"
        plot_target_episode(
            drone=drone, target=target, base=np.array(env_config.base_position),
            reach_distance=env_config.target_reach_distance, dome_size=env_config.flight_dome_size,
            caught=caught, dt=1.0 / env_config.agent_hz, save_prefix=prefix, title_suffix=suffix,
        )
        if args.video:
            animate_target_episode(
                drone=drone, target=target, base=np.array(env_config.base_position),
                reach_distance=env_config.target_reach_distance, dome_size=env_config.flight_dome_size,
                caught=caught, dt=1.0 / env_config.agent_hz, save_path=f"{prefix}.mp4", title_suffix=suffix,
            )
        if args.video3d:
            animate_target_episode_3d(
                drone=drone, target=target, base=np.array(env_config.base_position),
                reach_distance=env_config.target_reach_distance, dome_size=env_config.flight_dome_size,
                caught=caught, dt=1.0 / env_config.agent_hz, save_path=f"{prefix}_3d.mp4", title_suffix=suffix,
            )

    env.close()

    caught_flags = [r["caught"] for r in results]
    success_rate = float(np.mean(caught_flags))
    print(f"\nSuccess rate: {int(100 * success_rate)}%  "
          f"({sum(caught_flags)}/{len(caught_flags)}) — {step_tag}")

    if args.json:
        payload = {
            "checkpoint": str(ckpt_path),
            "ckpt_sha256": _sha256(ckpt_path),
            "step_tag": step_tag,
            "task": args.task,
            "env_config": args.config or default_cfg,
            "env_config_sha256": _sha256(Path(args.config or default_cfg)),
            "episodes": len(results),
            "eval_seed_base": args.seed,
            "greedy": not args.stochastic,
            "success_rate": success_rate,
            "successes": int(sum(caught_flags)),
            "mean_reward": float(np.mean([r["reward"] for r in results])),
            "mean_steps": float(np.mean([r["steps"] for r in results])),
            "mean_min_dist": float(np.mean([r["min_dist"] for r in results])),
            # Aggregated behaviour, so a gate can ask whether this success rate
            # came from pursuit or from luck without reading every episode.
            "behaviour": aggregate_behaviour([r["behaviour"] for r in results]),
            "results": results,
        }
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
