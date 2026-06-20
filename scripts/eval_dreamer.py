#!/usr/bin/env python3
"""Evaluate a trained DreamerV3 agent with PyBullet GUI visualization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from lightning import Fabric
from omegaconf import OmegaConf

# Video recording parameters
_VIDEO_W = 640
_VIDEO_H = 480
_VIDEO_FPS = 30

# Ensure project root is on the path.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import envs  # noqa: F401 — triggers gymnasium registration
from envs.config import EnvConfig
from envs.drone_inspection_env import DroneInspectionEnv
from envs.sheeprl_wrapper import SheepRLCompatWrapper
from eval.route_visualizer import plot_route_3d, plot_route_topdown


def main() -> None:
    """Load checkpoint and run agent with GUI."""
    parser = argparse.ArgumentParser(description="Evaluate trained DreamerV3 agent")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint .ckpt file",
    )
    parser.add_argument("--config", type=str, default=None, help="Env config YAML (overrides training config)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Open PyBullet GUI")
    parser.add_argument("--greedy", action="store_true", help="Use greedy (deterministic) actions")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--record", action="store_true", help="Record MP4 video of each episode (no GUI needed)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load training config saved alongside checkpoint.
    run_dir = ckpt_path.parent.parent  # version_0/checkpoint/ckpt.ckpt -> version_0/
    train_config_path = run_dir / "config.yaml"
    if not train_config_path.exists():
        print(f"Training config not found at {train_config_path}")
        sys.exit(1)

    cfg = OmegaConf.load(train_config_path)
    print(f"Loaded training config from {train_config_path}")

    # Create fabric and load checkpoint.
    fabric = Fabric(devices=1, accelerator="cpu", num_nodes=1)
    state = fabric.load(str(ckpt_path))
    print(f"Loaded checkpoint from {ckpt_path}")

    # Create environment (raw, for GUI rendering).
    if args.config:
        env_config = EnvConfig.from_yaml(args.config)
    else:
        env_config = EnvConfig.from_yaml("configs/easy.yaml")

    if args.record:
        try:
            import imageio
        except ImportError:
            print("ERROR: imageio is required for --record. Install with:")
            print("  pip install imageio[ffmpeg]")
            sys.exit(1)

    render_mode = "human" if args.render else None
    raw_env = DroneInspectionEnv(config=env_config, render_mode=render_mode)
    env = SheepRLCompatWrapper(raw_env)

    # Build agent from checkpoint.
    # The obs_space must match what sheeprl saw during training:
    # CNN keys are channel-first (C, H, W) uint8, after sheeprl's make_env transform.
    import gymnasium as gym
    from sheeprl.algos.dreamer_v3.agent import build_agent

    screen_size = cfg.env.screen_size
    is_grayscale = cfg.env.grayscale
    cnn_channels = 1 if is_grayscale else 3

    train_obs_space = gym.spaces.Dict({
        "depth": gym.spaces.Box(0, 255, (cnn_channels, screen_size, screen_size), dtype=np.uint8),
        "state": env.observation_space["state"],
    })

    actions_dim = [env.action_space.shape[0]]
    is_continuous = True
    obs_space = train_obs_space

    world_model, _, _, _, player = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        obs_space,
        world_model_state=state["world_model"],
        actor_state=state["actor"],
    )

    player.num_envs = 1
    player.eval()
    world_model.eval()

    # Run episodes.
    cnn_keys = list(cfg.algo.cnn_keys.encoder)

    def obs_to_torch(obs: dict) -> dict:
        """Convert env observation to tensors matching training format."""
        torch_obs = {}
        for k, v in obs.items():
            t = torch.from_numpy(v.copy()).to(fabric.device).float()
            if k in cnn_keys:
                # Convert (H, W, C) -> (1, 1, C, H, W) and normalize to [-0.5, 0.5].
                t = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.0 - 0.5
            else:
                t = t.view(1, 1, -1)
            torch_obs[k] = t
        return torch_obs

    save_dir = Path("results")
    base = np.array(env_config.base_position)
    reach = env_config.waypoint_reach_distance

    def capture_frame(backend) -> np.ndarray | None:
        """Capture an RGB frame from the active backend. Returns (H, W, 3) uint8."""
        # AirSim: use the photorealistic chase camera defined in settings.json
        if hasattr(backend, "get_video_frame"):
            return backend.get_video_frame(width=_VIDEO_W, height=_VIDEO_H)
        # PyBullet: render from a chase camera using TinyRenderer
        av = getattr(backend, "_aviary", None)
        if av is None:
            return None
        state = av.state(0)
        pos = state[3]
        eye = pos + np.array([0.0, -6.0, 4.0])
        proj = av.computeProjectionMatrixFOV(
            fov=60.0, aspect=_VIDEO_W / _VIDEO_H, nearVal=0.1, farVal=200.0
        )
        view = av.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=pos.tolist(),
            cameraUpVector=[0.0, 0.0, 1.0],
        )
        _, _, rgba, _, _ = av.getCameraImage(
            width=_VIDEO_W, height=_VIDEO_H,
            viewMatrix=list(view), projectionMatrix=list(proj),
            renderer=av.ER_TINY_RENDERER,
        )
        rgb = np.array(rgba, dtype=np.uint8).reshape(_VIDEO_H, _VIDEO_W, 4)[:, :, :3]
        return rgb

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        player.init_states()

        done = False
        total_reward = 0.0
        steps = 0
        frames: list = []

        while not done:
            # Preprocess observation for the player.
            torch_obs = obs_to_torch(obs)

            # Get action.
            with torch.no_grad():
                real_actions = player.get_actions(torch_obs, greedy=args.greedy)

            # Convert to numpy.
            action = torch.stack(real_actions, -1).cpu().numpy()
            action = action.reshape(env.action_space.shape)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if args.record:
                frame = capture_frame(raw_env.backend)
                if frame is not None:
                    frames.append(frame)

        wp_visited = info["waypoints_visited"]
        total_wp = info["total_waypoints"]
        status = "SUCCESS" if (terminated and wp_visited == total_wp) else ("COLLISION" if terminated else "TIMEOUT")
        print(
            f"Episode {ep + 1}: reward={total_reward:7.2f}  "
            f"steps={steps:4d}  WP={wp_visited}/{total_wp}  {status}"
        )

        # Save route plot for this episode (snapshot before next reset clears).
        if raw_env.positions and raw_env.waypoints is not None:
            positions = np.array(raw_env.positions)
            tag = f"ep{ep + 1:02d}_{status.lower()}_r{total_reward:+07.2f}_s{steps:03d}"
            label = (
                f"Episode {ep + 1} — {status}  "
                f"reward={total_reward:+.2f}  steps={steps}  WP={wp_visited}/{total_wp}"
            )
            plot_route_topdown(
                positions, raw_env.waypoints, raw_env.visited, base,
                save_path=save_dir / f"eval_route_{tag}_topdown.png",
                reach_radius=reach,
                title=label,
            )
            plot_route_3d(
                positions, raw_env.waypoints, raw_env.visited, base,
                save_path=save_dir / f"eval_route_{tag}_3d.png",
                reach_radius=reach,
                title=label,
            )

        if args.record and frames:
            video_path = save_dir / f"eval_{tag}.mp4"
            imageio.mimwrite(str(video_path), frames, fps=_VIDEO_FPS, macro_block_size=None)
            print(f"  Video saved → {video_path}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
