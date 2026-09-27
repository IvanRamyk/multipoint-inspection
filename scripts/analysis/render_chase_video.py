#!/usr/bin/env python3
"""Render a real-drone-model MP4 of a chase episode (PyBullet TinyRenderer RGB).

Rolls out a trained DreamerV3 policy in the chase env and captures RGB frames of
the actual QuadX drone meshes from an orbiting camera — unlike the abstract
trajectory-line animations. One-off presentation tool (lives in gitignored
results/, not the checksummed scripts/).

Usage:
  venv/bin/python results/render_chase_video.py <ckpt> --config configs/target/chase_easy2.yaml \
      --seed 1003 --out results/R1_report/assets/seed2_real_model.mp4
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pybullet as pb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import imageio_ffmpeg
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
from omegaconf import OmegaConf
from lightning.fabric import Fabric

from envs.core.config import EnvConfig
from envs.tasks.drone_chase_env import DroneChaseEnv
from envs.sheeprl_wrapper import SheepRLCompatWrapper


def grab_rgb(av, center, yaw, W, H, dist):
    view = av.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[float(center[0]), float(center[1]), float(center[2])],
        distance=dist, yaw=yaw, pitch=-28.0, roll=0.0, upAxisIndex=2)
    proj = av.computeProjectionMatrixFOV(fov=55.0, aspect=W / H, nearVal=0.1, farVal=200.0)
    w, h, rgba, _, _ = av.getCameraImage(
        W, H, viewMatrix=view, projectionMatrix=proj,
        renderer=pb.ER_TINY_RENDERER, flags=pb.ER_NO_SEGMENTATION_MASK)
    rgba = np.reshape(np.array(rgba, dtype=np.uint8), (h, w, 4))
    return rgba[:, :, :3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scripted", action="store_true",
                    help="Use the scripted lead-pursuit controller instead of a trained policy "
                         "(no checkpoint needed) — for visualising the task before a policy exists.")
    ap.add_argument("--seed", type=int, default=1003)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--find-catch", type=int, default=1,
                    help="Try this many consecutive seeds (from --seed) until the rendered "
                         "rollout actually catches; keep that episode. 1 = force the exact seed.")
    args = ap.parse_args()

    env_config = EnvConfig.from_yaml(args.config)
    raw_env = DroneChaseEnv(config=env_config)
    env = SheepRLCompatWrapper(raw_env)

    player = None
    if args.scripted:
        sys.path.insert(0, str(REPO / "tools"))
        from run_baselines import pursuit_action
        policy_label = "scripted pursuit"
    else:
        ck = Path(args.checkpoint)
        cfg = OmegaConf.load(ck.parent.parent / "config.yaml")
        fabric = Fabric(devices=1, accelerator="cpu", num_nodes=1)
        state = fabric.load(str(ck))
        import gymnasium as gym
        from sheeprl.algos.dreamer_v3.agent import build_agent
        # Build obs_space EXACTLY like scripts/eval_target_ckpt.py: always include the
        # depth key. build_agent sizes the model from cfg.algo.cnn_keys (empty for a
        # state-only checkpoint), so the extra key is ignored — but removing it changes
        # how the model is built and degrades the policy. Keep it identical to eval.
        screen = cfg.env.screen_size
        ch = 1 if cfg.env.grayscale else 3
        obs_space = gym.spaces.Dict({
            "depth": gym.spaces.Box(0, 255, (ch, screen, screen), dtype=np.uint8),
            "state": env.observation_space["state"],
        })
        world_model, _, _, _, player = build_agent(
            fabric, [env.action_space.shape[0]], True, cfg, obs_space,
            world_model_state=state["world_model"], actor_state=state["actor"])
        player.num_envs = 1
        player.eval(); world_model.eval()
        cnn_keys = list(cfg.algo.cnn_keys.encoder)
        policy_label = "DreamerV3"

    def to_torch(obs):
        out = {}
        for k, v in obs.items():
            t = torch.from_numpy(v.copy()).float()
            if k in cnn_keys:
                t = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.0 - 0.5
            else:
                t = t.view(1, 1, -1)
            out[k] = t
        return out

    # The chase dynamics are chaotic and frame-capture perturbs them, so a catch
    # measured elsewhere may not reproduce here. Retry consecutive seeds and keep
    # the first rollout that actually catches IN THIS render, so the saved video is
    # guaranteed to end in a catch (unless --find-catch 1 forces the exact seed).
    frames = []
    av = None
    caught = False
    for attempt in range(max(1, args.find_catch)):
        seed = args.seed + attempt
        obs, info = env.reset(seed=seed)
        # env.reset() (re)creates the aviary, so re-fetch it each attempt.
        av = raw_env.backend._aviary
        if player is not None:
            player.init_states()
        # Declutter: remove the orange waypoint-route markers so only the two drones remain.
        for mid in list(getattr(raw_env.backend, "_target_marker_ids", []) or []):
            try:
                av.removeBody(mid)
            except Exception:
                pass
        frames = []
        yaw = 45.0
        done = False; steps = 0; caught = False
        while not done and steps < args.max_frames:
            d = np.array(raw_env.backend.get_drone_state().position)
            t = np.array(raw_env.backend.get_target_state().position)
            center = (d + t) / 2.0
            # Tight follow so the quadrotor meshes are clearly visible; clamp so both
            # drones stay roughly in frame even when separated.
            dist = float(min(9.0, max(3.5, np.linalg.norm(d - t) * 0.85 + 2.5)))
            frames.append(grab_rgb(av, center, yaw, args.width, args.height, dist))
            yaw = (yaw + 1.1) % 360.0
            if args.scripted:
                action = pursuit_action(obs)
            else:
                with torch.no_grad():
                    real = player.get_actions(to_torch(obs), greedy=True)
                action = torch.stack(real, -1).cpu().numpy().reshape(env.action_space.shape)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc; steps += 1
            caught = bool(info.get("is_success", False))
        if caught:
            print(f"caught on seed {seed} (attempt {attempt + 1}) in {steps} steps")
            break
        print(f"seed {seed}: no catch in {steps} steps, retrying...")
    # a few extra frames at the catch moment
    if caught:
        d = np.array(raw_env.backend.get_drone_state().position)
        t = np.array(raw_env.backend.get_target_state().position)
        center = (d + t) / 2.0
        for _ in range(int(args.fps * 1.2)):
            frames.append(grab_rgb(av, center, yaw, args.width, args.height, 6.0))
            yaw = (yaw + 1.4) % 360.0

    print(f"captured {len(frames)} frames, caught={caught}, steps={steps}")
    fig = plt.figure(figsize=(args.width / 100, args.height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    im = ax.imshow(frames[0])
    tag = "CAUGHT" if caught else "chase"
    txt = ax.text(8, 24, "", color="white", fontsize=11, fontweight="bold",
                  bbox=dict(facecolor="black", alpha=0.4, pad=3))

    def upd(i):
        im.set_data(frames[i])
        txt.set_text(f"{policy_label} pursuer vs roaming target (chase_open) — {tag}  frame {i+1}/{len(frames)}")
        return im, txt

    anim = FuncAnimation(fig, upd, frames=len(frames), interval=1000 / args.fps, blit=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=2400))
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
