"""Wrapper and factory for sheeprl compatibility."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.core.config import EnvConfig
from envs.tasks.drone_inspection_env import DroneInspectionEnv
from envs.tasks.drone_target_env import DroneTargetEnv
from envs.tasks.drone_chase_env import DroneChaseEnv


class SheepRLCompatWrapper(gym.ObservationWrapper):
    """Convert depth from float32 meters to uint8 [0, 255] for sheeprl CNN encoder.

    sheeprl expects image observations as uint8 in [0, 255] with shape (H, W, C).
    The raw environment outputs depth as float32 in [0, max_depth] meters.
    """

    def __init__(self, env: gym.Env, max_depth: float = 100.0) -> None:
        super().__init__(env)
        self.max_depth = max_depth
        old_spaces = dict(env.observation_space.spaces)
        # State-only envs (observe_depth=False) have no depth key — nothing to
        # convert, the wrapper is then a passthrough for the state vector.
        self._has_depth = "depth" in old_spaces
        if self._has_depth:
            h, w, c = old_spaces["depth"].shape
            old_spaces["depth"] = gym.spaces.Box(
                low=0, high=255, shape=(h, w, c), dtype=np.uint8
            )
        self.observation_space = gym.spaces.Dict(old_spaces)

    def observation(self, obs: dict) -> dict:
        """Normalize depth to [0, 255] uint8 (no-op when there is no camera)."""
        if not self._has_depth:
            return obs
        depth = obs["depth"]
        depth_normalized = np.clip(depth / self.max_depth * 255, 0, 255).astype(
            np.uint8
        )
        obs["depth"] = depth_normalized
        return obs


def make_drone_inspection_env(
    config_path: str = "configs/easy.yaml",
    render_mode: str | None = None,
    **kwargs,
) -> gym.Env:
    """Factory function for sheeprl's Hydra instantiation.

    Creates DroneInspectionEnv wrapped with SheepRLCompatWrapper.

    Args:
        config_path: Path to YAML environment config.
        render_mode: Gymnasium render mode.

    Returns:
        Wrapped DroneInspectionEnv ready for sheeprl.
    """
    env = DroneInspectionEnv(config_path=config_path, render_mode=render_mode)
    env = SheepRLCompatWrapper(env)
    return env


def make_drone_target_env(
    config_path: str = "configs/target/l0_smoke.yaml",
    render_mode: str | None = None,
    record_video_every: int = 0,
    record_video_dir: str = "results/training_videos",
    record_video_max: int = 0,
    record_video_3d: bool = True,
    record_video_nest_under_run: bool = True,
    **kwargs,
) -> gym.Env:
    """Factory for sheeprl's Hydra instantiation of the moving-target task.

    Creates DroneTargetEnv wrapped with SheepRLCompatWrapper, and — when
    ``record_video_every > 0`` — an EpisodeVideoRecorder that dumps a trajectory
    video every N episodes (set via ``env.wrapper.record_video_every=N``).

    Args:
        config_path: Path to YAML environment config.
        render_mode: Gymnasium render mode.
        record_video_every: Record one episode video every N episodes (0 off).
        record_video_dir: Output directory for training videos.
        record_video_max: Cap on number of videos (0 = unlimited).
        record_video_3d: Render the orbiting 3D view (else 2D).

    Returns:
        Wrapped DroneTargetEnv ready for sheeprl.
    """
    env = DroneTargetEnv(config_path=config_path, render_mode=render_mode)
    env = SheepRLCompatWrapper(env)
    return _maybe_record(
        env, record_video_every, record_video_dir, record_video_max,
        record_video_3d, record_video_nest_under_run,
    )


def _maybe_record(env, every, out_dir, max_videos, use_3d, nest_under_run):
    """Wrap with EpisodeVideoRecorder when episode-video recording is enabled.

    Attaches to every env; the recorder's lock-file claim ensures exactly one
    process actually records (works for both sync and async vector envs).
    """
    if every and int(every) > 0:
        # Imported lazily so plain training runs never touch matplotlib/ffmpeg.
        from eval.episode_recorder import EpisodeVideoRecorder

        env = EpisodeVideoRecorder(
            env,
            record_every=int(every),
            out_dir=out_dir,
            max_videos=int(max_videos),
            use_3d=bool(use_3d),
            nest_under_run=bool(nest_under_run),
        )
    return env


def make_drone_chase_env(
    config_path: str = "configs/target/chase.yaml",
    render_mode: str | None = None,
    record_video_every: int = 0,
    record_video_dir: str = "results/training_videos",
    record_video_max: int = 0,
    record_video_3d: bool = True,
    record_video_nest_under_run: bool = True,
    **kwargs,
) -> gym.Env:
    """Factory for sheeprl's Hydra instantiation of the two-drone chase task.

    Creates DroneChaseEnv (pursuer + a physical target drone following a
    non-self-intersecting waypoint plan) wrapped with SheepRLCompatWrapper, and
    optionally the EpisodeVideoRecorder.

    Returns:
        Wrapped DroneChaseEnv ready for sheeprl.
    """
    env = DroneChaseEnv(config_path=config_path, render_mode=render_mode)
    env = SheepRLCompatWrapper(env)
    return _maybe_record(
        env, record_video_every, record_video_dir, record_video_max,
        record_video_3d, record_video_nest_under_run,
    )
