"""Wrapper and factory for sheeprl compatibility."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.config import EnvConfig
from envs.drone_inspection_env import DroneInspectionEnv


class SheepRLCompatWrapper(gym.ObservationWrapper):
    """Convert depth from float32 meters to uint8 [0, 255] for sheeprl CNN encoder.

    sheeprl expects image observations as uint8 in [0, 255] with shape (H, W, C).
    The raw environment outputs depth as float32 in [0, max_depth] meters.
    """

    def __init__(self, env: gym.Env, max_depth: float = 100.0) -> None:
        super().__init__(env)
        self.max_depth = max_depth
        old_spaces = dict(env.observation_space.spaces)
        h, w, c = old_spaces["depth"].shape
        old_spaces["depth"] = gym.spaces.Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )
        self.observation_space = gym.spaces.Dict(old_spaces)

    def observation(self, obs: dict) -> dict:
        """Normalize depth to [0, 255] uint8."""
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
