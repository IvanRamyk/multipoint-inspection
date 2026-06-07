"""Environment configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import yaml


@dataclass
class EnvConfig:
    """Configuration for DroneInspectionEnv.

    All physical units are in meters and seconds unless noted otherwise.
    """

    num_waypoints: int = 5
    flight_dome_size: float = 50.0
    waypoint_reach_distance: float = 1.0
    base_position: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    # Wind (Ornstein-Uhlenbeck process)
    wind_enabled: bool = True
    wind_strength: float = 0.25
    wind_ou_theta: float = 0.15
    wind_ou_sigma: float = 0.2

    # Obstacles
    num_obstacles: int = 5
    obstacle_size_range: Tuple[float, float] = (1.0, 3.0)

    # Timing
    max_episode_steps: int = 1000
    agent_hz: int = 30

    # Depth camera
    image_width: int = 64
    image_height: int = 64

    # Simulator backend: "pyflyt" (PyBullet, default) or "airsim" (Cosys-AirSim,
    # requires Linux/Windows + Unreal Engine 5; not available on macOS).
    backend: str = "pyflyt"

    # Potential-based reward shaping coefficient.
    # reward += reward_shaping * (prev_target_dist - curr_target_dist) each step
    # where target = nearest unvisited waypoint, or base if all visited.
    # 0.0 disables shaping.
    reward_shaping: float = 0.0

    # Collision penalty (negative). Default −100 matches the original spec;
    # override per-config to soften for sanity tasks where collisions are mostly
    # ground/dome hits rather than meaningful obstacles.
    collision_penalty: float = -100.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> EnvConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            An EnvConfig instance with values from the file.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Convert list values to tuples where needed
        if "base_position" in data and isinstance(data["base_position"], list):
            data["base_position"] = tuple(data["base_position"])
        if "obstacle_size_range" in data and isinstance(data["obstacle_size_range"], list):
            data["obstacle_size_range"] = tuple(data["obstacle_size_range"])
        return cls(**data)
