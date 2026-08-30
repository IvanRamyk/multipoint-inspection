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

    # -- Moving-target task (DroneTargetEnv) ----------------------------------
    # A single target moves through the dome; the drone must intercept it.
    # target_mode selects the motion model: "static" (never moves — for the L0
    # smoke test), "car2d" (Dubins car at fixed altitude), or "airplane3d"
    # (Dubins airplane that also wanders in altitude).
    target_mode: str = "car2d"
    target_speed: float = 1.0                 # m/s forward speed
    target_turn_rate_max: float = 0.8         # rad/s max heading rate
    target_ou_theta: float = 0.5              # OU mean-reversion for turn rate
    target_ou_sigma: float = 1.5              # OU volatility for turn rate
    target_altitude: float = 2.0              # fixed altitude (static / car2d)
    target_spawn_radius_frac: float = 0.5     # spawn dist from centre, × dome/2
    target_boundary_frac: float = 0.8         # steer inward beyond this × dome/2
    target_reach_distance: float = 1.5        # catch radius (mission success)
    # airplane3d altitude band:
    target_climb_rate_max: float = 1.0        # m/s max vertical speed
    target_altitude_min: float = 1.5
    target_altitude_max: float = 6.0
    # Reward for intercepting the target (episode terminates on success).
    target_catch_reward: float = 60.0
    # Partial observability knobs (harder levels): drop the target velocity from
    # the observation and/or add Gaussian noise (meters) to the observed target
    # position. Noise affects perception only — reward/catch use the true state.
    target_observe_velocity: bool = True
    target_position_noise: float = 0.0

    # -- Chase task (DroneChaseEnv): target is a SECOND real drone -------------
    # The target drone flies a non-self-intersecting waypoint plan (2-opt) using
    # velocity control, with the same physics as the pursuer.
    chase_num_waypoints: int = 8              # points in the target's route
    chase_min_separation: float = 3.0         # min xy spacing between waypoints
    chase_ordering: str = "2opt"              # "2opt" (natural) or "angular"
    chase_target_speed_cap: float = 0.4       # target velocity cap in [0,1] (×_MAX_SPEED)
    chase_target_gain: float = 0.5            # P-gain of the target's waypoint follower
    chase_waypoint_reach: float = 1.5         # target advances to next waypoint within this

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
