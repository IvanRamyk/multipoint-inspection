"""Gymnasium environment for intercepting a single moving target.

The drone must fly to a target that wanders through the flight dome following a
Dubins motion model (see :mod:`envs.targets.dubins`). The episode ends
successfully when the drone gets within ``config.target_reach_distance`` of the
target. This is the moving-target counterpart to :class:`DroneInspectionEnv`:
it shares the same backend, wind, depth-camera perception, and action space, but
swaps the static multi-waypoint mission for a single dynamic goal.

Observation (Dict):
    depth: (H, W, 1) float32 depth image.
    state: (15,) = drone pos (3) + drone vel (3) + wind (3)
                   + target relative pos (3) + target velocity (3).

Action: (3,) desired velocity in [-1, 1], mapped to m/s by the backend.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from envs.core.config import EnvConfig
from envs.core.sim_backend import SimBackend
from envs.core.wind_model import OUWindModel
from envs.backends.pyflyt_backend import PyFlytBackend
from envs.targets import make_target

# Reward constants.
_REWARD_TIME_PENALTY = -0.01

# State layout: [pos(3), vel(3), wind(3), target_rel(3), target_vel(3)].
_STATE_DIM = 15


def _make_backend(config: EnvConfig, render_mode: str | None) -> SimBackend:
    """Construct a simulator backend from config (mirrors DroneInspectionEnv)."""
    name = getattr(config, "backend", "pyflyt").lower()
    common = dict(
        image_width=config.image_width,
        image_height=config.image_height,
        agent_hz=config.agent_hz,
    )
    if name == "pyflyt":
        return PyFlytBackend(render=(render_mode == "human"), **common)
    if name == "airsim":
        from envs.backends.airsim_backend import AirSimBackend

        return AirSimBackend(**common)
    raise ValueError(f"Unknown backend {name!r}; expected 'pyflyt' or 'airsim'.")


class DroneTargetEnv(gymnasium.Env):
    """Single moving-target interception environment.

    Attributes:
        config: Environment configuration.
        backend: Simulator backend for physics and rendering.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: EnvConfig | None = None,
        config_path: str | None = None,
        backend: SimBackend | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = EnvConfig.from_yaml(config_path)
        else:
            self.config = EnvConfig()
        self.backend = backend or _make_backend(self.config, render_mode)

        self.observation_space = gymnasium.spaces.Dict(
            {
                "depth": gymnasium.spaces.Box(
                    0.0,
                    100.0,
                    shape=(self.config.image_height, self.config.image_width, 1),
                    dtype=np.float32,
                ),
                "state": gymnasium.spaces.Box(
                    -np.inf, np.inf, shape=(_STATE_DIM,), dtype=np.float32
                ),
            }
        )
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        self._dt = 1.0 / self.config.agent_hz
        self._wind = OUWindModel(
            theta=self.config.wind_ou_theta,
            sigma=self.config.wind_ou_sigma,
            strength=self.config.wind_strength,
            dt=self._dt,
        )
        self._target = make_target(self.config, dt=self._dt)

        self._step_count = 0
        self._positions: list[np.ndarray] = []
        self._target_positions: list[np.ndarray] = []
        self._prev_dist: float | None = None
        self._caught = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment and return the initial observation."""
        super().reset(seed=seed)
        rng = self.np_random

        target_pos = self._target.reset(rng)
        self._step_count = 0
        self._positions = []
        self._target_positions = []
        self._prev_dist = None
        self._caught = False

        obstacles = self._generate_obstacles(rng)
        base = np.array(self.config.base_position, dtype=np.float32)
        backend_seed = int(rng.integers(0, 2**31)) if seed is not None else None
        # Spawn a single marker for the target at its initial position.
        self.backend.reset(
            drone_start=base,
            waypoints=target_pos.reshape(1, 3),
            obstacles=obstacles,
            seed=backend_seed,
        )
        self.backend.set_waypoint_color(0, [1.0, 0.2, 0.1, 0.8])  # red = target

        wind_seed = int(rng.integers(0, 2**31)) if seed is not None else None
        self._wind.reset(seed=wind_seed)

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one agent step."""
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        self.backend.apply_action(action)

        if self.config.wind_enabled:
            self.backend.apply_wind(self._wind.step())

        self.backend.step_simulation()
        self._step_count += 1

        # Advance the target and update its marker.
        target_pos = self._target.step()
        self.backend.set_waypoint_position(0, target_pos)

        drone = self.backend.get_drone_state()
        self._positions.append(drone.position.copy())
        self._target_positions.append(target_pos.copy())

        reward = _REWARD_TIME_PENALTY
        terminated = False

        if drone.collision:
            reward += self.config.collision_penalty
            terminated = True

        curr_dist = float(np.linalg.norm(drone.position - target_pos))

        # Catch: drone within reach radius of the target.
        if not terminated and curr_dist < self.config.target_reach_distance:
            reward += self.config.target_catch_reward
            terminated = True
            self._caught = True

        # Potential-based shaping toward the (moving) target.
        if not terminated and self.config.reward_shaping > 0.0:
            if self._prev_dist is not None:
                reward += self.config.reward_shaping * (self._prev_dist - curr_dist)
        self._prev_dist = curr_dist

        truncated = self._step_count >= self.config.max_episode_steps

        obs = self._build_obs()
        info = self._build_info()
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        """Clean up simulator resources."""
        self.backend.close()

    # -- Properties for external access -----------------------------------

    @property
    def positions(self) -> list[np.ndarray]:
        """Drone positions recorded during the episode."""
        return self._positions

    @property
    def target_positions(self) -> list[np.ndarray]:
        """Target positions recorded during the episode."""
        return self._target_positions

    # -- Private helpers --------------------------------------------------

    def _build_obs(self) -> dict[str, np.ndarray]:
        """Construct the observation dictionary."""
        drone = self.backend.get_drone_state()
        depth = np.clip(self.backend.get_depth_image(), 0.0, 100.0)
        wind = self._wind.current if self.config.wind_enabled else np.zeros(3, dtype=np.float32)

        target_pos = self._target.position
        target_vel = self._target.velocity

        # Partial observability (harder levels): optional position noise and/or
        # hidden velocity. True target state is still used for reward/catch.
        if self.config.target_position_noise > 0.0:
            target_pos = target_pos + self.np_random.normal(
                0.0, self.config.target_position_noise, size=3
            ).astype(np.float32)
        if not self.config.target_observe_velocity:
            target_vel = np.zeros(3, dtype=np.float32)

        state_vec = np.concatenate(
            [
                drone.position,                 # 3
                drone.velocity,                 # 3
                wind,                           # 3
                target_pos - drone.position,    # 3 (target relative)
                target_vel,                     # 3
            ]
        ).astype(np.float32)
        return {"depth": depth, "state": state_vec}

    def _build_info(self) -> dict[str, Any]:
        """Build the info dictionary. ``is_success`` drives early stopping."""
        drone_pos = self._positions[-1] if self._positions else None
        dist = (
            float(np.linalg.norm(drone_pos - self._target.position))
            if drone_pos is not None
            else float("nan")
        )
        return {
            "is_success": self._caught,
            "distance_to_target": dist,
            "step_count": self._step_count,
        }

    def _generate_obstacles(self, rng: np.random.Generator) -> list[dict]:
        """Generate random box obstacles (empty when num_obstacles == 0)."""
        obstacles = []
        dome = self.config.flight_dome_size
        lo, hi = self.config.obstacle_size_range
        for _ in range(self.config.num_obstacles):
            pos = rng.uniform(
                low=[-dome / 2, -dome / 2, 1.0],
                high=[dome / 2, dome / 2, dome / 2],
            )
            obstacles.append({"position": pos, "size": float(rng.uniform(lo, hi))})
        return obstacles
