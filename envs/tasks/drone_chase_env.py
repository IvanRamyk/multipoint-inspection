"""Gymnasium environment for chasing a second, physically-simulated drone.

Unlike :class:`DroneTargetEnv` (kinematic Dubins target), here the target is a
real QuadX drone with identical physics that flies a non-self-intersecting
waypoint plan (see :mod:`envs.targets.waypoint_plan`) using velocity control.
The pursuer (the agent) must intercept it.

Observation (Dict):
    depth: (H, W, 1) float32 — pursuer's depth camera (the target drone is a
        physical body, so it is actually visible in the image).
    state: (15,) = pursuer pos (3) + pursuer vel (3) + wind (3)
                   + target relative pos (3) + target velocity (3).

Action: (3,) desired velocity in [-1, 1] for the pursuer.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from envs.core.config import EnvConfig
from envs.core.wind_model import OUWindModel
from envs.backends.pyflyt_chase_backend import PyFlytChaseBackend
from envs.targets.waypoint_plan import WaypointPlan

_REWARD_TIME_PENALTY = -0.01
_STATE_DIM = 15


class DroneChaseEnv(gymnasium.Env):
    """Pursue a second drone that flies a non-crossing waypoint route."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: EnvConfig | None = None,
        config_path: str | None = None,
        backend: PyFlytChaseBackend | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = EnvConfig.from_yaml(config_path)
        else:
            self.config = EnvConfig()
        if getattr(self.config, "backend", "pyflyt").lower() != "pyflyt":
            raise ValueError("DroneChaseEnv only supports the PyFlyt backend.")
        self.backend = backend or PyFlytChaseBackend(
            image_width=self.config.image_width,
            image_height=self.config.image_height,
            agent_hz=self.config.agent_hz,
            render=(render_mode == "human"),
        )

        self.observation_space = gymnasium.spaces.Dict(
            {
                "depth": gymnasium.spaces.Box(
                    0.0, 100.0,
                    shape=(self.config.image_height, self.config.image_width, 1),
                    dtype=np.float32,
                ),
                "state": gymnasium.spaces.Box(-np.inf, np.inf, shape=(_STATE_DIM,), dtype=np.float32),
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
        self._plan: WaypointPlan | None = None
        self._step_count = 0
        self._positions: list[np.ndarray] = []
        self._target_positions: list[np.ndarray] = []
        self._prev_dist: float | None = None
        self._caught = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random

        # Non-self-intersecting waypoint plan for the target drone.
        self._plan = WaypointPlan.sample(
            n=self.config.chase_num_waypoints,
            dome_size=self.config.flight_dome_size,
            rng=rng,
            min_separation=self.config.chase_min_separation,
            altitude_range=(self.config.target_altitude_min, self.config.target_altitude_max),
            method=self.config.chase_ordering,
            reach=self.config.chase_waypoint_reach,
            loop=True,
        )
        # Target starts at the first waypoint and heads to the next.
        target_start = self._plan.waypoints[0].astype(np.float32)
        self._plan._i = 1 if len(self._plan.waypoints) > 1 else 0

        self._step_count = 0
        self._positions = []
        self._target_positions = []
        self._prev_dist = None
        self._caught = False

        obstacles = self._generate_obstacles(rng)
        base = np.array(self.config.base_position, dtype=np.float32)
        backend_seed = int(rng.integers(0, 2**31)) if seed is not None else None
        self.backend.reset(
            drone_start=base,
            target_start=target_start,
            obstacles=obstacles,
            waypoints=self._plan.waypoints,
            seed=backend_seed,
        )

        wind_seed = int(rng.integers(0, 2**31)) if seed is not None else None
        self._wind.reset(seed=wind_seed)

        return self._build_obs(), self._build_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        self.backend.apply_action(action)

        # Drive the target drone toward its current waypoint (P-controller,
        # speed-capped) — real velocity control, same physics as the pursuer.
        self.backend.set_target_action(self._target_command())

        if self.config.wind_enabled:
            self.backend.apply_wind(self._wind.step())

        self.backend.step_simulation()
        self._step_count += 1

        drone = self.backend.get_drone_state()
        target = self.backend.get_target_state()
        self._positions.append(drone.position.copy())
        self._target_positions.append(target.position.copy())

        # Advance the target's plan if it reached its current waypoint.
        self._plan.advance_if_reached(target.position)

        reward = _REWARD_TIME_PENALTY
        terminated = False

        if drone.collision:
            reward += self.config.collision_penalty
            terminated = True

        curr_dist = float(np.linalg.norm(drone.position - target.position))
        if not terminated and curr_dist < self.config.target_reach_distance:
            reward += self.config.target_catch_reward
            terminated = True
            self._caught = True

        if not terminated and self.config.reward_shaping > 0.0:
            if self._prev_dist is not None:
                reward += self.config.reward_shaping * (self._prev_dist - curr_dist)
        self._prev_dist = curr_dist

        truncated = self._step_count >= self.config.max_episode_steps
        return self._build_obs(), float(reward), terminated, truncated, self._build_info()

    def close(self) -> None:
        self.backend.close()

    # -- Properties -------------------------------------------------------

    @property
    def positions(self) -> list[np.ndarray]:
        return self._positions

    @property
    def target_positions(self) -> list[np.ndarray]:
        return self._target_positions

    # -- Helpers ----------------------------------------------------------

    def _target_command(self) -> np.ndarray:
        """Velocity command (in [-1,1]^3) driving the target toward its waypoint."""
        target = self.backend.get_target_state()
        goal = self._plan.current()
        cmd = self.config.chase_target_gain * (goal - target.position)
        cap = self.config.chase_target_speed_cap
        return np.clip(cmd, -cap, cap).astype(np.float32)

    def _build_obs(self) -> dict[str, np.ndarray]:
        drone = self.backend.get_drone_state()
        target = self.backend.get_target_state()
        depth = np.clip(self.backend.get_depth_image(), 0.0, 100.0)
        wind = self._wind.current if self.config.wind_enabled else np.zeros(3, dtype=np.float32)

        target_pos = target.position
        target_vel = target.velocity
        if self.config.target_position_noise > 0.0:
            target_pos = target_pos + self.np_random.normal(
                0.0, self.config.target_position_noise, size=3
            ).astype(np.float32)
        if not self.config.target_observe_velocity:
            target_vel = np.zeros(3, dtype=np.float32)

        state_vec = np.concatenate(
            [
                drone.position,
                drone.velocity,
                wind,
                target_pos - drone.position,
                target_vel,
            ]
        ).astype(np.float32)
        return {"depth": depth, "state": state_vec}

    def _build_info(self) -> dict[str, Any]:
        drone_pos = self._positions[-1] if self._positions else None
        tgt_pos = self._target_positions[-1] if self._target_positions else None
        dist = (
            float(np.linalg.norm(drone_pos - tgt_pos))
            if drone_pos is not None and tgt_pos is not None
            else float("nan")
        )
        return {
            "is_success": self._caught,
            "distance_to_target": dist,
            "step_count": self._step_count,
        }

    def _generate_obstacles(self, rng: np.random.Generator) -> list[dict]:
        obstacles = []
        dome = self.config.flight_dome_size
        lo, hi = self.config.obstacle_size_range
        for _ in range(self.config.num_obstacles):
            pos = rng.uniform(low=[-dome / 2, -dome / 2, 1.0], high=[dome / 2, dome / 2, dome / 2])
            obstacles.append({"position": pos, "size": float(rng.uniform(lo, hi))})
        return obstacles
