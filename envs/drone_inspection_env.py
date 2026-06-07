"""Gymnasium environment for multi-waypoint drone inspection missions."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from envs.config import EnvConfig
from envs.pyflyt_backend import PyFlytBackend
from envs.sim_backend import SimBackend
from envs.wind_model import OUWindModel


# Reward constants.
_REWARD_WAYPOINT = 10.0
_REWARD_MISSION_COMPLETE = 50.0
_REWARD_COLLISION = -100.0
_REWARD_TIME_PENALTY = -0.01

# Minimum separation between waypoints in meters.
_MIN_WAYPOINT_SEPARATION = 3.0


def _make_backend(config: EnvConfig, render_mode: str | None) -> SimBackend:
    """Construct a simulator backend from config.

    Args:
        config: Environment configuration (uses ``config.backend``).
        render_mode: Gymnasium render mode; "human" opens a GUI (PyFlyt only).

    Returns:
        A SimBackend instance.

    Raises:
        ValueError: If ``config.backend`` is not a known backend name.
    """
    name = getattr(config, "backend", "pyflyt").lower()
    common = dict(
        image_width=config.image_width,
        image_height=config.image_height,
        agent_hz=config.agent_hz,
    )
    if name == "pyflyt":
        return PyFlytBackend(render=(render_mode == "human"), **common)
    if name == "airsim":
        # Lazy import: the airsim package is unavailable on macOS, so importing
        # it at module load would break PyFlyt-only environments.
        from envs.airsim_backend import AirSimBackend

        return AirSimBackend(**common)
    raise ValueError(
        f"Unknown backend {name!r}; expected 'pyflyt' or 'airsim'."
    )


class DroneInspectionEnv(gymnasium.Env):
    """Multi-waypoint drone inspection environment.

    The agent must visit all waypoints in any order and return to base.
    Observations are multimodal: a depth image and a state vector.
    Actions are 3D desired velocity commands in [-1, 1].

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

        N = self.config.num_waypoints
        state_dim = 12 + 4 * N

        self.observation_space = gymnasium.spaces.Dict(
            {
                "depth": gymnasium.spaces.Box(
                    0.0,
                    100.0,
                    shape=(self.config.image_height, self.config.image_width, 1),
                    dtype=np.float32,
                ),
                "state": gymnasium.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(state_dim,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = gymnasium.spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32
        )

        self._wind = OUWindModel(
            theta=self.config.wind_ou_theta,
            sigma=self.config.wind_ou_sigma,
            strength=self.config.wind_strength,
            dt=1.0 / self.config.agent_hz,
        )

        self._waypoints: np.ndarray | None = None
        self._visited: np.ndarray | None = None
        self._step_count = 0
        self._positions: list[np.ndarray] = []
        self._prev_target_dist: float | None = None
        self._prev_target: np.ndarray | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment and return initial observation.

        Args:
            seed: Random seed for reproducibility.
            options: Unused.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)

        rng = self.np_random

        # Generate waypoints with minimum separation.
        self._waypoints = self._generate_waypoints(rng)
        self._visited = np.zeros(self.config.num_waypoints, dtype=bool)
        self._step_count = 0
        self._positions = []
        self._prev_target_dist = None
        self._prev_target = None

        # Generate obstacles.
        obstacles = self._generate_obstacles(rng)

        base = np.array(self.config.base_position, dtype=np.float32)
        # Derive a deterministic seed for the backend from the env seed.
        backend_seed = int(rng.integers(0, 2**31)) if seed is not None else None
        self.backend.reset(
            drone_start=base,
            waypoints=self._waypoints,
            obstacles=obstacles,
            seed=backend_seed,
        )

        # Reset wind.
        wind_seed = int(rng.integers(0, 2**31)) if seed is not None else None
        self._wind.reset(seed=wind_seed)

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one agent step.

        Args:
            action: (3,) velocity command in [-1, 1].

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)

        # Apply action.
        self.backend.apply_action(action)

        # Apply wind. Each backend interprets the raw OU vector in its own
        # units (no-op on backends that don't model wind).
        if self.config.wind_enabled:
            wind_vec = self._wind.step()
            self.backend.apply_wind(wind_vec)

        # Step simulation.
        self.backend.step_simulation()
        self._step_count += 1

        drone = self.backend.get_drone_state()
        self._positions.append(drone.position.copy())

        reward = _REWARD_TIME_PENALTY
        terminated = False

        # Check collision.
        if drone.collision:
            reward += self.config.collision_penalty
            terminated = True

        # Check waypoint visits.
        if not terminated:
            for i, wp in enumerate(self._waypoints):
                if not self._visited[i]:
                    dist = np.linalg.norm(drone.position - wp)
                    if dist < self.config.waypoint_reach_distance:
                        self._visited[i] = True
                        reward += _REWARD_WAYPOINT
                        # Visual feedback: change color to blue (no-op on
                        # backends without marker visualization).
                        self.backend.set_waypoint_color(i, [0.0, 0.5, 1.0, 0.6])

        # Check mission complete: all visited and back at base.
        if not terminated and np.all(self._visited):
            base = np.array(self.config.base_position, dtype=np.float32)
            dist_to_base = np.linalg.norm(drone.position - base)
            if dist_to_base < self.config.waypoint_reach_distance:
                reward += _REWARD_MISSION_COMPLETE
                terminated = True

        # Potential-based shaping: Δdistance to current target.
        # Only applied while target is unchanged across consecutive steps;
        # target switches (waypoint visit, mission complete) reset the potential.
        if not terminated and self.config.reward_shaping > 0.0:
            target = self._next_target(drone.position)
            curr_dist = float(np.linalg.norm(drone.position - target))
            if (
                self._prev_target_dist is not None
                and self._prev_target is not None
                and np.array_equal(self._prev_target, target)
            ):
                reward += self.config.reward_shaping * (self._prev_target_dist - curr_dist)
            self._prev_target_dist = curr_dist
            self._prev_target = target

        truncated = self._step_count >= self.config.max_episode_steps

        obs = self._build_obs()
        info = self._build_info()
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        """Clean up simulator resources."""
        self.backend.close()

    # -- Properties for external access -----------------------------------

    @property
    def waypoints(self) -> np.ndarray | None:
        """Current episode waypoints (N, 3)."""
        return self._waypoints

    @property
    def visited(self) -> np.ndarray | None:
        """Boolean mask of visited waypoints."""
        return self._visited

    @property
    def positions(self) -> list[np.ndarray]:
        """List of drone positions recorded during the episode."""
        return self._positions

    # -- Private helpers --------------------------------------------------

    def _next_target(self, position: np.ndarray) -> np.ndarray:
        """Return the current shaping target: nearest unvisited waypoint, or base.

        Args:
            position: Drone position (3,).

        Returns:
            (3,) target position.
        """
        if np.all(self._visited):
            return np.array(self.config.base_position, dtype=np.float32)
        unvisited = np.where(~self._visited)[0]
        dists = np.linalg.norm(self._waypoints[unvisited] - position, axis=1)
        return self._waypoints[unvisited[int(np.argmin(dists))]]

    def _build_obs(self) -> dict[str, np.ndarray]:
        """Construct the observation dictionary."""
        drone = self.backend.get_drone_state()
        depth = self.backend.get_depth_image()
        # Clip depth to observation space bounds.
        depth = np.clip(depth, 0.0, 100.0)

        base = np.array(self.config.base_position, dtype=np.float32)
        wind = self._wind.current if self.config.wind_enabled else np.zeros(3, dtype=np.float32)

        # Build state vector.
        parts = [
            drone.position,                     # 3
            drone.velocity,                      # 3
            wind,                                # 3
            base - drone.position,               # 3 (base relative)
        ]
        for i in range(self.config.num_waypoints):
            rel = self._waypoints[i] - drone.position
            vis = np.array([1.0 if self._visited[i] else 0.0], dtype=np.float32)
            parts.append(rel.astype(np.float32))  # 3
            parts.append(vis)                       # 1

        state_vec = np.concatenate(parts).astype(np.float32)
        return {"depth": depth, "state": state_vec}

    def _build_info(self) -> dict[str, Any]:
        """Build the info dictionary."""
        return {
            "waypoints_visited": int(np.sum(self._visited)) if self._visited is not None else 0,
            "total_waypoints": self.config.num_waypoints,
            "step_count": self._step_count,
        }

    def _generate_waypoints(self, rng: np.random.Generator) -> np.ndarray:
        """Generate waypoints with minimum separation.

        Args:
            rng: Numpy random generator.

        Returns:
            (N, 3) array of waypoint positions.
        """
        N = self.config.num_waypoints
        dome = self.config.flight_dome_size
        waypoints = []
        max_attempts = 1000

        for _ in range(N):
            for attempt in range(max_attempts):
                wp = rng.uniform(
                    low=[-dome / 2, -dome / 2, 1.0],
                    high=[dome / 2, dome / 2, dome / 2],
                )
                # Check separation from existing waypoints.
                if all(
                    np.linalg.norm(wp - existing) >= _MIN_WAYPOINT_SEPARATION
                    for existing in waypoints
                ):
                    # Also ensure separation from base.
                    base = np.array(self.config.base_position)
                    if np.linalg.norm(wp - base) >= _MIN_WAYPOINT_SEPARATION:
                        waypoints.append(wp)
                        break
            else:
                # Fallback: accept last candidate.
                waypoints.append(wp)

        return np.array(waypoints, dtype=np.float32)

    def _generate_obstacles(self, rng: np.random.Generator) -> list[dict]:
        """Generate random obstacles.

        Args:
            rng: Numpy random generator.

        Returns:
            List of obstacle dicts with 'position' and 'size'.
        """
        obstacles = []
        dome = self.config.flight_dome_size
        lo, hi = self.config.obstacle_size_range

        for _ in range(self.config.num_obstacles):
            pos = rng.uniform(
                low=[-dome / 2, -dome / 2, 1.0],
                high=[dome / 2, dome / 2, dome / 2],
            )
            size = rng.uniform(lo, hi)
            obstacles.append({"position": pos, "size": float(size)})

        return obstacles
