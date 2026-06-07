"""Abstract simulator backend and drone state dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DroneState:
    """Snapshot of drone physical state.

    Attributes:
        position: (3,) xyz position in meters.
        velocity: (3,) velocity in m/s.
        orientation: (4,) quaternion (x, y, z, w).
        collision: Whether the drone is in collision.
    """

    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    collision: bool


class SimBackend(ABC):
    """Abstract interface for physics simulators.

    Implementations must provide drone control, sensing, and collision
    detection. This abstraction allows swapping PyFlyt for AirSim (or
    any other simulator) without changing the Gymnasium environment.
    """

    @abstractmethod
    def reset(
        self,
        drone_start: np.ndarray,
        waypoints: np.ndarray,
        obstacles: list[dict],
        seed: int | None = None,
    ) -> None:
        """Reset simulation with given configuration.

        Args:
            drone_start: (3,) starting position in meters.
            waypoints: (N, 3) waypoint positions in meters.
            obstacles: List of dicts with 'position' (3,), 'size' (float),
                       and 'shape' ('box' or 'sphere').
            seed: Optional random seed for deterministic simulation.
        """

    @abstractmethod
    def get_depth_image(self) -> np.ndarray:
        """Return (H, W, 1) float32 depth image in meters."""

    @abstractmethod
    def get_drone_state(self) -> DroneState:
        """Return current drone state."""

    @abstractmethod
    def apply_action(self, velocity: np.ndarray) -> None:
        """Apply desired velocity (3,) to drone.

        Args:
            velocity: Desired velocity vector, each component in [-1, 1].
        """

    @abstractmethod
    def step_simulation(self) -> None:
        """Advance simulation by one control step."""

    # -- Optional capabilities (concrete no-op defaults) ------------------
    # These are not part of the minimal contract; backends override them
    # when supported. Defaults let the env call them unconditionally.

    def apply_wind(self, wind_vec: np.ndarray) -> None:
        """Apply an external wind disturbance.

        Args:
            wind_vec: (3,) raw wind vector in world frame (units are
                backend-defined: PyFlyt treats it as a force scale, AirSim
                as a wind velocity in m/s). Default: no-op.
        """

    def set_waypoint_color(self, index: int, color: list[float]) -> None:
        """Recolor a waypoint marker for visualization. Default: no-op."""

    def close(self) -> None:
        """Release simulator resources. Default: no-op."""
