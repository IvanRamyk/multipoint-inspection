"""Dubins-style moving targets with naturally wandering trajectories.

A Dubins vehicle moves at constant forward speed with a bounded turn rate, so
its path is curvature-limited — it cannot teleport or reverse instantly, which
is what makes the motion look natural. To keep the target from driving in a
straight line forever, the turn rate is driven by a mean-reverting
Ornstein-Uhlenbeck process (the same idea as the wind model, but scalar). When
the target approaches the edge of the flight dome it steers back toward the
centre so it never escapes the play area.

Two models are provided:

* :class:`DubinsCar2D` — moves on a horizontal plane at a fixed altitude.
* :class:`DubinsAirplane3D` — adds a bounded vertical rate so it also wanders
  in altitude within a configured band.

Both expose the same interface: :meth:`reset` (given an RNG) and :meth:`step`
(advance by ``dt`` seconds), plus :attr:`position` and :attr:`velocity`
properties returning ``(3,)`` float32 world-frame vectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to the interval (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class _ScalarOU:
    """Scalar Ornstein-Uhlenbeck process, mean-reverting to zero."""

    def __init__(self, theta: float, sigma: float, dt: float) -> None:
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self._x = 0.0
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._x = 0.0

    def step(self) -> float:
        noise = float(self._rng.standard_normal())
        self._x += -self.theta * self._x * self.dt + self.sigma * np.sqrt(self.dt) * noise
        return self._x


class MovingTarget(ABC):
    """Base class for moving targets confined to the flight dome.

    Attributes:
        dome_size: Flight dome size in meters. The reachable xy region is a
            disk of radius ``dome_size / 2`` centred at the origin.
        speed: Constant forward (horizontal) speed in m/s.
        dt: Integration timestep in seconds (one agent step).
    """

    def __init__(self, dome_size: float, speed: float, dt: float) -> None:
        self.dome_size = dome_size
        self.speed = speed
        self.dt = dt
        self._pos = np.zeros(3, dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)

    @property
    def dome_radius(self) -> float:
        """Radius of the reachable xy disk."""
        return self.dome_size / 2.0

    @property
    def position(self) -> np.ndarray:
        """Current target position (3,) in world frame."""
        return self._pos.copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current target velocity (3,) in world frame."""
        return self._vel.copy()

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> np.ndarray:
        """Place the target and reset internal state. Returns initial position."""

    @abstractmethod
    def step(self) -> np.ndarray:
        """Advance the target by one timestep. Returns the new position."""


class StaticTarget(MovingTarget):
    """Degenerate target that never moves. Used by the L0 smoke config."""

    def __init__(self, dome_size: float, dt: float, spawn_radius_frac: float = 0.6, altitude: float = 2.0) -> None:
        super().__init__(dome_size=dome_size, speed=0.0, dt=dt)
        self.spawn_radius_frac = spawn_radius_frac
        self.altitude = altitude

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        r = self.spawn_radius_frac * self.dome_radius
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        self._pos = np.array(
            [r * np.cos(angle), r * np.sin(angle), self.altitude], dtype=np.float32
        )
        self._vel = np.zeros(3, dtype=np.float32)
        return self.position

    def step(self) -> np.ndarray:
        return self.position


class DubinsCar2D(MovingTarget):
    """Dubins car: constant-speed, curvature-limited motion at fixed altitude.

    Attributes:
        turn_rate_max: Maximum |heading rate| in rad/s.
        altitude: Fixed flight altitude in meters.
        boundary_frac: Fraction of ``dome_radius`` beyond which the target
            starts steering back toward the centre.
        boundary_gain: Proportional gain for the inward steering controller.
    """

    def __init__(
        self,
        dome_size: float,
        speed: float,
        dt: float,
        turn_rate_max: float = 0.8,
        ou_theta: float = 0.5,
        ou_sigma: float = 1.5,
        altitude: float = 2.0,
        spawn_radius_frac: float = 0.5,
        boundary_frac: float = 0.8,
        boundary_gain: float = 3.0,
    ) -> None:
        super().__init__(dome_size=dome_size, speed=speed, dt=dt)
        self.turn_rate_max = turn_rate_max
        self.altitude = altitude
        self.spawn_radius_frac = spawn_radius_frac
        self.boundary_frac = boundary_frac
        self.boundary_gain = boundary_gain
        self._theta = 0.0
        self._ou = _ScalarOU(theta=ou_theta, sigma=ou_sigma, dt=dt)

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        self._ou.reset(rng)
        r = self.spawn_radius_frac * self.dome_radius
        spawn_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        x = r * np.cos(spawn_angle)
        y = r * np.sin(spawn_angle)
        self._theta = float(rng.uniform(0.0, 2.0 * np.pi))
        self._pos = np.array([x, y, self.altitude], dtype=np.float32)
        self._vel = np.array(
            [self.speed * np.cos(self._theta), self.speed * np.sin(self._theta), 0.0],
            dtype=np.float32,
        )
        return self.position

    def _turn_rate(self) -> float:
        """Turn rate this step: inward steering near the boundary, else OU noise."""
        x, y = float(self._pos[0]), float(self._pos[1])
        r = np.hypot(x, y)
        soft = self.boundary_frac * self.dome_radius
        if r > soft:
            # Steer toward the origin, scaled by how far past the soft boundary
            # we are, so the pushback grows smoothly toward the hard edge.
            desired = np.arctan2(-y, -x)
            err = _wrap_to_pi(desired - self._theta)
            urgency = min(1.0, (r - soft) / max(soft, 1e-6))
            omega = self.boundary_gain * err * (0.5 + urgency)
        else:
            omega = self._ou.step()
        return float(np.clip(omega, -self.turn_rate_max, self.turn_rate_max))

    def step(self) -> np.ndarray:
        omega = self._turn_rate()
        self._theta = _wrap_to_pi(self._theta + omega * self.dt)
        vx = self.speed * np.cos(self._theta)
        vy = self.speed * np.sin(self._theta)
        self._pos = self._pos + np.array([vx * self.dt, vy * self.dt, 0.0], dtype=np.float32)
        self._pos[2] = self.altitude
        self._vel = np.array([vx, vy, 0.0], dtype=np.float32)
        return self.position


class DubinsAirplane3D(DubinsCar2D):
    """Dubins airplane: a Dubins car plus a bounded, wandering vertical rate.

    Altitude wanders within ``[altitude_min, altitude_max]`` via a second OU
    process on vertical velocity, with soft steering back toward the band centre
    near the altitude limits.

    Attributes:
        climb_rate_max: Maximum |vertical speed| in m/s.
        altitude_min / altitude_max: Altitude band in meters.
    """

    def __init__(
        self,
        dome_size: float,
        speed: float,
        dt: float,
        turn_rate_max: float = 0.8,
        ou_theta: float = 0.5,
        ou_sigma: float = 1.5,
        spawn_radius_frac: float = 0.5,
        boundary_frac: float = 0.8,
        boundary_gain: float = 3.0,
        climb_rate_max: float = 1.0,
        altitude_min: float = 1.5,
        altitude_max: float = 6.0,
        vertical_ou_theta: float = 0.5,
        vertical_ou_sigma: float = 1.0,
    ) -> None:
        mid_alt = 0.5 * (altitude_min + altitude_max)
        super().__init__(
            dome_size=dome_size,
            speed=speed,
            dt=dt,
            turn_rate_max=turn_rate_max,
            ou_theta=ou_theta,
            ou_sigma=ou_sigma,
            altitude=mid_alt,
            spawn_radius_frac=spawn_radius_frac,
            boundary_frac=boundary_frac,
            boundary_gain=boundary_gain,
        )
        self.climb_rate_max = climb_rate_max
        self.altitude_min = altitude_min
        self.altitude_max = altitude_max
        self._vz = 0.0
        self._vou = _ScalarOU(theta=vertical_ou_theta, sigma=vertical_ou_sigma, dt=dt)

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        super().reset(rng)
        self._vou.reset(rng)
        z0 = float(rng.uniform(self.altitude_min, self.altitude_max))
        self._pos[2] = z0
        self._vz = 0.0
        return self.position

    def _vertical_rate(self) -> float:
        """Vertical speed this step: steer back inside the band, else OU noise."""
        z = float(self._pos[2])
        mid = self.altitude
        band = 0.5 * (self.altitude_max - self.altitude_min)
        soft = 0.7 * band
        if abs(z - mid) > soft:
            # Push back toward the band centre.
            vz = -np.sign(z - mid) * self.climb_rate_max
        else:
            vz = self._vou.step()
        return float(np.clip(vz, -self.climb_rate_max, self.climb_rate_max))

    def step(self) -> np.ndarray:
        omega = self._turn_rate()
        self._theta = _wrap_to_pi(self._theta + omega * self.dt)
        vx = self.speed * np.cos(self._theta)
        vy = self.speed * np.sin(self._theta)
        vz = self._vertical_rate()
        self._pos = self._pos + np.array(
            [vx * self.dt, vy * self.dt, vz * self.dt], dtype=np.float32
        )
        self._pos[2] = float(np.clip(self._pos[2], self.altitude_min, self.altitude_max))
        self._vel = np.array([vx, vy, vz], dtype=np.float32)
        return self.position


def make_target(config, dt: float) -> MovingTarget:
    """Construct a moving target from an :class:`~envs.core.config.EnvConfig`.

    Args:
        config: Env config carrying ``target_*`` fields.
        dt: Integration timestep in seconds (typically ``1 / agent_hz``).

    Returns:
        A :class:`MovingTarget` matching ``config.target_mode``.

    Raises:
        ValueError: If ``config.target_mode`` is unknown.
    """
    mode = getattr(config, "target_mode", "car2d").lower()
    dome = config.flight_dome_size
    if mode == "static":
        return StaticTarget(
            dome_size=dome,
            dt=dt,
            spawn_radius_frac=config.target_spawn_radius_frac,
            altitude=config.target_altitude,
        )
    if mode == "car2d":
        return DubinsCar2D(
            dome_size=dome,
            speed=config.target_speed,
            dt=dt,
            turn_rate_max=config.target_turn_rate_max,
            ou_theta=config.target_ou_theta,
            ou_sigma=config.target_ou_sigma,
            altitude=config.target_altitude,
            spawn_radius_frac=config.target_spawn_radius_frac,
            boundary_frac=config.target_boundary_frac,
        )
    if mode == "airplane3d":
        return DubinsAirplane3D(
            dome_size=dome,
            speed=config.target_speed,
            dt=dt,
            turn_rate_max=config.target_turn_rate_max,
            ou_theta=config.target_ou_theta,
            ou_sigma=config.target_ou_sigma,
            spawn_radius_frac=config.target_spawn_radius_frac,
            boundary_frac=config.target_boundary_frac,
            climb_rate_max=config.target_climb_rate_max,
            altitude_min=config.target_altitude_min,
            altitude_max=config.target_altitude_max,
        )
    raise ValueError(
        f"Unknown target_mode {mode!r}; expected 'static', 'car2d', or 'airplane3d'."
    )
