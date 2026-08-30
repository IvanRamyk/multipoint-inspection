"""Ornstein-Uhlenbeck wind model for stochastic wind simulation."""

from __future__ import annotations

import numpy as np


class OUWindModel:
    """3D Ornstein-Uhlenbeck wind process.

    Generates temporally correlated wind vectors that are bounded and
    physically plausible. The process mean-reverts to zero.

    Attributes:
        theta: Mean reversion rate (higher = faster reversion to zero).
        sigma: Volatility (higher = stronger gusts).
        strength: Maximum wind magnitude as fraction of drone max speed.
        dt: Timestep in seconds.
    """

    def __init__(
        self,
        theta: float = 0.15,
        sigma: float = 0.2,
        strength: float = 0.25,
        dt: float = 1.0 / 30.0,
    ) -> None:
        self.theta = theta
        self.sigma = sigma
        self.strength = strength
        self.dt = dt
        self._wind = np.zeros(3, dtype=np.float32)
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int | None = None) -> None:
        """Reset wind to zero and optionally reseed RNG.

        Args:
            seed: Random seed for reproducibility.
        """
        self._rng = np.random.default_rng(seed)
        self._wind = np.zeros(3, dtype=np.float32)

    def step(self) -> np.ndarray:
        """Advance one timestep and return the current wind vector.

        Returns:
            (3,) wind vector in arbitrary force units, scaled by strength.
        """
        if self._rng is None:
            self._rng = np.random.default_rng()

        noise = self._rng.standard_normal(3).astype(np.float32)
        self._wind += (
            self.theta * (0.0 - self._wind) * self.dt
            + self.sigma * np.sqrt(self.dt) * noise
        )
        # Clamp magnitude to strength
        mag = np.linalg.norm(self._wind)
        if mag > self.strength:
            self._wind = self._wind * (self.strength / mag)

        return self._wind.copy()

    @property
    def current(self) -> np.ndarray:
        """Current wind vector without advancing the process."""
        return self._wind.copy()
