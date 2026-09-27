"""Two-drone PyFlyt backend for the pursuit/chase task.

Extends :class:`PyFlytBackend` to spawn a SECOND QuadX drone (the target) with
identical physics. Drone index 0 is the pursuer (agent-controlled, reuses the
parent's depth camera, collision check, wind bias); index 1 is the target,
driven by velocity setpoints the environment computes from a waypoint plan.

The pursuer's collision check (inherited) only flags contact with the ground
plane and obstacles — never the target drone — so "catching" the target is an
intentional close approach, not a crash.
"""

from __future__ import annotations

import numpy as np

from envs.core.sim_backend import DroneState
from envs.backends.pyflyt_backend import PyFlytBackend, _MAX_SPEED


class PyFlytChaseBackend(PyFlytBackend):
    """PyFlyt backend with a pursuer (idx 0) and a physical target drone (idx 1)."""

    def __init__(self, *args, drone_model: str = "cf2x", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._drone_model = drone_model
        self._target_id: int | None = None
        self._target_marker_ids: list[int] = []

    def reset(  # type: ignore[override]
        self,
        drone_start: np.ndarray,
        target_start: np.ndarray,
        obstacles: list[dict],
        waypoints: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        """Reset with two drones.

        Args:
            drone_start: (3,) pursuer start position.
            target_start: (3,) target start position.
            obstacles: list of {'position','size'} boxes.
            waypoints: optional (K,3) target route, spawned as visual markers.
            seed: optional deterministic seed.
        """
        from PyFlyt.core import Aviary

        if self._aviary is not None:
            self._aviary.disconnect()

        start_pos = np.array([drone_start, target_start], dtype=np.float64)
        start_orn = np.zeros((2, 3), dtype=np.float64)

        # Both drones use the same model. drone_model selects the QuadX airframe
        # (e.g. "cf2x" nano vs "primitive_drone" ~45 cm); resolved by PyFlyt from
        # its models/vehicles/ dir. Passed per-drone via drone_options.
        drone_options = [{"drone_model": self._drone_model}, {"drone_model": self._drone_model}]

        self._aviary = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            drone_type="quadx",
            drone_options=drone_options,
            render=self._render,
            physics_hz=240,
            seed=seed,
        )
        control_hz = 120  # PyFlyt default
        self._steps_per_agent_step = control_hz // self.agent_hz

        self._drone_id = self._aviary.drones[0].Id      # pursuer
        self._target_id = self._aviary.drones[1].Id     # target

        # Velocity control (mode 6: vx, vy, yaw_rate, vz) for BOTH drones.
        self._aviary.set_mode(6)
        hover = np.array([0.0, 0.0, 0.0, 0.0])
        self._aviary.set_setpoint(0, hover)
        self._aviary.set_setpoint(1, hover)
        for _ in range(10):
            self._aviary.step()

        # Obstacles.
        self._obstacle_ids = []
        for obs in obstacles:
            oid = self._spawn_box(
                position=obs["position"],
                half_extents=[obs["size"] / 2.0] * 3,
                color=[0.5, 0.5, 0.5, 1.0],
            )
            self._obstacle_ids.append(oid)

        # Optional route markers (visual only) for the target's plan.
        self._target_marker_ids = []
        if waypoints is not None:
            for wp in waypoints:
                self._target_marker_ids.append(
                    self._spawn_sphere(position=wp, radius=0.25, color=[1.0, 0.6, 0.0, 0.4])
                )

        self._aviary.register_all_new_bodies()

        self._proj_matrix = list(
            self._aviary.computeProjectionMatrixFOV(
                fov=90.0,
                aspect=self.image_width / self.image_height,
                nearVal=0.1,
                farVal=100.0,
            )
        )
        self._last_cmd_vel = None

    def get_target_state(self) -> DroneState:
        """Return the target drone's state (index 1). No collision flag."""
        av = self._aviary
        state = av.state(1)
        pos = state[3].copy()
        vel_body = state[2].copy()
        orn_euler = state[1]
        quat = np.array(av.getQuaternionFromEuler(orn_euler.tolist()), dtype=np.float32)
        rot = np.array(av.getMatrixFromQuaternion(quat.tolist())).reshape(3, 3)
        vel_world = rot @ vel_body
        return DroneState(
            position=pos.astype(np.float32),
            velocity=vel_world.astype(np.float32),
            orientation=quat,
            collision=False,
        )

    def set_target_action(self, velocity: np.ndarray) -> None:
        """Command the target drone a desired velocity (each component in [-1, 1])."""
        vx = float(velocity[0]) * _MAX_SPEED
        vy = float(velocity[1]) * _MAX_SPEED
        vz = float(velocity[2]) * _MAX_SPEED
        self._aviary.set_setpoint(1, np.array([vx, vy, 0.0, vz]))
