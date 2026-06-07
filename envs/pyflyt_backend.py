"""PyFlyt/PyBullet backend for drone simulation."""

from __future__ import annotations

import numpy as np

from envs.sim_backend import DroneState, SimBackend


# Maximum drone speed in m/s, used to scale [-1, 1] actions.
_MAX_SPEED = 5.0

# Depth camera parameters.
_CAM_NEAR = 0.1
_CAM_FAR = 100.0

# Wind force scaling (Newtons per unit of OU wind output). Lives here so the
# raw OU vector handed in by the env is interpreted in PyFlyt's own units;
# net force is identical to the previous (env-side scaled) behavior.
_WIND_FORCE_SCALE = 10.0


class PyFlytBackend(SimBackend):
    """Simulator backend using PyFlyt (PyBullet) for drone physics.

    Uses the PyFlyt Aviary to manage a single QuadX drone with cascaded
    PID controllers. Obstacles are spawned as PyBullet collision bodies.
    Depth images are rendered with the CPU-based TinyRenderer.

    Attributes:
        image_width: Depth image width in pixels.
        image_height: Depth image height in pixels.
        agent_hz: Control frequency from the RL agent's perspective.
    """

    def __init__(
        self,
        image_width: int = 64,
        image_height: int = 64,
        agent_hz: int = 30,
        render: bool = False,
    ) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.agent_hz = agent_hz
        self._render = render

        self._aviary = None
        self._drone_id: int | None = None
        self._obstacle_ids: list[int] = []
        self._waypoint_ids: list[int] = []
        self._steps_per_agent_step: int = 1

        # Precompute projection matrix (stays constant).
        self._proj_matrix: list[float] | None = None

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
            obstacles: List of dicts with 'position' (3,), 'size' (float).
            seed: Optional random seed for deterministic simulation.
        """
        from PyFlyt.core import Aviary

        # Tear down previous simulation if any.
        if self._aviary is not None:
            self._aviary.disconnect()

        start_pos = np.array([drone_start], dtype=np.float64)
        start_orn = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        self._aviary = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            drone_type="quadx",
            render=self._render,
            physics_hz=240,
            seed=seed,
        )

        # Figure out how many aviary steps per agent step.
        # Aviary default control_hz is 120, physics_hz is 240.
        control_hz = 120  # PyFlyt default
        self._steps_per_agent_step = control_hz // self.agent_hz

        self._drone_id = self._aviary.drones[0].Id

        # Set velocity control mode: vx, vy, yaw_rate, vz
        self._aviary.set_mode(6)

        # Hover setpoint to stabilise before first real action.
        self._aviary.set_setpoint(0, np.array([0.0, 0.0, 0.0, 0.0]))

        # Let the drone stabilise for a short time.
        for _ in range(10):
            self._aviary.step()

        # Spawn obstacles.
        self._obstacle_ids = []
        for obs in obstacles:
            oid = self._spawn_box(
                position=obs["position"],
                half_extents=[obs["size"] / 2.0] * 3,
                color=[0.5, 0.5, 0.5, 1.0],
            )
            self._obstacle_ids.append(oid)

        # Spawn waypoint markers.
        self._waypoint_ids = []
        for wp in waypoints:
            wid = self._spawn_sphere(
                position=wp,
                radius=0.3,
                color=[0.0, 1.0, 0.0, 0.6],
            )
            self._waypoint_ids.append(wid)

        # Register new bodies so contact_array covers obstacles.
        self._aviary.register_all_new_bodies()

        # Precompute projection matrix.
        self._proj_matrix = list(
            self._aviary.computeProjectionMatrixFOV(
                fov=90.0,
                aspect=self.image_width / self.image_height,
                nearVal=_CAM_NEAR,
                farVal=_CAM_FAR,
            )
        )

    def get_depth_image(self) -> np.ndarray:
        """Return (H, W, 1) float32 depth image in meters."""
        av = self._aviary
        state = av.state(0)
        pos = state[3]  # ground-frame position
        orn_euler = state[1]  # roll, pitch, yaw

        # Compute forward and up vectors from drone orientation.
        quat = av.getQuaternionFromEuler(orn_euler.tolist())
        rot = np.array(av.getMatrixFromQuaternion(quat)).reshape(3, 3)
        forward = rot @ np.array([1.0, 0.0, 0.0])
        up = rot @ np.array([0.0, 0.0, 1.0])

        target = pos + forward
        view_matrix = list(
            av.computeViewMatrix(
                cameraEyePosition=pos.tolist(),
                cameraTargetPosition=target.tolist(),
                cameraUpVector=up.tolist(),
            )
        )

        _, _, _, depth_buffer, _ = av.getCameraImage(
            width=self.image_width,
            height=self.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=av.ER_TINY_RENDERER,
        )

        # Convert non-linear depth buffer to linear depth in meters.
        depth_buffer = np.array(depth_buffer, dtype=np.float32)
        depth_meters = (
            _CAM_FAR * _CAM_NEAR / (_CAM_FAR - (_CAM_FAR - _CAM_NEAR) * depth_buffer)
        )
        return depth_meters.reshape(self.image_height, self.image_width, 1)

    def get_drone_state(self) -> DroneState:
        """Return current drone state."""
        av = self._aviary
        state = av.state(0)
        pos = state[3].copy()
        vel = state[2].copy()  # body-frame velocity
        orn_euler = state[1]
        quat = np.array(
            av.getQuaternionFromEuler(orn_euler.tolist()), dtype=np.float32
        )

        # Convert body-frame velocity to world frame.
        rot = np.array(av.getMatrixFromQuaternion(quat.tolist())).reshape(3, 3)
        vel_world = rot @ vel

        # Check collisions: drone vs ground or any obstacle.
        collision = False
        if av.contact_array is not None:
            drone_id = self._drone_id
            # Check ground.
            if av.contact_array[drone_id][av.planeId]:
                collision = True
            # Check obstacles.
            for oid in self._obstacle_ids:
                if av.contact_array[drone_id][oid]:
                    collision = True
                    break

        return DroneState(
            position=pos.astype(np.float32),
            velocity=vel_world.astype(np.float32),
            orientation=quat,
            collision=collision,
        )

    def apply_action(self, velocity: np.ndarray) -> None:
        """Apply desired velocity (3,) to drone.

        Args:
            velocity: Desired velocity vector, each component in [-1, 1].
                      Mapped to (vx, vy, vz) in ground frame.
        """
        vx = float(velocity[0]) * _MAX_SPEED
        vy = float(velocity[1]) * _MAX_SPEED
        vz = float(velocity[2]) * _MAX_SPEED
        # Mode 6 setpoint: (vx, vy, yaw_rate, vz)
        self._aviary.set_setpoint(0, np.array([vx, vy, 0.0, vz]))

    def step_simulation(self) -> None:
        """Advance simulation by one agent timestep (1/agent_hz seconds)."""
        for _ in range(self._steps_per_agent_step):
            self._aviary.step()

    def apply_wind(self, wind_vec: np.ndarray) -> None:
        """Apply an external wind force to the drone.

        Args:
            wind_vec: (3,) raw OU wind vector, world frame. Scaled to a
                force in Newtons internally by ``_WIND_FORCE_SCALE``.
        """
        wind_force = np.asarray(wind_vec, dtype=np.float64) * _WIND_FORCE_SCALE
        self._aviary.applyExternalForce(
            objectUniqueId=self._drone_id,
            linkIndex=-1,
            forceObj=wind_force.tolist(),
            posObj=[0.0, 0.0, 0.0],
            flags=self._aviary.WORLD_FRAME,
        )

    def set_waypoint_color(self, index: int, color: list[float]) -> None:
        """Change the visual color of a waypoint marker.

        Args:
            index: Waypoint index.
            color: RGBA color list.
        """
        if index < len(self._waypoint_ids):
            self._aviary.changeVisualShape(
                self._waypoint_ids[index], -1, rgbaColor=color
            )

    def close(self) -> None:
        """Disconnect the simulation."""
        if self._aviary is not None:
            try:
                self._aviary.disconnect()
            except Exception:
                pass
            self._aviary = None

    # -- Private helpers --------------------------------------------------

    def _spawn_box(
        self,
        position: np.ndarray,
        half_extents: list[float],
        color: list[float],
    ) -> int:
        """Spawn a static box obstacle and return its body id."""
        av = self._aviary
        col = av.createCollisionShape(
            shapeType=av.GEOM_BOX, halfExtents=half_extents
        )
        vis = av.createVisualShape(
            shapeType=av.GEOM_BOX, halfExtents=half_extents, rgbaColor=color
        )
        body_id = av.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=list(position),
        )
        return body_id

    def _spawn_sphere(
        self,
        position: np.ndarray,
        radius: float,
        color: list[float],
    ) -> int:
        """Spawn a visual-only sphere marker and return its body id."""
        av = self._aviary
        vis = av.createVisualShape(
            shapeType=av.GEOM_SPHERE, radius=radius, rgbaColor=color
        )
        body_id = av.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=list(position),
        )
        return body_id
