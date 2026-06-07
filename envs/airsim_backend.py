"""Cosys-AirSim / AirSim backend for drone simulation (Phase 4).

Implements the same ``SimBackend`` contract as ``PyFlytBackend`` so the
Gymnasium env can swap simulators with no other code changes.

Platform
--------
AirSim / Cosys-AirSim runs inside Unreal Engine and is **not available on
macOS**. Use a Linux or Windows host with a built UE5 environment. The
``airsim`` (or ``cosysairsim``) Python package is imported lazily in
``__init__`` so this module can be imported anywhere without the package.

Coordinate frames
------------------
AirSim uses **NED** (North-East-Down): +Z points *down*. The rest of this
project uses an ENU-style Z-up frame (matching PyFlyt). Every position /
velocity crossing the backend boundary is converted with ``_ned_to_enu`` /
``_enu_to_ned`` (flip Y and Z). Getting this wrong makes the drone "fall
upward" — it is the single most common AirSim integration bug.

Required ``settings.json``
--------------------------
Place at ``~/Documents/AirSim/settings.json`` (Linux: ``~/Documents/AirSim``)::

    {
      "SettingsVersion": 2.0,
      "SimMode": "Multirotor",
      "ClockSpeed": 5.0,
      "Vehicles": {
        "drone": {
          "VehicleType": "SimpleFlight",
          "Cameras": {
            "front": {
              "CaptureSettings": [
                {"ImageType": 1, "Width": 64, "Height": 64, "FOV_Degrees": 90}
              ],
              "X": 0.30, "Y": 0.0, "Z": 0.0,
              "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
            }
          }
        }
      }
    }

``ImageType: 1`` is ``DepthPlanar``. ``ClockSpeed`` > 1 speeds up the sim;
stepping is done deterministically via pause + ``simContinueForTime``.
"""

from __future__ import annotations

import numpy as np

from envs.sim_backend import DroneState, SimBackend


# Maximum drone speed in m/s, used to scale [-1, 1] actions. Matches PyFlyt.
_MAX_SPEED = 5.0

# Depth clip range in meters (observation space upper bound is 100).
_DEPTH_MAX = 100.0


def _ned_to_enu(vec) -> np.ndarray:
    """Convert an AirSim NED Vector3r to the project's Z-up ENU frame."""
    return np.array([vec.x_val, -vec.y_val, -vec.z_val], dtype=np.float32)


def _enu_to_ned(xyz) -> tuple[float, float, float]:
    """Convert a Z-up ENU (x, y, z) to AirSim NED (x, y, z)."""
    return float(xyz[0]), -float(xyz[1]), -float(xyz[2])


class AirSimBackend(SimBackend):
    """Simulator backend using AirSim / Cosys-AirSim for drone physics.

    Drives a single ``SimpleFlight`` multirotor over AirSim's RPC API. The
    sim is paused between agent steps and advanced a fixed ``dt`` with
    ``simContinueForTime`` for deterministic, framework-independent stepping.

    Attributes:
        image_width: Depth image width in pixels.
        image_height: Depth image height in pixels.
        agent_hz: Control frequency from the RL agent's perspective.
        camera_name: Name of the depth camera as declared in settings.json.
        vehicle_name: Vehicle name as declared in settings.json.
    """

    def __init__(
        self,
        image_width: int = 64,
        image_height: int = 64,
        agent_hz: int = 30,
        camera_name: str = "front",
        vehicle_name: str = "drone",
    ) -> None:
        # Lazy import: airsim is unavailable on macOS / without UE.
        try:
            import airsim  # type: ignore
        except ImportError:  # pragma: no cover - exercised only on UE hosts
            try:
                import cosysairsim as airsim  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "AirSimBackend requires the 'airsim' or 'cosysairsim' "
                    "Python package, available only on a Linux/Windows host "
                    "with Unreal Engine. Install with `pip install airsim` "
                    "(or cosysairsim) on the UE machine."
                ) from exc

        self._airsim = airsim
        self.image_width = image_width
        self.image_height = image_height
        self.agent_hz = agent_hz
        self.camera_name = camera_name
        self.vehicle_name = vehicle_name
        self._dt = 1.0 / agent_hz

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # One reusable depth-image request.
        self._depth_request = airsim.ImageRequest(
            self.camera_name,
            airsim.ImageType.DepthPlanar,
            pixels_as_float=True,
            compress=False,
        )

    def reset(
        self,
        drone_start: np.ndarray,
        waypoints: np.ndarray,
        obstacles: list[dict],
        seed: int | None = None,
    ) -> None:
        """Reset simulation with given configuration.

        Args:
            drone_start: (3,) starting position in meters (ENU, Z-up).
            waypoints: (N, 3) waypoint positions (logical only; no physical
                markers are needed for training).
            obstacles: List of obstacle dicts. Dynamic spawning requires UE
                assets (Cosys ``simSpawnObject``); left to the UE level for
                v1 — obstacles baked into the map are picked up by collision.
            seed: Unused by AirSim (physics is deterministic given commands).
        """
        airsim = self._airsim

        self.client.reset()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)

        # Teleport to the start pose (ENU -> NED).
        nx, ny, nz = _enu_to_ned(drone_start)
        pose = airsim.Pose(
            airsim.Vector3r(nx, ny, nz),
            airsim.to_quaternion(0.0, 0.0, 0.0),
        )
        self.client.simSetVehiclePose(pose, ignore_collision=True,
                                      vehicle_name=self.vehicle_name)

        # Pause so steps advance deterministically via simContinueForTime.
        self.client.simPause(True)

    def get_depth_image(self) -> np.ndarray:
        """Return (H, W, 1) float32 depth image in meters."""
        responses = self.client.simGetImages(
            [self._depth_request], vehicle_name=self.vehicle_name
        )
        resp = responses[0]
        depth = np.array(resp.image_data_float, dtype=np.float32)
        if depth.size != resp.height * resp.width:
            return np.zeros((self.image_height, self.image_width, 1),
                            dtype=np.float32)
        depth = depth.reshape(resp.height, resp.width, 1)
        return np.clip(depth, 0.0, _DEPTH_MAX)

    def get_drone_state(self) -> DroneState:
        """Return current drone state (converted to ENU, Z-up)."""
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        kin = state.kinematics_estimated

        position = _ned_to_enu(kin.position)
        velocity = _ned_to_enu(kin.linear_velocity)

        # Orientation quaternion (x, y, z, w). NED->ENU sign flip on y,z.
        o = kin.orientation
        orientation = np.array(
            [o.x_val, -o.y_val, -o.z_val, o.w_val], dtype=np.float32
        )

        collision = self.client.simGetCollisionInfo(
            vehicle_name=self.vehicle_name
        ).has_collided

        return DroneState(
            position=position,
            velocity=velocity,
            orientation=orientation,
            collision=bool(collision),
        )

    def apply_action(self, velocity: np.ndarray) -> None:
        """Apply desired velocity (3,) to drone.

        Args:
            velocity: Desired velocity, each component in [-1, 1]. Mapped to
                (vx, vy, vz) in m/s and converted ENU -> NED.
        """
        vx = float(velocity[0]) * _MAX_SPEED
        vy = float(velocity[1]) * _MAX_SPEED
        vz = float(velocity[2]) * _MAX_SPEED
        nvx, nvy, nvz = _enu_to_ned((vx, vy, vz))
        self.client.moveByVelocityAsync(
            nvx, nvy, nvz, duration=self._dt, vehicle_name=self.vehicle_name
        )

    def step_simulation(self) -> None:
        """Advance the (paused) simulation by one agent timestep."""
        self.client.simContinueForTime(self._dt)

    def apply_wind(self, wind_vec: np.ndarray) -> None:
        """Set global wind. Interprets the raw OU vector as a velocity (m/s).

        Args:
            wind_vec: (3,) raw OU wind vector in ENU; converted to NED.
        """
        wx, wy, wz = _enu_to_ned(wind_vec)
        self.client.simSetWind(self._airsim.Vector3r(wx, wy, wz))

    def close(self) -> None:
        """Release control and unpause the simulator."""
        try:
            self.client.simPause(False)
            self.client.armDisarm(False, self.vehicle_name)
            self.client.enableApiControl(False, self.vehicle_name)
        except Exception:
            pass
