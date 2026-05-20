"""Drone inspection environment package."""

from gymnasium.envs.registration import register

register(
    id="DroneInspection-v0",
    entry_point="envs.drone_inspection_env:DroneInspectionEnv",
    kwargs={"config_path": "configs/easy.yaml"},
)

register(
    id="DroneInspection-medium-v0",
    entry_point="envs.drone_inspection_env:DroneInspectionEnv",
    kwargs={"config_path": "configs/medium.yaml"},
)

register(
    id="DroneInspection-sheeprl-v0",
    entry_point="envs.sheeprl_wrapper:make_drone_inspection_env",
    kwargs={"config_path": "configs/easy.yaml"},
)
