"""Drone RL environments package.

Registers all Gymnasium environments. Entry points are import strings resolved
lazily at ``gym.make`` time, so importing this package is cheap and does not
pull in heavy backends.
"""

from gymnasium.envs.registration import register

# -- Multi-waypoint inspection task (original) ---------------------------------

register(
    id="DroneInspection-v0",
    entry_point="envs.tasks.drone_inspection_env:DroneInspectionEnv",
    kwargs={"config_path": "configs/inspection/easy.yaml"},
)

register(
    id="DroneInspection-medium-v0",
    entry_point="envs.tasks.drone_inspection_env:DroneInspectionEnv",
    kwargs={"config_path": "configs/inspection/medium.yaml"},
)

register(
    id="DroneInspection-sheeprl-v0",
    entry_point="envs.sheeprl_wrapper:make_drone_inspection_env",
    kwargs={"config_path": "configs/inspection/easy.yaml"},
)

# -- Moving-target interception task (Dubins) ----------------------------------

register(
    id="DroneTarget-sheeprl-v0",
    entry_point="envs.sheeprl_wrapper:make_drone_target_env",
    kwargs={"config_path": "configs/target/l0_smoke.yaml"},
)

# -- Two-drone chase task (target is a real drone on a waypoint plan) ----------

register(
    id="DroneChase-sheeprl-v0",
    entry_point="envs.sheeprl_wrapper:make_drone_chase_env",
    kwargs={"config_path": "configs/target/chase.yaml"},
)
