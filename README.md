# DreamerV3 for Multi-Waypoint Drone Inspection

A model-based reinforcement learning agent that learns to plan and execute multi-waypoint inspection missions with a quadrotor — combining combinatorial route planning, continuous flight control, depth-based obstacle avoidance, and adaptation to stochastic wind, all in a single learned policy.

This is **not** drone racing or trajectory following. The agent must decide the *order* in which to visit N waypoints, navigate in 3D using a depth camera, deal with wind disturbances, and return to base — essentially a **Travelling Salesman Problem with continuous dynamics, uncertainty, and physical constraints**.

## Why DreamerV3

[DreamerV3](https://arxiv.org/abs/2301.04104) (Hafner et al., 2023) is a model-based RL algorithm with three pieces:

1. Learn a **world model** from collected experience.
2. **Dream** — roll the policy out through the learned model for many imagined trajectories.
3. Optimize the policy on those imagined trajectories instead of real environment steps.

For this task this matters because:

| Property | Model-free (PPO / SAC) | DreamerV3 |
|---|---|---|
| Sample efficiency | Millions of real env steps | 10–50× fewer real steps |
| Planning horizon | 1-step value bootstrapping | Imagines 15 steps ahead |
| Multi-modal obs | Treated as flat input | CNN (depth) + MLP (state) fused in latent space |
| Sparse rewards | Struggles | World model propagates signal through imagination |

We use [`sheeprl`](https://github.com/Eclectic-Sheep/sheeprl), a PyTorch implementation of DreamerV3 with native Gymnasium integration.

## Architecture

### Observation space (multimodal, Gymnasium Dict)

| Component | Shape | Encoded by |
|---|---|---|
| Depth image | `(64, 64, 1)` | CNN |
| State vector — drone position, velocity, wind estimate, base-relative position, per-waypoint relative position + visited flag | `(12 + 4N,)` | MLP |

Both modalities are encoded separately, then fused in the world model's recurrent latent state.

### Action space

Continuous 3D desired-velocity vector in `[-1, 1]^3`. A low-level cascaded PID controller (provided by the simulator) translates this to motor commands. The RL agent learns *where to fly*, not *how to spin propellers*.

### Reward function

```
reward = -0.01                                          # time penalty (per step)
       + 10.0   if drone enters an unvisited waypoint   # sparse
       + 50.0   if all visited AND back at base         # sparse, terminates
       + config.collision_penalty   on collision        # configurable, terminates
       + config.reward_shaping × Δdistance_to_target    # optional dense shaping
```

`reward_shaping` (potential-based, telescoping) and `collision_penalty` are knobs in `EnvConfig` for taming sparse-reward exploration on small-scale debug tasks.

### Simulator layer

The Gymnasium env talks to a `SimBackend` abstract interface, currently implemented by **PyFlyt** (PyBullet physics + depth camera, ~1000–5000 env steps/sec headless). A `Cosys-AirSim` backend (Unreal Engine 5) is planned for Phase 4 photorealistic validation; same interface, same agent code.

## Installation

Requires Python 3.10+. Tested on macOS and Linux.

```bash
git clone <repo>
cd dreamer
python3 -m venv venv
source venv/bin/activate
pip install -e .              # base env (PyFlyt, gymnasium, numpy, matplotlib)
pip install -e ".[dreamer]"   # adds sheeprl + torch + tensorboard for training/eval
```

## Quickstart

### Sanity-check the environment (random agent)

```bash
python scripts/test_env.py --config configs/easy.yaml --episodes 5
python scripts/test_env.py --config configs/easy.yaml --episodes 1 --render  # PyBullet GUI
```

### Train

```bash
# CPU debug (small model, slow but works on any machine)
python scripts/train_dreamer.py algo.total_steps=15000

# GPU (recommended for real experiments)
python scripts/train_dreamer.py \
    fabric.accelerator=gpu \
    algo=dreamer_v3_S \
    algo.total_steps=5000000 \
    env.num_envs=4
```

Hydra-style CLI overrides flow through to sheeprl. To switch which env config gets used by training, edit the `config_path:` line in `configs/sheeprl/env/drone_inspection.yaml`.

### Watch a trained agent in the PyBullet GUI

```bash
python scripts/eval_dreamer.py <checkpoint.ckpt> \
    --config configs/sanity.yaml \
    --episodes 5 --render --greedy
```

After the run, per-episode top-down and 3D route plots land in `results/`. The reach-radius spheres around each waypoint and around base are drawn so you can see exactly when the success conditions fire.

### Monitor training

```bash
tensorboard --logdir logs/
```

Useful scalar tags: `Rewards/rew_avg`, `Game/ep_len_avg`, `Loss/world_model_loss`, `Loss/policy_loss`, `Loss/value_loss`, `Time/sps_train`, `Params/replay_ratio`.

## Repository layout

```
envs/
  config.py                  EnvConfig dataclass (YAML-loadable). All env knobs.
  drone_inspection_env.py    Gymnasium env: reward, termination, waypoint generation.
  pyflyt_backend.py          PyBullet physics, depth camera, wind forces, collision.
  sim_backend.py             Abstract SimBackend interface (for AirSim swap).
  sheeprl_wrapper.py         Adapter to sheeprl's Dict-obs / CNN+MLP routing.
  wind_model.py              Ornstein–Uhlenbeck stochastic wind.

configs/
  sanity.yaml                N=1 minimal debug task (the convergence proof).
  easy.yaml / medium.yaml / hard.yaml   Progressively harder curriculum.
  sheeprl/exp/drone_inspection.yaml     Algo + Fabric + buffer + metric config.
  sheeprl/env/drone_inspection.yaml     Points sheeprl wrapper at one of the configs above.

scripts/
  train_dreamer.py           Launches sheeprl training with project configs.
  eval_dreamer.py            Loads a checkpoint, runs episodes, optional GUI.
  smoke_test_dreamer.py      End-to-end pipeline validation.
  test_env.py                Random-agent baseline.

eval/
  metrics.py                 Per-episode metrics.
  route_visualizer.py        Matplotlib top-down + 3D route plots with reach spheres.

logs/runs/dreamer_v3/...     Training run dirs (TensorBoard events + checkpoints).
```

## Sanity training result (Phase 2 deliverable)

The `configs/sanity.yaml` recipe — N=1 waypoint, 5m dome, dense reward shaping — converges on a laptop CPU in ~8 hours:

| Metric | Value |
|---|---|
| Total env steps | 15,000 |
| Final `Rewards/rew_avg` | +60 (≈ theoretical max) |
| Final `Game/ep_len_avg` | 17–27 steps (mission completed in <1 second of simulated flight) |
| Final `Loss/world_model_loss` | 7.6 (from ≈2,200 at the start) |
| 5/5 eval success rate | rewards 59.95–60.71 |

This proves the full pipeline (env + world model + actor-critic + sheeprl integration) learns end-to-end. The next phase scales it up.

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 — Environment | done | Gymnasium env, PyFlyt backend, depth rendering, wind, random-agent baseline. |
| 2 — DreamerV3 integration | done | Multimodal obs routed through CNN + MLP encoders into the RSSM; sanity task converges. |
| 3 — Full experiments | planned | GPU runs on easy / medium / hard curriculum, ablations (depth on/off, wind on/off, N=3/5/7/10), wandb logging. |
| 4 — Sim-to-sim | planned | Implement `AirSimBackend` against the same `SimBackend` interface; validate in Unreal Engine 5.5; paper. |

## References

- Hafner et al., **"Mastering Diverse Domains through World Models"** (DreamerV3), 2023. <https://arxiv.org/abs/2301.04104>
- [`sheeprl`](https://github.com/Eclectic-Sheep/sheeprl) — PyTorch implementation of DreamerV3 and other algorithms used here.
- [`PyFlyt`](https://github.com/jjshoots/PyFlyt) — PyBullet-based drone simulation.
- [Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim) — Unreal Engine drone simulation (planned Phase 4 backend).
