# AGENTS.md — DreamerV3 for Multi-Waypoint Drone Inspection

## What this project is

Research project: a **DreamerV3** (model-based RL with learned world models) agent that solves a multi-waypoint drone inspection mission — a TSP with continuous 3D flight, depth-camera perception, stochastic wind, and obstacle avoidance. The novelty pitch is "combinatorial planning + low-level control jointly learned in one policy" — not drone racing, not trajectory following.

- **Sim**: PyFlyt (PyBullet physics, depth camera, PID-controlled QuadX drone). Cosys-AirSim (UE5) is planned for Phase 4 photorealistic validation, not yet implemented.
- **Algo**: DreamerV3 via `sheeprl` v0.5.7 (PyTorch). XS model for CPU dev; S/M planned for GPU.
- **Obs space (Dict)**: `depth: (64,64,1) float32` + `state: (12+4N,) float32` (drone pos/vel, wind, base-relative pos, per-waypoint relative pos + visited flag).
- **Action space**: continuous 3D desired-velocity in `[-1, 1]^3`; PID converts to motor commands.

## Project layout

```
envs/
  config.py                  EnvConfig dataclass (YAML-loadable). All env knobs live here.
  drone_inspection_env.py    Gymnasium env. Reward function, episode termination, waypoint gen.
  pyflyt_backend.py          PyBullet physics, depth cam, collision check, wind force.
  sim_backend.py             Abstract SimBackend interface (for future AirSim swap).
  sheeprl_wrapper.py         Adapter to sheeprl's Dict-obs / CNN+MLP routing.
  wind_model.py              Ornstein–Uhlenbeck wind process.

configs/
  sanity.yaml                N=1 sanity task. WORKING recipe — see §Working sanity recipe.
  easy.yaml / medium.yaml / hard.yaml   Progressively harder; for Phase 3.
  sheeprl/exp/drone_inspection.yaml     Algo + Fabric + buffer + metric config. Uses XS model on CPU by default.
  sheeprl/env/drone_inspection.yaml     Points sheeprl wrapper at a YAML under configs/. Edit `config_path:` to switch.

scripts/
  train_dreamer.py           Launches `python -m sheeprl exp=drone_inspection`; passes through CLI overrides.
  eval_dreamer.py            Loads a checkpoint, runs N episodes, optional --render (PyBullet GUI).
  smoke_test_dreamer.py      Validates the full pipeline end-to-end.
  test_env.py                Random-agent baseline.

eval/
  metrics.py                 Per-episode metric helpers.
  route_visualizer.py        Matplotlib top-down + 3D route plots.

logs/runs/dreamer_v3/DroneInspection-sheeprl-v0/<timestamp>_..._<seed>/version_0/
  config.yaml                Full Hydra-merged config snapshot. eval_dreamer.py reads this.
  events.out.tfevents.*      TensorBoard scalars.
  checkpoint/ckpt_<step>_0.ckpt   ~105 MB per checkpoint.

venv/                        Python 3.11 venv. Key deps: sheeprl 0.5.7, pyflyt, torch, tensorboard 2.20.
```

## Reward function

In `envs/drone_inspection_env.py`. Per step:

```
reward = -0.01                                         # _REWARD_TIME_PENALTY (always)
       + 10.0   if drone enters a not-yet-visited waypoint (episode continues)
       + 50.0   if all visited AND drone back at base (episode terminates)
       + config.collision_penalty   if collision (episode terminates)
       + config.reward_shaping × (prev_target_dist − curr_target_dist)
```

Shaping is **potential-based**: target = nearest unvisited waypoint, or base if all visited. Reset on `env.reset()` and on target switch (waypoint visited). Telescoping property: total shaping per episode = `k × (initial_dist − final_dist)` — can't be milked by oscillation.

`collision_penalty` and `reward_shaping` are `EnvConfig` fields. Defaults `-100.0` and `0.0` respectively, matching the original spec. Override per-config YAML for tasks that need different reward landscapes.

## Working sanity recipe (N=1 — the one that converged)

`configs/sanity.yaml`:
```yaml
num_waypoints: 1
flight_dome_size: 5.0
waypoint_reach_distance: 3.0    # = _MIN_WAYPOINT_SEPARATION → waypoint spawns AT edge of reach radius
wind_enabled: false
num_obstacles: 0
max_episode_steps: 300
agent_hz: 30
image_width: 64
image_height: 64
reward_shaping: 2.0
collision_penalty: 0.0
```

Converges to **rew ≈ +60, ep_len ≈ 17–27** in ~7k env steps on CPU (~3.5h wall on a recent Mac under light load). `+60` is essentially the theoretical max (`+10 waypoint + +50 mission − tiny time penalty`). Final 15k-step checkpoint exists under `logs/runs/dreamer_v3/.../version_0/checkpoint/ckpt_15000_0.ckpt` of the most recent run dir.

## How to run

### Train (CPU debug)
```bash
./venv/bin/python scripts/train_dreamer.py algo.total_steps=15000 metric.log_every=200 checkpoint.every=2000
```
Hydra-style overrides flow through. To switch the env config, edit `configs/sheeprl/env/drone_inspection.yaml` line `config_path:` and relaunch.

### Train (GPU — Phase 3, not yet executed)
```bash
./venv/bin/python scripts/train_dreamer.py \
  fabric.accelerator=gpu algo=dreamer_v3_S algo.total_steps=5000000 env.num_envs=4
```

### Watch a trained agent in the PyBullet GUI
```bash
LATEST_CKPT=$(ls -td logs/runs/dreamer_v3/DroneInspection-sheeprl-v0/*/version_0/checkpoint 2>/dev/null | head -1 | xargs -I{} ls -t {}/ckpt_*.ckpt | head -1)
./venv/bin/python scripts/eval_dreamer.py "$LATEST_CKPT" \
  --config configs/sanity.yaml --episodes 5 --render --greedy
```
**Always pass `--config`** matching what the agent was trained on. The script defaults to `configs/easy.yaml`, which won't match an agent trained on `sanity.yaml`. `--greedy` uses deterministic argmax actions for cleaner playback. Route plots get dropped at `results/eval_route_{topdown,3d}.png`.

### TensorBoard
```bash
./venv/bin/tensorboard --logdir logs/runs/dreamer_v3/DroneInspection-sheeprl-v0 --port 6006
```
Useful scalar tags: `Rewards/rew_avg`, `Game/ep_len_avg`, `Loss/world_model_loss`, `Loss/policy_loss`, `Loss/value_loss`, `Time/sps_train`, `Time/sps_env_interaction`, `Params/replay_ratio`.

### Read TB scalars programmatically (when GUI inconvenient)
```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator("<version_0 path>", size_guidance={'scalars': 0}); ea.Reload()
for e in ea.Scalars('Rewards/rew_avg'):
    print(e.step, e.value)
```

### Caffeinate during long CPU runs (macOS)
```bash
caffeinate -i -w <training_pid> &
```
Auto-exits when the training PID dies.

## Gotchas

1. **Sparse rewards collapse the actor to "do nothing".** Default config (`collision_penalty=-100`, `reward_shaping=0`) makes any movement look catastrophic before the agent has a chance to discover the +10 waypoint reward, so the actor learns to sit still forever. The fix that worked: dense potential-based shaping + softened collision penalty.

2. **`waypoint_reach_distance < _MIN_WAYPOINT_SEPARATION` makes sanity tasks effectively unwinnable.** `_MIN_WAYPOINT_SEPARATION = 3.0` is hardcoded in `drone_inspection_env.py`. If reach radius is smaller, random initial-policy actions almost never accidentally land within reach, so the actor never observes positive reward, so the world model only learns "sit still" dynamics, so imagination never finds a path to the goal. This is the chicken-and-egg that ate 5 reward-tuning iterations. Set `reach ≥ _MIN_WAYPOINT_SEPARATION` for sanity / debug tasks.

3. **The dome is NOT a hard physical boundary.** The drone routinely drifts well past `flight_dome_size` without triggering collision. Use `flight_dome_size` as a *waypoint-spawn* constraint only — it doesn't bound the agent's trajectory.

4. **PID-damped random commands barely produce directed motion.** Net drift per random-policy episode is large but stochastic — explanation for why sparse-reward sanity tasks need physical proximity (small dome + large reach), not just stronger shaping.

5. **CPU training is debug-only.** XS DreamerV3 does ~30–60 env-steps/min on a recent Mac under light load, degrading 10× under memory pressure. Real Phase 3 ablations need GPU.

6. **`eval_dreamer.py` silently defaults `--config` to `configs/easy.yaml`** — see §Watch a trained agent. Always pass `--config` explicitly.

7. **`checkpoint.every` counts env-steps, not gradient steps.** With 15k total and `every=2000`, you get checkpoints at 2k, 4k, …, 14k, 15k.

8. **Waypoints are random per episode**, generated in `_generate_waypoints` via rejection sampling on the dome × `_MIN_WAYPOINT_SEPARATION` constraint. With `dome=5, min_sep=3`, the achievable region is a thin annular shell — the agent learned a distribution, not a single fixed position.

## Status

| Phase | Status |
|---|---|
| 1: Env + PyFlyt backend + random-agent baseline | done |
| 2: DreamerV3 integration + sanity convergence (N=1) | **done** (sanity recipe → rew ≈ +60) |
| 3: GPU runs on easy/medium/hard curriculum + ablations + wandb | not started |
| 4: AirSim backend + photorealistic validation + paper | not started |

Sanity convergence proves the pipeline learns. Phase 3 needs a GPU: target is `configs/easy.yaml` (N=3, dome=30m) at 1M+ steps with standard `collision_penalty=-100` and either no shaping or much weaker shaping (the dome is big enough that random exploration will eventually hit waypoints).

## Background docs

`else/SUPERVISOR_REPORT.md` — supervisor elevator pitch.
`else/project-spec.md` — original detailed design.
`else/agent-prompt.md`, `else/phase2-prompt.md` — bootstrap prompts used to scaffold the code.

These predate the iterative debugging that produced the working sanity recipe. Treat them as design intent, not current ground truth.
