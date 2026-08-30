# AGENTS.md — DreamerV3 for Drone Inspection and Interception

## What this project is

Research project: a **DreamerV3** (model-based RL with learned world models) agent flying a
quadcopter in PyFlyt. Two task families live here:

- **Interception** (`envs/tasks/drone_target_env.py`, `drone_chase_env.py`) — the current line of
  work. The drone must catch a moving target: either a Dubins-model target (`DroneTargetEnv`) or a
  second real drone flying a 2-opt waypoint route under the same physics (`DroneChaseEnv`).
- **Inspection** (`envs/tasks/drone_inspection_env.py`) — the original task. A multi-waypoint
  mission: TSP with continuous 3D flight, depth perception, stochastic wind, obstacles. The pitch
  was "combinatorial planning + low-level control jointly learned in one policy."

Common ground:

- **Sim**: PyFlyt (PyBullet physics, depth camera, PID-controlled QuadX in velocity-control mode).
  Cosys-AirSim (UE5) exists as a backend but is validation-only, not used for training.
- **Algo**: DreamerV3 via `sheeprl` v0.5.7 (PyTorch). XS model for CPU dev and GPU runs so far.
- **Action space**: continuous 3D desired velocity in `[-1, 1]^3`, scaled by `_MAX_SPEED = 5.0` m/s;
  PID converts to motor commands.
- **Obs space (Dict)**: `depth: (64,64,1)` + `state`. For the interception tasks `state` is 15-dim:
  `[pos(3), vel(3), wind(3), target_rel(3), target_vel(3)]`. For inspection it is `12+4N`.

## Project layout

```
envs/
  core/
    config.py            EnvConfig dataclass (YAML-loadable). EVERY env knob lives here.
    sim_backend.py       Abstract SimBackend interface + DroneState dataclass.
    wind_model.py        Ornstein-Uhlenbeck wind process.
  backends/
    pyflyt_backend.py       PyBullet physics, depth cam, collision, wind-as-velocity-bias.
    pyflyt_chase_backend.py Adds a SECOND drone (the target) with its own velocity control.
    airsim_backend.py       Cosys-AirSim (UE5). Linux/Windows only; validation, not training.
  targets/
    dubins.py            Dubins car2d / airplane3d motion models for DroneTargetEnv.
    waypoint_plan.py     Non-self-intersecting (2-opt) route generation for the chase target.
  tasks/
    drone_target_env.py     Intercept a Dubins target.
    drone_chase_env.py      Intercept a second real drone. The frontier task.
    drone_inspection_env.py Multi-waypoint inspection mission.
  sheeprl_wrapper.py     Dict-obs adapter + make_drone_{inspection,target,chase}_env factories.

configs/
  target/       Interception ladders. l0_smoke..l4_partial (Dubins target),
                chase_veryeasy / chase_mid / chase_easy / chase_easy2 / chase (two-drone).
  inspection/   Inspection ladder: sanity*, easy, medium, hard.
  sheeprl/exp/{drone_chase,drone_target,drone_inspection}.yaml   Algo + Fabric + buffer + metrics.
  sheeprl/env/{drone_chase,drone_target,drone_inspection}.yaml    Registers the env id, points the
                wrapper at a config via wrapper.config_path (overridable from the CLI — see below).

scripts/
  train_dreamer.py       Launches `python -m sheeprl exp=<exp>`; passes Hydra overrides through.
                         Also implements performance-based EARLY STOPPING (--stop-* flags).
  eval_target_ckpt.py    Eval a TRAINED checkpoint on target/chase. Prints success rate; plots/videos.
  eval_target.py         Eval scripted policies (pursuit / random) — no checkpoint needed. Env sanity.
  watch_train.py         Run alongside training: periodically renders the newest checkpoint to video
                         and refreshes the learning curve. Reads only; never touches the train loop.
  eval_dreamer.py        Inspection-task checkpoint eval.
  test_env.py            Random-agent baseline + gymnasium env_checker.
  smoke_test_dreamer.py  Full-pipeline end-to-end validation.
  plot_learning_curve.py / plot_results.py / plot_dubins.py / plot_waypoint_plan.py

eval/
  target_visualizer.py   Top-down + 3D + animated MP4 renders of an interception episode.
  route_visualizer.py    Inspection route plots.
  episode_recorder.py    Frame capture during training rollouts.

deploy/                  vast.ai GPU orchestration + AirSim/UE5 setup. See deploy/README.md.

logs/runs/dreamer_v3/<env-id>/<timestamp>_..._<seed>/version_0/
  config.yaml            Full Hydra-merged config snapshot. The eval scripts read this.
  events.out.tfevents.*  TensorBoard scalars.
  checkpoint/ckpt_<step>_0.ckpt   ~105 MB per checkpoint.

venv/                    Python 3.11 venv (sheeprl needs <3.12). sheeprl 0.5.7, pyflyt, torch.
```

## Reward functions

**Interception** (`drone_target_env.py`, `drone_chase_env.py`), per step:

```
reward = -0.01                                  # _REWARD_TIME_PENALTY (always)
       + config.target_catch_reward             # target within target_reach_distance → terminates
       + config.collision_penalty               # collision → terminates
       + config.reward_shaping × (prev_dist − curr_dist)    # potential-based, toward the target
```

**Inspection** (`drone_inspection_env.py`), per step:

```
reward = -0.01
       + 10.0   entering a not-yet-visited waypoint (episode continues)
       + 50.0   all visited AND back at base (terminates)
       + config.collision_penalty   if collision (terminates)
       + config.reward_shaping × (prev_target_dist − curr_target_dist)
```

Shaping is **potential-based** in both: it telescopes to `k × (initial_dist − final_dist)` over an
episode, so it can't be milked by oscillating. For inspection the potential's target switches to
the next unvisited waypoint (and the shaping baseline resets on switch).

## How to run

### Select a curriculum rung from the CLI (no file editing)

`wrapper.config_path` is a plain Hydra key, so the env config is an override:

```bash
./venv/bin/python scripts/train_dreamer.py exp=drone_chase \
  env.wrapper.config_path=configs/target/chase_mid.yaml seed=1
```

Paths resolve relative to the sheeprl process CWD (the repo root locally, `/workspace/dreamer` on a
vast.ai box).

### Train with early stopping

`train_dreamer.py` can stop a run once it is good enough, so `algo.total_steps` acts only as a
safety ceiling. The monitor tails **this run's** tfevents (it diffs against the event files that
existed before launch, so a previously-converged run can never trigger a false stop), and on
trigger waits up to `--stop-grace` seconds for a fresh checkpoint before terminating.

```bash
./venv/bin/python scripts/train_dreamer.py exp=drone_chase \
  env.wrapper.config_path=configs/target/chase_easy2.yaml \
  algo.total_steps=2000000 env.num_envs=4 fabric.accelerator=gpu fabric.precision=16-mixed \
  metric.log_every=1000 checkpoint.every=50000 \
  --stop-metric Rewards/rew_avg --stop-threshold 45 --stop-window 5 --stop-patience 3
```

### Evaluate a trained checkpoint

```bash
./venv/bin/python scripts/eval_target_ckpt.py <ckpt.ckpt> \
  --task chase --config configs/target/chase_easy2.yaml --episodes 30 --seed 1000 --video3d
```

**Always pass `--config`** matching what the agent trained on. Success rate is printed at the end;
`--json PATH` writes it machine-readably.

### Check the env without a policy

```bash
./venv/bin/python scripts/eval_target.py --config configs/target/l2_moving.yaml --policy pursuit
```

A scripted lead-pursuit policy should catch the target. If it can't, the task is broken, not the RL.

### Watch a run improve while it trains

```bash
./venv/bin/python scripts/watch_train.py --task chase --interval 180
```

### TensorBoard

```bash
./venv/bin/tensorboard --logdir logs/runs/dreamer_v3 --port 6006
```

Useful tags: `Rewards/rew_avg`, `Game/ep_len_avg`, `Loss/world_model_loss`, `Loss/policy_loss`,
`Loss/value_loss`, `State/kl`, `State/post_entropy`, `Time/sps_train`, `Params/replay_ratio`.
Read them programmatically with `EventAccumulator` (see `_read_scalar` in `train_dreamer.py`).

**Success rate is NOT in TensorBoard.** `info["is_success"]` exists in the env but sheeprl does not
log it. `Rewards/rew_avg` is only a proxy — a real success number requires a rollout eval.

### Caffeinate during long CPU runs (macOS)

```bash
caffeinate -i -w <training_pid> &
```

## Gotchas

1. **Sparse rewards collapse the actor to "do nothing".** With `collision_penalty=-100` and
   `reward_shaping=0`, any movement looks catastrophic before the agent discovers the positive
   reward, so the actor learns to sit still forever. The fix that worked: dense potential-based
   shaping + a softened collision penalty.

2. **Reach radius below the minimum waypoint separation makes sanity tasks unwinnable.**
   `_MIN_WAYPOINT_SEPARATION = 3.0` is hardcoded in `drone_inspection_env.py`. If the reach radius
   is smaller, a random initial policy almost never lands inside it, so the actor never sees
   positive reward, so the world model only learns "sit still" dynamics, so imagination never finds
   a path. This chicken-and-egg ate five reward-tuning iterations.

3. **The dome is NOT a hard physical boundary.** The drone routinely drifts well past
   `flight_dome_size` without a collision. It constrains *spawning*, not the trajectory.

4. **PID-damped random commands barely produce directed motion.** Net drift per random-policy
   episode is large but undirected — which is why sparse-reward sanity tasks need physical
   proximity (small dome, large reach), not merely stronger shaping.

5. **CPU training is debug-only.** XS DreamerV3 does ~30–60 env-steps/min on a recent Mac, 10×
   worse under memory pressure. Real runs need a GPU.

6. **The eval scripts default `--config` to something that probably isn't your task.** Always pass
   it explicitly; a mismatched config silently evaluates the wrong environment.

7. **`checkpoint.every` counts env-steps, not gradient steps.**

8. **Waypoints and target routes are randomized per episode.** The agent learns a distribution, not
   a fixed layout. With a small dome and large min-separation the achievable region is a thin
   annular shell.

9. **Wind must be a velocity-setpoint bias, never an external force.** The QuadX flies in
   velocity-control mode (mode 6); an external force fights the inner controller and drives it
   unstable. `pyflyt_backend.apply_wind()` adds the wind vector, in m/s, to the commanded velocity
   instead — the drone is pushed off course while the controller stays well-posed.

10. **Amplified shaping amplifies divergence tails.** On chase, `reward_shaping: 2.0` produced
    episode rewards around −150 on misses; 1.0 is the stable choice. See the comment in
    `configs/target/chase_mid.yaml`.

11. **Warm-starting across curriculum rungs is unproven here.** A direct warm-start from
    `chase_veryeasy` into `chase_easy` dropped success to 10–20%, and sheeprl's
    `checkpoint.resume_from` restores the old run's whole config, which conflicts with changing the
    env config per rung. Default to training each rung from scratch.

12. **One training run per box.** `deploy/train_remote.sh` kills any existing `train` tmux session
    and `pkill`s sheeprl before launching, so multiple seeds means multiple instances.

## Status

| Phase | Status |
|---|---|
| 1: Env + PyFlyt backend + random-agent baseline | done |
| 2: DreamerV3 integration + inspection sanity convergence | done (rew ≈ +60) |
| 3: Interception — Dubins target then two-drone chase | in progress |
| 4: AirSim photorealistic validation | not started |

Interception is where the work is. The chase ladder reached roughly 70% greedy catch rate on
`configs/target/chase_easy2.yaml` (no wind) via `chase_veryeasy → chase_mid → chase_easy2`, on a
single seed, locally. That number is **not yet reproduced across seeds** and is the immediate
target. `configs/target/chase.yaml` (wind on, faster target, tighter catch radius) is the harder
headline config.

## Background docs

`else/` holds the original design intent (`project-spec.md`, `SUPERVISOR_REPORT.md`, and the
bootstrap prompts). These predate the iterative debugging that produced the working recipes — treat
them as intent, not ground truth. This file is the ground truth.
