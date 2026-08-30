# Baseline findings — measured 2026-08-30, before any GPU spend

Scripted baselines on the real environment, CPU, `tools/run_baselines.py`. These cost
nothing and they invalidated the campaign's original premise before a dollar was spent.

## What was measured

| config | policy | eps | success | alignment | fastest catch |
|---|---|---|---|---|---|
| `chase_easy2` | scripted pursuit | 10 | **100%** | 0.69 | 43 |
| `chase_easy2` | random | 10 | **0%** | −0.75 | — |
| probe: speed cap 0.3 → **0.5** (parity with pursuit's usable command) | pursuit | 8 | **100%** | 0.72 | 40 |
| probe: **no target velocity** observed + **1.5 m position noise** | pursuit | 8 | **100%** | 0.68 | 42 |
| probe: speed cap → **0.7** (~3.5 m/s, above pursuit's ~2.5 m/s ceiling) | pursuit | 8 | **100%** | 0.73 | 40 |

Probe configs were deleted after measuring; their JSON reports are kept here as the
evidence. Nothing about the tracked rung configs was changed.

## Finding 1 — the behaviour metrics work on real data

Scripted pursuit measures **0.69** pursuit alignment; a random policy measures **−0.75**
(PID-damped random commands drift away from the target, so it is not merely uncorrelated).
That separation is wider than the synthetic test suggested, and it puts the spec's
precommitted `--min-alignment 0.45` cleanly between the two. The threshold is calibrated,
not guessed.

`fastest_catch` of 40–43 steps also validates `--min-catch-steps 40` as the boundary
between a real chase and a spawn artefact — a competent controller needs about that long.

Random at 0% is the other half of the check: the task cannot be won by accident, so a
learned success rate means something.

## Finding 2 — the task is trivial for a hand-written controller, at every setting

Scripted lead pursuit is six lines. It reads `obs["state"]`, which carries the target's
exact relative position and velocity — **the same information the DreamerV3 agent gets**,
since the experiment config encodes `state` through the MLP encoder. So this is not a
privileged baseline.

It solves `chase_easy2` **100% of the time**. It still solves it 100% of the time when the
target moves at parity with the pursuer's usable speed, when the target moves *faster* than
the pursuer's usable speed, and when the target's velocity is hidden and its observed
position is noised by 1.5 m.

That reframes the original goal. Reproducing ~70% on `chase_easy2` would mean showing that
DreamerV3 does, somewhat less reliably, what a trivial controller does perfectly. The first
question a reader asks is "why not use the controller", and there is no good answer.

## Finding 3 — why nothing in the config space fixes it

The target flies a **fixed closed waypoint route that ignores the pursuer**
(`WaypointPlan.sample(..., loop=True)`, and `_target_command` steers only toward the next
waypoint). So the target's speed is close to irrelevant: it comes back around. The optimal
strategy is to move onto the route and wait, which any distance-reducing policy discovers,
and 600 steps is ample.

In other words `DroneChaseEnv` is a *rendezvous* task, not an interception task. Nothing in
`EnvConfig` changes that, because evasion is not a parameter — the target has no term that
depends on the pursuer's position.

## Recommendation

Make the target evade, then re-measure. `_target_command` is six lines; adding a repulsion
term from the pursuer, scaled by how close it is, is roughly ten:

```python
cmd = self.config.chase_target_gain * (goal - target.position)
if self.config.chase_target_evasion > 0.0:
    away = target.position - self.backend.get_drone_state().position
    dist = float(np.linalg.norm(away))
    if dist > 1e-6:
        # Flee harder the closer the pursuer is, so evasion decides interceptions
        # without distorting the route at long range.
        urgency = min(self.config.chase_evasion_radius / dist, 1.0)
        cmd += self.config.chase_target_evasion * urgency * (away / dist)
```

plus two `EnvConfig` fields defaulting to `0.0`, so every existing config and every past
result keeps its exact current behaviour.

This is a change under `envs/`, which the constitution reserves for a human checkpoint. It
is also the change `todo.md` has been pointing at all along ("максимізація урону по рухомих
цілях" presumes a target worth chasing).

Once it is in, `tools/run_baselines.py` answers the question that decides the campaign: at
what evasion strength does scripted pursuit stop being able to intercept? The rung just
below that is where a learned policy has something to prove, and where "RL beats a
hand-written pursuit controller" becomes a claim worth showing.
