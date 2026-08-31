# Campaign: chase-pursuit — force genuine pursuit, not loiter

## Why

Two prior iterations established the problem, precisely:

- **R1 (`chase_easy2`)**: DreamerV3 reproduces the catch-rate across 3 seeds (76% ± 11%) but the
  behaviour gate fails — **mean pursuit alignment 0.28** vs scripted pursuit 0.69. The policy
  *loiters and intercepts a predictable non-evasive target*, it does not chase. (Route is random
  per episode but the target ignores the pursuer, and the 2 m catch radius + small dome make
  waiting near-optimal.)
- **`chase_open` (iteration 2)**: added a non-telescoping distance penalty, a large dome, and a
  tight 0.75 m catch radius. Baseline came back `suitable` (scripted 80%, random 10%) and training
  was healthy — but a **bounded spawn (~4–7 m start)** still handed out cheap fast catches (scripted
  caught in 23 steps). So it did not yet force *sustained* pursuit. Documented in
  `results/chase_open_report/CHASE_OPEN_REPORT.md`.

**This campaign's single job:** make loitering and close-spawn catches fail, so the only way to
succeed is to *actively chase across distance*, and then measure whether the learned policy's
pursuit alignment rises above R1's 0.28 toward the scripted reference on this task.

The scientific question is sharp: **does forcing (distance penalty + far spawn + faster target)
produce genuine pursuit, or does DreamerV3 still find a degenerate strategy?** Either answer is a
result. If alignment stays low even here, that is strong evidence the missing ingredient is an
*evasive* target, not reward/geometry shaping.

## What changed from the previous runs (and why)

| axis | R1 `chase_easy2` | `chase_open` | **`chase_pursuit` (this)** | why |
|---|---|---|---|---|
| distance term | telescoping shaping only | + `distance_penalty` 0.02 | **keep** `distance_penalty` 0.02 | per-step cost for being far — makes waiting bleed reward |
| dome / waypoints | 20 m | 500 m | **keep** 500 m | target roams far; waiting → far |
| spawn radius | coupled to dome | **10 m** (close) | **25 m** (far) | the close spawn was the flaw — start far so a real chase is required |
| catch radius | 2.0 m | 0.75 m | **keep** 0.75 m | demands precise interception |
| target speed | 0.3 (~1.5 m/s) | 0.3 | **0.4 (~2 m/s)** | still < pursuer (~2.5 m/s) so catchable, but closing now needs better alignment |
| reward_shaping | 1.0 | 1.0 | **keep** 1.0 | bootstraps learning |

Net: the only new changes vs `chase_open` are **spawn radius 10 → 25** and **target speed
0.3 → 0.4**. Both push toward sustained pursuit; both risk unwinnability, so a baseline gates them.

## Config (AMBITIOUS — distance 100, human-directed)

`configs/target/chase_pursuit_d100.yaml`: `chase_spawn_radius: 100` (target starts ~65 m away on
average), `max_episode_steps: 1500` (~50 s — long enough that a ~2.5 m/s pursuer can physically
cross 100 m; at 600 steps it is unwinnable), `chase_target_speed_cap: 0.3`, `distance_penalty: 0.01`
(scaled down for the larger distances / longer episode). Unchanged: dome 500, catch 0.75,
shaping 1.0, 8 waypoints, 30 Hz.

**Why this is the right ambitious choice:** at a ~65 m start, *loitering wins nothing* (measured:
scripted must fly straight at the target for hundreds of steps; a non-pursuing policy gets 0%). So
the far spawn STRUCTURALLY forces pursuit — the exact failure mode R1/chase_open could not escape.
Cost: episodes are 2.5× longer and the task is much harder, so learning may need many more steps
(see P1) and some seeds may not solve it — a partial/negative result here is still informative.

## Ground rules

- Budget: inherits the running guard (**$15 cap, 12 h TTL**). Concurrency **≤ 5** (human-approved
  above the default 3 for this exploratory line).
- Task `chase`, `exp=drone_chase`, `algo=dreamer_v3_XS`, `env.num_envs=4`, GPU, 16-mixed,
  `checkpoint.every=1000`, `metric.log_every=1000`, `total_steps=200000` (ceiling).
- **Reward scale note:** the distance penalty makes `Rewards/rew_avg` negative, so the `≥ 40`
  early-stop is **inert**. Runs are stopped manually once converged (reward-curve flat / behaviour
  stable), then the newest checkpoint is evaluated — as in R1.
- Evaluation: `tools/evaluate_ckpt.py`, greedy, **30 episodes**, seed base **1000**, trajectories on.

## Mandatory baseline gate (before any GPU spend)

Run `tools/run_baselines.py` on `chase_pursuit.yaml` first. It must return **`suitable`**:

- **Scripted pursuit success ≥ 50%.** If lower, the config is likely unwinnable in a 20 s episode —
  soften and re-baseline: first `chase_spawn_radius 25 → 15`, then `target speed 0.4 → 0.35`. Do not
  train an unwinnable task.
- **Record scripted pursuit's alignment on this exact config** — it is the reference for "genuine
  pursuit" here (expect ~0.35–0.40; it is *not* R1's 0.69, because a far, faster, fleeing target
  raises the lead/lag angle even for a perfect chaser).

## Rungs

### P0 — baseline + calibration (free, no GPU)
Run the baseline. Accept only if `suitable` (scripted ≥ 50%). Write scripted's success and
alignment here as the references. If not suitable, soften per the rule above and repeat.

**Measured (2026-08-31, chase_pursuit_d100): scripted success = 70%, scripted alignment = 0.604,
random = 0% → `suitable`.** So the pursuit reference for this config is **0.604** — notably HIGHER
than chase_open's 0.39, because a 65 m start forces the pursuer to fly straight at the target for a
long time. Loitering scores 0% here.

### P1 — pursuit across 5 seeds
`chase_pursuit_d100` × seeds `{1,2,3,4,5}`, from scratch. `total_steps` ceiling **300000** (this is a
much harder task than R1; a 65 m chase may need far more steps than chase_easy2's ~29 k, and some
seeds may not solve it within budget/TTL — that is itself informative). Monitor; stop+evaluate each
when converged or when its alignment trend has clearly plateaued.

Acceptance (precommitted, using the measured scripted reference 0.604):
- **Primary — pursuit alignment:** mean `mean_pursuit_alignment` ≥ **0.48** (= 0.80 × 0.604). This
  is the whole point: the learned policy must clearly beat R1's loiter (0.28) and approach the
  scripted reference (0.60) for this task. Report it beside R1's 0.28 and scripted's 0.60.
- **Secondary — winnable:** mean success ≥ **30%**, worst seed ≥ **15%**. Success is secondary
  (pursuit quality is the goal); this floor only confirms the policy actually catches. Given the
  difficulty, if success is low but **alignment is high**, that is a *positive* pursuit result
  (it chases well even if it doesn't always close the tight 0.75 m gap in time).
- **Behaviour, recalibrated for a far chase:** `--min-initial-separation 30.0` (mean start ~65 m),
  `--min-catch-steps 40`, `--max-idle-fraction 0.5`, `--min-separation-closed 0.5`.
- Reviewer verdict `pass` (blind re-eval seed base 5000, reads the contact sheet).

Fail policy: if alignment does **not** clear 0.48, that is the headline finding — **"reward +
geometry alone did not produce scripted-level pursuit"** — report it, do not seed-shop. Note the
distinction from R1: here loitering is impossible (0% baseline), so a low learned alignment would
mean the policy failed to learn the chase at all (likely under-trained), not that it chose to
loiter — re-check whether it needed more steps before concluding.

## Non-goals
- No hyperparameter tuning (DreamerV3 is one fixed config).
- No reward term that rewards alignment directly (that Goodharts the very metric we measure).
- No `envs/` code change beyond the two already-added knobs (`distance_penalty`,
  `chase_spawn_radius`); the evasive-target term is the *next* campaign, not this one.

## What this hands to the next campaign
1. Whether reward + geometry shaping alone can produce genuine pursuit on a non-evasive target.
2. If not: a clean, precommitted negative result motivating an evasive-target environment change.
3. A pursuit-quality baseline (scripted alignment) calibrated for far/fast chases.
