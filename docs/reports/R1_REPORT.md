# Drone Chase — R1 Reproduction Study (DreamerV3)

**Campaign:** `2026-08-chase-hardening`, rung **R1**  ·  **Config:** `chase_easy2`  ·  **Date:** 2026-08-31
**Config hash:** `sha256:58591240ab2a15495c3d1cfcf437f10d`  ·  **Seeds:** 1, 2, 3 (each trained from scratch)

---

## TL;DR (the headline)

> **DreamerV3 reproduces the chase catch-rate across 3 seeds (76% ± 11%), but the learned policy does not actually pursue — it loiters on the target's route and lets the target come into range.** The success rate passes every threshold; the *behaviour* gate (pursuit alignment ≥ 0.45) fails on all three seeds (mean 0.28). This is the intended, honest result of a control condition: it shows the task is trivially winnable by waiting, which is exactly why the next study needs an **evasive** target.

![R1 summary](assets/R1_summary_comparison.png)

*Left: catch success per seed vs the ≥55% gate and scripted-pursuit reference (100%). Right: pursuit alignment per seed vs the ≥0.45 gate and scripted-pursuit reference (0.69). Success clears its bar; pursuit quality does not.*

---

## 1. The task and the environment

**Task — "chase":** a quad-rotor **pursuer** must catch a second quad-rotor **target**. The target
is a *real* drone with the same physics, flying a fixed, non-self-intersecting **closed waypoint
route** (it ignores the pursuer). "Catch" = the pursuer gets within the catch radius of the target.

**Simulator:** PyFlyt (PyBullet-based) QuadX drones in velocity-control mode, 30 Hz control.

**Geometry & spawn (`chase_easy2`):**

| parameter | value |
|---|---|
| flight dome | 20 m diameter (radius 10 m) |
| pursuer spawn | fixed at centre `(0, 0, 1 m)` |
| target spawn | first of an 8-waypoint 2-opt route, uniform in the dome disk, altitude 1.5–6 m |
| initial pursuer↔target separation | ~7.3 m mean (measured) |
| target speed cap | 0.3 (~1.5 m/s) |
| catch radius | 2.0 m |
| episode length | up to 600 steps (20 s) |
| reward | +60 catch, −0.01/step time penalty, dense potential shaping (1.0), −10 collision |

Because the target flies a **fixed route that ignores the pursuer**, it always comes back around —
so "get onto the route and wait" is a viable strategy. That property is central to the result below.

---

## 2. Why this is a control condition (baselines, measured before any training)

Scripted baselines were measured on the real environment *before* spending on training, reading
exactly the same observation the agent gets:

| policy | success | pursuit alignment | mean steps-to-catch |
|---|---|---|---|
| **scripted lead-pursuit** (6 lines) | **100%** | **0.69** | 75 |
| **random** (PID-damped) | **0%** | −0.75 | — |

So the task is **not accidentally winnable** (random = 0%) but is **trivially winnable by a
reactive controller** (scripted = 100%). This bounds the claim: R1 can only show whether *RL
reproduces* the result across seeds — **not** that RL is needed for interception. The scripted
alignment (0.69) and steps-to-catch (75) are the reference for what genuine pursuit looks like,
and they calibrate the behaviour gate: 0.45 sits cleanly between random (−0.75) and scripted (0.69).

---

## 3. How the study was organised (methodology)

- **Algorithm:** DreamerV3 (size XS), one fixed config across all seeds (DreamerV3's whole claim is
  one config across many tasks — no per-seed tuning).
- **Hardware:** single-GPU cloud instances (RTX 2080 Ti / 4060 Ti), 16-mixed precision.
- **Seeds:** 3 independent runs (seeds 1, 2, 3), each **trained from scratch**. Three seeds is the
  minimum for a claim — deep RL varies seed-to-seed, and one seed is an optimistic estimate.
- **Training signal:** early stop when the training reward proxy (`Rewards/rew_avg`) sustains ≥ 40.
  In practice the reward was highly volatile (catches give +60, misses go negative), so the
  windowed average rarely held ≥ 40 for long; runs were therefore stopped once clearly converged
  (all seeds peaked ≈ 64) and their checkpoints evaluated. Checkpoints were saved every 1000 steps.
- **Acceptance = a rollout evaluation, not a training metric.** Each seed's checkpoint was evaluated
  over **30 greedy episodes** (fixed eval seeds), producing a success rate **and** a behaviour block.
- **Precommitted gates** (written before any run, calibrated against the baselines above):

| gate | threshold | rationale |
|---|---|---|
| mean success | ≥ 55% | below the single-seed 70% being reproduced — 3-seed ≥55% is a stronger claim |
| worst-seed success | ≥ 40% | one collapsed seed must not be hidden by the mean |
| **mean pursuit alignment** | **≥ 0.45** | separates a pursuer (scripted 0.69) from a wanderer (random −0.75) |
| initial separation | ≥ 5.0 m | a catch from <5 m is mostly spawn geometry |
| steps-to-catch | ≥ 40 | anything faster didn't really cross the gap |
| idle fraction | ≤ 0.5 | catches a policy that just hovers |
| separation-closed fraction | ≥ 0.5 | the pursuer must actually close distance |

**Numbers veto, eyes advise:** the behaviour gates are hard pass/fail; the videos and contact sheet
are for human sanity-checking, not for overriding the numbers.

---

## 4. Results

### Per seed (30 greedy episodes each)

| seed | success | pursuit alignment | steps-to-catch | station-keeping | sep-closed |
|---|---|---|---|---|---|
| seed 1 | 83% (25/30) | 0.275 | 289 | 0.176 | 0.71 |
| seed 2 | **90% (27/30)** | 0.364 | 180 | 0.068 | 0.72 |
| seed 3 | 53% (16/30) | 0.202 | 173 | 0.018 | 0.64 |
| **scripted (ref)** | 100% | 0.69 | 75 | 0.00 | — |

### Aggregate (3 seeds)

| metric | value | gate | verdict |
|---|---|---|---|
| mean success | **75.6% ± 11.3%** | ≥ 55% | ✅ PASS |
| worst seed | 53.3% | ≥ 40% | ✅ PASS |
| **mean pursuit alignment** | **0.28** | ≥ 0.45 | ❌ **FAIL** |
| initial separation | 7.76 m | ≥ 5.0 | ✅ PASS |
| steps-to-catch | 214 | ≥ 40 | ✅ PASS |
| idle fraction | 0.006 | ≤ 0.5 | ✅ PASS |
| separation-closed | 0.69 | ≥ 0.5 | ✅ PASS |

**Aggregate gate = FAIL (on pursuit alignment).** This was **not overridden**: the miss is large
(0.28 vs 0.45), consistent across all three seeds, and confirmed by the videos — the gate is
correctly reporting real behaviour, not a mis-set threshold.

---

## 5. Interpretation — reproduces *catch-rate*, not *pursuit*

The catch rate reproduces (mean 76%, every seed above the 40% floor). But every behaviour signal
says the policy **waits** rather than **chases**:

- **Alignment 0.28** vs scripted 0.69 — the drone's velocity is only weakly aimed at the target.
- **Steps-to-catch 214** vs scripted 75 — catches take ~3× longer: the drone lets the target's
  fixed route bring it into range rather than intercepting it.
- The videos (below) show the pursuer holding near the centre while the target flies *into* the
  catch zone.

This is the expected consequence of a target that ignores the pursuer: **waiting is close to
optimal**, so RL learns to wait. The success rate alone would look great and be misleading — which
is precisely what the behaviour gate exists to catch. It also motivates the next study: an
**evasive** target, where waiting no longer works and genuine pursuit becomes necessary.

---

## 6. How they trained (learning curves)

All three seeds show the same volatile ascent, peaking near reward 64 (well above the catch
threshold) but oscillating rather than settling — a symptom of the discrete catch/miss reward.

| seed 1 | seed 2 | seed 3 |
|---|---|---|
| ![s1](assets/seed1_learning_curve.png) | ![s2](assets/seed2_learning_curve.png) | ![s3](assets/seed3_learning_curve.png) |

---

## 7. Episode videos and contact sheet

Best genuine catch per seed (excluding spawn-freebie episodes where the target spawned inside the
catch radius). 3D orbit + top-down `.mp4` in `assets/`:

| seed | best catch | files |
|---|---|---|
| seed 1 | 43 steps | `assets/seed1_best_catch_43steps_3d.mp4`, `..._topdown.mp4` |
| seed 2 | 49 steps | `assets/seed2_best_catch_49steps_3d.mp4`, `..._topdown.mp4` |
| seed 3 | 61 steps | `assets/seed3_best_catch_61steps_3d.mp4` |

**Honest note on the videos:** even the *best* episodes show the pursuer moving little while the
target flies toward it — consistent with the 0.28 alignment. They are genuine catches, not the
impressive long-range chases the raw success rate might suggest.

Contact sheet (seed 1, 9 episodes — mix of catches and misses, showing the looping/waiting paths):

![contact sheet](assets/seed1_contact_sheet.png)

---

## 8. Caveats and honest limitations

- **Not evidence that RL is required.** Scripted lead-pursuit solves this task 100%. R1 establishes
  reproducibility of the RL result, nothing stronger.
- **Single-seed spread is wide** (53%–90%): seed variance is real; seed 3 is a genuinely weaker run.
- **Early-stopping rarely triggered** because of reward volatility; seeds were evaluated at their
  converged checkpoints. Greedy evaluation smooths the training-time oscillation (e.g. seed 1
  oscillated 20↔62 in training but evaluated at a steady 83%).
- **GPU non-determinism:** re-rolling the same episode can catch or miss (especially seed 3), so
  the per-episode videos are representative, not bit-reproducible.

---

## 9. What this hands to the next study

1. A working, seed-reproducible RL result on a **known-answer** control task.
2. A measured cost model and calibrated behaviour thresholds from real GPU curves.
3. The key motivation: **the fixed-route target makes waiting optimal**, so the next campaign must
   make the target **evade** — that is where pursuit alignment becomes a real, non-trivial objective.

---

## Artifact index (`results/R1_report/assets/`)

- `R1_summary_comparison.png` — the headline results figure
- `seed{1,2,3}_learning_curve.png` — training curves
- `seed{1,2,3}_best_catch_*_3d.mp4` / `_topdown.mp4` — episode videos
- `seed1_contact_sheet.png` — 9-episode overview
- `seed{1,2,3}_eval.json` — raw per-episode eval data (30 episodes each)
- `aggregate_R1.json` — the 3-seed aggregate + gate results
