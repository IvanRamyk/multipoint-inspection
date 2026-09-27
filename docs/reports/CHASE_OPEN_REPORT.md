# Drone Chase — Iteration 2: `chase_open` (pursuit-forcing attempt)

**Campaign:** `2026-08-chase-hardening`, rung **OPEN**  ·  **Config:** `chase_open`  ·  **Date:** 2026-08-31
**Status:** stopped early by design decision (see §5) — **no learned-policy evaluation**. This
documents the iteration and *why* it was superseded, before the runs were reaped.

---

## TL;DR

R1 showed DreamerV3 reproduces the chase catch-rate but **loiters instead of pursuing**
(alignment 0.28 vs scripted 0.69). `chase_open` was a deliberate attempt to make loitering fail
and force genuine pursuit, via three reward/task changes. Its scripted baseline came back
**`suitable`** (pursuit 80%, random 10%), and early training was **healthy** (world-model loss
collapsing, episodes terminating). But inspection showed a design flaw: the **bounded spawn
(~4–7 m start)** still hands out cheap fast catches, so the task does not yet force *sustained*
pursuit. We stop here and iterate with a harder spawn/speed setup (see the new spec).

---

## 1. The task (recap)

A quad-rotor **pursuer** (spawns at centre) must catch a second quad-rotor **target** that flies a
**randomly re-sampled** (each episode), **non-evasive** waypoint route in PyFlyt. Drone model is a
Crazyflie-2 (`cf2x`), ~0.1 m. "Catch" = pursuer within the catch radius.

## 2. What we changed vs the previous runs (R1 `chase_easy2`)

This is the core of this report — every change and *why*, driven by R1's loiter finding.

| axis | R1 `chase_easy2` | R2/R3 probes | **`chase_open`** | why we changed it |
|---|---|---|---|---|
| **reward: distance term** | potential-based only (`reward_shaping·Δdist`, telescopes → policy-invariant) | same | **added `distance_penalty` 0.02** (non-telescoping `−k·dist` every step) | the telescoping term bootstraps learning but does not make pursuit *optimal*; a per-step penalty makes *waiting* bleed reward continuously |
| **dome / waypoint spread** | 20 m (radius 10) | 20 m | **500 m** (waypoints ~250 m) | small dome → a waiter is never far from the looping target; a large space means if you wait, the target roams far and the penalty bites |
| **spawn vs waypoints** | coupled (target starts on the dome-wide route) | coupled | **decoupled**: new `chase_spawn_radius` 10 (close start) while waypoints spread wide | keep the episode winnable (close start) but make the route lead *away*, so staying close requires chasing |
| **catch radius** | 2.0 m (~20× drone) | 2.0 (wind) / 2.0 (fast) | **0.75 m** (~7.5×) | 2 m lets the drone "drift within range"; a tight radius demands precise interception |
| **target speed** | 0.3 (~1.5 m/s) | 0.3 / 0.4 | 0.3 (~1.5 m/s) | kept slower than the pursuer (~2.5 m/s) so a fleeing target is still catchable |
| **early-stop** | rew_avg ≥ 40 | same | (unchanged, but **inert** — see §4) | the distance penalty makes rew_avg negative, so the ≥40 stop never fires |

Also fixed en route (infrastructure, applies to all runs): checkpoints every 1000 steps + a
guaranteed checkpoint at stop; vast CLI `--order`/`-y` fixes; `train_remote.sh` launch bugs.

## 3. Baseline (measured before spending — `assets/baseline_chase_open.json`)

| policy | success | pursuit alignment |
|---|---|---|
| scripted lead-pursuit | **80%** | **0.39** |
| random | **10%** | −0.46 |

**Verdict: `suitable`** — winnable (pursuit 80%) and non-trivial (random 10%). **Key recalibration:**
even a competent scripted pursuer only reaches alignment **0.39** here (vs 0.69 on `chase_easy2`),
because a fleeing target in open space carries a larger lead/lag angle. So on this task, "genuine
pursuit" ≈ 0.39 — the R1 gate of 0.45 does **not** transfer; a `chase_open` alignment gate should
reference ~0.39.

## 4. Early training health (before stopping)

One health check on seed 2 at ~step 3 k: **healthy**. World-model loss collapsed ~1600 → 34 in the
first 3 k steps (rapidly absorbing the new env); episodes terminated at ~100 steps (the agent is
interacting with the target, not timing out). Episode reward ~ −75 — **expected and by design**
(the non-telescoping distance penalty makes absolute reward negative; only its *trend* matters).
Consequence: the `rew_avg ≥ 40` early-stop is inert on this task; evaluation would be driven by
manual stop-when-converged, as in R1.

## 5. Why we stopped and iterate (the design finding)

Measured initial separations under `chase_spawn_radius = 10`: seeds spawn only **~3.8–7.1 m** apart.
With a faster pursuer and a 0.75 m radius, many episodes are won quickly simply because the target
happened to start close — the scripted demo below catches in **23 steps (0.8 s)**. That is still a
*close-spawn* catch, not the long, sustained pursuit we want to force. Diagnosis: the bounded spawn
that keeps the big dome winnable also re-introduces cheap catches.

**Levers for the next iteration:** (1) a larger spawn radius so the chase must cross real distance;
(2) a faster target (0.3 → 0.4, still below the pursuer) so closing demands better alignment.

## 6. Artifact: scripted-pursuit demo (`assets/chase_open_scripted_demo.mp4`)

Scripted pursuit on `chase_open`, caught in 23 steps. Real PyBullet drone meshes (waypoint markers
removed). **Honest caveat:** the cf2x drone is ~0.1 m, so it appears as small marks on the ground
plane — the sim's model is minimal; the abstract trajectory renders read more clearly.

![scripted demo frame](assets/scripted_demo_frame.png)

## 7. What this iteration establishes

- The reward/task machinery for pursuit-forcing works: distance penalty applies, spawn/waypoints
  decouple, baseline is `suitable`, training is healthy.
- **Open question left unanswered here:** does the distance penalty raise learned-policy alignment
  above R1's 0.28 toward scripted's 0.39? Not measured — the config was superseded before a policy
  matured, because the close spawn undermined the test.
- The fix is a spawn/speed change, not a reward change — carried into the next experiment's spec.

## Artifact index (`results/chase_open_report/assets/`)
- `baseline_chase_open.json` — scripted/random baseline (the `suitable` verdict)
- `chase_open.yaml` — the exact config used
- `chase_open_scripted_demo.mp4` + `scripted_demo_frame.png` — scripted pursuit on the task
