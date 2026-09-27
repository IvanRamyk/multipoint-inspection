# Drone Chase — Experiment Log (DreamerV3)

Single source of truth for every chase-task experiment, in one format. Each entry follows the
same template: **Question → Setup → Result → Verdict → Artifacts**. Chronological; the last one
(E5) is the solved result.

**The task.** A quad-rotor *pursuer* must intercept a second quad-rotor *target* that flies a
randomly re-sampled (per episode), **non-evasive** waypoint route in PyFlyt/PyBullet. Both drones
share the same physics under velocity control (mode 6, 30 Hz). "Catch" = the pursuer comes within
the *catch radius* (center-to-center) of the target. Policies are DreamerV3 (via sheeprl); no
pursuit law is hand-coded. A scripted lead-pursuit controller is the reference baseline throughout.

## Summary

| # | Experiment | When | Drone | Obs | Catch r | Target | Model | Best success | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| E1 | Curriculum ladder | Jul 2026 | cf2x ~9 cm | state+depth | 2.0–2.5 m | 1.0–1.5 m/s | S | **70%** (greedy, easy2) | curriculum works; generous radius/small dome |
| E2 | R1 reproduction | Aug 31 | cf2x | state+depth | 2.0 m | 1.5 m/s | XS | 75.6% ±11.3% (3 seeds) | reproduces catch-rate but **loiters** (align 0.28 < 0.45) — FAIL behaviour gate |
| E3 | chase_open (pursuit-forcing) | Aug 31 | cf2x | state+depth | 0.75 m | 1.5 m/s | — | *no learned eval* | design flaw: bounded spawn still gives cheap catches — superseded |
| E4 | chase-pursuit (65 m far-spawn) | Aug 31–Sep 1 | cf2x | state+depth | ~1 m | ~2 m/s | S | 2.2% (3 seeds) | genuine **active pursuit** but undertrained (align 0.13 < 0.48) — clean negative |
| E5 | **chase-big (redesign)** | Sep 2026 | primitive_drone ~45 cm | **state-only** | 1.2 m | ~2 m/s | S | **99.3%** (500 ep, 0 crashes) | **SOLVED** — matches tuned scripted on success, faster + safer |

**Arc of best success:** 2% (original cf2x vision task) → 70% (curriculum) → *(reframe: loiter vs
pursuit)* → 77% → 94% → 97% → **99.3%** (realistic redesign).

---

## E1 — Curriculum ladder (cf2x, from-scratch → warm-start)

**Question.** Can DreamerV3 learn the two-drone chase at all, and what training recipe makes it
stable?

**Setup.**
- Drone `cf2x` (~9 cm nano). Model DreamerV3 **S**. Obs = 15-dim state + depth camera. 8 async envs.
- Configs climbed a difficulty ladder: `chase_veryeasy` (dome 12, 4 wp, ~1.0 m/s, reach 2.5) →
  `chase_mid` (dome 16, 6 wp, ~1.25 m/s, reach 2.0) → `chase_easy2` (dome 20, 8 wp, ~1.5 m/s,
  reach 2.0). Episodes 500–600 steps. Reward: +60 catch, −0.01/step, potential shaping (1.0),
  −10 collision. Hardware: vast.ai RTX 3060 / 3060 Ti (~$0.05–0.07/hr).

**Result.**

| iter | task | from | best greedy |
|---|---|---|---|
| 1–3 | chase / chase_easy (from scratch) | — | catches in *windows* then collapses (unstable) |
| 4 | chase_veryeasy | scratch | **50%** (first stable plateau) |
| 5 | chase_easy | iter4 ckpt | training stable, but greedy ~10% |
| 6 | chase_mid | iter4 ckpt | **60%** (best-window `ckpt_30000`) |
| 7 | chase_easy2 | iter6 ckpt | **70%** (best-window `ckpt_47568`) |

Best-window greedy varied wildly per checkpoint (e.g. iter6: latest ckpt 10% vs best-window 60%) —
the oscillating +60 training reward hides the true rate.

**Verdict.** DreamerV3-S from scratch on a hard chase collapses/oscillates. The recipe that works:
a **curriculum ladder, warm-starting each rung from the previous rung's best checkpoint**, plus
**dense multi-checkpoint greedy eval** (never trust the latest ckpt). Caveats: generous catch
radius (2.0–2.5 m, ~20–25× the drone) and small dome — the task is easy relative to the redesign.

**Artifacts.** Checkpoints `results/chase_iter{4,6,7}/`; catch videos `results/CHASE_ITER{4,5,7}_*`.
*(All under gitignored `results/`.)*

---

## E2 — R1 reproduction study (`chase_easy2`, 3 seeds)

**Question.** Does the ~70% single-seed curriculum result reproduce across independent seeds, and
does the policy actually *pursue*?

**Setup.**
- Campaign `2026-08-chase-hardening`, rung **R1**, config `chase_easy2` (hash `58591240…`).
- Drone `cf2x`. Model DreamerV3 **XS**. Seeds 1, 2, 3 each from scratch. Dome 20 m, pursuer spawns
  at centre, initial separation ~7.3 m, catch 2.0 m, target ~1.5 m/s, episode ≤600 steps (20 s).
- **Precommitted gates** (calibrated to baselines): mean success ≥55%, worst seed ≥40%, **pursuit
  alignment ≥0.45**, initial separation ≥5 m, steps-to-catch ≥40, idle ≤0.5, sep-closed ≥0.5.
- Baselines measured first: scripted lead-pursuit 100% / align 0.69 / 75 steps; random 0% / −0.75.

**Result.**

| metric | value | gate | verdict |
|---|---|---|---|
| mean success | **75.6% ±11.3%** (seeds 83/90/53%) | ≥55% | PASS |
| worst seed | 53.3% | ≥40% | PASS |
| **pursuit alignment** | **0.28** | ≥0.45 | **FAIL** |
| steps-to-catch | 214 (vs scripted 75) | ≥40 | PASS |
| idle / sep-closed | 0.006 / 0.69 | — | PASS |

**Verdict.** The catch-*rate* reproduces across seeds, **but the policy loiters rather than
pursues** — alignment 0.28 vs scripted 0.69, catches take ~3× longer. Root cause: a fixed,
non-evasive route makes "get onto the route and wait" near-optimal. Not overridden (miss is large,
consistent, video-confirmed). Motivates a task where waiting fails.

**Artifacts.** `results/R1_report/assets/` — learning curves, per-seed eval JSON, best-catch
videos, contact sheet. Eval records: `experiments/campaigns/2026-08-chase-hardening/evals/`.

---

## E3 — `chase_open` (pursuit-forcing attempt)

**Question.** Does making loitering *bleed reward* (open space + per-step distance penalty) force
genuine pursuit?

**Setup.**
- Campaign `2026-08-chase-hardening`, rung **OPEN**, config `chase_open`. Drone `cf2x`.
- Changes vs R1: added non-telescoping **`distance_penalty` 0.02**; dome 20 m → **500 m**;
  **decoupled spawn** (`chase_spawn_radius` 10) from wide waypoints; catch radius 2.0 → **0.75 m**.
- Baseline measured: scripted 80% / align 0.39 (open space carries a larger lead angle, so
  "genuine pursuit" recalibrates to ~0.39 here, not R1's 0.69).

**Result.** **No learned-policy evaluation** — stopped by design decision. Early training was
healthy (world-model loss 1600→34 in 3 k steps; episodes terminate ~100 steps; reward ~−75 by
design since the distance penalty makes absolute reward negative). But measured initial separations
under the bounded spawn were only **~3.8–7.1 m**, so many episodes are won by a cheap close-spawn
catch (scripted demo caught in 23 steps / 0.8 s) — the task does **not** force *sustained* pursuit.

**Verdict.** The reward/task machinery works (distance penalty, spawn/waypoint decoupling, suitable
baseline), but the bounded spawn re-introduces cheap catches. Fix is a **larger spawn radius + a
faster target**, not a reward change — carried into E4.

**Artifacts.** `results/chase_open_report/assets/` — `chase_open.yaml`, baseline JSON, scripted demo.

---

## E4 — `chase-pursuit` (65 m far-spawn)

**Question.** With a genuinely far start (~65 m) and the pursuit-forcing reward, does DreamerV3
reach scripted-level *pursuit* (alignment)?

**Setup.**
- Campaign `2026-08-chase-pursuit`, rung **P1**, config `chase_pursuit_d100_v2` (hash `60601447…`).
- Drone `cf2x`. Model DreamerV3 **S**. Initial separation **68.8 m**. Catch ~1 m, target ~2 m/s.
- Launched 7 seeds; **3 evaluated** (s2, s4, s5) — 4 lost to vast.ai provisioning flakiness.
- Precommitted gates: mean ≥30%, worst ≥15%, **alignment ≥0.48**, sep ≥30 m. Cost $4.52.

**Result.**

| metric | value | gate | verdict |
|---|---|---|---|
| mean success | **2.2% ±1.1%** (seeds 3.3/3.3/0%) | ≥30% | FAIL |
| worst seed | 0.0% | ≥15% | FAIL |
| **pursuit alignment** | **0.129** (early-pursuit 0.33–0.48) | ≥0.48 | FAIL |
| idle / sep-closed | 0.002 / 0.62 | — | genuine motion |

**Verdict.** A clean, precommitted **negative result** — but an informative one: the policy shows
**genuine active pursuit** (near-zero idle, closes 62% of separation, early-pursuit alignment up to
0.48), *not* the E2 loiter. The binding constraint was **throughput**: ~1.5 steps/s on cheap boxes
capped training at 25–72 k steps, far below the ~300 k the 65 m task needs — undertrained, not
misdesigned. Hands off: either rerun to ~300 k on faster boxes, or redesign the task (→ E5).

**Artifacts.** `experiments/campaigns/2026-08-chase-pursuit/` — `evals/AGG_P1.json`, per-seed evals,
health, trajectories.

---

## E5 — `chase-big` (realistic redesign) — **SOLVED**

**Question.** With a realistic airframe, sensible catch tolerance, redesigned reward and adequate
training, can the learned policy solve the chase and match a hand-tuned classical controller?

**Setup.**
- Campaign `2026-09-chase-big`, final config `chase_pursuit_big_v3.yaml`.
- Drone **`primitive_drone`** (~45 cm motor-to-motor, 1 kg, PyFlyt-pretuned). Model DreamerV3 **S**.
  **State-only obs** (15-dim, no camera — the target's relative pos+vel are already in the state,
  and dropping depth cut the world model 7.94M→2.44M params and sped training).
- Start ~38 m avg (10–55 m), pursuer cap ~5 m/s (achieves ~3), target ~2 m/s (`speed_cap` 0.40),
  **catch 1.2 m** (~2.7× the drone), episode 2500 steps (83 s).
- **Reward:** catch **150 flat + up to 150 time-decay** (catch early is strictly optimal),
  collision **−1000**, telescoping shaping 1.0, per-step `distance_penalty` 0.01, and a **dense
  pursuer ground-avoidance penalty** (`pursuer_ground_margin` 2.0, `pursuer_ground_penalty` 8.0).
- Cost ~$6 vast.ai total.

**Result.**
- Best checkpoint `ckpt_332000`: **99.3% success** (497/500 episodes across three seed banks),
  **zero crashes**.
- **Definitive 1000-episode head-to-head** vs a hand-tuned scripted lead-pursuit (clip 0.7), on
  identical start positions:

| | DreamerV3 (learned) | scripted lead-pursuit (tuned) |
|---|---|---|
| success | **99.3%** | 98.9% |
| avg catch time | **17.2 s** | 18.8 s |
| crashes | **1** | 8 |

- **Development arc:** cf2x vision 2% → realistic redesign 77% → higher target altitude + more time
  94% → continued training 97% → **dense ground penalty 99.3%**.
- **The crash-avoidance breakthrough.** At the ~97% plateau, categorizing misses showed **all**
  failures were pursuer **crashes** (mid-chase ground collisions), not near-misses — the policy
  caught 100% of episodes it survived. The −1000 collision penalty is *sparse* (no gradient until
  already crashing); adding the **dense** ground-avoidance penalty drove crashes 2.5%→0 and success
  97%→99.3%. Lesson: *categorize the failure mode, don't just chase the success number.*
- **Obs ablation (60 ep):** zero x/y → 98% (horizontal position ~redundant), shift x/y by 200 m →
  83% (partial map-coordinate overfitting), zero altitude z → **3%** (height is critical — it
  carries the ground-avoidance mechanism). Next iteration should use 10 obs numbers (drop x/y and
  the always-zero wind).
- **Residual ~0.7%:** a few hardest far-start episodes where the pursuer tracks to 1.37–1.75 m but
  can't tuck inside 1.2 m within the clock — not wander/crash, just final-approach precision. The
  farthest starts (~55 m) are still caught in 13–17 s, so it's precision-limited, not distance-limited.

**Scripted baseline (the opponent), for the record.** Proportional lead-pursuit, one line:
`action = clip(0.4·(target_rel + target_vel), −0.7, +0.7)` — aim at the target plus a lead for its
velocity, proportional gain, thrust clipped to ~70% (tuned optimum: 0.5→78%, 0.7→98.9%, 0.9→80%,
1.0→65% as the QuadX controller destabilizes). It plans no trajectory and has no notion of the
ground — hence its 8 crashes vs the learned policy's 1.

**Verdict.** **Solved.** On raw success the learned policy is a statistical tie with the best
hand-tuned controller (both ~99% on a non-evasive target given enough time), but it is decidedly
**more efficient (~9% faster, it intercepts rather than tail-chases) and safer (~8× fewer crashes,
having learned collision avoidance the scripted law lacks)**. The value of RL here is efficiency +
safety, not raw catch rate.

**Artifacts.** `experiments/campaigns/2026-09-chase-big/` (evals, `head2head_1000.json`,
trajectories); videos `results/chase_big_report/videos/` (`S_vs_scripted_sidebyside.mp4`,
`S_994pct_final.mp4`, `FARCATCH_*`, `MISS_*`); banked checkpoint `resume_ckpts/best_332k_994pct.ckpt`.
Analysis tooling in `scripts/analysis/`. *(Videos/checkpoints/trajectories are gitignored.)*

---

## Cross-cutting lessons

1. **Categorize the failure MODE, don't chase the success number.** The "97% precision ceiling"
   was wrong — all misses were crashes (E5). One `categorize_misses.py` run reframed the campaign.
2. **Eval noise floor.** The sim is chaotic + non-deterministic; a 100-ep eval carries ±3–4%
   variance. Single 100-ep evals of a 97% policy spiked to 98–99% by luck. Confirm any "new best"
   on ≥300–400 episodes across seeds. And per-checkpoint greedy varies wildly (E1: 15%→60%) —
   eval a spread of checkpoints, never just the latest.
3. **Warm-start needs the replay buffer.** Curriculum warm-starts (E1, E5) work, but expect a
   transient dip into a changed task that recovers once the buffer refills; save the buffer
   (`buffer.checkpoint: true`) for seamless resumes.
4. **Behaviour gates catch what success hides.** E2's 76% looked great but the policy loitered;
   the alignment gate exposed it. Numbers veto, eyes advise.
5. **Throughput is a first-class constraint.** E4 was undertrained (~1.5 sps → 25–72 k steps) —
   a negative result driven by compute budget, not method. The state-only model in E5 is tiny, so
   cheap GPUs are correct (powerful cards ran at ~15–19% utilisation).

## Reproduction

Train the solved task:

```bash
./venv/bin/python scripts/train_dreamer.py exp=drone_chase_fast_S \
  env.wrapper.config_path=configs/target/chase_pursuit_big_v3.yaml \
  algo.total_steps=2000000 env.num_envs=8 fabric.accelerator=gpu fabric.precision=16-mixed
```

Evaluate a checkpoint (use ≥300–400 episodes near the ceiling):

```bash
./venv/bin/python scripts/eval_target_ckpt.py <ckpt.ckpt> \
  --task chase --config configs/target/chase_pursuit_big_v3.yaml --episodes 400 --seed 1000
```

Learned-vs-scripted on identical starts: `scripts/analysis/head2head_1000.py`.
Failure-mode categorization: `scripts/analysis/categorize_misses.py`. Obs ablation:
`scripts/analysis/ablate_obs.py`.
