# Drone Pursuit with DreamerV3 — Realistic Chase Task

**Task.** A quad-rotor *pursuer* must intercept a second quad-rotor *target* that flies a
randomly re-sampled, non-evasive waypoint route in a PyFlyt/PyBullet simulator. Both drones are
the same airframe (`primitive_drone`, ~45 cm motor-to-motor, 1 kg) under velocity control
(mode 6, 30 Hz). "Catch" = the pursuer comes within **1.2 m** (center-to-center) of the target.
Observation is **state-only** (pursuer pose + target relative position & velocity); the policy is
**DreamerV3 (size S)**, trained from scratch by RL — no pursuit law is hand-coded.

Episodes: target starts ~38 m away on average (range ~10–55 m), moves at ~2 m/s (pursuer ~5 m/s
cap), 1500-step (50 s) time limit.

---

## Headline result

**The learned policy reaches ~97% catch success on the realistic task, matching a hand-tuned
classical pursuit controller — and catches ~20–40% faster.**

| policy | success (400-ep) | avg catch time (fair, common episodes) |
|---|---|---|
| **DreamerV3 (learned)** | **97%** | **12.2 s** |
| scripted lead-pursuit, tuned | ~95% | 15.4 s |
| scripted lead-pursuit, default | ~78% | 21.0 s |

The learned policy is at **parity on success** with the best hand-tuned controller (both near the
task's structural ceiling), but is **decisively more efficient**: on episodes all policies catch,
it intercepts in **12.2 s vs 15.4 s** (tuned) / **21.0 s** (default). Qualitatively it *cuts the
target off* (interception) rather than tail-chasing.

---

## The development arc

| iteration | change | best success |
|---|---|---|
| cf2x nano (9 cm), vision-based | original task | **2%** (loiter, not pursuit) |
| big-drone redesign | 45 cm airframe, state-only, sensible 1.2 m catch, redesigned reward, achievable speeds | **77%** |
| v2 | + raise target altitude (kill ground-collisions) + more episode time | **94%** |
| continued training (~290k steps) | warm-start + more steps | **97%** |

Reward: catch 150 + up to 150 time-decay bonus (catch *early* is strictly optimal), collision
−1000 (anti-crash), telescoping distance shaping, small per-step distance penalty. No term rewards
alignment directly (that would Goodhart the metric).

---

## Why not 99%? (well-established)

The residual ~3% are the **hardest far-start geometries**: the pursuer tracks the target down to
~1.25 m but can't tuck inside the 1.2 m tolerance within the time limit (final-approach control
precision on a moving target). Evidence this is a **structural task ceiling**, not a training gap:

1. A **hand-tuned scripted expert also caps ~95%** on the identical task — two independent methods
   hit the same wall.
2. **More episode time did not help** (96%→97%); the misses stabilize at ~1.25 m, they don't
   simply run out of clock.

**Methodological note (important).** The simulator is chaotic, so a 100-episode eval carries
**±3–4% noise**. Near the ceiling this is deceptive: single 100-episode evals of the 97% policy
spiked to 98% and even 99% by luck, but a **400-episode eval settles at 97.0%** (95% CI
95.3–98.7%). All headline numbers here use ≥400 episodes.

---

## Videos (`results/chase_big_report/videos/`)

- **`S_vs_scripted_sidebyside.mp4`** — learned vs tuned-scripted on the *same* 43 m episode:
  learned catches in **9.2 s**, scripted in **13.4 s** (interception vs tail-chase).
- **`S_97pct_coord_1_start53m.mp4` / `_2_start41m.mp4` / `_3_start7m.mp4`** — coordinate
  trajectory tracks of the 97% policy across near/mid/far starts.
- **`S_97pct_realmesh.mp4`** — real PyBullet drone-mesh render of the 97% policy catching.
- Earlier milestones: `S_v2_94pct_realmesh.mp4`, `S_84k_realmesh_catch.mp4`.

## Artifacts
- Best checkpoint: `ckpt_290000` (97%, 400-ep). Evals: `experiments/campaigns/2026-09-chase-big/evals/`.
- Total compute for the whole campaign: ~$5 of vast.ai GPU (cheap RTX-class boxes; the state-only
  model is tiny and under-utilizes powerful GPUs).
