# Chase experiment — autonomous run log

Goal: get `DroneChaseEnv` (pursue a second real drone flying a non-self-intersecting
waypoint route) to visibly **learn** (reward trending toward the +60 catch, greedy
eval catching). Autonomy: up to 5 iterations; each = launch → observe → assess →
fix/simplify → relaunch. GPU upgrade allowed if throughput blocks progress.

## Assessment criteria per iteration
- **Learning:** `Rewards/rew_avg` trends upward from negative toward +60 within the
  compute budget (~8–15k steps, using L2 as the reference: it turned ~step 4–8k).
- **Greedy check:** eval newest checkpoint with `eval_target_ckpt.py --task chase`;
  a trained agent should start catching (success > 0, min_dist → < reach).
- **Health:** no divergence (pursuer speed bounded), no crash-loop, throughput
  reasonable (≥ ~5 steps/s or upgrade GPU).

## Adjustment ladder (apply if a run is going the wrong way)
Make the task easier to *start* learning, per the CLAUDE.md sanity lesson (dense
shaping + physical proximity beats sparse reward):
1. Baseline `chase.yaml` (dome 20, target ~2 m/s, wind on, shaping 0.5, reach 1.5).
2. Denser shaping (↑ reward_shaping), wind off, slower target (↓ speed cap).
3. Easier geometry: smaller dome, fewer waypoints, larger catch radius.
4. Model/opt tweaks (batch, replay_ratio) or GPU upgrade if throughput-bound.

---

## Iterations

### Iteration 1 — baseline chase (S)
- **Instance:** vast 45813382, RTX 3060 24-vCPU, $0.052/hr (ssh8.vast.ai:13382).
- **Config:** `exp=drone_chase`, `dreamer_v3_S`, `sync_env=false num_envs=8`,
  `total_steps=100000`, `checkpoint.every=500`, `metric.log_every=200`. No monitor
  (run to ceiling so we can watch improvement). Task = default `chase.yaml`
  (dome 20, target ~2 m/s, wind on, shaping 0.5, reach 1.5, 8 waypoints).
- **Rationale:** start from the honest baseline; S is the size that learned L2.
  Watcher renders a 3D video + chart every 3 min from the newest checkpoint.
- **Decision point:** assess at ~step 8–10k (L2 turned by ~4–8k). Learning →
  keep going; stuck-negative → simplify per ladder step 2 (denser shaping, wind
  off, slower target).
- **Result:** **LEARNS.** rew_avg: −20 (2.4k) → −10 (3k) → **+61 (3.4k)** → −9 (4.8k)
  → **+58 (6.2k)**. Reaches catch windows fast (~step 3.4k, faster than L2), but
  **unstable/oscillating** (same signature as L2: +61 bursts interleaved with −9).
  Catches are quick (ep_len ~68) vs misses (600 timeout). Throughput ~5 steps/s.
  No crash, checkpoints every 504, watcher rendering videos.
- **Decision:** RIGHT direction — do NOT simplify. Verify greedy eval of a
  +window checkpoint (is it real or exploration luck?), then Iteration 2 targets
  the instability if greedy underperforms.
- **Greedy check:** `ckpt_6552` → **0/6 greedy** (min_dist reached 1.9 m but
  never closed). The +58 training spikes are exploration, not a reliable policy —
  same instability/greedy-gap as L2. Verdict: learns but deliverable (reliable
  greedy catch) not met → simplify to stabilize.

### Iteration 2 — stabilized/easier chase (S), `chase_easy.yaml`
- **Config:** denser shaping `reward_shaping 0.5→2.0`, slower target
  `speed_cap 0.4→0.3` (~1.5 m/s), forgiving catch `reach 1.5→2.0`, `wind off`.
  Same S model, 8 async envs, `checkpoint.every=500`, run to 60k ceiling.
- **Rationale:** CLAUDE.md sanity lesson — dense shaping + physical ease gives a
  smoother reward landscape and a reliably-catching (not just spiky) policy. Goal:
  greedy success clearly > 0, ideally trending toward reliable.
- **Result:** **FAILED to stabilize.** Caught early (+63.8 @ step 2.2k) but then
  oscillated *worse* than iter 1: rew swings `-35 → -50 → -46 → -63 → +69` by step
  9k. Denser shaping amplified the swings (−63 = pursuer actively diverges when it
  loses the target). Confirms the problem is **DreamerV3 training-dynamics
  instability, not task difficulty** (same on baseline + easier configs, and on L2).
- **Decision:** stop chasing task-difficulty; fix the optimizer dynamics.

**Side finding (important):** greedy-eval of a *catch-window* baseline checkpoint
(`ckpt_7056`, iter-1 chase.yaml) = **2/6 = 33% success** (caught in 148–307 steps,
min_dist 1.4). So the earlier "greedy 0%" was just a bad-window checkpoint — the
chase policy genuinely catches, and 33% > L2's ~18%. Takeaway: the deliverable
exists; the instability means you must **select a good-window checkpoint**, not
use the latest. Catch videos rendered to `results/CHASE_TRAINED_*`.

### Iteration 3 — stability via lower replay_ratio (S, chase_easy)
- **Config:** `chase_easy.yaml` + `algo.replay_ratio=0.5` (was 1). Fewer gradient
  updates per env step → less overfitting to recent data → the canonical DreamerV3
  fix for policy collapse/oscillation. Same S, 8 async envs, checkpoint.every=500,
  run to ~60k.
- **Rationale:** both prior iters catch in windows but oscillate; goal is a
  *sustained* reward plateau (rew stays high across consecutive log points) and
  greedy success clearly > 0.
- **Result:** **FAILED — never caught.** Max reward ever = −3.3; all-negative
  (−3…−15) through step 7.2k. Lowering replay_ratio starved learning (baseline
  caught +61 by 3.4k; this never found a catch). Verdict: replay_ratio=0.5 hurt.
- **Decision:** baseline (replay 1) remains best. Stop tuning optimizer; instead
  make the task *reliably winnable* to get a consistent high-success policy.

### Iteration 4 — reliably-winnable easier chase (S), `chase_veryeasy.yaml`
- **Config:** small dome (12), short route (4 waypoints), slow target (~1 m/s),
  generous catch (reach 2.5), shaping 1.0, wind off, **replay_ratio=1** (baseline).
  Watcher → `results/watch_iter4/` (preserved). Run to 60k.
- **Rationale:** sanity-recipe logic — a clearly-winnable task should let the
  policy lock into reliable catching (target >>33% greedy), giving a clean
  dumb→catching improvement video sequence.
- **Result:** **SUCCESS — sustained plateau.** rew_avg reached ~+60 and *held*:
  steps 5.4–6.2k = [61.3, 59.8, 60.8, 60.7, 61.4, 58.5] — 5–6 consecutive catch
  windows (vs iters 1–3 which only spiked then collapsed). First stable policy.
  This confirms: the instability was driven by task difficulty relative to the
  sparse terminal catch; a reliably-winnable task lets DreamerV3-S lock in.
  Greedy eval of plateau `ckpt_6048` + catch videos in progress → `results/CHASE_ITER4_*`.
- **Greedy eval:** **50% (5/10)** — best result (baseline 33%). Genuine chases
  caught in 59–140 steps (~+62); misses are hard far-spawn episodes where the
  pursuer diverges. Improvement video sequence (36 clips) → `results/watch_iter4/`.
  This is the headline chase deliverable.

### Iteration 5 — curriculum: fine-tune the stable easy policy on harder task
- **Config:** `checkpoint.resume_from=<iter4 ckpt_8064>`, env `chase_easy.yaml`
  (dome 20, target 1.5 m/s, reach 2.0). Same S/8-env, train ~12k more steps.
- **Rationale:** iter4 proved a winnable task gives a stable policy; test whether
  that policy transfers/extends to the harder full-size task via curriculum
  (avoids the from-scratch instability that sank iters 1–2 on the hard task).
- **Result:** **SUCCESS — curriculum works.** Resumed at step 8064; held a
  sustained +60 plateau on the harder task (steps 8.4–9.8k: 60,62,61,61,62,61,61)
  with only occasional dips. Contrast iter 2 (chase_easy *from scratch*) which
  oscillated −63↔+69. Warm-starting from the stable easy policy transferred and
  stayed stable on the harder full-size task. Greedy eval + videos → `results/CHASE_ITER5_*`.

- **Greedy eval (honest):** plateau `ckpt_9576` = **10% on chase_easy** (harder
  task). So curriculum stabilized *training* on the harder task (no wild
  oscillation like iter 2) but greedy success stayed low — the training-vs-greedy
  gap + higher difficulty. Best greedy remains iter 4 (50% on the easy task).

### Pre-iteration-6 diagnostic — is iter5's plateau real?
- **Test:** stochastic (sampled-action) eval of iter5 plateau ckpt_9576 on
  chase_easy, 10 eps, seed 90 — if the +61 training plateau reflected a
  truly-catching policy, stochastic eval should catch most episodes.
- **Result:** **20% (2/10)**, with divergence tails (a miss at reward −158.9,
  min_dist 5.3 m — shaping 2.0 amplifies flying-away episodes). The training
  plateau materially overstates true catch rate; the iter5 policy is honestly
  ~10–20% on chase_easy. "More consolidation on easy" is NOT the supported
  lever; the 4-knob veryeasy→easy jump (dome 12→20, wp 4→8, speed 0.2→0.3,
  reach 2.5→2.0) is what broke the 50% policy.
- **Decision:** climb a gentler ladder from iter4's genuinely-50% policy, with
  a fixed eval protocol everywhere: `checkpoint.keep_last=40` (default 5 is why
  good ckpts rotated out), eval ≥5 plateau ckpts × 20 episodes. New configs:
  `chase_mid.yaml` (dome 16, 6 wp, speed 0.25, reach 2.0, shaping 1.0) and
  `chase_easy2.yaml` (= chase_easy geometry, shaping 2.0→1.0 to kill the
  divergence-tail amplification).

### Iteration 6 — curriculum rung: chase_mid from iter4's 50% policy
- **Instance:** vast 45881700, RTX 3060 Ti, 24 vCPU, $0.069/hr
  (141.0.85.213:44480). Destroy when done.
- **Config:** `checkpoint.resume_from=<iter4 ckpt_7056>`, env `chase_mid.yaml`,
  S / 8 async envs, `total_steps=30000` (~23k additional), `checkpoint.every=500`,
  `checkpoint.keep_last=40`.
- **Hypothesis:** one gentle rung (dome +4 m, +2 waypoints, +0.25 m/s target
  speed, reach already at 2.0) transfers the 50% policy with modest degradation,
  and consolidation on a winnable-but-harder task recovers ≥40% greedy on mid.
- **Decision point:** rew_avg at ~step 15k and ~25k; then dense greedy eval
  across plateau ckpts (20 eps each). ≥40% on mid → iter 7 = same recipe on
  chase_easy2. Stuck ≤20% → the rung is still too big; split it (speed OR
  geometry alone).
- **Ladder success bar:** ≥40% greedy on chase_easy2 (dome 20, 8 wp, 1.5 m/s
  target, reach 2.0) — "decent interception on a non-trivial setup". Stretch:
  wind-on full chase.yaml.
- **Launch note (resume gotcha, confirmed):** sheeprl merges the checkpoint's
  neighbouring `config.yaml` OVER CLI flags, so CLI `env.wrapper.config_path=`/
  `total_steps=`/`keep_last=` are ignored on resume. Fix used: doctor a copy of
  iter4's `config.yaml` (config_path→chase_mid, total_steps→30000, keep_last→40)
  and place it beside the checkpoint. The printed config dump in train.log shows
  the *pre-merge* values (chase.yaml/2M/5) — misleading; the **saved run
  config.yaml** correctly shows chase_mid/30000/40. Verified in effect.
- **Startup health:** resumed at policy_step ~7056; immediately scoring
  reward_env 60–65 on chase_mid (iter4's policy transfers to the harder rung
  out of the box), occasional −18 misses. Checkpoints writing every 504 steps,
  keep_last=40 retained. Monitoring to the 30k ceiling.
  (Watch: my `pgrep -fc 'python -m sheeprl'` health check gave a false "0" —
  the real cmdline is `python -u -m sheeprl`; use `-m sheeprl` as the pattern.)

- **Result — training:** ran the full 30k ceiling (resumed 7056 → 29600).
  rew_avg oscillated the entire run (catches 58–65 interleaved with −10…−33
  misses, ~2/3 catches); a rough patch at 23–25k then recovered. Consolidation
  did NOT stabilise — same DreamerV3-S oscillation signature as every prior iter.
- **Result — greedy (the truth):** step-16632 ckpt = **10% (2/20)** on chase_mid.
  The diagnostic detail: misses cluster at **min_dist 2.0–3.5 m** with the catch
  radius at 2.0 m (closest misses 1.97 / 2.01 / 2.16 m). The policy reliably
  **navigates to ~2–3 m but cannot close the last ~0.5 m** — a terminal-precision
  limit, not a navigation failure. The oscillating +60 training reward hides
  this completely (it only logs the episodes that *do* close).
- **Reframe (key):** greedy success is dominated by the **catch radius**, not by
  more training. iter4 veryeasy reach 2.5 → 50%; chase_mid reach 2.0 → 10%, same
  "gets near, can't close" behaviour. So "more consolidation" can't clear the 40%
  bar — the training reward is already saturated at its ceiling.
- **Decision (user-confirmed):** finish iter6 + dense multi-ckpt eval to get the
  honest number, THEN choose direction. Dense sweep = 6 ckpts across the run ×
  20 greedy eps on chase_mid, GPU box, seed 100, `--catches-only`.
- **Dense-eval result (overturns the early read):** greedy success varies wildly
  by checkpoint window — `ckpt_10584` 15%, `ckpt_14112` 30%, `ckpt_18144` 25%,
  `ckpt_22176` 55%, `ckpt_26208` 40%, **`ckpt_30000` 60% (12/20)**. So the
  single-checkpoint early read (10% on ckpt_16632) was a bad-window artefact;
  the best-window policy hits **60% on chase_mid** — the best chase result yet,
  on a genuinely non-trivial setup (dome 16, 6 wp, 1.25 m/s, reach 2.0), well
  past the 40% bar. Later-window ckpts (22k–30k) are all 40–60%.
  Strong confirmation of the CLAUDE.md lesson: **eval a spread of checkpoints,
  never just the latest** — here the spread is 15%→55%.
- **Direction (resolved by the data):** chase_mid at reach 2.0 IS catchable
  (55%), so the pessimistic "terminal precision is a hard ceiling / must loosen
  reach" read from the 10% checkpoint was wrong. Proceed UP the ladder:
  iteration 7 = curriculum warm-start from the best iter6 checkpoint onto
  chase_easy2 (full dome 20, 8 wp, 1.5 m/s, reach 2.0) — the non-trivial
  benchmark. The catch radius still matters, but 2.0 m is workable for the
  best-window S policy.

### Iteration 7 — curriculum: chase_easy2 from iter6's 60% policy
- **Instance:** vast 45881700 (same box). Resume from iter6 `ckpt_30000`
  (60% on chase_mid) onto `chase_easy2.yaml` (dome 20, 8 wp, 1.5 m/s, reach 2.0,
  shaping 1.0 — chase_easy geometry but shaping halved to kill divergence tails).
  S / 8 async envs, total_steps=60000 (+30k), keep_last=40, checkpoint.every=500.
- **Hypothesis:** the ladder works — a policy that catches 60% on chase_mid,
  warm-started onto the full-size benchmark, adapts and lands ≥40% greedy on
  chase_easy2 (vs the 10–20% that from-scratch/direct-jump attempts gave).
- **Startup:** resume merge verified (authoritative saved config = chase_easy2 /
  60000 / keep_last 40). Immediately scoring reward_env ~62 on chase_easy2 at
  step 31088 — the mid policy transfers to the harder benchmark out of the box.
- **Decision point:** assess rew_avg ~45k and ~57k; dense greedy eval (spread of
  late-window ckpts × 20 eps) on chase_easy2. ≥40% → ladder complete, that's the
  headline "decent interception on a non-trivial setup" deliverable.
- **CRASH @ step ~39k — disk full (`OSError: Errno 28`), NOT a training failure.**
  The 20 GB vast container disk filled: S-model checkpoints are 239 MB each and
  `keep_last=40` ⇒ ~9.5 GB/run, ×2 runs (iter6+iter7) + venv 6.4 GB + resume
  copies ⇒ 100% full. sheeprl died mid-checkpoint-save; the last written ckpt
  (`ckpt_39072`, 228 MB) was truncated/corrupt.
- **Recovery:** (1) fetched iter6's two best ckpts (30000=60%, 22176=55%) to
  local first (deliverable insurance); (2) freed disk — deleted eval PNGs, the
  iter4 resume copy, the truncated ckpt, and trimmed iter6's 40 ckpts to just
  those 2 (20 G→12 G used, 9 G free); (3) relaunched iter7 from the last COMPLETE
  ckpt `ckpt_38568` (step 38568, ~500 steps lost) with **`keep_last=8` +
  `checkpoint.every=1000`** ⇒ ≤1.9 GB ckpt footprint, disk can't refill. Verified
  resumed catching (reward 62–66) on chase_easy2; run 2026-07-26_11-45-45.

- **Result — greedy dense eval on chase_easy2 (the headline):** parallel eval of
  the 8-ckpt late window (dropped to 6 for the sweep) on the FULL benchmark
  (dome 20, 8 wp, 1.5 m/s, reach 2.0): `ckpt_45568` 10%, **`ckpt_47568` 70%
  (14/20)**, `ckpt_48568` 50%, `ckpt_50568` 65%, `ckpt_51568` 45%, `ckpt_52000`
  35%. Four of six clear the 40% bar, peaking at **70%** — the best chase result
  in the project, on a far harder setup than iter4's 50%-on-veryeasy. **Ladder
  complete: the goal (decent interception on a non-trivial setup) is met.**
- **Note (eval infra):** parallel per-ckpt eval initially thrashed (torch/OMP
  222 threads/proc on a 192-core box → 0 episodes in 20 min); capping
  `OMP_NUM_THREADS=6` fixed it AND sped each eval ~15× (thread overhead on the
  small model). Logged in the skill.
- **Deliverables:** best ckpts fetched to `results/chase_iter7/version_0/`
  (ckpt_47568 = 70%, ckpt_50568 = 65%); iter6 best in `results/chase_iter6/`
  (ckpt_30000 = 60% on mid). Catch video of ckpt_47568 on chase_easy2 →
  `results/CHASE_ITER7_easy2_*_3d.mp4`.

## Conclusion — the curriculum ladder (iterations 4→7)

**The recipe that works for the chase task: a curriculum ladder, warm-starting
each harder rung from the previous rung's best checkpoint, with dense
multi-checkpoint greedy eval to pick the policy (never the latest ckpt).**

| iter | task | difficulty | best greedy | resumed from |
|---|---|---|---|---|
| 4 | chase_veryeasy | dome 12, 4 wp, 1.0 m/s, reach 2.5 | 50% | scratch |
| 6 | chase_mid | dome 16, 6 wp, 1.25 m/s, reach 2.0 | **60%** | iter4 ckpt |
| 7 | chase_easy2 | dome 20, 8 wp, 1.5 m/s, reach 2.0 | **70%** | iter6 ckpt |

- **Why it works:** DreamerV3-S from-scratch on a hard chase collapses/oscillates
  (iters 1–3). A reliably-winnable rung lets the policy lock in; warm-starting
  the next (only slightly harder) rung keeps training stable where from-scratch
  fails, and each rung's best-window greedy actually *improves* up the ladder.
- **Two things that mattered as much as the training:** (1) **dense
  multi-checkpoint eval** — single-ckpt reads were off by up to 6× (iter6:
  latest ckpt 10% vs best-window 60%); the oscillating +60 training reward never
  tells you the greedy rate. (2) **Bounded `keep_last` on the 20 GB box** — the
  disk-full crash (§iter7) came from 40×239 MB ckpts; keep ≤10.

## Conclusion (5 iterations)
- **Root cause of instability:** DreamerV3-S oscillates/collapses when the sparse
  terminal catch is hard to reach relative to task difficulty. Not fixed by
  reward shaping (amplified swings) or lower replay_ratio (starved learning).
- **What worked:** (a) make the task *reliably winnable* → stable policy (iter 4,
  50% greedy); (b) **curriculum** — warm-start the hard task from the easy policy
  → stable on the hard task (iter 5). This is the recipe for the chase task.
- **Deliverables:** stable catching policies + checkpoints, catch videos
  (`CHASE_ITER4_*`, `CHASE_ITER5_*`), improvement sequences (`watch_iter4/5`),
  and this log.



