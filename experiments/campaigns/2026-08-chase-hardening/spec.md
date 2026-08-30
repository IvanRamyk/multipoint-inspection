# Campaign: chase-hardening — reproduce the chase result across seeds, on GPU

## Why

`configs/target/chase_easy2.yaml` reached roughly 70% greedy catch rate via the ladder
`chase_veryeasy → chase_mid → chase_easy2` — on **one seed, on CPU, locally**. Nothing about it
has been checked: not whether another seed reproduces it, and not whether the policy pursues the
target or is handed catches by a route that happens to pass nearby. Deep RL fails on a quarter to
a third of seeds on tasks the same code solves reliably elsewhere, so one seed is close to no
evidence.

This campaign has two jobs, and the first one matters more.

**1. Prove the autonomous workflow works, end to end, on real GPU runs.** This is the first
unattended session: the guard, the launcher, the health checks, the evaluator, the behaviour
gates, the reviewer, and the loop's own decision-making have only ever run against a fake
backend. Validating them against a task with a **known answer** is the point. If the loop
mishandles something, a familiar task makes that unambiguous; a novel task would leave you unable
to tell whether the pipeline or the environment was at fault.

**2. Turn the single-seed number into a defensible one.** Three seeds, behaviour metrics
confirming it is pursuit rather than luck, then push one axis at a time toward
`configs/target/chase.yaml`, which is the headline config and has never been trained.

### What the baselines say, and why the campaign still runs

`baselines/FINDINGS.md` records a measured fact worth knowing before reading any result from this
campaign: **scripted lead pursuit — six lines, reading exactly the same observation the agent gets
— solves `chase_easy2` 100% of the time**, and keeps doing so at parity target speed, above
parity, and with the target's velocity hidden and its position noised. Random scores 0%, so the
task is not accidentally winnable; it is specifically trivially winnable by a reactive controller.
The cause is structural: the target flies a fixed closed waypoint route that ignores the pursuer,
so it always comes back around and getting onto the route and waiting is enough.

That bounds what this campaign may claim. It is **not** evidence that RL is needed for
interception, and the report must not imply it. What it establishes is that the pipeline learns
this task reliably across seeds, with pursuit-like behaviour, at a cost we can now measure —
which is exactly the base the next campaign needs.

Making the target evade is the next campaign's job, not this one's. It needs a change under
`envs/` (~10 lines in `_target_command`, sketched in `baselines/FINDINGS.md`), and pairing an
unproven environment change with an unproven autonomous loop on the same night is how you end up
unable to debug either.

The deliverable: a per-rung table of mean ± standard error with the worst seed, scripted pursuit's
number beside each rung for honesty, a contact sheet per rung, one episode video of the best
checkpoint, and a measured cost model for planning the next campaign.

## Ground rules

- Budget: **$15 hard cap** (external guard). Wall clock: **12 h TTL**.
- Concurrency: at most **3 instances**.
- Overrides: **2 for the campaign** (`tools/request_override.py`). Integrity failures are never
  overridable.
- Task `chase`, experiment config `exp=drone_chase`.
- Fixed training config on every rung, so rungs compare: `algo=dreamer_v3_XS`,
  `env.num_envs=4`, `fabric.accelerator=gpu`, `fabric.precision=16-mixed`,
  `algo.total_steps=1500000` (a ceiling, not a target), `metric.log_every=1000`,
  `checkpoint.every=50000`.
- Early stop: `--stop-metric Rewards/rew_avg --stop-threshold 40 --stop-window 5
  --stop-patience 3`. A catch is +60 against −0.01/step, so a sustained 40 means it is catching
  most episodes. Runs should stop well before the ceiling; if none do, that is itself the finding.
- Evaluation: `tools/evaluate_ckpt.py`, greedy, **30 episodes**, eval seed base **1000**, with
  `--trajectories` so a contact sheet can be built. The reviewer re-evaluates 20 episodes at seed
  base **5000** and reads the contact sheet.
- Reporting: mean ± standard error over seeds `{1, 2, 3}` **and the worst seed**. Every rung
  reports scripted pursuit's number on the same config beside it.
- Every rung trains **from scratch**. Cross-rung warm-starting previously dropped success to
  10–20% and is out of scope.

## Behaviour criteria (every rung)

Hard gates, checked by `tools/aggregate_seeds.py` beside the success bars.

```
--min-alignment 0.45          mean cosine(velocity, bearing to target)
--min-initial-separation 5.0  metres
--min-catch-steps 40          mean steps to interception
--max-idle-fraction 0.5
--min-separation-closed 0.5
```

Calibrated against measurement, not taste: on the real env scripted pursuit measures **0.69**
alignment and random measures **−0.75**, so 0.45 separates a pursuer from a wanderer without
demanding the policy match a hand-written controller. 5.0 m is a quarter of the dome, below which
a catch is mostly spawn geometry. Scripted pursuit's fastest catch is 40 steps, so anything faster
did not really cross the gap.

These matter more here than on a harder task, precisely because the route passes near the drone:
this config is where "wait and let it come to you" is a live strategy, and the gates are what
catch it.

## Rungs

### R0 — calibration (gate: the cost model, not a result)

One instance, `chase_easy2`, seed 1, **30 minutes** of training, then read `Time/sps_train`.

- Accept: throughput measured; cost per seed to the 1.5M ceiling written in here.
- If projected three-seed R1 cost exceeds **$7**, replan — lower the ceiling or drop to two seeds
  and say so in the report. Do not launch a wave you cannot finish.
- **This run continues as R1 seed 1.** Do not discard it.

Measured: `sps_train = ____`, `$____` per seed, `$____` for R1.

### R1 — reproduce (the headline gate)

`configs/target/chase_easy2.yaml` × seeds `{1, 2, 3}`, from scratch.

- Accept: mean success **≥ 55%**, every seed **≥ 40%**, all behaviour criteria met, reviewer
  verdict `pass`.
- Stretch: mean ≥ 70%, matching the original single-seed claim.
- The bar sits below the 70% being reproduced on purpose: a single seed is an optimistic estimate,
  and 55% across three seeds is a stronger claim than 70% on one.
- Report it as "reproduces across seeds", never as "solves interception" — scripted pursuit gets
  100% here.
- Fail policy: if two or three seeds clear 40%, retry the failing seed **once**. If at most one
  clears it, conclude **"not reproducible"** and report that as the finding. Do not keep
  reseeding; a result that needs seed shopping is not a result.

### R2 — wind (config-only)

New `configs/target/chase_easy2_wind.yaml`: `chase_easy2` plus `wind_enabled: true`,
`wind_strength: 0.2`, matching `chase.yaml`'s wind. Wind is a velocity-setpoint bias and already
implemented correctly — do not touch `envs/`. Seeds `{1, 2}`.

- Accept: mean **≥ 45%** and within **15 points** of R1's mean, behaviour criteria met.
- Run `tools/run_baselines.py` on the new config first; if it comes back `flagged`, fix the config
  before spending.

### R3 — faster target (config-only)

New `configs/target/chase_easy2_fast.yaml`: `chase_easy2` with `chase_target_speed_cap`
`0.3 → 0.4` (~1.5 → ~2 m/s), matching `chase.yaml`. Seeds `{1, 2}`.

- Accept: mean **≥ 35%**, behaviour criteria met.
- A lower bar than R2 deliberately: at 2 m/s the target's speed disadvantage against the pursuer
  largely disappears.
- Independent axis from R2, so the two may run in parallel when headroom allows.

### R4 — the headline config (opportunistic, needs ≥ $4 headroom)

`configs/target/chase.yaml` as written: wind on, target at 0.4, catch radius 1.5,
`reward_shaping: 0.5`. Seeds `{1, 2}`.

- Accept: mean **≥ 30%** with behaviour criteria met. This config has never been trained, so
  anything here is new ground.
- It changes three axes at once relative to R1, so a failure does not attribute. Its value is a
  headline number, not an ablation.

## Budget allocation

| rung | instances | projected | running total |
|---|---|---|---|
| R0 | 1 (continues into R1) | ~$0.5 | $0.5 |
| R1 | 3 | ~$6 | $6.5 |
| R2 + R3 | 4, in two waves or three at once | ~$5 | $11.5 |
| R4 | 2 | ~$3 | $14.5 |

Reserve **$2 unspent**; never launch a wave whose projection exceeds headroom minus the reserve.
Drop R4 first, then R3. If R1 needs a retry seed, R4 is cancelled — reproducing the result matters
more than a headline number on top of it.

## Non-goals

- **No changes to `envs/`.** The evasion term that would make this task non-trivial belongs to the
  next campaign, deliberately not this one. If a rung seems to need an env change, halt for review.
- **No hyperparameter tuning.** DreamerV3's claim is one fixed config across 150+ tasks; a rung
  that cannot learn without tuning is telling you about the task. Journal it and stop that axis.
- **No warm-starting between rungs.**
- **No claim that RL is required for this task.** The baselines say otherwise and the report must
  say so too.
- **No AirSim**, no fuel constraint, no evasive target, no partial observability.

## What this campaign hands to the next one

1. Whether the autonomous loop runs a full night unattended without human rescue.
2. A measured cost model: dollars per seed per million steps on real hardware.
3. Calibrated health thresholds from real GPU curves rather than synthetic ones.
4. Whether DreamerV3 reproduces on this task at all — the control condition for judging a harder
   one.
