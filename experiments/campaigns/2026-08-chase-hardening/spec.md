# Campaign: chase-hardening — make the chase worth winning, then win it

> **Status: blocked on one human decision.** The baselines invalidated this campaign's
> original premise before any GPU spend — see `baselines/FINDINGS.md`. Read the fork below,
> pick a track, and this spec is ready to run.

## Why

`configs/target/chase_easy2.yaml` reached roughly 70% greedy catch rate on a single seed,
locally, and the plan was to reproduce that across seeds on GPU.

The scripted baselines say that is not worth doing. **Scripted lead pursuit — six lines,
reading exactly the same observation the agent gets — solves `chase_easy2` 100% of the time.**
It still solves it 100% of the time with the target at parity speed, above parity speed, and
with the target's velocity hidden and its position noised by 1.5 m. A random policy scores 0%,
so the task is not accidentally winnable; it is specifically *trivially* winnable by a
reactive controller.

The reason is structural rather than a matter of difficulty settings. The target follows a
fixed closed waypoint route that ignores the pursuer, so it always comes back around, and the
optimal strategy is to get onto the route and wait. `DroneChaseEnv` is a rendezvous task, not
an interception task, and no `EnvConfig` field changes that because evasion is not a parameter.

So reproducing 70% here would demonstrate that DreamerV3 does, less reliably, what a trivial
controller does perfectly. That is not showable. The campaign's real first job is to make the
task one a hand-written controller cannot solve.

## The fork — pick one before starting

**Track A (recommended): make the target evade.** Approve the ~10-line change to
`_target_command` in `envs/tasks/drone_chase_env.py` sketched in `baselines/FINDINGS.md`, plus
two `EnvConfig` fields defaulting to `0.0` so every existing config and past result keeps its
exact behaviour. Then R1 below finds the evasion strength at which scripted pursuit fails, and
the campaign shows a learned policy succeeding where it cannot. This is the result worth
showing, and it is the direction `todo.md` already points.

**Track B: keep `envs/` frozen and claim less.** Skip R1, start at R2, and state the result
honestly as "DreamerV3 learns the rendezvous task from state observations, reaching X% where
scripted pursuit reaches 100%". Cheaper and lower risk, but the headline is weak and the
obvious question has no answer.

Everything below assumes **Track A**. Under Track B, delete R1 and R3 and treat R2 as the
whole campaign with the bar lowered to "reproduces across seeds".

## Ground rules

- Budget: **$15 hard cap** (external guard). Wall clock: **12 h TTL**.
- Concurrency: at most **3 instances**.
- Overrides: **2 for the campaign**. Integrity failures are never overridable.
- Task `chase`, experiment config `exp=drone_chase`.
- Fixed training config on every rung, so rungs compare: `algo=dreamer_v3_XS`,
  `env.num_envs=4`, `fabric.accelerator=gpu`, `fabric.precision=16-mixed`,
  `algo.total_steps=1500000` (a ceiling, not a target), `metric.log_every=1000`,
  `checkpoint.every=50000`.
- Early stop: `--stop-metric Rewards/rew_avg --stop-threshold 40 --stop-window 5
  --stop-patience 3`. A catch is +60 against −0.01/step, so a sustained 40 means it is
  catching most episodes. If no run ever stops early, that is itself the finding.
- Evaluation: `tools/evaluate_ckpt.py`, greedy, **30 episodes**, eval seed base **1000**, with
  `--trajectories` so a contact sheet can be built. The reviewer re-evaluates 20 episodes at
  seed base **5000** and reads the contact sheet.
- Reporting: mean ± standard error over seeds `{1, 2, 3}` **and the worst seed**. Every rung
  reports the scripted-pursuit number for the same config beside it — that comparison is the
  point of this campaign.
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

These are calibrated against measurement, not taste: on the real env scripted pursuit measures
**0.69** alignment and random measures **−0.75**, so 0.45 separates a pursuer from a wanderer
without demanding the policy match a hand-written controller. 5.0 m is a quarter of the dome,
below which a catch is mostly spawn geometry. Scripted pursuit's fastest catch is 40 steps, so
anything faster did not really cross the gap.

## Rungs

### R0 — calibration (gate: the cost model, not a result)

One instance, the R2 config, seed 1, **30 minutes** of training, then read `Time/sps_train`.

- Accept: throughput measured; cost per seed to the 1.5M ceiling written in here.
- If projected three-seed cost for R2 exceeds **$7**, replan — lower the ceiling or drop to two
  seeds and say so in the report.
- **This run continues as R2 seed 1.** Do not discard it.

Measured: `sps_train = ____`, `$____` per seed, `$____` for a three-seed wave.

### R1 — find the evasion strength that defeats scripted pursuit (free, CPU, before the session)

Not a training rung. With the evasion change in, sweep `chase_target_evasion` locally:

```
for e in 0.0 0.2 0.4 0.6 0.8; do
  # write configs/target/chase_evade_$e.yaml = chase_easy2 + chase_target_evasion: $e
  tools/run_baselines.py --env-config configs/target/chase_evade_$e.yaml --task chase \
    --episodes 20 --seed 1000 --json baselines/chase_evade_$e.json
done
```

- Accept: an evasion strength where scripted pursuit lands **between 20% and 70%** — hard
  enough that reaction is not sufficient, winnable enough that the task is not impossible.
  That config becomes R2.
- If pursuit stays above 70% at every strength, evasion is not the binding constraint; report
  that and stop rather than guessing at another axis.
- If it drops below 20% everywhere, the task is likely unwinnable; back off the strength or
  widen `target_reach_distance`.

Chosen: `chase_target_evasion = ____`, scripted pursuit `____%`, alignment `____`.

### R2 — beat the controller (the headline gate)

The config chosen in R1, seeds `{1, 2, 3}`, from scratch.

- Accept: mean success **above scripted pursuit's number on the same config**, every seed
  within 15 points of the mean, all behaviour criteria met, reviewer verdict `pass`.
- This is a *relative* bar on purpose. An absolute percentage means nothing on a task nobody
  has run; beating the hand-written controller means something on any task.
- Fail policy: if two or three seeds beat pursuit, retry the failing seed **once**. If at most
  one does, conclude **"does not beat scripted pursuit"** and report it. That is a real finding
  and the campaign ends on it — do not go seed shopping.

### R3 — how much evasion can it take (opportunistic, needs ≥ $6 headroom)

The next evasion strength up from R2, where scripted pursuit is near 0%. Seeds `{1, 2}`.

- Accept: mean **≥ 25%** with behaviour criteria met. Anything above pursuit's ~0% is the
  interesting claim; the absolute number is secondary.
- One axis changed from R2, so a failure here attributes cleanly to evasion strength.

### R4 — wind on top (opportunistic, needs ≥ $4 headroom)

R2's config plus `wind_enabled: true`, `wind_strength: 0.2`, matching `chase.yaml`. Seeds
`{1, 2}`. Wind is a velocity-setpoint bias and already correct — do not touch `envs/`.

- Accept: mean within **15 points** of R2, behaviour criteria met.

## Budget allocation

| rung | instances | projected | running total |
|---|---|---|---|
| R1 | 0 (CPU, local, free) | $0 | $0 |
| R0 | 1 (continues into R2) | ~$0.5 | $0.5 |
| R2 | 3 | ~$6 | $6.5 |
| R3 | 2 | ~$4 | $10.5 |
| R4 | 2 | ~$3 | $13.5 |

Reserve **$2 unspent**; never launch a wave whose projection exceeds headroom minus the
reserve. Drop R4 first, then R3. If R2 needs a retry seed, both are cancelled — beating the
controller once, credibly, matters more than two extra numbers on top.

## Non-goals

- **No `envs/` changes beyond the approved evasion term.** Anything else halts for review.
- **No hyperparameter tuning.** A rung that cannot learn without tuning is telling you about
  the task; journal it and stop that axis.
- **No warm-starting between rungs.**
- **No AirSim**, no fuel constraint, no partial observability. The baselines already showed
  partial observability does not make this task hard, so it is not the next axis either.
