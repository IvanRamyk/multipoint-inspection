# Autonomous experiment workflow — operator runbook

A spec-driven loop that runs RL experiments unattended: it provisions GPUs, launches
DreamerV3 runs, decides whether each one is learning, kills the bad ones, grades the good
ones across seeds, and gates every rung behind an independent review. You write a spec in the
evening and read a report in the morning.

## Why it is built this way

Two facts drive the whole design.

**RL fails silently.** A broken env, an unreachable reward, or a diverged world model all
produce a running process and a flat curve, never a crash. So no single number is trusted:
`tools/run_health.py` weighs reward improvement against its own noise *and* whether episodes
ever terminate, and the acceptance gate is a rollout evaluation, not a training metric.
Success rate is not even in TensorBoard — `Rewards/rew_avg` is a proxy used for early stopping
and nothing else.

**Autonomous agents loosen the constraints they are under.** The documented failure mode is an
agent editing its own runner to extend a timeout instead of fixing what was slow. So the spend
cap does not live in the repo: `tools/setup_guard.sh` copies the guard to `~/.dreamer-guard/`
and runs it there, in its own process, holding its own copy of the code. Nothing an agent
edits can change what it does for the rest of the session.

## Running a session

```bash
# 1. Author the campaign. Copy the template, write the "Why" and the ground rules.
cp -r experiments/campaigns/TEMPLATE experiments/campaigns/2026-09-my-campaign

# 2. Have the planner propose the rung ladder and numbers, then read and edit plan.md.
#    (Ask Claude to run the experiment-planner agent on the campaign.)

# 3. Start the enforcement layer YOURSELF. Starting it is how you authorize the spend.
bash tools/setup_guard.sh --cap 15 --ttl 12

# 4. Start the loop and go to sleep.
#    /loop 30m /loop-experiments

# 5. In the morning: read the final report at the end of the campaign's journal.md.

# Teardown, any time:
bash tools/setup_guard.sh --status     # what has been spent
bash tools/setup_guard.sh --stop       # stop the guard (does NOT destroy instances)
bash tools/reap_instances.sh           # destroy everything, stop billing
```

To stop the loop mid-session: interrupt the session, or run `tools/reap_instances.sh`. The
guard's `KILLED` flag also halts all launching; only a human should remove it.

## Verify before you spend

Cheapest first. Each level exercises strictly more of the stack.

1. **Fake trainer (free, minutes).** `tools/launch_run.py --backend fake --fake-profile
   {rising,flat,diverging,nan}` writes real TensorBoard events tracing each curve shape, so you
   can confirm `run_health.py` returns `healthy`, `stalled`, `diverged`, `dead` respectively,
   and that evals aggregate and gate correctly. `--dry-run` prints the plan without touching
   anything.
2. **Guard rehearsal (free).** Run `tools/budget_guard.py --state-dir /tmp/guardtest --cap 0.002
   --ttl 99 --poll 5 --vastai <stub>` against a stub CLI and confirm it destroys and latches
   `KILLED`. Use a throwaway state dir — the hook correctly refuses to let anything clear the
   real kill switch.
3. **CPU sanity (free, 1–2 h).** A mini-spec with `--backend local` and a few thousand steps.
   Real sheeprl, real tfevents, real eval rollouts, real early stopping — everything but vast.ai.
4. **GPU rehearsal (~$1–2, ~1 h).** One instance, an easy rung, a short ceiling, guard at
   `--cap 2 --ttl 2`, driven by the real loop. This is also how you calibrate the health
   thresholds and measure throughput for the cost model.

## The pieces

| Path | What it is |
|---|---|
| `constitution.md` | Non-negotiable rules every agent inherits. Only a human edits it. |
| `campaigns/TEMPLATE/` | Copy this per campaign. `spec.md` holds the precommitted criteria. |
| `campaigns/<c>/journal.md` | Append-only lab notebook and the final report. |
| `campaigns/<c>/runs.jsonl` | Run registry. Every run is registered before money is spent. |
| `state/current.json` | The orchestrator's checkpoint — first thing it reads each wake. |
| `state/budget.json` | Written only by the guard. Read-only to everything else. |
| `state/KILLED` | Session over. Launching refuses while it exists. |
| `tools/` | Deterministic layer. Agents call these; they must not rewrite them. |
| `eval/behaviour_metrics.py` | Turns trajectories into the pursuit-quality numbers. |
| `tools/run_baselines.py` | Scripted pursuit / random, plus the trivial-and-unwinnable check. |
| `tools/make_contact_sheet.py` | Tiles episodes into the one PNG the reviewer looks at. |
| `.claude/agents/` | The roster: planner, runner, monitor, evaluator, reviewer, env-engineer. |
| `.claude/commands/loop-experiments.md` | The wake procedure. One decision per wake. |

## How a good number gets rejected

A success rate says the target was reached. It does not say the policy pursued anything, and
that gap is where a run looks great in a table and disappointing on video. Three layers close it.

**Behaviour metrics (hard gate).** Every eval carries a `behaviour` block computed from the
recorded trajectories. The load-bearing number is `mean_pursuit_alignment` — the cosine between
the drone's velocity and its bearing to the target. Real pursuit runs 0.7–0.9; a policy that
hovers until the target arrives sits near 0 no matter how often it "succeeds".
`station_keeping_fraction`, `travel_ratio`, `initial_separation` and `steps_to_catch` catch the
rest. These are precommitted in `spec.md` and enforced by `aggregate_seeds.py`, so:

| behaviour | success rate | verdict |
|---|---|---|
| genuine pursuit | 67% | pass |
| holds position, target flies in | 72% | **fail** — alignment 0.04 |
| catches handed over at spawn | 98% | **fail** — 1.7 m gap, caught in 19 steps |

**Baselines (before spending).** `tools/run_baselines.py` runs scripted pursuit and random on a
config. If pursuit already wins almost always, the rung proves nothing; if random wins often, it
proves less than nothing; if pursuit cannot win at all, no training will. Scripted pursuit's
alignment is also the reference for what "looks like pursuit" means on that config.

**Contact sheet (advisory).** `tools/make_contact_sheet.py` tiles episode trajectories into one
PNG, and the reviewer reads it — a single image, a few thousand tokens, no video decoding. It is
looking for what a human notices in a demo: earned catches, competent flight, a target that
actually ran. A visual impression may only raise `suspicious`, which halts for a human; it can
never fail a rung by itself. The sheet is a reconstruction from position tracks, and taste is
not a precommitted criterion. **Numbers veto, eyes advise.**

When a behaviour gate fails, the fix is usually an env parameter rather than more training. The
symptom-to-knob table lives in `.claude/agents/env-engineer.md`.

## When the gate is wrong

Thresholds are written before a single run exists, so some of them are guesses, and discarding a
genuinely good result over a threshold that was slightly off is real waste. The orchestrator can
continue past a failed gate **twice per campaign**, via `tools/request_override.py`.

It is not a way to lower the bar. The spec is never edited, `aggregate_seeds.py` keeps reporting
FAIL, and the override is written as an exception *against* that failure — carrying the measured
value, the threshold, the relative shortfall, the agent's argument, and the blind reviewer's own
verdict. It leads the final report, so a reader who stops after the first paragraph still knows
which results rest on a judgment call.

Two refusals are absolute. **Integrity failures** cannot be overridden by any argument: a
checksum mismatch, an eval against a different config, a synthetic eval, or fewer seeds than the
minimum mean the number is untrustworthy rather than borderline, so there is nothing to exercise
judgment about. And a **spent budget** ends it: two missed gates in one campaign means either the
thresholds or the task are wrong, and both are the human's call.

```
tools/request_override.py --campaign <c> --list     # what's left
bash experiments/verify_overrides.sh                # the limits still bind
```

## How the numbers stay honest

- **Three seeds minimum**, reported as mean ± standard error *plus the worst seed*. A rung
  where the mean clears the bar but one seed collapsed has not passed — `aggregate_seeds.py`
  enforces the per-seed floor separately from the mean.
- **Criteria are precommitted** in `spec.md` before the first launch. `launch_run.py` refuses
  to start a run in a campaign with no spec.
- **The reviewer is deliberately blind** to the journal and to every other agent's reasoning.
  It re-evaluates on different eval seeds, verifies the config hash matches the rung's spec,
  checks `tools/checksums.sha256` and the git state of `tools/`, `deploy/`, and `envs/`, and
  inspects episode geometry for reward artefacts before letting the ladder advance.
- **Config hashing** joins the seeds of one experiment cell: the hash covers the env YAML's
  bytes plus the sorted Hydra overrides with the seed removed. Relaunching a covered cell is
  refused, so a double-woken orchestrator cannot double-spend.
