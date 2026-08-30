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
| `.claude/agents/` | The roster: planner, runner, monitor, evaluator, reviewer, env-engineer. |
| `.claude/commands/loop-experiments.md` | The wake procedure. One decision per wake. |

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
