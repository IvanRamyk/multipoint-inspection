# Campaign: <short name>

> Copy this directory to `experiments/campaigns/<YYYY-MM>-<slug>/` and fill it in.
> The header is yours to write; the planner agent proposes the rung ladder and the
> numbers, and you approve or edit them. Nothing may be launched until this file
> exists — `tools/launch_run.py` refuses without it.

## Why

<What question is this campaign answering, and what would a showable result look
like? Two or three sentences. If the honest answer is "I want to know whether X
works", say that — a campaign that ends in a clean negative is a success.>

## Ground rules

- Budget: **$<N> hard cap** (external guard). Wall clock: **<N> h TTL**.
- Concurrency: at most **<N> instances** at once.
- Task: `<chase | target>`, experiment config `exp=drone_<task>`.
- Fixed training config (identical across every rung, so rungs are comparable):
  `algo=dreamer_v3_XS`, `env.num_envs=4`, `fabric.accelerator=gpu`,
  `fabric.precision=16-mixed`, `algo.total_steps=<ceiling>`,
  `metric.log_every=1000`, `checkpoint.every=50000`.
- Early stop: `--stop-metric Rewards/rew_avg --stop-threshold <N>
  --stop-window 5 --stop-patience 3`.
- Evaluation: `tools/evaluate_ckpt.py`, greedy, **<N> episodes**, eval seed base
  `1000`. The reviewer re-evaluates at seed base `5000` with its own episode count.
- Reporting: mean ± standard error over seeds `{1,2,3}`, **plus the worst seed**.
  No cherry-picking, no post-hoc threshold changes.

## Rungs

Each rung is one experiment cell (one env config) evaluated over several seeds.
Acceptance is checked by `tools/aggregate_seeds.py` and then by the
adversarial reviewer; both must pass before the ladder advances.

### R0 — calibration (gate: the cost model)

One instance, `<config>`, seed 1, capped at ~30 minutes of training. Purpose is to
measure `Time/sps_train` and project the cost per seed to the ceiling.

- Accept: throughput measured and projected cost for the next rung computed.
- If the projected next-rung cost exceeds `$<N>`, stop and replan rather than
  launching it.
- This run continues as the first seed of R1 — do not throw it away.

### R1 — <name> (gate: <what this establishes>)

- Env config: `configs/target/<config>.yaml`
- Seeds: `{1, 2, 3}`, trained from scratch.
- Accept: mean success ≥ `<N>%`, every seed ≥ `<N>%`, reviewer verdict `pass`.
- Stretch: mean ≥ `<N>%`.
- Fail policy: <how many retries, and what conclusion to draw if it still fails>.

### R2 — <name>

- Env config: `configs/target/<config>.yaml` (new: `<what changed vs R1>`)
- Seeds: `{1, 2}`.
- Accept: mean ≥ `<N>%` and within `<N>` points of R1's mean.

<Add rungs as needed. Mark any rung that should only run if budget remains as
"opportunistic" and give the headroom it requires.>

## Budget allocation

<Rung-by-rung projected spend, and the ordering. State which rungs may run in
parallel. Never launch a wave whose projected cost exceeds remaining headroom
minus the $2 reserve.>

## Non-goals

- No changes to `envs/` code. (If a rung seems to need one, halt for review.)
- No hyperparameter tuning beyond the fixed config above. If a rung cannot learn
  without tuning, journal it and stop that axis.
- <Anything else deliberately out of scope.>
