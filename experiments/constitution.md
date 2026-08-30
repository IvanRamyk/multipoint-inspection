# Constitution — autonomous RL experimentation

Non-negotiable rules for every agent in this workflow. Read this before acting.
If a rule here conflicts with a campaign spec, this file wins. Only a human may
change this file.

## 1. Domain facts you must not rediscover the hard way

These cost real debugging time to learn. `AGENTS.md` holds the full list; these
are the ones that bite an automated loop specifically.

1. **Wind is a velocity-setpoint bias, never an external force.** The QuadX flies
   in velocity-control mode; an external force fights the inner controller and
   diverges. `envs/backends/pyflyt_backend.py:apply_wind()` is correct — do not
   "improve" it into a force.
2. **Sparse reward collapses the actor to sitting still.** A large collision
   penalty with no shaping makes any movement look catastrophic before the agent
   ever sees a positive reward. Keep dense potential-based shaping on new rungs.
3. **Reward shaping above ~1.0 amplifies divergence tails** on chase (episode
   rewards around −150 on misses). 1.0 is the stable value.
4. **The flight dome is not a physical boundary.** It constrains spawning only;
   the drone drifts past it freely. Do not treat dome size as a safety limit.
5. **A catch radius smaller than the minimum waypoint separation can make a task
   unwinnable** — a random initial policy never lands inside it, so no positive
   reward is ever observed and the world model only learns "sit still".
6. **Cross-rung warm-starting is unproven here** and previously dropped success to
   10–20%. Train each rung from scratch unless a human approves otherwise.
7. **CPU training is debug-only** (~30–60 env-steps/min). Never draw a conclusion
   from a CPU run.
8. **`checkpoint.every` counts env-steps, not gradient steps.**
9. **One training run per box.** `deploy/train_remote.sh` kills any existing run
   on the instance, so N seeds means N instances.

## 2. What you may and may not change

- **Allowed unattended**: creating a new env-config YAML under `configs/target/`,
  changing Hydra overrides, choosing seeds, deciding to kill or retry a run.
- **Requires a human checkpoint**: any change under `envs/`, `scripts/`, or
  `pyproject.toml`. Produce the diff and a rationale, set `halted: true` in
  `experiments/state/current.json`, journal "HUMAN REVIEW NEEDED", and stop
  launching new work. Keep babysitting runs that are already alive.
- **Never edit, under any circumstance**: `tools/**`, `deploy/**`,
  `experiments/state/budget.json`, `experiments/constitution.md`,
  `tools/checksums.sha256`. These are the measurement and enforcement layer. Code
  that grades an experiment must not be editable by the agent being graded.
- **Never call `vastai create` or `vastai destroy` directly.** Use
  `tools/launch_run.py` and `tools/reap_instances.sh`. Direct calls produce
  instances the registry does not know about, which is how money gets lost.

## 3. How success is measured

1. **Success rate comes only from `tools/evaluate_ckpt.py`.** It is not in
   TensorBoard. `Rewards/rew_avg` is a training proxy used for early stopping and
   nothing else. Never report a reward number as a success rate.
2. **Three seeds minimum before any claim.** Report mean ± standard error *and*
   the worst seed, via `tools/aggregate_seeds.py`. A rung where the mean passes
   but one seed collapsed has not passed.
3. **Acceptance criteria are precommitted.** They live in the campaign's
   `spec.md`, written before the first launch. If a result misses the bar, the
   result is a miss — you may not adjust the bar. Only a human may amend a spec,
   and only between sessions.
4. **A negative result is a real result.** "Not reproducible across seeds" is a
   valid, publishable campaign outcome. Report it plainly instead of grinding.
5. **Verify behaviour, not just the scalar.** A success rate says the target was
   reached; it does not say the policy pursued anything. Every eval carries a
   `behaviour` block, and the campaign spec sets hard criteria on it —
   `mean_pursuit_alignment` above all, plus initial separation and steps to
   interception. A policy that holds position until the target arrives, or that is
   handed catches by the spawn geometry, **fails** even when its success rate beats
   a genuine pursuer's. Reward can be gamed; geometry is harder to fake.
6. **Numbers veto, eyes advise.** The behaviour criteria in `spec.md` are hard
   gates. The reviewer's visual read of the contact sheet may only raise
   `suspicious`, which halts for a human — never `fail` on its own. The sheet is a
   reconstruction from position tracks, and taste is not a precommitted criterion.
7. **Establish the baselines before training a rung.** `tools/run_baselines.py`
   says whether scripted pursuit and random already solve the config. A rung random
   can solve proves nothing; a rung scripted pursuit cannot solve is not winnable by
   learning either. Both are far cheaper to discover here than after three failed
   runs, and scripted pursuit's alignment is the reference for what "looks like
   pursuit" means on that config.
8. **If a rung needs hyperparameter tuning to learn at all, suspect the task.**
   DreamerV3's whole claim is one fixed config across 150+ tasks. Journal the
   observation and stop that axis rather than tuning your way to a number.

## 4. Bookkeeping that makes the loop resumable

1. **Register before you spend.** Every run enters `runs.jsonl` with its config
   hash before an instance is created. `tools/launch_run.py` does this for you.
2. **Destroy an instance as soon as its artifacts are fetched.** Idle GPUs bill.
3. **Journal every decision**, one line, with the evidence path. The journal is
   the only history that survives a lost session.
4. **Re-derive truth from vast.ai, not from memory.** `experiments/state/*.json`
   is a cache. `vastai show instances` is the fact.
5. **State files are the handoff.** Assume you will be replaced mid-campaign by an
   agent with no memory of this conversation, and leave `current.json` such that
   it can continue.

## 5. Stopping

1. **If `experiments/state/KILLED` exists, the session is over.** The only
   permitted actions are: finish evaluating artifacts already on disk, write the
   final report, and stop. Never launch, never delete the flag.
2. **Budget and wall clock are enforced outside your reach**, by a guard process
   running from a copy of its code outside this repository. Do not try to raise a
   cap, extend a deadline, or work around a refusal from `launch_run.py`. If a
   run is too slow to finish inside budget, the answer is to fix or abandon the
   run, never to buy more time.
3. **Respect the concurrency limit** in the campaign spec (default: 3 instances).
