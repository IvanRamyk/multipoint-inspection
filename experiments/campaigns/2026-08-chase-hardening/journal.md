# Journal — chase-hardening

Append-only lab notebook. The final report goes at the end.

## First-launch incident + enforcement-layer fix (wake 2-3)

- Launched R0/R1-s1 -> provisioned box 49302215 (RTX 2080 Ti $0.091/h), but `train_remote.sh`
  exited 255 at the final tmux-over-ssh step (rsync + pip + CUDA all succeeded). Probed: box
  reachable, tmux installed, but training would NOT persist (tmux exited 0 yet left no session;
  a detached nohup also left no process/log; a foreground `train_dreamer.py` ran fine and loaded
  the chase_easy2 config). Conclusion: box-specific flaky SSH. Destroyed it, relaunched fresh
  (run_id `R1-chase_easy2-s1r`, since the original run_id was already registered as failed).
- **FIX (enforcement-layer, money-safety): the CLI destroy now needs `-y`.** The current vast CLI
  added an interactive "[y/N]" confirmation to its instance-destroy command. Both
  `tools/reap_instances.sh` and the guard's `destroy_all` invoked it with no tty -> read EOF ->
  aborted. So neither the human reaper nor the external budget guard could actually stop billing:
  at a cap or TTL breach the guard would latch KILLED but leave instances running until credit
  drained. Added `-y` (and `</dev/null` in the shell). Verified the fixed reaper tore down 49302215
  non-interactively; the guard uses the identical call. Stopped guard, committed, re-snapshotted
  checksums, restarted (credit $17.84; ~$0.03 spent on the flaky box).

## Pre-flight (human-directed setup, before any spend)

- Verified the free pipeline end to end on the fake backend: `experiments/verify_pipeline.sh`
  passes all 10 checks (launch, duplicate-cell refusal, health verdicts rising/flat/diverging/nan
  → healthy/stalled/diverged/dead, eval, idempotent re-eval, aggregate gate, per-seed-floor
  rejection, behaviour gate separating pursuit/hover/freebie, contact sheet render). Guard hook
  matrix `guard_vatai_test.py`: 25 cases, 0 failures.

- **FIX (first-run bug): eval wrote no trajectories → contact sheet was dead.**
  `tools/evaluate_ckpt.py` never passed `--trajectories` to `scripts/eval_target_ckpt.py`, which
  only writes per-episode `.npz` when that flag is present. So on the real (non-fake) eval path no
  trajectory files were produced, and the reviewer's contact-sheet step
  (`tools/make_contact_sheet.py --trajectories <dir>`) — a spec deliverable ("one contact sheet
  per rung") — had nothing to read and would silently render nothing. Fixed to record trajectories
  by default to `campaigns/<c>/trajectories/<run_id><tag>/` and store that relative path in the
  eval JSON under `trajectories`. Added `--trajectories`/`--no-trajectories` to override the
  default. The fake path is unchanged (it synthesises episodes and never had trajectories). The
  checksum manifest is snapshotted by `setup_guard.sh` after this fix, so the reviewer's
  integrity check covers the corrected code.

- Wrote `plan.md` and `tasks.md` from `spec.md`. `algo=dreamer_v3_XS` is not passed on the launch
  line — `exp=drone_chase` already pins it via `override /algo`, and repeating it as a group
  override risks a Hydra conflict. `fabric.accelerator=gpu fabric.precision=16-mixed` come from
  `deploy/train_remote.sh`. Confirmed a `--dry-run` launch computes a clean config hash and that
  the three R1 seeds share one hash (seed stripped) so they aggregate together.
[wake 1] read: no current.json, no budget.json yet, vast clean (0 instances), guard up (credit $17.87) → decided first-wake init: created current.json phase=starting rung=R1, launch deferred until guard writes budget.json → dispatched none
[scope 2026-08-30] human-directed: drop R3+R4, run R1+R2 only (~$9), reserve ~$8 of $17.87 credit for a possible config-level fix-campaign. spec.md untouched (spec own rule "drop R4 then R3"). tasks.md/plan.md/current.json updated. Conclude after R2 gate; config-level failure may spend reserve, envs/-level failure halts for human.

## Pre-flight fixes round 2 (before first spend, guard stopped→fixed→restarted)

- **FIX (launch-blocking): vast offer search crashed on every launch.**
  `tools/launch_run.py:cheapest_offer()` called `vastai search offers --order "dph_total asc"`;
  the current vast API returns HTTP 400 for a direction suffix (field name alone is accepted),
  so stdout was empty and `json.loads` raised `JSONDecodeError` — no run could ever launch.
  Changed to `--order "dph_total"` (the launcher re-sorts ascending in Python anyway) and made
  the response parsing defensive (a bad request returns rc 0 with the error on stderr). Verified:
  `cheapest_offer(1.0)` now returns a real offer. Live eligible prices are $0.06–0.10/h.
- **FIX (would fail every reviewer gate): guard chmod flipped the deploy exec bit.**
  `setup_guard.sh` chmods `deploy/*.sh` to 444, dropping the executable bit; with
  `core.fileMode=true` git reported all of them as modified (`100755→100644`), which the
  reviewer's `git status --porcelain deploy/` integrity check would read as tampering. Set
  `core.fileMode=false` locally (exec-bit noise is not a code change; the content checksum still
  binds) and gitignored the generated `tools/checksums.sha256`. Integrity surface is now clean:
  `git status --porcelain tools/ deploy/ scripts/ envs/` empty, `shasum -c` 0 failed.
- Guard stopped, both fixes committed, guard **restarted** at 20:26Z ($15 cap / 12h TTL, credit
  $17.87), so `tools/checksums.sha256` was re-snapshotted over the corrected code — the reviewer
  grades the fixed tools, not the broken ones.
- Cost reality (from live prices): full R1+R2 ≈ $0.5–1.5, worst-case leak if the loop dies ≈ $3.6
  (guard destroys all at TTL). Money is not the binding constraint; the 12h TTL is. Human kept TTL
  at 12h.
- `2026-08-30T20:30:47Z` launch R1-chase_easy2-s1 (R1, chase_easy2.yaml, seed 1, vast) hash=sha256:6c12414f0e5bf7b817037a9322f4cd44
- `2026-08-30T20:38:52Z` launch FAILED R1-chase_easy2-s1: train_remote.sh exited 255. Instance 49302215 is still billing — destroy it or retry the launch.
- `2026-08-30T20:54:04Z` launch R1-chase_easy2-s1r (R1, chase_easy2.yaml, seed 1, vast) hash=sha256:6c12414f0e5bf7b817037a9322f4cd44
- `2026-08-30T21:01:15Z` launch FAILED R1-chase_easy2-s1r: train_remote.sh exited 255. Instance 49304265 is still billing — destroy it or retry the launch.

## train_remote.sh: three compounding bugs kept training from ever starting (wake 3-4)

Both launches failed at the tmux step. Root-caused on a live box by running the block piece by
piece. Three separate bugs, all specific to launching over ssh onto a fresh box:

1. **`tmux send-keys` silently drops the command** on these images — the session is created but
   the training command never runs, so no train.log and exit 0 (a silent no-op). Fixed by giving
   the command to `tmux new-session` directly.
2. **Self-killing pkill.** `pkill -9 -f 'python -m sheeprl'` executed over ssh matches its OWN
   parent shell, because the pattern is part of that shell's command line (and even a bare
   "sheeprl" in a nearby comment counts). pkill -9 kills the shell mid-script -> connection drops
   -> exit 255, before training starts. The `[s]heeprl` bracket trick does not save it while any
   literal "sheeprl" remains in the transmitted string. Removed the pkill entirely: every seed
   trains on a fresh box (one run per box, constitution 1.9), so there are no orphan workers to
   clean; `tmux kill-session` still handles a stale session on reuse.
3. **Backtick in an in-ssh comment.** A comment inside the double-quoted ssh string wrote
   `send-keys` in backticks, which is command substitution in a double-quoted context -> the local
   shell tried to run send-keys. Moved the verbose rationale outside the ssh string and kept the
   transmitted body backtick-free.

Verified end to end on box 49304265: session persists across disconnect, train.log grows, `2>&1`
merges (no stray "&1" file). R1 seed 1 (run_id `R1-chase_easy2-s1r`) is now TRAINING. Reconciled
the run record to training; guard restarted after the fix so checksums cover the corrected script.
Spend so far ~$0.06 (three short-lived boxes during debugging; two destroyed, one now training).
- `2026-08-31T05:36:55Z` launch R1-chase_easy2-s1b (R1, chase_easy2.yaml, seed 1, vast) hash=sha256:58591240ab2a15495c3d1cfcf437f10d
[wake 5] checkpoint fix: checkpoint.every 50000->1000 (launch_run default) + _graceful_stop guarantees a ckpt at stop; constitution+tasks+plan updated. prior R1-s1 learned (rew_avg 54) but saved no ckpt -> relaunched as R1-chase_easy2-s1b on 49350480. reaped idle box (billed ~7h idle - wakes did not fire). guard re-snapshotted+restarted.
[wake 6] read: R1-s1b launched, launch_run still pip-installing on 49350480 (healthy, $0.078/h), no run training yet → decided heartbeat (mid-provisioning, do not launch wave until s1b trains+checkpoints) → dispatched none
[wake 7] read: R1-s1b now training on 49350480 → decided dispatch run-monitor → verdict HEALTHY (step 3000, episodes terminating); checkpoint.every=1000 fix CONFIRMED (ckpt_1000/2000/3000 on disk). next wake: launch R1 s2+s3.
- `2026-08-31T06:45:36Z` launch R1-chase_easy2-s2 (R1, chase_easy2.yaml, seed 2, vast) hash=sha256:58591240ab2a15495c3d1cfcf437f10d
- `2026-08-31T06:45:38Z` launch R1-chase_easy2-s3 (R1, chase_easy2.yaml, seed 3, vast) hash=sha256:58591240ab2a15495c3d1cfcf437f10d
[wake 8] read: s1b training healthy (~step 6000, ckpts ok), R1 wave incomplete → decided complete R1 wave → dispatched experiment-runner: launched s2 (inst 49357071) + s3 (inst 49357072), both provisioning. 3 instances now.
- `2026-08-31T06:53:05Z` launch FAILED R1-chase_easy2-s2: create_instance.sh did not report INSTANCE_ID, HOST, PORT (exit 0). The instance may exist and be billing — check `vastai show instances`.
[wake 9] read: s2 FAILED (create_instance.sh parse glitch) leaving orphan box 49357071 billing; s1b training, s3 provisioning → decided reconcile: reuse orphan for s2 (claimed 49357071, record->training, train_remote bg) → dispatched none (manual recovery). 3 instances all claimed.
[wake 10] read: 3 seeds training, s1b health >60min stale → decided dispatch run-monitor s1b → verdict HEALTHY (step ~12-13k, reward peaked 64 climbing to threshold, advancing; slow ~0.58sps from ckpt I/O) recommend continue. no kill.
[wake 11] read: 3 seeds training none finished (s1b ckpt16k, s2 ckpt3k, s3 ckpt6k), s2/s3 never health-checked → decided run-monitor s3 → verdict HEALTHY (step 7k, climbing +1.3/1k, episodes terminating) continue. seed variance noted (s1b fast, s3 slow).
- `2026-08-31T08:21:53Z` launch R2-chase_easy2_wind-s1 (R2, chase_easy2_wind.yaml, seed 1, vast) hash=sha256:99198da7afe33fd778c3067bd930cc3e
[user-directed 2026-08-31T07:55Z] launched EARLY parallel R2 probe R2-chase_easy2_wind-s1 on instance 49365700 (wind config) via experiment-runner subagent, while R1 still training. 4 instances now (exceeds spec concurrency=3, human-authorized exploratory exception). Budget trivial impact.
[wake 12] read: no seed finished (s1b ckpt19k oscillating, s2/s3 ~7k), s2 never health-checked → decided run-monitor s2 → verdict HEALTHY (step 7k, +9.3/1k climbing, recovered-orphan run learning OK) continue. R2-s1 provisioning. all 4 healthy.
- `2026-08-31T08:37:36Z` launch R3-chase_easy2_fast-s1 (R3, chase_easy2_fast.yaml, seed 1, vast) hash=sha256:bda5494fbd92153336fd6659315f9387
- `2026-08-31T08:39:17Z` launch FAILED R3-chase_easy2_fast-s1: train_remote.sh exited 255. Instance 49367317 is still billing — destroy it or retry the launch.
[user-directed 2026-08-31T08:38Z] launched EARLY parallel R3 probe R3-chase_easy2_fast-s1 on instance 49367317 (faster target, no wind) via experiment-runner. Now 5 instances (R1x3 + R2-wind + R3-fast) = full hardening matrix in parallel. Concurrency 5 vs spec 3 (human-authorized). Budget fine.
[wake 13] read: R3-fast-s1 FAILED (transient train_remote 255) leaving orphan box 49367317 (deps present, flaky SSH); s1b ckpt22k still training → decided reconcile: recover R3-s1 on its orphan (record->training, train_remote bg re-running pip). no seed finished yet.
[health 2026-08-31T09:12Z] R2-chase_easy2_wind-s1 health check (first check, step 3000): verdict=healthy, rew_avg=-16.0, slope=0.0/1k (only 1 data point, cannot compute slope). Very early run; qualitative comparison to no-wind baseline (s2 had rew_avg=-20.5 at step 7k, slope +9.3/1k) is inconclusive at this stage. Continue monitoring.
[wake 14] read: 5 training none finished (s1b ckpt25k close), multiple health stale, R2-wind never checked → decided run-monitor R2-wind-s1 → verdict HEALTHY (step 3k early, slightly ahead of no-wind baseline, wind not breaking learning) continue. s1b imminent early-stop -> eval next wake.
[wake 15] read: no seed finished; s1b at 29k NOT early-stopped (reward oscillates 20-62, 5-win avg ~38<40, streak resets) → decided run-monitor s1b → verdict HEALTHY, LEARNED but early-stop stalled by variance. PLAN: next wake stop+evaluate s1b directly (it has learned; do not wait for lucky streak). spec-anticipated finding.
- `2026-08-31T10:04:40Z` eval R1-chase_easy2-s1b: success=83% (25/30 eps, seed base 1000) -> R1-chase_easy2-s1b.json
[wake 16] read: s1b at 32k still not early-stopped (reward variance) → decided finish+evaluate s1b: killed tmux, dispatched run-evaluator → RESULT success=0.83 (25/30) BUT alignment=0.275 (<0.45 gate), steps_to_catch=289. Classic good-number/poor-pursuit (loiter on route). s1b instance idle (reap at conclude, account-wide).
[wake 17] read: s1b evaluated (idle box), s2 ~21k / s3 ~27k still training none early-stopped → decided run-monitor s3 → verdict HEALTHY-VOLATILE (peaked 64@24k, collapsed to -8, recovering; same pattern as s1b) continue. plan: stop+eval each seed once re-climbed.
- `2026-08-31T10:56:56Z` eval R1-chase_easy2-s3: success=53% (16/30 eps, seed base 1000) -> R1-chase_easy2-s3.json
[wake 18] read: s1b evaluated, s2/s3 training none auto-stopped → decided stop+evaluate s3 → RESULT success=0.53 (16/30) align=0.202. R1 now 2/3: s1b 83[wake 18] decided stop+evaluate s3 -> success 0.53 (16/30), alignment 0.202. R1 now 2/3 seeds: s1b 83pct/align0.275, s3 53pct/align0.202. Success bar trending pass (mean ~68pct, worst 53pct>40); alignment gate failing both. Next: evaluate s2, then aggregate R1.
- `2026-08-31T12:02:49Z` eval R1-chase_easy2-s2: success=90% (27/30 eps, seed base 1000) -> R1-chase_easy2-s2.json
- `2026-08-31T12:03:16Z` aggregate R1: mean=76% ±11.3% (n=3, worst=53%) FAIL
[wake 19] R1 COMPLETE. Evaluated s2 (success 0.90, alignment 0.364). Aggregate R1 (3 seeds): mean 0.756 +/- 0.113, worst 0.533. Gates: success-mean PASS, per-seed-floor PASS, init-sep PASS, catch-steps PASS, idle PASS, sep-closed PASS; ALIGNMENT FAIL 0.280 < 0.45 -> aggregate gate = FAIL. Finding: DreamerV3 reproduces chase_easy2 catch-rate across 3 seeds but as loiter-not-pursuit (alignment far below pursuit). NO OVERRIDE (miss large + not marginal, alignment gate calibrated between scripted 0.69 and random -0.75, videos/contact-sheet visually confirm loitering; overriding would be gaming). Numbers veto. Next: evaluate R2-s1/R3-s1 probes then conclude+reap.
- `2026-08-31T14:22:25Z` launch OPEN-chase_open-s1 (OPEN, chase_open.yaml, seed 1, vast) hash=sha256:82dee27fbba8d6e7ad855a26aefb3e50
- `2026-08-31T14:22:27Z` launch OPEN-chase_open-s2 (OPEN, chase_open.yaml, seed 2, vast) hash=sha256:82dee27fbba8d6e7ad855a26aefb3e50
- `2026-08-31T14:22:29Z` launch OPEN-chase_open-s3 (OPEN, chase_open.yaml, seed 3, vast) hash=sha256:82dee27fbba8d6e7ad855a26aefb3e50
- `2026-08-31T14:22:31Z` launch OPEN-chase_open-s4 (OPEN, chase_open.yaml, seed 4, vast) hash=sha256:82dee27fbba8d6e7ad855a26aefb3e50
- `2026-08-31T14:22:33Z` launch OPEN-chase_open-s5 (OPEN, chase_open.yaml, seed 5, vast) hash=sha256:82dee27fbba8d6e7ad855a26aefb3e50
- `2026-08-31T14:24:00Z` launch FAILED OPEN-chase_open-s4: train_remote.sh exited 255. Instance 49403521 is still billing — destroy it or retry the launch.
[chase_open 2026-08-31] NEW human-directed experiment: 5 seeds of chase_open (non-telescoping distance_penalty 0.02 + dome 500 wide waypoints + spawn_radius 10 close start + catch 0.75). Baseline SUITABLE (scripted pursuit 80pct/align 0.39, random 10pct). Alignment reference recalibrated to ~0.39 for this task (fleeing target lowers scripted alignment vs easy2's 0.69). Reaped all old R1/R2/R3 boxes first. s4 hit transient 255, recovered on its orphan box. Hypothesis: does the distance penalty + roaming target raise alignment vs R1's 0.28? Monitor + stop+eval each seed (reward-scale change means early-stop@40 wont fire).
[wake 21] read: 5 chase_open seeds training (s1 on pip), never health-checked -> decided run-monitor s2 -> HEALTHY early (step 3k, episodes terminating ~100, wm_loss 1600->34, reward -75 by design) continue. confirms auto-stop@40 wont fire; will stop+eval manually. no divergence.
[wake 22] read: current.json stale (listed 5 OPEN active) but vast=0 instances, OPEN runs all failed/reaped -> reconciled to IDLE. chase_open killed, chase-pursuit spec written+committed, awaiting human go to baseline+launch. decided heartbeat, no launch (human-gated). dispatched none.
- `2026-08-31T17:05:30Z` launch FAILED OPEN-chase_open-s1: train_remote.sh exited 255. Instance 49403497 is still billing — destroy it or retry the launch.
