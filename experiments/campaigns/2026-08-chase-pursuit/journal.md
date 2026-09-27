- `2026-08-31T16:08:52Z` launch P1-chase_pursuit_d100-s1 (P1, chase_pursuit_d100.yaml, seed 1, vast) hash=sha256:a58471659ed33a289cce646272acbe93
- `2026-08-31T16:08:54Z` launch P1-chase_pursuit_d100-s2 (P1, chase_pursuit_d100.yaml, seed 2, vast) hash=sha256:a58471659ed33a289cce646272acbe93
- `2026-08-31T16:08:56Z` launch P1-chase_pursuit_d100-s3 (P1, chase_pursuit_d100.yaml, seed 3, vast) hash=sha256:a58471659ed33a289cce646272acbe93
- `2026-08-31T16:08:58Z` launch P1-chase_pursuit_d100-s4 (P1, chase_pursuit_d100.yaml, seed 4, vast) hash=sha256:a58471659ed33a289cce646272acbe93
- `2026-08-31T16:09:00Z` launch P1-chase_pursuit_d100-s5 (P1, chase_pursuit_d100.yaml, seed 5, vast) hash=sha256:a58471659ed33a289cce646272acbe93
- `2026-08-31T16:10:03Z` launch FAILED P1-chase_pursuit_d100-s1: train_remote.sh exited 255. Instance 49414386 is still billing — destroy it or retry the launch.
[launch 2026-08-31] chase-pursuit P1: 5 seeds of chase_pursuit_d100 (AMBITIOUS distance 100, human-directed). Baseline suitable: scripted 70pct/align 0.604, random 0pct. At ~65m start loitering=0pct so pursuit is structurally forced (scripted align 0.60 vs 0.39 on close-spawn chase_open). Primary gate: mean alignment >= 0.48. Hard/long task (1500-step episodes, 300k ceiling); some seeds may not solve within budget - informative either way. s1 hit transient 255, recovered on its orphan box.
- `2026-08-31T16:16:40Z` launch FAILED P1-chase_pursuit_d100-s5: create_instance.sh did not report INSTANCE_ID, HOST, PORT (exit 0). The instance may exist and be billing — check `vastai show instances`.
[wake 24] read: s5 FAILED (create_instance parse glitch) orphan 49414399; s1/s2/s4 training, s3 provisioning -> decided reconcile+recover s5 (train_remote on its box). 2/5 launches glitched this round, both recovered. all 5 now accounted.
- `2026-08-31T17:16:26Z` health P1-chase_pursuit_d100-s2 step=7000 verdict=healthy reward_recent=-38.93 slope=+0.54/1k ep_len_recent=45.1 wm_loss=10.77 (dropping from min -290 at start; reward rising strongly; episodes already short at ~45 steps — catching the target early; recommend continue)
[wake 25] read: chase_pursuit seeds never health-checked -> decided run-monitor s2 -> reward rising (-290->-39), wm_loss dropping, BUT ep_len ~45 impossible as catches at 65m => SUSPECTED reward-hack (crash to end episode, dodge distance penalty). next wake: evaluate one checkpoint to confirm crash-vs-catch before burning more compute.
- `2026-08-31T17:42:41Z` eval P1-chase_pursuit_d100-s2: success=0% (0/30 eps, seed base 1000) -> P1-chase_pursuit_d100-s2_diag.json
[wake 26] read: verify crash-hack -> diagnostic eval s2@10k = 0% success, min_sep 68 (1.1% closed), align -0.006, episodes 14-50 steps = crashing. evaluator: UNDERTRAINED (40k env-steps far too few for 65m chase) not confirmed hack; recommends continue. DECIDED: continue + re-check distance-closing at ~step 50-80k to distinguish learning-vs-stuck. no kill.
- `2026-08-31T18:01:51Z` launch FAILED P1-chase_pursuit_d100-s3: train_remote.sh exited 255. Instance 49414394 is still billing — destroy it or retry the launch.
- `2026-08-31T18:03:08Z` launch P1-chase_pursuit_d100_v2-s1 (P1, chase_pursuit_d100_v2.yaml, seed 1, vast) hash=sha256:606014471f8713d6bc749229f33f4cdc
- `2026-08-31T18:03:10Z` launch P1-chase_pursuit_d100_v2-s2 (P1, chase_pursuit_d100_v2.yaml, seed 2, vast) hash=sha256:606014471f8713d6bc749229f33f4cdc
- `2026-08-31T18:03:12Z` launch P1-chase_pursuit_d100_v2-s3 (P1, chase_pursuit_d100_v2.yaml, seed 3, vast) hash=sha256:606014471f8713d6bc749229f33f4cdc
- `2026-08-31T18:03:14Z` launch P1-chase_pursuit_d100_v2-s4 (P1, chase_pursuit_d100_v2.yaml, seed 4, vast) hash=sha256:606014471f8713d6bc749229f33f4cdc
- `2026-08-31T18:03:16Z` launch P1-chase_pursuit_d100_v2-s5 (P1, chase_pursuit_d100_v2.yaml, seed 5, vast) hash=sha256:606014471f8713d6bc749229f33f4cdc
- `2026-08-31T18:05:16Z` launch FAILED P1-chase_pursuit_d100_v2-s3: train_remote.sh exited 255. Instance 49425772 is still billing — destroy it or retry the launch.
[wake 27] user: redesign reward so crashing is super-costly. Confirmed hack mechanism (collision -10 + terminate = escape distance penalty). FIX: chase_pursuit_d100_v2 collision_penalty -10->-1000. Reaped v1 d100 (5). Baseline v2 suitable (scripted 70- `2026-08-31T18:10:48Z` launch FAILED P1-chase_pursuit_d100_v2-s1: create_instance.sh did not report INSTANCE_ID, HOST, PORT (exit 0). The instance may exist and be billing — check `vastai show instances`.
[wake 28] v2: 2/5 launches failed again (s1 create-glitch/booting, s3 255); s2/s4/s5 training. decided recover s3 on orphan 49425772 (train_remote). s1 box still booting - recover next wake. then diagnostic to test if collision -1000 fix turns distance penalty into pursuit.
- `2026-08-31T19:15:06Z` eval P1-chase_pursuit_d100_v2-s2: success=0% (0/30 eps, seed base 1000) -> P1-chase_pursuit_d100_v2-s2_diag.json
[wake 29] diagnostic v2 s2@7k vs v1: REWARD FIX WORKS - crashing eliminated (ep_len 32->505), distance-closing started (closed 1.1%->13.9%, some eps 40-66%). but align ~0, 0 success at 7k (early). s1 box stuck 'created' -> abandoned (4 seeds suffice). DECIDED continue + re-check pursuit development at ~step 40-80k. collision -1000 did its job.
- `2026-08-31T20:05:29Z` eval P1-chase_pursuit_d100_v2-s2: success=7% (2/30 eps, seed base 1000) -> P1-chase_pursuit_d100_v2-s2_diag2.json
[wake 30] re-diagnose s2 trend 7k->13k: PURSUIT DEVELOPING - separation_closed 14->52pct, min_sep 60->33m, align 0.01->0.15 (early 0.28), success 0->6.7pct (2 catches), ep_len full. reward design WORKS and improves with training. decided continue + re-check toward 0.48 gate. reset s2 status->training (diag side-effect).
[wake 31] read vast reality on user status-check -> found s1 stuck 'created' + s3 dead (gpu0/no-tmux/no-log) both billing; tried destroy -> BLOCKED by guard (no selective reap, account-wide only) -> ACCEPTED bounded idle spend (TTL-guarded), marked s1+s3 failed, active_runs now s2/s4/s5 (3 live). Pursuit developing well. -> dispatched none
- `2026-08-31T20:59:58Z` launch P1-chase_pursuit_d100_v2-s6 (P1, chase_pursuit_d100_v2.yaml, seed 6, vast) hash=sha256:c7d820687733f9abba6bd4080d5c5896
- `2026-08-31T21:08:14Z` launch P1-chase_pursuit_d100_v2-s7 (P1, chase_pursuit_d100_v2.yaml, seed 7, vast) hash=sha256:c7d820687733f9abba6bd4080d5c5896
- `2026-08-31T21:10:11Z` launch FAILED P1-chase_pursuit_d100_v2-s7: train_remote.sh exited 255. Instance 49443356 is still billing — destroy it or retry the launch.
[wake 32] user: keep many trainings running + auto-escalate on 100%. Refilled pool: launched s6 (OK, training), s7 hit vast 255 -> orphan box 49443356; tried train_remote recovery x2, box drops SSH mid-install (bad box) -> abandoned (TTL-bounded, guard blocks destroy). 4 live seeds s2/s4/s5/s6. -> dispatched experiment-runner (s6)
[wake 33] read budget($3.02/$15, ~4.6h TTL) + vast: 4 live seeds healthy (s2/s4/s5/s6), 3 dead idle boxes TTL-bounded -> decided health-check stale run -> dispatched run-monitor(s5): HEALTHY step41k reward improving +39/1k, ep_len ~1473 near ceiling (timing out, pursuit developing). continue.
[wake 34] read budget($3.52/$15, ~3.8h TTL) + vast: 4 live seeds, s6 showed gpu=None -> decided health-check s6 -> dispatched run-monitor(s6): HEALTHY step12k reward improving +58/1k, ep_len 452, gpu=None was vast telemetry gap (real 23% util). continue.
- `2026-08-31T23:27:03Z` eval P1-chase_pursuit_d100_v2-s5: success=0% (0/30 eps, seed base 1000) -> P1-chase_pursuit_d100_v2-s5_diag3.json
[wake 35] read budget($3.97/$15,~3h TTL)+vast(4 live) -> decided measure the core metric -> dispatched run-evaluator DIAG3 on s5@61k: alignment 0.19 (<0.48 gate, <R1 0.28) success 0/30, BUT early_align 0.50, closes 74% gap, idle~0, gets to 1.4m -> ACTIVE imprecise undertrained pursuit not loiter. reset s5->training.
- `2026-09-01T00:25:34Z` eval P1-chase_pursuit_d100_v2-s6: success=0% (0/30 eps, seed base 1000) -> P1-chase_pursuit_d100_v2-s6.json
- `2026-09-01T00:25:35Z` eval P1-chase_pursuit_d100_v2-s5: success=0% (0/30 eps, seed base 1000) -> P1-chase_pursuit_d100_v2-s5.json
- `2026-09-01T00:25:54Z` eval P1-chase_pursuit_d100_v2-s2: success=3% (1/30 eps, seed base 1000) -> P1-chase_pursuit_d100_v2-s2.json
- `2026-09-01T00:26:01Z` eval P1-chase_pursuit_d100_v2-s4: success=3% (1/30 eps, seed base 1000) -> P1-chase_pursuit_d100_v2-s4.json
- `2026-09-01T00:27:11Z` aggregate P1: mean=2% ±1.1% (n=3, worst=0%) FAIL
- `2026-09-01T00:27:29Z` aggregate P1: mean=2% ±1.1% (n=3, worst=0%) FAIL


================================================================================
# FINAL REPORT — campaign 2026-08-chase-pursuit (rung P1) — concluded 2026-09-01T00:29:59.584164Z
================================================================================

## Overrides
NONE. No gate was overridden. The rung fails the precommitted gates cleanly and
the miss is large (not marginal), so no override was justified — this is a
genuine, honest negative result.

## Headline
DreamerV3 (XS) with reward + geometry shaping (distance penalty + 100 m far spawn
+ heavy collision penalty) did NOT produce scripted-level pursuit on the 65 m
chase within the night's TTL budget. Mean pursuit alignment 0.13 (gate 0.48),
mean success ~2% (gate 30%, worst-seed floor 15%). All gates FAIL. Leading
explanation per the spec's own fail-policy: UNDER-TRAINING — seeds reached only
25k-72k steps against a 300k ceiling on a task ~2.5x harder than R1, because the
cheap RTX-2080-Ti boxes ran at ~1-1.5 sps. The reward machinery is NOT broken:
the policy actively pursues (idle~0, mean speed 4.4 m/s, closes 60-74% of the
69 m gap, early-episode alignment up to 0.49 — above the gate) but loses heading
lock mid-episode and never closes the tight 0.75 m catch. This is active-but-
imprecise pursuit, categorically different from R1's loiter.

## Per-seed final evals (config chase_pursuit_d100_v2.yaml, 30 eps, greedy, seed base 1000)
| seed | ckpt step | pursuit_alignment | success | sep_closed | early_align | note |
|------|-----------|-------------------|---------|------------|-------------|------|
| s2   | 45000     | 0.169             | 3.3%    | 0.656      | 0.321       | in-cell |
| s4   | 42000     | 0.008             | 3.3%    | 0.583      | 0.200       | in-cell |
| s5   | 72000     | 0.209             | 0.0%    | 0.609      | 0.485       | in-cell; furthest trained |
| s6   | 25000     | -0.107            | 0.0%    | 0.297      | -0.073      | DIFFERENT config_hash (launch omitted --stop-* overrides); excluded from aggregate |

## Aggregate (in-cell seeds s2,s4,s5; config_hash sha256:606014471f8713d6bc749229f33f4cdc)
- mean pursuit_alignment = 0.129 ± ~0.06   vs gate 0.48   -> FAIL
- mean success = 2.2%                        vs gate 30%    -> FAIL
- worst-seed success = 0.0% (s5)             vs floor 15%   -> FAIL
- reference points: scripted 0.604, R1 loiter 0.28. Best learned seed (s5 0.209) is below R1's loiter number.
- Blind result-reviewer: NOT RUN — reviewer gates a claimed PASS; on an unambiguous fail there is nothing to advance. Spec fail-policy = report the finding, do not seed-shop.

## Seed attrition (vast.ai launch flakiness, NOT the method)
Launched 7 seeds; 4 produced evals (s2,s4,s5,s6). s1 stuck in 'created' (never booted),
s3 died early (idle box), s7 unrecoverable (box dropped SSH mid-install). 3/7 lost to
transient vast provisioning failures.

## Cost
Spend $4.52 of $15 cap (~30%). TTL ~2.1h remaining at conclude (session ~10h/12h).
Note: 3 dead idle boxes (s1/s3/s7) billed until this reap because the guard blocks
selective per-box destroy (account-wide reap only) — bounded, ~$1 of the total.

## Artifacts
- eval JSONs: experiments/campaigns/2026-08-chase-pursuit/evals/P1-chase_pursuit_d100_v2-s{2,4,5,6}.json
- s5 mid-training diagnostic: evals/P1-chase_pursuit_d100_v2-s5_diag3.json (step 61k)
- aggregate: evals/AGG_P1.json
- trajectories (for videos): experiments/campaigns/2026-08-chase-pursuit/trajectories/
- health snapshots: experiments/campaigns/2026-08-chase-pursuit/health/
- baseline/calibration: scripted 0.604 alignment, 70% success (spec P0, measured 2026-08-31)

## What this hands to the next campaign
1. Reward + geometry shaping is NOT sufficient to reach scripted-level pursuit on a
   65 m non-evasive chase WITHIN a small step budget — but it does produce genuine
   active pursuit (not loiter), so the design is directionally right.
2. The binding constraint tonight was THROUGHPUT: ~1.5 sps on cheap boxes caps steps
   at ~25-72k, far below the ~300k the task likely needs. Next run needs either faster
   GPUs (higher $/h) or higher env throughput, or a longer wall-clock budget.
3. Clean precommitted negative result -> either (a) rerun d100_v2 to ~300k on faster
   boxes, or (b) proceed to the evasive-target campaign. This is a human budget call.
4. s6 launch bug to fix in tooling next time: recovery/relaunch that omits --stop-*
   overrides produces a different config_hash and falls outside the aggregation cell.
================================================================================
[wake 36] read all 4 seeds evaluated + aggregate FAIL (align 0.13<0.48, success 2%<30%) -> decided CONCLUDE (clean negative, no override justified) -> reaped all 7 instances (0 remain), wrote final report.
