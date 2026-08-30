# Plan — chase-hardening

Derived mechanically from `spec.md`. The spec is the precommitted contract and is
checksummed; this file and `tasks.md` only translate it into launchable commands. Where the
spec left a blank to be *measured* (R0's cost model), that measurement is recorded in
`journal.md` — never by editing the checksummed `spec.md`.

## Fixed across every rung

- Task `chase`, `--exp drone_chase` (which already pins `override /algo: dreamer_v3_XS`, so
  `algo=` is **not** passed again — a redundant group override risks a Hydra conflict).
- `--backend vast`, `--steps 1500000` (ceiling), `--num-envs 4`, `--log-every 1000`,
  `--checkpoint-every 50000`. `fabric.accelerator=gpu fabric.precision=16-mixed` are injected by
  `deploy/train_remote.sh`, so they are not repeated on the launch line.
- Early stop: `--stop-metric Rewards/rew_avg --stop-threshold 40 --stop-window 5 --stop-patience 3`.
- Guard rails on every launch: `--max-dph 1.0 --min-headroom 2.0`.
- Every rung trains **from scratch**. No warm-starting (constitution §1.6).

## Evaluation (every run)

`tools/evaluate_ckpt.py --campaign 2026-08-chase-hardening --run-id <id> --task chase
--episodes 30 --seed 1000`. Trajectories are written by default now (fix below), so the
contact sheet has `.npz` to read. Reviewer re-evaluates with `--episodes 20 --seed 5000 --tag review`.

## Behaviour gate (every rung, from spec)

`--min-alignment 0.45 --min-initial-separation 5.0 --min-catch-steps 40 --max-idle-fraction 0.5
--min-separation-closed 0.5`. Calibrated against baselines: scripted pursuit measures alignment
~0.69, random ~−0.75, so 0.45 separates a pursuer from a wanderer.

## Rungs and acceptance (verbatim from spec)

| rung | config | seeds | min-seeds | mean bar | worst-seed floor | extra |
|---|---|---|---|---|---|---|
| R0 | chase_easy2 | {1} | — | — | — | calibration: measure `Time/sps_train` at ~30 min on the R1-s1 run; write cost/seed to the 1.5M ceiling in `journal.md`. If projected 3-seed R1 > $7, replan (lower ceiling or 2 seeds) and say so. **This run continues as R1 seed 1.** |
| R1 | chase_easy2 | {1,2,3} | 3 | ≥ 0.55 | ≥ 0.40 | headline gate. Stretch ≥ 0.70. Fail policy: if ≥2 seeds clear 0.40, retry the failing seed once; if ≤1 clears, conclude "not reproducible". |
| R2 | chase_easy2_wind | {1,2} | 2 | ≥ 0.45 | — (spec sets none) | also require mean within 15 pts of R1 (reviewer/orchestrator check — not an aggregator flag). Run baselines first; fix config if `flagged` on suitability grounds. |
| R3 | chase_easy2_fast | {1,2} | 2 | ≥ 0.35 | — | independent axis from R2; may run in parallel when headroom allows. |
| R4 | chase | {1,2} | 2 | ≥ 0.30 | — | opportunistic, needs ≥ $4 headroom. Never trained before. Changes 3 axes vs R1, so a failure does not attribute. |

The spec sets no per-seed floor for R2–R4, only a mean bar and (R2) the ±15-point tie to R1.
I do **not** invent one — acceptance criteria are precommitted and I may not add gates the human
did not write (constitution §3.3). The behaviour gate still applies to every rung.

## Budget allocation (from spec)

| rung | instances | projected | running total |
|---|---|---|---|
| R0 | 1 (→ R1 s1) | ~$0.5 | $0.5 |
| R1 | 3 | ~$6 | $6.5 |
| R2 + R3 | 4 | ~$5 | $11.5 |
| R4 | 2 | ~$3 | $14.5 |

Reserve **$2 unspent**. Drop R4 first, then R3. If R1 needs a retry seed, R4 is cancelled.
Concurrency ≤ 3.

## Scope (human-directed 2026-08-30): R3 and R4 dropped, reserve for a fix-campaign

Total vast credit is $17.87 and campaign 1's full ladder is ~$14.5, leaving no room for an
autonomous rerun if a rung fails. Per the human's call, campaign 1 runs **R1 + R2 only** (~$9),
banking ~$8 for a possible *config-level* fix-campaign. This is the spec's own budget mechanism
("Drop R4 first, then R3"), so `spec.md` is untouched. R3/R4 remain in tasks.md as reference,
marked DROPPED — the orchestrator skips them and Concludes after R2.

The fix-reserve has two escape hatches, in priority order: if R1 or R2 fails from a cause that
is expressible as a new `configs/target/*.yaml` (allowed unattended), author a small fix
campaign and spend the reserve; if the cause needs an `envs/` code change, **halt for the human**
(constitution §2) and do not spend. Never manufacture a fix to chase a number — "not
reproducible" is a real result (constitution §3.5).

## Ordering

1. **R0/R1-s1 alone first** — do not launch the R1 wave before the cost model is measured on one
   instance (spec: "Do not launch a wave you cannot finish"). After the cost model clears $7,
   launch R1 seeds 2 and 3.
2. R1 gate (aggregate → reviewer) must pass before R2.
3. R2 (wind) after R1 passes.
4. After R2's gate, **Conclude**. R3 and R4 are not launched (reserved).

## Infra fix applied before this campaign (first real run)

`tools/evaluate_ckpt.py` did not pass `--trajectories` to `scripts/eval_target_ckpt.py`, so the
real eval path wrote **no** `.npz` files and the reviewer's contact sheet (spec deliverable) had
nothing to build from. Fixed to write trajectories by default to
`campaigns/<c>/trajectories/<run_id><tag>/` and record that path in the eval JSON. Details in
`journal.md` and the session's fix summary. The manifest in `tools/checksums.sha256` is snapshotted
by `setup_guard.sh` *after* this fix, so the reviewer's integrity check covers the fixed code.
