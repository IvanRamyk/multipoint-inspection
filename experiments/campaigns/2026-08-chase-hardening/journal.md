# Journal — chase-hardening

Append-only lab notebook. The final report goes at the end.

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
