# Tasks — chase-hardening

Ordered checklist the orchestrator ticks off. Each launchable run carries the exact
`tools/launch_run.py` invocation — the runner never composes one. All commands run from the repo
root with the project venv (`venv/bin/python`). Tick `[x]` only after the run is `evaluated` and
the rung's gate is decided by the reviewer.

> **SCOPE (human-directed, 2026-08-30): R3 and R4 are DROPPED to reserve budget.**
> Total vast credit is $17.87. Campaign 1 runs **R1 + R2 only** (~$9), banking ~$8 for a
> possible autonomous *config-level* fix-campaign if a rung fails. This uses the spec's own
> budget rule ("Drop R4 first, then R3") — `spec.md` is unchanged. After R2's reviewer gate,
> **Conclude** (do not launch R3/R4). A config-level failure may spend the reserve on a fix
> spec; an `envs/`-level cause halts for the human instead.

Common launch flags (every run):
`--exp drone_chase --backend vast --steps 1500000 --num-envs 4 --log-every 1000
--checkpoint-every 1000 --stop-metric Rewards/rew_avg --stop-threshold 40 --stop-window 5
--stop-patience 3 --max-dph 1.0 --min-headroom 2.0`

---

## R0 — calibration (gate: cost model). Launch seed 1 ALONE first.

- [ ] `R0/R1-s1` — launch, then after ~30 min read `Time/sps_train` and write cost/seed to the
      1.5M ceiling into `journal.md`. If projected 3-seed R1 > $7, replan before launching s2/s3.
```
venv/bin/python tools/launch_run.py --campaign 2026-08-chase-hardening --rung R1 \
  --env-config configs/target/chase_easy2.yaml --seed 1 \
  --exp drone_chase --backend vast --steps 1500000 --num-envs 4 --log-every 1000 \
  --checkpoint-every 1000 --stop-metric Rewards/rew_avg --stop-threshold 40 \
  --stop-window 5 --stop-patience 3 --max-dph 1.0 --min-headroom 2.0
```

## R1 — reproduce (headline gate). Launch s2 and s3 only after the R0 cost model clears $7.

- [ ] `R1-chase_easy2-s2`
```
venv/bin/python tools/launch_run.py --campaign 2026-08-chase-hardening --rung R1 \
  --env-config configs/target/chase_easy2.yaml --seed 2 \
  --exp drone_chase --backend vast --steps 1500000 --num-envs 4 --log-every 1000 \
  --checkpoint-every 1000 --stop-metric Rewards/rew_avg --stop-threshold 40 \
  --stop-window 5 --stop-patience 3 --max-dph 1.0 --min-headroom 2.0
```
- [ ] `R1-chase_easy2-s3`
```
venv/bin/python tools/launch_run.py --campaign 2026-08-chase-hardening --rung R1 \
  --env-config configs/target/chase_easy2.yaml --seed 3 \
  --exp drone_chase --backend vast --steps 1500000 --num-envs 4 --log-every 1000 \
  --checkpoint-every 1000 --stop-metric Rewards/rew_avg --stop-threshold 40 \
  --stop-window 5 --stop-patience 3 --max-dph 1.0 --min-headroom 2.0
```
- [ ] R1 aggregate + gate (after all three seeds `evaluated`):
```
venv/bin/python tools/aggregate_seeds.py --campaign 2026-08-chase-hardening --rung R1 \
  --min-seeds 3 --min-mean 0.55 --min-seed 0.40 \
  --min-alignment 0.45 --min-initial-separation 5.0 --min-catch-steps 40 \
  --max-idle-fraction 0.5 --min-separation-closed 0.5
```
- [ ] R1 reviewer verdict `pass` (result-reviewer, re-eval seed base 5000, 20 eps).

## R2 — wind (config-only). Only after R1 passes. Run baselines first.

Baseline already measured: `baselines/chase_easy2_wind.json` (pursuit 100%, `flagged` only for
triviality, which is expected and not a suitability block). If a *fresh* baseline run comes back
`flagged` for a new reason, fix the config before spending.

- [ ] `R2-chase_easy2_wind-s1`
```
venv/bin/python tools/launch_run.py --campaign 2026-08-chase-hardening --rung R2 \
  --env-config configs/target/chase_easy2_wind.yaml --seed 1 \
  --exp drone_chase --backend vast --steps 1500000 --num-envs 4 --log-every 1000 \
  --checkpoint-every 1000 --stop-metric Rewards/rew_avg --stop-threshold 40 \
  --stop-window 5 --stop-patience 3 --max-dph 1.0 --min-headroom 2.0
```
- [ ] `R2-chase_easy2_wind-s2`
```
venv/bin/python tools/launch_run.py --campaign 2026-08-chase-hardening --rung R2 \
  --env-config configs/target/chase_easy2_wind.yaml --seed 2 \
  --exp drone_chase --backend vast --steps 1500000 --num-envs 4 --log-every 1000 \
  --checkpoint-every 1000 --stop-metric Rewards/rew_avg --stop-threshold 40 \
  --stop-window 5 --stop-patience 3 --max-dph 1.0 --min-headroom 2.0
```
- [ ] R2 aggregate + gate (mean ≥ 0.45; reviewer also checks within 15 pts of R1's mean):
```
venv/bin/python tools/aggregate_seeds.py --campaign 2026-08-chase-hardening --rung R2 \
  --min-seeds 2 --min-mean 0.45 \
  --min-alignment 0.45 --min-initial-separation 5.0 --min-catch-steps 40 \
  --max-idle-fraction 0.5 --min-separation-closed 0.5
```
- [ ] R2 reviewer verdict `pass`.

## R3 — faster target — DROPPED (reserved for fix-campaign). Do not launch.

_Kept below for reference only; the orchestrator must skip this rung and Conclude after R2._

## R3 (reference, not launched) — faster target (config-only).

- [ ] `R3-chase_easy2_fast-s1`
```
venv/bin/python tools/launch_run.py --campaign 2026-08-chase-hardening --rung R3 \
  --env-config configs/target/chase_easy2_fast.yaml --seed 1 \
  --exp drone_chase --backend vast --steps 1500000 --num-envs 4 --log-every 1000 \
  --checkpoint-every 1000 --stop-metric Rewards/rew_avg --stop-threshold 40 \
  --stop-window 5 --stop-patience 3 --max-dph 1.0 --min-headroom 2.0
```
- [ ] `R3-chase_easy2_fast-s2`
```
venv/bin/python tools/launch_run.py --campaign 2026-08-chase-hardening --rung R3 \
  --env-config configs/target/chase_easy2_fast.yaml --seed 2 \
  --exp drone_chase --backend vast --steps 1500000 --num-envs 4 --log-every 1000 \
  --checkpoint-every 1000 --stop-metric Rewards/rew_avg --stop-threshold 40 \
  --stop-window 5 --stop-patience 3 --max-dph 1.0 --min-headroom 2.0
```
- [ ] R3 aggregate + gate (mean ≥ 0.35):
```
venv/bin/python tools/aggregate_seeds.py --campaign 2026-08-chase-hardening --rung R3 \
  --min-seeds 2 --min-mean 0.35 \
  --min-alignment 0.45 --min-initial-separation 5.0 --min-catch-steps 40 \
  --max-idle-fraction 0.5 --min-separation-closed 0.5
```
- [ ] R3 reviewer verdict `pass`.

## R4 — headline config — DROPPED (reserved for fix-campaign). Do not launch.

_Kept below for reference only; the orchestrator must skip this rung and Conclude after R2._

## R4 (reference, not launched) — headline config (opportunistic).

- [ ] `R4-chase-s1`
```
venv/bin/python tools/launch_run.py --campaign 2026-08-chase-hardening --rung R4 \
  --env-config configs/target/chase.yaml --seed 1 \
  --exp drone_chase --backend vast --steps 1500000 --num-envs 4 --log-every 1000 \
  --checkpoint-every 1000 --stop-metric Rewards/rew_avg --stop-threshold 40 \
  --stop-window 5 --stop-patience 3 --max-dph 1.0 --min-headroom 2.0
```
- [ ] `R4-chase-s2`
```
venv/bin/python tools/launch_run.py --campaign 2026-08-chase-hardening --rung R4 \
  --env-config configs/target/chase.yaml --seed 2 \
  --exp drone_chase --backend vast --steps 1500000 --num-envs 4 --log-every 1000 \
  --checkpoint-every 1000 --stop-metric Rewards/rew_avg --stop-threshold 40 \
  --stop-window 5 --stop-patience 3 --max-dph 1.0 --min-headroom 2.0
```
- [ ] R4 aggregate + gate (mean ≥ 0.30):
```
venv/bin/python tools/aggregate_seeds.py --campaign 2026-08-chase-hardening --rung R4 \
  --min-seeds 2 --min-mean 0.30 \
  --min-alignment 0.45 --min-initial-separation 5.0 --min-catch-steps 40 \
  --max-idle-fraction 0.5 --min-separation-closed 0.5
```
- [ ] R4 reviewer verdict `pass`.

## Per-run eval (evaluator, once each run finishes)
```
venv/bin/python tools/evaluate_ckpt.py --campaign 2026-08-chase-hardening \
  --run-id <run_id> --task chase --episodes 30 --seed 1000
```
