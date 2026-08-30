#!/usr/bin/env bash
# Exercise the whole orchestration pipeline on the fake backend. No spend, no
# training, no GPU — it writes real TensorBoard events and drives every tool the
# autonomous loop uses, then deletes what it made.
#
# Run this after changing anything under tools/, and before turning the loop
# loose on a real campaign.
#
#   bash experiments/verify_pipeline.sh
#
# Also worth running alongside it:
#   python3 .claude/hooks/guard_vastai_test.py    # the permission hook's matrix
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# sheeprl needs Python <3.12 and fake_trainer needs torch's SummaryWriter, so
# prefer the project venv and fall back to whatever python3 is around.
PY="${PY:-./venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

C=_verify
FAILURES=0
note() { printf '  %s\n' "$*"; }
fail() { printf '  FAIL: %s\n' "$*"; FAILURES=$((FAILURES + 1)); }

cleanup() {
  rm -rf "experiments/campaigns/$C"
  rm -rf logs/runs/dreamer_v3/*/*_fake_* 2>/dev/null
}
trap cleanup EXIT

cleanup
mkdir -p "experiments/campaigns/$C"/{evals,verdicts,health}
cat > "experiments/campaigns/$C/spec.md" <<'SPEC'
# Campaign: _verify — pipeline check only
## Why
Verify the machinery works. Deliberately not a research result; its numbers are
synthetic and mean nothing.
## Ground rules
Budget $0 (fake backend only). Eval 20 episodes at seed base 1000; the reviewer
re-evaluates at 5000.
## Rungs
### R1
- Env config: configs/target/chase_easy2.yaml
- Seeds {1,2,3}. Accept: mean >= 0.60 and every seed >= 0.45.
SPEC

echo "1. launch three seeds (fake backend)"
for s in 1 2 3; do
  if "$PY" tools/launch_run.py --campaign "$C" --rung R1 \
      --env-config configs/target/chase_easy2.yaml --seed "$s" \
      --backend fake --steps 400000 --fake-profile rising >/dev/null 2>&1; then
    note "seed $s launched"
  else
    fail "seed $s did not launch"
  fi
done

echo "2. duplicate cell must be refused"
if "$PY" tools/launch_run.py --campaign "$C" --rung R1 \
    --env-config configs/target/chase_easy2.yaml --seed 1 \
    --backend fake --steps 400000 >/dev/null 2>&1; then
  fail "relaunching a covered (config, seed) cell was allowed — double-spend risk"
else
  note "refused, as it must be"
fi

echo "3. health verdicts must match each curve shape"
declare -a want=(rising:healthy flat:stalled diverging:diverged nan:dead)
for pair in "${want[@]}"; do
  prof="${pair%%:*}"; expect="${pair##*:}"
  d=$("$PY" tools/fake_trainer.py --profile "$prof" --seed 7 --steps 400000 2>/dev/null | tail -1)
  got=$("$PY" tools/run_health.py "$d" --min-steps 100000 2>/dev/null \
        | "$PY" -c "import sys,json; print(json.load(sys.stdin)['verdict'])")
  if [ "$got" = "$expect" ]; then note "$prof -> $got"; else fail "$prof -> $got (wanted $expect)"; fi
done

echo "4. evaluate each run"
for s in 1 2 3; do
  "$PY" tools/evaluate_ckpt.py --campaign "$C" --run-id "R1-chase_easy2-s$s" \
    --task chase --episodes 20 --seed 1000 --fake --fake-success-rate 0.7 >/dev/null 2>&1 \
    && note "seed $s evaluated" || fail "seed $s eval failed"
done

echo "5. re-evaluation must skip rather than redo"
# Captured to a variable rather than piped into `grep -q`: with pipefail set,
# grep -q closes the pipe on its first match and the Python process dies of
# SIGPIPE, which would look like a failing check.
REEVAL_OUT=$("$PY" tools/evaluate_ckpt.py --campaign "$C" --run-id R1-chase_easy2-s1 \
  --task chase --episodes 20 --seed 1000 --fake 2>&1)
if printf '%s' "$REEVAL_OUT" | grep -q "already exists"; then
  note "skipped, as it must"
else
  fail "re-evaluation was not idempotent"
fi

echo "6. aggregate must pass on three good seeds"
if "$PY" tools/aggregate_seeds.py --campaign "$C" --rung R1 \
    --min-seeds 3 --min-mean 0.60 --min-seed 0.45 >/dev/null 2>&1; then
  note "gate passed"
else
  fail "gate should have passed"
fi

echo "7. a collapsed seed must fail the gate even when the mean clears the bar"
"$PY" tools/launch_run.py --campaign "$C" --rung R1 \
  --env-config configs/target/chase_easy2.yaml --seed 4 \
  --backend fake --steps 400000 >/dev/null 2>&1
"$PY" tools/evaluate_ckpt.py --campaign "$C" --run-id R1-chase_easy2-s4 \
  --task chase --episodes 20 --seed 1000 --fake --fake-success-rate 0.05 >/dev/null 2>&1
# The mean bar is lowered to 0.35 for THIS check only, so that the mean definitely
# clears it and the per-seed floor is the only thing that can fail. Twenty episodes
# per seed leave enough sampling noise that tuning the fake rates to straddle a
# 0.60 bar would make the test flaky rather than isolating.
out=$("$PY" tools/aggregate_seeds.py --campaign "$C" --rung R1 \
  --min-seeds 3 --min-mean 0.35 --min-seed 0.45 2>&1)
if printf '%s' "$out" | grep -q "per-seed floor" \
   && ! printf '%s' "$out" | grep -q "below required"; then
  note "rejected on the per-seed floor alone: $(printf '%s' "$out" | grep -o 'mean=[0-9]*%.*')"
else
  fail "the per-seed floor did not reject a collapsed seed on its own"
fi

echo "8. behaviour metrics must separate pursuit from luck"
if "$PY" eval/behaviour_metrics_test.py >/dev/null 2>&1; then
  note "the metric set labels pursuer, hoverer, wanderer and spawn-freebie correctly"
else
  fail "behaviour metrics no longer discriminate — run eval/behaviour_metrics_test.py"
fi

echo "9. the behaviour gate must reject a good success rate from the wrong behaviour"
BGATES=(--min-seeds 3 --min-mean 0.60 --min-seed 0.45
        --min-alignment 0.45 --min-initial-separation 5.0 --min-catch-steps 40
        --max-idle-fraction 0.5 --min-separation-closed 0.5)
behaviour_case() {
  local label="$1" behaviour="$2" rate="$3" expect="$4"
  local c="${C}_$behaviour"
  rm -rf "experiments/campaigns/$c"
  mkdir -p "experiments/campaigns/$c/evals"
  printf '# Campaign: %s\nBehaviour gate check.\n' "$c" > "experiments/campaigns/$c/spec.md"
  for s in 1 2 3; do
    "$PY" tools/launch_run.py --campaign "$c" --rung R1 \
      --env-config configs/target/chase_easy2.yaml --seed "$s" \
      --backend fake --steps 400000 >/dev/null 2>&1
    "$PY" tools/evaluate_ckpt.py --campaign "$c" --run-id "R1-chase_easy2-s$s" \
      --task chase --episodes 20 --seed 1000 --fake \
      --fake-success-rate "$rate" --fake-behaviour "$behaviour" >/dev/null 2>&1
  done
  local got=fail
  "$PY" tools/aggregate_seeds.py --campaign "$c" --rung R1 "${BGATES[@]}" >/dev/null 2>&1 && got=pass
  if [ "$got" = "$expect" ]; then note "$label -> $got"; else fail "$label -> $got (wanted $expect)"; fi
  rm -rf "experiments/campaigns/$c"
}
behaviour_case "genuine pursuit at 75%" pursuit 0.75 pass
behaviour_case "hoverer at 85% — a BETTER score than the pursuer" hover 0.85 fail
behaviour_case "spawn freebies at 95%" freebie 0.95 fail

echo "10. contact sheet renders"
SHEET_TRAJ=$(mktemp -d)
"$PY" - <<'PYEOF' "$SHEET_TRAJ" >/dev/null 2>&1
import sys, numpy as np, pathlib
out = pathlib.Path(sys.argv[1])
rng = np.random.default_rng(0)
for i in range(4):
    steps = 120
    t = np.stack([np.linspace(6, 1, steps), np.linspace(-4, 4, steps), np.full(steps, 3.0)], 1)
    d = [np.array([-5.0, -5.0, 3.0])]
    for k in range(1, steps):
        v = t[k] - d[-1]
        d.append(d[-1] + v / max(np.linalg.norm(v), 1e-9) * 0.1)
    np.savez_compressed(out / f"ckpt_1000_0_ep{i+1}.npz", drone=np.array(d), target=t,
                        caught=(i % 2 == 0), reach_distance=2.0, dome_size=16.0, dt=1/30)
PYEOF
if "$PY" tools/make_contact_sheet.py --trajectories "$SHEET_TRAJ" \
     --out "$SHEET_TRAJ/sheet.png" --max-episodes 4 --cols 2 >/dev/null 2>&1 \
   && [ -s "$SHEET_TRAJ/sheet.png" ]; then
  note "rendered $(du -h "$SHEET_TRAJ/sheet.png" | cut -f1) contact sheet"
else
  fail "contact sheet did not render"
fi
rm -rf "$SHEET_TRAJ"

echo
if [ "$FAILURES" -eq 0 ]; then
  echo "pipeline verified — all checks passed."
else
  echo "pipeline NOT verified — $FAILURES check(s) failed."
fi
exit "$FAILURES"
