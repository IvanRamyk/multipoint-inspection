#!/usr/bin/env bash
# Can the agent continue past a failed gate — and only in the ways it should?
#
# The override is the one place where a deterministic gate becomes a judgment
# call, so its limits need testing as much as the gates do: the budget must bind,
# integrity failures must stay unoverridable, and the measurement must keep
# reporting FAIL afterwards. Run it from the repo root.
#
#   bash experiments/verify_overrides.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-./venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
C=_ovr
FAILURES=0
note() { printf '  %s\n' "$*"; }
fail() { printf '  FAIL: %s\n' "$*"; FAILURES=$((FAILURES+1)); }

cleanup() { rm -rf "experiments/campaigns/$C"; rm -rf logs/runs/dreamer_v3/*/*_fake_* 2>/dev/null; }
trap cleanup EXIT
cleanup

mkdir -p "experiments/campaigns/$C"/{evals,verdicts,overrides}
printf '# Campaign: %s\nOverride test.\n' "$C" > "experiments/campaigns/$C/spec.md"

# Three seeds of a hoverer: high success rate, behaviour gate fails on alignment.
for s in 1 2 3; do
  "$PY" tools/launch_run.py --campaign "$C" --rung R1 \
    --env-config configs/target/chase_easy2.yaml --seed "$s" \
    --backend fake --steps 400000 >/dev/null 2>&1
  "$PY" tools/evaluate_ckpt.py --campaign "$C" --run-id "R1-chase_easy2-s$s" \
    --task chase --episodes 20 --seed 1000 --fake \
    --fake-success-rate 0.85 --fake-behaviour hover >/dev/null 2>&1
done
"$PY" tools/aggregate_seeds.py --campaign "$C" --rung R1 \
  --min-seeds 3 --min-mean 0.60 --min-seed 0.45 --min-alignment 0.45 >/dev/null 2>&1

echo "1. a behaviour failure can be overridden"
if OUT=$("$PY" tools/request_override.py --campaign "$C" --rung R1 --kind advance-rung \
    --category near-miss --justification "Alignment metric misreads this config" 2>&1); then
  note "granted; $(printf '%s' "$OUT" | grep -o '[0-9] left in this campaign' || echo '?')"
  printf '%s' "$OUT" | grep -o 'worst miss:.*' | sed 's/^/       /'
else
  fail "a plain behaviour failure was refused: $(printf '%s' "$OUT" | head -2)"
fi

echo "2. the spec is untouched and the gate still reports FAIL"
if "$PY" tools/aggregate_seeds.py --campaign "$C" --rung R1 \
     --min-seeds 3 --min-mean 0.60 --min-seed 0.45 --min-alignment 0.45 >/dev/null 2>&1; then
  fail "the gate started passing — the override must not change the measurement"
else
  note "gate still FAILs, as it must"
fi

echo "3. the second override is allowed (budget is 2)"
cp "experiments/campaigns/$C/evals/AGG_R1.json" "experiments/campaigns/$C/evals/AGG_R2.json"
if "$PY" tools/request_override.py --campaign "$C" --rung R2 --kind advance-rung \
    --category edge-case --justification "Second one" \
    --from-json "experiments/campaigns/$C/evals/AGG_R2.json" >/dev/null 2>&1; then
  note "granted"
else
  fail "the second override was refused"
fi

echo "4. the third must be refused"
cp "experiments/campaigns/$C/evals/AGG_R1.json" "experiments/campaigns/$C/evals/AGG_R3.json"
if OUT=$("$PY" tools/request_override.py --campaign "$C" --rung R3 --kind advance-rung \
    --category edge-case --justification "Third" \
    --from-json "experiments/campaigns/$C/evals/AGG_R3.json" 2>&1); then
  fail "a third override was granted despite a budget of 2"
else
  note "refused: $(printf '%s' "$OUT" | head -1)"
fi

echo "5. integrity failures can never be overridden"
"$PY" - <<'PYEOF'
import json, pathlib
p = pathlib.Path("experiments/campaigns/_ovr/evals/AGG_INT.json")
p.write_text(json.dumps({
    "rung": "INT", "mean": 0.9, "min": 0.88, "n_seeds": 3, "criteria": {},
    "failures": ["only 2 seeds evaluated, 3 required"],
}, indent=2))
PYEOF
if OUT=$("$PY" tools/request_override.py --campaign "$C" --rung INT --kind advance-rung \
    --category edge-case --justification "Two seeds is plenty" \
    --from-json "experiments/campaigns/$C/evals/AGG_INT.json" 2>&1); then
  fail "an insufficient-seeds failure was overridden"
else
  note "refused: $(printf '%s' "$OUT" | grep -o 'missing evidence[^"]*' | head -1)"
fi

echo "6. --list reports the budget"
"$PY" tools/request_override.py --campaign "$C" --list \
  | "$PY" -c "import sys,json; d=json.load(sys.stdin); print(f\"       budget={d['budget']} used={d['used']} remaining={d['remaining']}\")"

echo
[ "$FAILURES" -eq 0 ] && echo "override behaviour verified" || echo "$FAILURES failure(s)"
exit "$FAILURES"
