#!/usr/bin/env python3
"""Classify the health of a training run from its TensorBoard scalars.

RL fails silently. A broken environment, a reward that cannot be reached, or a
world model that has diverged all produce a *running* process and a flat curve,
not a crash. So this tool never answers with a bare boolean: it returns a verdict
plus the full evidence dict it used, and the caller (a subagent, then the
orchestrator) sees the numbers alongside the label.

Verdicts, in precedence order:
  dead      NaN in a loss, or no new scalar data for --stale-minutes.
  diverged  world-model loss blown up relative to its own trailing median, or
            KL far above the run's early baseline.
  stalled   reward slope non-positive over the recent window AND episodes are
            pinned at the truncation limit — i.e. it is not merely slow to
            improve, it is never finishing an episode successfully.
  slow      throughput below --min-sps, so the run cannot reach its ceiling
            inside its cost budget.
  healthy   none of the above.

Thresholds are deliberately relative to the run's own history rather than
absolute. DreamerV3 publishes no early-stop numbers, and absolute loss scales
shift with reward magnitude and observation size, so a fixed ceiling would be
wrong on the next task. Calibrate --divergence-factor and --kl-factor against
known-good historical runs before trusting them.

    tools/run_health.py logs/runs/dreamer_v3/<env>/<run>/version_0 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_REWARD_TAG = "Rewards/rew_avg"
_EPLEN_TAG = "Game/ep_len_avg"
_WM_LOSS_TAG = "Loss/world_model_loss"
_KL_TAG = "State/kl"
_SPS_TAG = "Time/sps_train"

_LOSS_TAGS = (
    "Loss/world_model_loss",
    "Loss/policy_loss",
    "Loss/value_loss",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
)


def read_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """All scalar series in a run directory, keyed by tag.

    Uses the same EventAccumulator pattern as scripts/train_dreamer.py, with
    size_guidance=0 so nothing is downsampled — we are computing slopes.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return out


def slope(points: list[tuple[int, float]]) -> float:
    """Least-squares slope of value against step. Zero for degenerate input."""
    if len(points) < 2:
        return 0.0
    n = len(points)
    xs = [float(s) for s, _ in points]
    ys = [float(v) for _, v in points]
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def stderr(values: list[float]) -> float:
    """Standard error of the mean. Zero for fewer than two samples."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var / n)


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])


def newest_event_mtime(run_dir: Path) -> float:
    times = [p.stat().st_mtime for p in run_dir.glob("events.out.tfevents.*")]
    return max(times) if times else 0.0


def max_episode_steps(run_dir: Path) -> int | None:
    """The truncation limit for this run, from its merged Hydra config.

    Needed to tell "episodes are long" from "episodes always hit the time limit",
    which is the difference between a slow learner and one that never succeeds.
    """
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return None
    env = cfg.get("env") or {}
    for key in ("max_episode_steps",):
        if env.get(key):
            return int(env[key])
    wrapper = env.get("wrapper") or {}
    inner = wrapper.get("config_path")
    if inner:
        candidate = Path(inner)
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parent.parent / inner
        if candidate.exists():
            try:
                import yaml

                data = yaml.safe_load(candidate.read_text()) or {}
                if data.get("max_episode_steps"):
                    return int(data["max_episode_steps"])
            except Exception:
                return None
    return None


def analyse(run_dir: Path, args: argparse.Namespace) -> dict:
    scalars = read_scalars(run_dir)
    reward = scalars.get(_REWARD_TAG, [])
    eplen = scalars.get(_EPLEN_TAG, [])
    wm_loss = scalars.get(_WM_LOSS_TAG, [])
    kl = scalars.get(_KL_TAG, [])
    sps = scalars.get(_SPS_TAG, [])

    last_step = reward[-1][0] if reward else 0
    age_minutes = (time.time() - newest_event_mtime(run_dir)) / 60.0 if reward else float("inf")
    limit = max_episode_steps(run_dir)

    evidence: dict = {
        "run_dir": str(run_dir),
        "last_step": last_step,
        "n_points": len(reward),
        "event_age_minutes": None if math.isinf(age_minutes) else round(age_minutes, 1),
        "max_episode_steps": limit,
    }

    if reward:
        values = [v for _, v in reward]
        window = max(2, int(len(reward) * args.window_frac))
        recent = reward[-window:]
        evidence.update({
            "reward_last": round(values[-1], 3),
            "reward_mean": round(sum(values) / len(values), 3),
            "reward_min": round(min(values), 3),
            "reward_max": round(max(values), 3),
            "reward_recent_mean": round(sum(v for _, v in recent) / len(recent), 3),
            "reward_slope_per_1k_steps": round(slope(recent) * 1000.0, 4),
            "window_points": len(recent),
        })
        # Window-over-window improvement and the noise it has to beat. The earlier
        # window is everything before the recent one, so on a learning run this is
        # strongly positive and on a stalled run it sits inside the noise.
        earlier = [v for _, v in reward[: max(1, len(reward) - len(recent))]]
        recent_vals = [v for _, v in recent]
        if earlier and len(recent_vals) >= 2:
            improvement = (sum(recent_vals) / len(recent_vals)) - (sum(earlier) / len(earlier))
            evidence["reward_improvement"] = round(improvement, 4)
            evidence["reward_improvement_stderr"] = round(
                max(stderr(recent_vals), stderr(earlier)), 4
            )
    if eplen:
        ep_values = [v for _, v in eplen]
        ep_recent = ep_values[-max(2, int(len(ep_values) * args.window_frac)):]
        evidence["ep_len_last"] = round(ep_values[-1], 1)
        evidence["ep_len_recent_mean"] = round(sum(ep_recent) / len(ep_recent), 1)
    if sps:
        sps_recent = [v for _, v in sps][-5:]
        evidence["sps_train_recent"] = round(sum(sps_recent) / len(sps_recent), 2)
    if wm_loss:
        wm_values = [v for _, v in wm_loss]
        wm_recent = wm_values[-max(2, int(len(wm_values) * args.window_frac)):]
        trailing = wm_values[: max(1, len(wm_values) - len(wm_recent))]
        evidence["wm_loss_recent_mean"] = round(sum(wm_recent) / len(wm_recent), 4)
        evidence["wm_loss_trailing_median"] = round(median(trailing), 4)
    if kl:
        kl_values = [v for _, v in kl]
        baseline_n = max(1, len(kl_values) // 5)
        evidence["kl_baseline_median"] = round(median(kl_values[:baseline_n]), 4)
        evidence["kl_recent_mean"] = round(sum(kl_values[-baseline_n:]) / baseline_n, 4)

    nan_tags = [
        tag for tag in _LOSS_TAGS
        for _, v in scalars.get(tag, [])
        if math.isnan(v) or math.isinf(v)
    ]
    evidence["nan_tags"] = sorted(set(nan_tags))

    verdict, reasons = _verdict(evidence, args)
    return {
        "verdict": verdict,
        "reasons": reasons,
        "evidence": evidence,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _verdict(ev: dict, args: argparse.Namespace) -> tuple[str, list[str]]:
    """Apply the precedence-ordered rules. Returns (verdict, human reasons)."""
    reasons: list[str] = []

    if ev["n_points"] == 0:
        return "unknown", ["no scalar data yet — run may still be starting up"]

    if ev["nan_tags"]:
        return "dead", [f"non-finite values in {', '.join(ev['nan_tags'])}"]
    age = ev.get("event_age_minutes")
    if age is not None and age > args.stale_minutes:
        return "dead", [f"no new scalars for {age:.0f} min (>{args.stale_minutes})"]

    wm_recent = ev.get("wm_loss_recent_mean")
    wm_trailing = ev.get("wm_loss_trailing_median")
    if wm_recent is not None and wm_trailing not in (None, 0.0):
        if wm_recent > args.divergence_factor * abs(wm_trailing):
            reasons.append(
                f"world-model loss {wm_recent} is >{args.divergence_factor}x its "
                f"trailing median {wm_trailing}"
            )
    kl_recent = ev.get("kl_recent_mean")
    kl_base = ev.get("kl_baseline_median")
    if kl_recent is not None and kl_base not in (None, 0.0):
        if kl_recent > args.kl_factor * abs(kl_base):
            reasons.append(
                f"KL {kl_recent} is >{args.kl_factor}x its early baseline {kl_base}"
            )
    if reasons:
        return "diverged", reasons

    # Stalled needs BOTH a flat reward trend and episodes that never terminate.
    # Either alone is ambiguous: a flat curve early is normal, and long episodes
    # are fine if the reward is still climbing.
    if ev["last_step"] >= args.min_steps:
        # "Flat" is decided by improvement against the series' own noise, not by
        # the sign of the slope. A genuinely stalled run's slope jitters either
        # side of zero, so a `slope <= 0` test misses roughly half of them, and an
        # absolute slope threshold cannot be chosen without knowing the task's
        # reward scale. Requiring window-over-window improvement to clear a
        # multiple of its own standard error is scale-free and stable.
        improvement = ev.get("reward_improvement")
        noise = ev.get("reward_improvement_stderr")
        if improvement is None or noise is None:
            flat = ev.get("reward_slope_per_1k_steps", 0.0) <= 0.0
        else:
            flat = improvement <= args.flat_sigma * noise
        limit = ev.get("max_episode_steps")
        ep_recent = ev.get("ep_len_recent_mean")
        pinned = (
            limit is not None and ep_recent is not None
            and ep_recent >= args.pinned_frac * limit
        )
        if limit is None:
            # Without the truncation limit the pinned-episode half of the rule
            # cannot be evaluated, so 'stalled' can never fire. Say so out loud
            # rather than silently reporting healthy.
            reasons.append(
                "could not resolve max_episode_steps from the run config — the "
                "stalled check is degraded to reward-slope only"
            )
        if flat and pinned:
            return "stalled", [
                f"reward improvement {ev.get('reward_improvement')} is within "
                f"{args.flat_sigma} standard errors "
                f"({ev.get('reward_improvement_stderr')}) of zero",
                f"episodes pinned at {ep_recent}/{limit} steps — never terminating successfully",
            ]
        if flat:
            reasons.append(
                f"reward improvement {ev.get('reward_improvement')} is inside the noise, but "
                "episodes still terminate — watch, do not kill"
            )

    sps = ev.get("sps_train_recent")
    if sps is not None and args.min_sps > 0 and sps < args.min_sps:
        return "slow", [f"throughput {sps} steps/s below required {args.min_sps}"]

    return "healthy", reasons or ["all monitored signals within expected ranges"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Classify a training run's health from TB scalars.")
    ap.add_argument("run_dir", help="A version_0 directory containing events.out.tfevents.*")
    ap.add_argument("--json", default=None, help="Also write the verdict to this path.")
    ap.add_argument("--window-frac", type=float, default=0.3,
                    help="Fraction of the most recent points treated as 'now' (default 0.3).")
    ap.add_argument("--stale-minutes", type=float, default=45.0,
                    help="No new scalars for this long means the run is dead.")
    ap.add_argument("--min-steps", type=int, default=300_000,
                    help="Do not judge 'stalled' before this many env steps.")
    ap.add_argument("--flat-sigma", type=float, default=1.0,
                    help="Reward counts as flat when window-over-window improvement is within "
                         "this many standard errors of zero. Scale-free, so it transfers across "
                         "tasks with different reward magnitudes.")
    ap.add_argument("--pinned-frac", type=float, default=0.95,
                    help="Mean episode length above this fraction of the truncation limit "
                         "means episodes are never ending early, i.e. never succeeding.")
    ap.add_argument("--divergence-factor", type=float, default=3.0,
                    help="World-model loss above this multiple of its trailing median = diverged.")
    ap.add_argument("--kl-factor", type=float, default=5.0,
                    help="KL above this multiple of its early baseline = diverged.")
    ap.add_argument("--min-sps", type=float, default=0.0,
                    help="Required training throughput. 0 disables the check.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}", file=sys.stderr)
        return 2

    report = analyse(run_dir, args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
