#!/usr/bin/env python3
"""Train DreamerV3 on a drone task using sheeprl, with optional early stopping.

Without stop flags this is a thin pass-through to sheeprl (same as before):

    ./venv/bin/python scripts/train_dreamer.py exp=drone_target \\
        algo.total_steps=15000 metric.log_every=200 checkpoint.every=2000

With ``--stop-threshold`` it launches sheeprl as a subprocess and runs a monitor
thread that tails the run's TensorBoard scalars. Once the chosen metric holds at
or above the threshold for ``--stop-patience`` consecutive log points, it stops
training *gracefully*: it waits (up to ``--stop-grace`` seconds) for sheeprl to
write its next periodic checkpoint so a usable checkpoint exists at the stop
point, then terminates the run. ``algo.total_steps`` remains a safety ceiling.

    ./venv/bin/python scripts/train_dreamer.py exp=drone_target \\
        checkpoint.every=2000 metric.log_every=200 \\
        --stop-threshold 50 --stop-window 5 --stop-patience 3

Any argument the parser does not recognise is forwarded verbatim to sheeprl as a
Hydra override.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Default TensorBoard scalar to watch. For the moving-target task a catch is
# worth ~+60, so a high rew_avg is a faithful proxy for "reliably succeeding".
_DEFAULT_STOP_METRIC = "Rewards/rew_avg"
_LOGS_ROOT = Path("logs/runs")


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split our stop flags from Hydra overrides forwarded to sheeprl."""
    parser = argparse.ArgumentParser(
        description="Train DreamerV3 with optional performance-based early stopping.",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--stop-threshold", type=float, default=None,
                        help="Stop once the metric reaches this value. Omit to disable early stopping.")
    parser.add_argument("--stop-metric", type=str, default=_DEFAULT_STOP_METRIC,
                        help=f"TensorBoard scalar tag to watch (default: {_DEFAULT_STOP_METRIC}).")
    parser.add_argument("--stop-window", type=int, default=5,
                        help="Number of most recent log points to average before comparing.")
    parser.add_argument("--stop-patience", type=int, default=3,
                        help="Consecutive new log points the averaged metric must stay above threshold.")
    parser.add_argument("--stop-grace", type=float, default=600.0,
                        help="Seconds to wait for a fresh checkpoint after the threshold is met, before terminating.")
    parser.add_argument("--poll-interval", type=float, default=15.0,
                        help="Seconds between metric polls.")
    known, passthrough = parser.parse_known_args(argv)
    if known.help:
        parser.print_help()
        print("\nAll other arguments are forwarded to sheeprl as Hydra overrides.")
        sys.exit(0)
    return known, passthrough


def _build_env(project_root: Path) -> dict[str, str]:
    """Environment for the sheeprl subprocess (search path + PYTHONPATH)."""
    env = os.environ.copy()
    configs_path = project_root / "configs" / "sheeprl"
    env["SHEEPRL_SEARCH_PATH"] = f"file://{configs_path}"
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{pythonpath}"
    return env


def _list_event_files() -> set[Path]:
    """All TensorBoard event files currently under the logs root."""
    if not _LOGS_ROOT.exists():
        return set()
    return set(_LOGS_ROOT.rglob("events.out.tfevents.*"))


def _find_new_event_file(existing_before: set[Path]) -> Path | None:
    """The event file created by THIS run, i.e. one not present before launch.

    Reading the "newest file by mtime" across all runs is unsafe: a previous
    run's converged tfevents (e.g. reward already above threshold) would make
    the monitor stop the fresh run seconds after startup. Instead we diff
    against the set of event files that existed before this run was launched
    and take the newest genuinely-new file. Returns None until it appears.
    """
    new_files = [p for p in _list_event_files() if p not in existing_before]
    if not new_files:
        return None
    return max(new_files, key=lambda p: p.stat().st_mtime)


def _read_scalar(event_file: Path, tag: str) -> list[tuple[int, float]]:
    """Read (step, value) pairs for a scalar tag from an event file."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(event_file.parent), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def _newest_checkpoint_mtime(run_dir: Path) -> float:
    """Mtime of the newest checkpoint under a run dir, or 0.0 if none."""
    ckpts = list(run_dir.rglob("ckpt_*.ckpt"))
    return max((c.stat().st_mtime for c in ckpts), default=0.0)


def _monitor(proc: subprocess.Popen, args: argparse.Namespace, existing_before: set[Path]) -> None:
    """Watch THIS run's metrics and stop it gracefully once good enough.

    ``existing_before`` is the set of event files that existed before launch;
    the monitor only ever reads the new file created by this run, so a stale
    converged run can never trigger an early stop here.
    """
    streak = 0
    last_step = -1
    event_file: Path | None = None  # pinned once this run's file appears
    print(
        f"[monitor] early stop armed: {args.stop_metric} >= {args.stop_threshold} "
        f"(avg of last {args.stop_window}, sustained {args.stop_patience} points)"
    )
    while proc.poll() is None:
        time.sleep(args.poll_interval)
        if event_file is None:
            event_file = _find_new_event_file(existing_before)
            if event_file is None:
                continue  # this run's tfevents not created yet
            print(f"[monitor] tracking run events: {event_file.parent}")
        scalars = _read_scalar(event_file, args.stop_metric)
        if len(scalars) < args.stop_window:
            continue
        step, _ = scalars[-1]
        if step == last_step:
            continue  # no new data since last poll
        last_step = step
        recent = [v for _, v in scalars[-args.stop_window:]]
        avg = sum(recent) / len(recent)
        if avg >= args.stop_threshold:
            streak += 1
            print(f"[monitor] step {step}: {args.stop_metric} avg={avg:.2f} "
                  f">= {args.stop_threshold} ({streak}/{args.stop_patience})")
        else:
            if streak:
                print(f"[monitor] step {step}: {args.stop_metric} avg={avg:.2f} — streak reset")
            streak = 0
        if streak >= args.stop_patience:
            _graceful_stop(proc, event_file.parent.parent, args.stop_grace)
            return
    print("[monitor] training process exited before threshold was reached.")


def _graceful_stop(proc: subprocess.Popen, run_dir: Path, grace: float) -> None:
    """Stop training, but never discard the policy we just trained.

    The point of stopping is to keep the learned policy, so we must not terminate
    with zero checkpoints on disk. Two separate waits:

    * for a checkpoint *newer* than the stop decision — bounded by ``grace``;
    * for the *first* checkpoint to exist at all — effectively unbounded (a large
      hard cap only guards against a truly broken run), because terminating a
      successful run before it ever checkpointed throws the run away.

    With a sane ``checkpoint.every`` (<= the earliest possible stop) the first
    checkpoint already exists, so this returns as soon as a fresh one lands.
    """
    print(f"[monitor] performance target reached. Waiting up to {grace:.0f}s "
          f"for a checkpoint newer than the stop decision...")
    decision_ts = time.time()
    baseline = _newest_checkpoint_mtime(run_dir)
    deadline = decision_ts + grace
    hard_cap = decision_ts + max(grace, 1.0) * 6.0
    while proc.poll() is None:
        newest = _newest_checkpoint_mtime(run_dir)
        if newest > baseline:
            print("[monitor] fresh checkpoint written — stopping now.")
            break
        if time.time() >= deadline:
            if newest > 0.0:
                print("[monitor] grace elapsed; a usable checkpoint exists — stopping now.")
                break
            if time.time() >= hard_cap:
                print("[monitor] hard cap reached with NO checkpoint on disk — stopping; "
                      "this run produced nothing evaluable (check checkpoint.every).")
                break
            print("[monitor] no checkpoint on disk yet — extending the wait so the "
                  "stopped policy is actually saved.")
        time.sleep(5.0)
    _terminate(proc)


def _terminate(proc: subprocess.Popen) -> None:
    """SIGTERM the process group, escalate to SIGKILL if it lingers."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print("[monitor] process did not exit on SIGTERM — sending SIGKILL.")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    import envs  # noqa: F401 — triggers gymnasium.register

    args, passthrough = _parse_args(sys.argv[1:])

    # Default experiment if the caller didn't specify one.
    if not any(a.startswith("exp=") for a in passthrough):
        passthrough = ["exp=drone_target", *passthrough]

    cmd = [sys.executable, "-m", "sheeprl", *passthrough]
    env = _build_env(project_root)

    print(f"Running: {' '.join(cmd)}")
    if args.stop_threshold is None:
        # No early stopping: behave like a plain launcher.
        subprocess.run(cmd, check=True, env=env)
        return

    # Snapshot existing event files BEFORE launch so the monitor can identify
    # this run's own tfevents and never read a stale (already-converged) one.
    existing_before = _list_event_files()
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    monitor = threading.Thread(target=_monitor, args=(proc, args, existing_before), daemon=True)
    monitor.start()
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[train] interrupted by user — terminating training.")
        _terminate(proc)
    monitor.join(timeout=5.0)
    sys.exit(proc.returncode or 0)


if __name__ == "__main__":
    main()
