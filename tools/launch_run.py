#!/usr/bin/env python3
"""Launch one training run. The only sanctioned way to start an experiment.

Everything an autonomous session must not get wrong lives here rather than in an
agent's head: the budget precondition, the kill-switch check, the config hash, and
registering the run *before* any money is spent. An agent that calls this cannot
accidentally start an unregistered or unaffordable run; an agent that bypasses it
is visible as an untracked instance at the orchestrator's next reconcile.

Backends:
  vast   provision a vast.ai instance and train on GPU (the real thing).
  local  train on this machine's CPU in the background. Free, debug-only.
  fake   synthesise a run directory with plausible curves. Free, no training.

    tools/launch_run.py --campaign my-campaign --rung R1 \\
        --env-config configs/target/chase_easy2.yaml --seed 1 \\
        --exp drone_chase --steps 2000000 --stop-threshold 45
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _state as st  # noqa: E402

OFFER_QUERY = (
    "reliability > 0.98 num_gpus=1 gpu_ram >= 16 cuda_vers >= 12.0 "
    "inet_down > 100 disk_space >= 20"
)


def vastai_bin() -> str:
    return os.environ.get("VASTAI") or shutil.which("vastai") or str(st.REPO_ROOT / "venv/bin/vastai")


def cheapest_offer(max_dph: float | None) -> dict:
    """Cheapest offer meeting the training requirements.

    Queried as JSON rather than by parsing deploy/vast_search.sh's human output —
    a launcher that misreads a price could pick a $9/hour box.
    """
    out = subprocess.run(
        [vastai_bin(), "search", "offers", OFFER_QUERY,
         # The vast API rejects a direction suffix ("dph_total asc" -> 400); the
         # field name alone is accepted. Direction does not matter here because we
         # re-sort by dph_total ascending in Python below.
         "--order", "dph_total", "--limit", "20", "--raw"],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(f"vastai search failed: {out.stderr.strip()[:300]}")
    # A bad request returns rc 0 with the error object on stderr and empty stdout,
    # so returncode alone is not enough — parse defensively and fail loudly.
    body = out.stdout.strip() or out.stderr.strip()
    try:
        offers = json.loads(body)
    except json.JSONDecodeError:
        raise RuntimeError(f"vastai search returned no JSON: {body[:300]}")
    if isinstance(offers, dict):
        if offers.get("error"):
            raise RuntimeError(f"vastai search error: {offers.get('msg')}")
        offers = offers.get("offers", [])
    if not offers:
        raise RuntimeError("no vast.ai offers matched the requirements")
    offers.sort(key=lambda o: float(o.get("dph_total") or 1e9))
    if max_dph is not None:
        affordable = [o for o in offers if float(o.get("dph_total") or 1e9) <= max_dph]
        if not affordable:
            cheapest = float(offers[0].get("dph_total"))
            raise RuntimeError(
                f"cheapest offer is ${cheapest:.3f}/h, above the --max-dph ceiling ${max_dph:.3f}/h"
            )
        offers = affordable
    return offers[0]


def create_instance(offer_id: str) -> tuple[str, str, str]:
    """Provision and wait for SSH, reusing deploy/create_instance.sh.

    That script already encodes the non-obvious part — an ssh-url appears before
    sshd accepts connections, so it verifies a real login before declaring ready.
    We parse its summary lines instead of reimplementing the wait.
    """
    proc = subprocess.run(
        ["bash", str(st.REPO_ROOT / "deploy/create_instance.sh"), offer_id],
        capture_output=True, text=True, cwd=str(st.REPO_ROOT), check=False,
    )
    combined = proc.stdout + proc.stderr
    sys.stderr.write(combined)
    fields = {}
    for key in ("INSTANCE_ID", "HOST", "PORT"):
        match = re.findall(rf"^\s*{key}=(\S+)\s*$", combined, re.MULTILINE)
        if match:
            fields[key] = match[-1]
    missing = [k for k in ("INSTANCE_ID", "HOST", "PORT") if k not in fields]
    if missing:
        raise RuntimeError(
            f"create_instance.sh did not report {', '.join(missing)} (exit {proc.returncode}). "
            "The instance may exist and be billing — check `vastai show instances`."
        )
    return fields["INSTANCE_ID"], fields["HOST"], fields["PORT"]


def build_overrides(args: argparse.Namespace) -> list[str]:
    """Hydra overrides plus train_dreamer.py's own stop flags, in launch order."""
    overrides = [
        f"exp={args.exp}",
        f"seed={args.seed}",
        f"env.wrapper.config_path={args.env_config}",
        f"algo.total_steps={args.steps}",
        f"env.num_envs={args.num_envs}",
        f"metric.log_every={args.log_every}",
        f"checkpoint.every={args.checkpoint_every}",
    ]
    overrides += list(args.override or [])
    if args.stop_threshold is not None:
        overrides += [
            "--stop-metric", args.stop_metric,
            "--stop-threshold", str(args.stop_threshold),
            "--stop-window", str(args.stop_window),
            "--stop-patience", str(args.stop_patience),
        ]
    return overrides


def preflight(args: argparse.Namespace, cfg_hash: str) -> None:
    """Refuse to spend when the session says stop, or when a cell is already done.

    Idempotence matters more than it looks: the orchestrator may wake twice on the
    same pending action after a lost write, and a duplicate launch is real money.
    """
    if st.is_killed():
        raise SystemExit(f"REFUSING: kill switch is set.\n{st.kill_reason()}")

    if args.backend == "vast":
        budget = st.read_budget()
        if not budget:
            raise SystemExit(
                "REFUSING: no budget.json — the external guard is not running. "
                "Start it with tools/setup_guard.sh before launching GPU work."
            )
        head = st.headroom()
        if head is None:
            raise SystemExit("REFUSING: budget.json has no cap/spent fields.")
        if head < args.min_headroom:
            raise SystemExit(
                f"REFUSING: headroom ${head:.2f} is below the ${args.min_headroom:.2f} reserve "
                f"(spent ${budget.get('spent')} of ${budget.get('cap')})."
            )

    if args.force:
        return
    live = ("launched", "training", "fetched", "evaluated")
    for rec in st.read_runs(args.campaign):
        same_cell = rec.get("config_hash") == cfg_hash and rec.get("seed") == args.seed
        if same_cell and rec.get("status") in live:
            raise SystemExit(
                f"REFUSING: {rec['run_id']} already covers this cell "
                f"(hash {cfg_hash}, seed {args.seed}) with status {rec['status']}. "
                "Use a different seed, or --force to override."
            )


def launch_vast(args: argparse.Namespace, overrides: list[str], run_id: str) -> dict:
    offer = cheapest_offer(args.max_dph)
    dph = float(offer.get("dph_total"))
    print(f"==> offer {offer.get('id')}: {offer.get('gpu_name')} at ${dph:.3f}/h")

    instance_id, host, port = create_instance(str(offer.get("id")))
    st.register_instance({
        "instance_id": instance_id,
        "host": host,
        "port": port,
        "dph_total": dph,
        "gpu_name": offer.get("gpu_name"),
        "run_id": run_id,
        "campaign": args.campaign,
        "started_at": st.utcnow(),
    })

    cmd = ["bash", str(st.REPO_ROOT / "deploy/train_remote.sh"), host, port, *overrides]
    print(f"==> {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(st.REPO_ROOT), check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"train_remote.sh exited {proc.returncode}. Instance {instance_id} is still "
            "billing — destroy it or retry the launch."
        )
    return {"instance_id": instance_id, "host": host, "port": port, "dph_total": dph}


def launch_local(args: argparse.Namespace, overrides: list[str], run_id: str) -> dict:
    """CPU training in a detached background process, logging to a file."""
    python = str(st.REPO_ROOT / "venv/bin/python")
    if not Path(python).exists():
        python = sys.executable
    log_path = st.campaign_dir(args.campaign) / f"{run_id}.local.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python, str(st.REPO_ROOT / "scripts/train_dreamer.py"), *overrides]
    print(f"==> {' '.join(cmd)}  (log: {log_path})")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, cwd=str(st.REPO_ROOT), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {"local_pid": proc.pid, "local_log": str(log_path)}


def launch_fake(args: argparse.Namespace, overrides: list[str], run_id: str) -> dict:
    cmd = [
        sys.executable, str(st.REPO_ROOT / "tools/fake_trainer.py"),
        "--profile", args.fake_profile, "--seed", str(args.seed),
        "--env-config", args.env_config, "--steps", str(args.steps),
    ]
    print(f"==> {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(st.REPO_ROOT), capture_output=True, text=True, check=False)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError("fake_trainer.py failed")
    return {"local_run_dir": proc.stdout.strip().splitlines()[-1]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Launch one registered training run.")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--rung", required=True, help="Ladder rung id from the campaign spec, e.g. R1.")
    ap.add_argument("--env-config", required=True, help="Env YAML, e.g. configs/target/chase_easy2.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-id", default=None, help="Defaults to <rung>-<config stem>-s<seed>.")
    ap.add_argument("--exp", default="drone_chase", help="sheeprl experiment config name.")
    ap.add_argument("--backend", choices=("vast", "local", "fake"), default="vast")

    ap.add_argument("--steps", type=int, default=2_000_000, help="algo.total_steps safety ceiling.")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--checkpoint-every", type=int, default=50_000)
    ap.add_argument("--override", action="append", default=[],
                    help="Extra Hydra override, repeatable.")

    ap.add_argument("--stop-metric", default="Rewards/rew_avg")
    ap.add_argument("--stop-threshold", type=float, default=None,
                    help="Enable early stopping at this metric value.")
    ap.add_argument("--stop-window", type=int, default=5)
    ap.add_argument("--stop-patience", type=int, default=3)

    ap.add_argument("--min-headroom", type=float, default=2.0,
                    help="Refuse to launch when remaining budget is below this reserve.")
    ap.add_argument("--max-dph", type=float, default=1.0,
                    help="Refuse offers above this hourly price.")
    ap.add_argument("--fake-profile", default="rising",
                    choices=("rising", "flat", "diverging", "nan"))
    ap.add_argument("--force", action="store_true",
                    help="Launch even if this (config, seed) cell is already covered.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would happen, touch nothing, spend nothing.")
    args = ap.parse_args()

    env_config = Path(args.env_config)
    if not env_config.is_absolute():
        env_config = st.REPO_ROOT / env_config
    if not env_config.exists():
        raise SystemExit(f"env config not found: {args.env_config}")

    spec = st.campaign_dir(args.campaign) / "spec.md"
    if not spec.exists() and not args.dry_run:
        raise SystemExit(
            f"REFUSING: no spec at {spec}. Acceptance criteria must be written down "
            "before a run is launched, not after it finishes."
        )

    overrides = build_overrides(args)
    cfg_hash = st.config_hash(env_config, overrides)
    run_id = args.run_id or f"{args.rung}-{env_config.stem}-s{args.seed}"

    if args.dry_run:
        print(json.dumps({
            "run_id": run_id, "campaign": args.campaign, "rung": args.rung,
            "backend": args.backend, "env_config": args.env_config,
            "config_hash": cfg_hash, "seed": args.seed, "overrides": overrides,
            "would_check": {
                "killed": st.is_killed(), "headroom": st.headroom(),
                "min_headroom": args.min_headroom,
            },
        }, indent=2))
        return 0

    preflight(args, cfg_hash)

    record = {
        "run_id": run_id,
        "campaign": args.campaign,
        "rung": args.rung,
        "backend": args.backend,
        "env_config": args.env_config,
        "env_config_sha256": st.file_sha256(env_config),
        "overrides": overrides,
        "config_hash": cfg_hash,
        "seed": args.seed,
        "instance_id": None,
        "host": None,
        "port": None,
        "remote_run_dir": None,
        "local_run_dir": None,
        "started_at": st.utcnow(),
        "status": "launched",
        "stop_reason": None,
        "cost_usd_est": 0.0,
        "eval": None,
    }
    # Registered before anything is provisioned: a launch that dies halfway leaves
    # a visible 'launched' record to reconcile, not an invisible charge.
    st.append_run(args.campaign, record)
    st.journal(args.campaign, f"launch {run_id} ({args.rung}, {env_config.name}, seed {args.seed}, "
                              f"{args.backend}) hash={cfg_hash}")

    try:
        if args.backend == "vast":
            extra = launch_vast(args, overrides, run_id)
        elif args.backend == "local":
            extra = launch_local(args, overrides, run_id)
        else:
            extra = launch_fake(args, overrides, run_id)
    except BaseException as exc:
        st.update_run(args.campaign, run_id, status="failed", stop_reason=f"launch error: {exc}")
        st.journal(args.campaign, f"launch FAILED {run_id}: {exc}")
        raise

    st.update_run(args.campaign, run_id, status="training", **extra)
    print(f"\n==> {run_id} is training. status=training")
    print(json.dumps({"run_id": run_id, "config_hash": cfg_hash, **extra}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
