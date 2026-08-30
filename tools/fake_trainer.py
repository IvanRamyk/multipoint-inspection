#!/usr/bin/env python3
"""Synthesise a training run directory with real TensorBoard events.

Verification aid, not a trainer. It writes the same directory shape sheeprl
produces — events.out.tfevents.*, config.yaml, checkpoint/ckpt_<step>_0.ckpt —
with curves traced from a chosen profile, so the whole orchestration loop
(health checks, evaluation, aggregation, gating, journalling) can be exercised
end to end for free and in seconds.

Profiles map onto the verdicts run_health.py must distinguish:
  rising     reward climbs and episodes shorten     -> healthy
  flat       reward stuck, episodes pinned at limit -> stalled
  diverging  world-model loss and KL blow up        -> diverged
  nan        losses go non-finite                   -> dead

Prints the created version_0 directory as its last line, which launch_run.py
records as the run's local_run_dir.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _state as st  # noqa: E402

ENV_ID = "DroneChase-sheeprl-v0"


def curve(profile: str, frac: float, rng: random.Random) -> dict[str, float]:
    """One log point's scalars at progress ``frac`` in [0, 1]."""
    noise = rng.gauss(0.0, 1.0)

    if profile == "rising":
        reward = -10.0 + 70.0 * (1.0 - math.exp(-3.0 * frac)) + 2.0 * noise
        ep_len = 590.0 - 380.0 * (1.0 - math.exp(-2.5 * frac)) + 8.0 * noise
        wm_loss = 12.0 * math.exp(-1.2 * frac) + 0.4 * abs(noise)
        kl = 4.0 + 2.0 * frac + 0.2 * noise
    elif profile == "flat":
        reward = -6.0 + 1.5 * noise
        ep_len = 598.0 + 1.0 * noise          # pinned at the 600-step limit
        wm_loss = 10.0 + 0.5 * abs(noise)
        kl = 4.2 + 0.2 * noise
    elif profile == "diverging":
        reward = -5.0 - 40.0 * frac ** 2 + 3.0 * noise
        ep_len = 500.0 + 60.0 * frac + 8.0 * noise
        wm_loss = 9.0 * math.exp(4.0 * frac)  # blows past any trailing median
        kl = 4.0 * math.exp(3.5 * frac)
    elif profile == "nan":
        if frac > 0.5:
            bad = float("nan")
            return {"reward": bad, "ep_len": bad, "wm_loss": bad, "kl": bad}
        reward = -5.0 + 2.0 * noise
        ep_len = 550.0 + 10.0 * noise
        wm_loss = 11.0
        kl = 4.1
    else:
        raise ValueError(f"unknown profile: {profile}")

    return {"reward": reward, "ep_len": max(1.0, ep_len), "wm_loss": wm_loss, "kl": kl}


def main() -> int:
    ap = argparse.ArgumentParser(description="Write a synthetic sheeprl run directory.")
    ap.add_argument("--profile", required=True,
                    choices=("rising", "flat", "diverging", "nan"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200_000, help="Final env step.")
    ap.add_argument("--points", type=int, default=40, help="Number of log points.")
    ap.add_argument("--checkpoints", type=int, default=3)
    ap.add_argument("--env-config", default="configs/target/chase_easy2.yaml")
    ap.add_argument("--logs-root", default=None,
                    help="Defaults to the repo's logs/runs/dreamer_v3.")
    args = ap.parse_args()

    from torch.utils.tensorboard import SummaryWriter

    logs_root = Path(args.logs_root) if args.logs_root else st.REPO_ROOT / "logs/runs/dreamer_v3"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = logs_root / ENV_ID / f"{stamp}_fake_{args.profile}_{ENV_ID}_{args.seed}" / "version_0"
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    writer = SummaryWriter(log_dir=str(run_dir))
    for i in range(1, args.points + 1):
        frac = i / args.points
        step = int(args.steps * frac)
        c = curve(args.profile, frac, rng)
        writer.add_scalar("Rewards/rew_avg", c["reward"], step)
        writer.add_scalar("Game/ep_len_avg", c["ep_len"], step)
        writer.add_scalar("Loss/world_model_loss", c["wm_loss"], step)
        writer.add_scalar("Loss/policy_loss", 0.5 * c["wm_loss"], step)
        writer.add_scalar("Loss/value_loss", 0.3 * c["wm_loss"], step)
        writer.add_scalar("State/kl", c["kl"], step)
        writer.add_scalar("Time/sps_train", 120.0 + rng.gauss(0, 5), step)
    writer.flush()
    writer.close()

    # A minimal merged config, shaped like the one sheeprl writes. The eval and
    # health tools read screen_size, grayscale, the cnn/mlp keys and the inner
    # env config path out of this.
    (run_dir / "config.yaml").write_text(
        "seed: {seed}\n"
        "env:\n"
        "  id: {env_id}\n"
        "  screen_size: 64\n"
        "  grayscale: true\n"
        "  num_envs: 1\n"
        "  max_episode_steps: null\n"
        "  wrapper:\n"
        "    config_path: {cfg}\n"
        "algo:\n"
        "  name: dreamer_v3\n"
        "  total_steps: {steps}\n"
        "  cnn_keys:\n"
        "    encoder: [depth]\n"
        "  mlp_keys:\n"
        "    encoder: [state]\n"
        "fake: true\n".format(seed=args.seed, env_id=ENV_ID, cfg=args.env_config, steps=args.steps)
    )

    ckpt_dir = run_dir / "checkpoint"
    ckpt_dir.mkdir(exist_ok=True)
    for i in range(1, args.checkpoints + 1):
        step = int(args.steps * i / args.checkpoints)
        # Not a loadable checkpoint — deliberately. Anything that tries to
        # actually restore a policy from a fake run should fail loudly, and only
        # tools/evaluate_ckpt.py --fake is allowed to look at these.
        (ckpt_dir / f"ckpt_{step}_0.ckpt").write_bytes(b"FAKE-CKPT\n" + bytes(64))

    print(f"profile={args.profile} points={args.points} steps={args.steps}", file=sys.stderr)
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
