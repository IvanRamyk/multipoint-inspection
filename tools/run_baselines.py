#!/usr/bin/env python3
"""Measure what a scripted policy achieves on a config, before spending on RL.

Two questions this answers that a learned policy's success rate cannot.

**Is the task trivial?** If proportional pursuit catches the target almost every
time on a near-straight path, the rung proves nothing about learning — and if a
random policy catches it often, it proves less than nothing. A rung has to be hard
enough to be worth a GPU. This is the failure mode the pipeline had no check for.

**Is the task winnable at all?** If scripted pursuit cannot catch the target, the
target is faster than the pursuer can manage or the catch radius is too tight, and
no amount of training will fix it. Finding that out here costs seconds; finding it
out from three failed 3-hour runs costs hours and dollars.

It also produces the behaviour metrics for both baselines, which is what turns
"does the learned policy look like pursuit" into a comparison against a policy
that provably does.

    tools/run_baselines.py --env-config configs/target/chase_easy2.yaml \\
        --task chase --episodes 20 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _state as st  # noqa: E402
from envs.core.config import EnvConfig  # noqa: E402
from eval.behaviour_metrics import aggregate, episode_metrics  # noqa: E402

# State layout: [pos(3), vel(3), wind(3), target_rel(3), target_vel(3)].
_TARGET_REL = slice(9, 12)
_TARGET_VEL = slice(12, 15)


def pursuit_action(obs: dict, gain: float = 0.4, clip: float = 0.5, lead: float = 1.0) -> np.ndarray:
    """Proportional lead pursuit — fly at the target, aiming slightly ahead.

    Kept identical to scripts/eval_target.py so the baseline number here and the
    one a human sees there cannot drift apart. The clip matters: commanding
    near-max velocity on every axis at once destabilises PyFlyt's velocity
    controller, so this stays in the well-behaved regime and is still faster than
    the target.
    """
    state = obs["state"]
    rel = state[_TARGET_REL].astype(np.float64)
    vel = state[_TARGET_VEL].astype(np.float64)
    return np.clip(gain * (rel + lead * vel), -clip, clip).astype(np.float32)


def run_policy(env, raw_env, policy: str, episodes: int, seed_base: int, config: EnvConfig) -> dict:
    per_episode = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed_base + ep)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action = pursuit_action(obs) if policy == "pursuit" else env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        drone = np.array(raw_env.positions)
        target = np.array(raw_env.target_positions)
        caught = bool(info["is_success"])
        metrics = episode_metrics(
            drone, target, caught=caught, dt=1.0 / config.agent_hz,
            reach_distance=config.target_reach_distance,
            dome_size=config.flight_dome_size,
            velocities=np.array(raw_env.velocities) if getattr(raw_env, "velocities", None) else None,
        )
        metrics["reward"] = round(float(total_reward), 3)
        per_episode.append(metrics)

    caught_flags = [m["caught"] for m in per_episode]
    return {
        "policy": policy,
        "episodes": episodes,
        "seed_base": seed_base,
        "success_rate": round(float(np.mean(caught_flags)), 4),
        "successes": int(sum(caught_flags)),
        "behaviour": aggregate(per_episode),
        "results": per_episode,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run scripted baselines on an env config.")
    ap.add_argument("--env-config", required=True)
    ap.add_argument("--task", choices=("chase", "target"), default="chase")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000,
                    help="Seed base. Use the SAME base as the learned policy's eval so the "
                         "comparison is on identical episodes.")
    ap.add_argument("--policies", default="pursuit,random")
    ap.add_argument("--json", default=None, help="Write the report here.")
    ap.add_argument("--trivial-pursuit-rate", type=float, default=0.95,
                    help="Scripted pursuit above this success rate flags the config as trivial.")
    ap.add_argument("--trivial-random-rate", type=float, default=0.25,
                    help="Random above this success rate flags the config as trivial.")
    ap.add_argument("--unwinnable-pursuit-rate", type=float, default=0.30,
                    help="Scripted pursuit below this success rate flags the config as unwinnable.")
    args = ap.parse_args()

    from envs.tasks.drone_chase_env import DroneChaseEnv
    from envs.tasks.drone_target_env import DroneTargetEnv

    env_cls = DroneChaseEnv if args.task == "chase" else DroneTargetEnv
    cfg_path = Path(args.env_config)
    if not cfg_path.is_absolute():
        cfg_path = st.REPO_ROOT / cfg_path
    config = EnvConfig.from_yaml(cfg_path)

    report: dict = {
        "env_config": args.env_config,
        "env_config_sha256": st.file_sha256(cfg_path),
        "task": args.task,
        "episodes": args.episodes,
        "seed_base": args.seed,
        "baselines": {},
        "measured_at": st.utcnow(),
    }

    for policy in [p.strip() for p in args.policies.split(",") if p.strip()]:
        raw_env = env_cls(config=config)
        print(f"==> {policy} on {cfg_path.name} ({args.episodes} episodes)")
        result = run_policy(raw_env, raw_env, policy, args.episodes, args.seed, config)
        raw_env.close()
        report["baselines"][policy] = result
        b = result["behaviour"]
        print(f"    success={result['success_rate']:.0%}  "
              f"alignment={b.get('mean_pursuit_alignment')}  "
              f"path_efficiency={b.get('path_efficiency')}  "
              f"fastest_catch={b.get('fastest_catch_steps')}")

    pursuit = report["baselines"].get("pursuit", {}).get("success_rate")
    random_rate = report["baselines"].get("random", {}).get("success_rate")

    flags = []
    if pursuit is not None and pursuit >= args.trivial_pursuit_rate:
        flags.append(
            f"trivial: scripted pursuit already succeeds {pursuit:.0%} of the time, so a learned "
            "policy matching it demonstrates nothing"
        )
    if random_rate is not None and random_rate >= args.trivial_random_rate:
        flags.append(
            f"trivial: a RANDOM policy succeeds {random_rate:.0%} of the time — the catch radius "
            "or the arena makes accidental interception likely"
        )
    if pursuit is not None and pursuit <= args.unwinnable_pursuit_rate:
        flags.append(
            f"possibly unwinnable: scripted pursuit only manages {pursuit:.0%}. The target may be "
            "too fast or the catch radius too tight for the pursuer's speed limit"
        )

    report["flags"] = flags
    report["verdict"] = "flagged" if flags else "suitable"
    # The margin a learned policy has to beat to have shown anything at all.
    report["floor_for_learned_policy"] = random_rate
    report["reference_for_learned_policy"] = pursuit

    if args.json:
        st.write_json_atomic(Path(args.json), report)

    print()
    if flags:
        print(f"VERDICT: flagged — this config is a poor rung")
        for f in flags:
            print(f"  - {f}")
    else:
        print("VERDICT: suitable — hard enough to be worth training, and winnable")
        if random_rate is not None:
            print(f"  a learned policy must clearly beat random ({random_rate:.0%}) to mean anything")
        if pursuit is not None:
            print(f"  scripted pursuit reference: {pursuit:.0%}")
    if args.json:
        print(f"  path={args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
