#!/usr/bin/env python3
"""Quick validation that DreamerV3 + DroneInspectionEnv pipeline works."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Ensure project root is on the path.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main() -> None:
    """Run smoke tests for the DreamerV3 integration."""
    print("=" * 60)
    print("DreamerV3 Smoke Test")
    print("=" * 60)

    # Step 1: Check env registration.
    print("\n[1/4] Checking environment registration...")
    import envs  # noqa: F401 — triggers registration
    import gymnasium as gym
    import numpy as np

    env = gym.make("DroneInspection-sheeprl-v0")
    obs, info = env.reset()
    assert "depth" in obs, "Missing 'depth' in observations"
    assert "state" in obs, "Missing 'state' in observations"
    assert obs["depth"].dtype == np.uint8, f"Depth should be uint8, got {obs['depth'].dtype}"
    print(f"  depth shape: {obs['depth'].shape}, dtype: {obs['depth'].dtype}")
    print(f"  state shape: {obs['state'].shape}, dtype: {obs['state'].dtype}")
    env.close()
    print("  PASSED")

    # Step 2: Check sheeprl import.
    print("\n[2/4] Checking sheeprl installation...")
    import sheeprl

    print(f"  sheeprl version: {sheeprl.__version__}")
    print("  PASSED")

    # Step 3: Check torch.
    print("\n[3/4] Checking PyTorch...")
    import torch

    print(f"  torch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print("  PASSED")

    # Step 4: Short training run.
    print("\n[4/4] Running DreamerV3 for 2000 steps (this may take a few minutes on CPU)...")

    configs_path = project_root / "configs" / "sheeprl"
    env_vars = os.environ.copy()
    env_vars["SHEEPRL_SEARCH_PATH"] = f"file://{configs_path}"
    pythonpath = env_vars.get("PYTHONPATH", "")
    env_vars["PYTHONPATH"] = f"{project_root}{os.pathsep}{pythonpath}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sheeprl",
            "exp=drone_inspection",
            "algo.total_steps=2000",
            "algo.learning_starts=200",
            "algo.per_rank_batch_size=4",
            "algo.per_rank_sequence_length=8",
            "metric.log_every=500",
            "checkpoint.every=100000",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=env_vars,
    )

    if result.returncode == 0:
        print("  Training completed without errors")
        # Check for NaN in output.
        if "nan" in result.stdout.lower():
            print("  WARNING: NaN detected in training output")
        else:
            print("  No NaN detected")
        print("  PASSED")
    else:
        print("  Training FAILED")
        print(f"  STDOUT (last 1000 chars): {result.stdout[-1000:]}")
        print(f"  STDERR (last 1000 chars): {result.stderr[-1000:]}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("Ready for full training on GPU.")
    print("=" * 60)


if __name__ == "__main__":
    main()
