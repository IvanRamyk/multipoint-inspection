#!/usr/bin/env python3
"""Train DreamerV3 on DroneInspectionEnv using sheeprl."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch sheeprl training with custom configs."""
    # Ensure our envs package is importable (triggers gymnasium.register).
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    import envs  # noqa: F401

    # Point sheeprl to our custom configs.
    configs_path = project_root / "configs" / "sheeprl"
    os.environ["SHEEPRL_SEARCH_PATH"] = f"file://{configs_path}"

    # Ensure envs is importable for the subprocess too.
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{project_root}{os.pathsep}{pythonpath}"

    # Build sheeprl command.
    cmd = [
        sys.executable,
        "-m",
        "sheeprl",
        "exp=drone_inspection",
    ]

    # Pass through any additional CLI args.
    cmd.extend(sys.argv[1:])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
