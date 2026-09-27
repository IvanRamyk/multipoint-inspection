#!/usr/bin/env python3
"""launch_hard.py — launch_run's twin that picks the MOST POWERFUL GPU within budget.

launch_run.py deliberately picks the *cheapest* offer meeting the requirements. This
variant instead sorts offers by vast's `dlperf` (deep-learning perf score) descending
and takes the strongest one within `--max-dph` — for when you want speed over price
(human-authorised exploratory runs).

It is a thin shim: it monkeypatches only the offer-selection function and then hands
off to launch_run.main(), so it INHERITS every safety property unchanged —
  * budget headroom + KILLED preflight,
  * register-the-run-before-spending,
  * config hashing, journalling, and the same run/instance bookkeeping.
The guard's $cap and TTL still bound total spend; this only changes *which* GPU
inside that envelope.

Usage (every launch_run flag works, plus the two hardware knobs below):
  tools/launch_hard.py --campaign C --rung R --env-config X.yaml --seed 1 \
      --exp drone_chase_fast --max-dph 0.6 --min-gpu-ram 24 [--gpu-name 4090]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import launch_run as lr  # noqa: E402


def powerful_offer_factory(min_gpu_ram: float, gpu_name: str | None):
    """Return a cheapest_offer-compatible fn that picks the highest-dlperf offer."""

    def powerful_offer(max_dph: float | None) -> dict:
        # vast's search query takes gpu_ram in GB (like launch_run's "gpu_ram >= 16"),
        # even though the returned JSON field is in MB.
        query = (
            f"reliability > 0.98 num_gpus=1 gpu_ram >= {min_gpu_ram:g} cuda_vers >= 12.0 "
            "inet_down > 100 disk_space >= 20"
        )
        out = subprocess.run(
            [lr.vastai_bin(), "search", "offers", query,
             "--order", "dlperf", "--limit", "40", "--raw"],
            capture_output=True, text=True, check=False,
        )
        if out.returncode != 0:
            raise RuntimeError(f"vastai search failed: {out.stderr.strip()[:300]}")
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
            raise RuntimeError(
                f"no vast.ai offers with gpu_ram >= {min_gpu_ram} GB matched"
            )
        if gpu_name:
            offers = [o for o in offers
                      if gpu_name.lower() in str(o.get("gpu_name", "")).lower()]
            if not offers:
                raise RuntimeError(f"no offers matched --gpu-name '{gpu_name}'")
        if max_dph is not None:
            affordable = [o for o in offers if float(o.get("dph_total") or 1e9) <= max_dph]
            if not affordable:
                best_dph = min(float(o.get("dph_total") or 1e9) for o in offers)
                raise RuntimeError(
                    f"cheapest matching powerful offer is ${best_dph:.3f}/h, above the "
                    f"--max-dph ceiling ${max_dph:.3f}/h — raise --max-dph"
                )
            offers = affordable
        # Most powerful within budget.
        offers.sort(key=lambda o: float(o.get("dlperf") or 0.0), reverse=True)
        best = offers[0]
        print(
            f"==> HARD offer {best.get('id')}: {best.get('gpu_name')} "
            f"dlperf={best.get('dlperf')} gpu_ram={int((best.get('gpu_ram') or 0)/1024)}GB "
            f"at ${float(best.get('dph_total') or 0):.3f}/h "
            f"({best.get('dlperf_per_dphtotal')} perf/$)"
        )
        return best

    return powerful_offer


def main() -> int:
    # Parse (and strip) the two hardware knobs; everything else is launch_run's.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--min-gpu-ram", type=float, default=24.0,
                     help="GPU VRAM floor in GB (default 24 -> 3090/4090/A100 class).")
    pre.add_argument("--gpu-name", default=None,
                     help="Optional substring filter, e.g. '4090' or 'A100'.")
    hw, rest = pre.parse_known_args()

    # Swap ONLY the offer selector; launch_run.launch_vast resolves this at call time.
    lr.cheapest_offer = powerful_offer_factory(hw.min_gpu_ram, hw.gpu_name)

    # Hand the remaining args to the unchanged launcher.
    sys.argv = [sys.argv[0]] + rest
    return lr.main()


if __name__ == "__main__":
    sys.exit(main())
