#!/usr/bin/env python3
"""Connection probe for an AirSim PIE session.

Run AFTER pressing Play in the AirSim-enabled UE project. Verifies:
  1. RPC connection (port 41451)
  2. Multirotor state readout
  3. The 'front' depth camera produces a 64x64 image

Usage:
    ./venv/bin/python scripts/airsim_probe.py [--ip 127.0.0.1] [--vehicle drone] [--camera front]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1", help="AirSim host (use box IP if remote)")
    ap.add_argument("--vehicle", default="drone")
    ap.add_argument("--camera", default="front")
    args = ap.parse_args()

    try:
        import airsim
    except ImportError:
        print("FAIL: airsim client not installed in this venv.")
        return 2

    print(f"[1/3] connecting to AirSim at {args.ip}:41451 ...")
    client = airsim.MultirotorClient(ip=args.ip)
    try:
        client.confirmConnection()
    except Exception as exc:
        print(f"FAIL: could not connect ({type(exc).__name__}: {exc})")
        print("      -> Is the UE project in Play (PIE) mode? Is it the AirSim project,")
        print("         not a Cesium-only map? Is SimMode=Multirotor in settings.json?")
        return 1
    print("      connection OK")

    print("[2/3] reading multirotor state ...")
    try:
        st = client.getMultirotorState(vehicle_name=args.vehicle)
        p = st.kinematics_estimated.position
        print(f"      vehicle '{args.vehicle}' NED pos = "
              f"({p.x_val:.2f}, {p.y_val:.2f}, {p.z_val:.2f})")
        coll = client.simGetCollisionInfo(vehicle_name=args.vehicle).has_collided
        print(f"      collided = {coll}")
    except Exception as exc:
        print(f"FAIL: state read failed ({type(exc).__name__}: {exc})")
        print(f"      -> Check vehicle name. Available may differ from '{args.vehicle}'.")
        return 1

    print(f"[3/3] grabbing depth image from camera '{args.camera}' ...")
    try:
        req = airsim.ImageRequest(args.camera, airsim.ImageType.DepthPlanar,
                                  pixels_as_float=True, compress=False)
        resp = client.simGetImages([req], vehicle_name=args.vehicle)[0]
        n = len(resp.image_data_float)
        if n == 0:
            print("      WARN: empty image buffer (camera name wrong, or first-frame).")
            print("            Available camera names depend on settings.json 'Cameras'.")
            return 1
        depth = np.array(resp.image_data_float, dtype=np.float32).reshape(
            resp.height, resp.width, 1)
        print(f"      depth shape = {depth.shape}, "
              f"min={depth.min():.2f}m max={depth.max():.2f}m")
    except Exception as exc:
        print(f"FAIL: depth capture failed ({type(exc).__name__}: {exc})")
        return 1

    print("\nALL OK — AirSim is reachable and the depth camera works.")
    print("Next: run a full env rollout with backend=airsim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
