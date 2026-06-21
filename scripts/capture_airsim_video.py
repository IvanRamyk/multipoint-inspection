#!/usr/bin/env python3
"""Capture a video clip from AirSim's chase camera while training is running.

Connects to the live AirSim instance and records the RGB chase camera for a
fixed duration, saving an MP4. Runs alongside training without interfering.

Usage:
    venv/bin/python scripts/capture_airsim_video.py \
        [--ip 127.0.0.1] [--duration 30] [--fps 15] [--out clip.mp4]
"""
from __future__ import annotations

import argparse
import time
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds to record")
    ap.add_argument("--fps", type=int, default=15, help="capture frame rate")
    ap.add_argument("--out", default="airsim_clip.mp4")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--camera", default="chase")
    ap.add_argument("--vehicle", default="drone")
    args = ap.parse_args()

    try:
        import cosysairsim as airsim
    except ImportError:
        print("FAIL: cosysairsim not installed.")
        return 2

    try:
        import imageio
    except ImportError:
        print("FAIL: imageio not installed.")
        return 2

    print(f"Connecting to AirSim at {args.ip}:41451 ...")
    client = airsim.MultirotorClient(ip=args.ip)
    client.confirmConnection()
    print("Connected.")

    req = airsim.ImageRequest(
        args.camera, airsim.ImageType.Scene, pixels_as_float=False, compress=False
    )

    n_frames = int(args.duration * args.fps)
    interval = 1.0 / args.fps
    frames = []

    print(f"Recording {args.duration}s @ {args.fps}fps ({n_frames} frames) -> {args.out}")
    for i in range(n_frames):
        t0 = time.monotonic()
        try:
            resps = client.simGetImages([req], vehicle_name=args.vehicle)
            resp = resps[0]
            if resp.width == 0 or resp.height == 0:
                # Camera not ready yet — use a black frame
                frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            else:
                img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
                frame = img.reshape(resp.height, resp.width, 3)
                # AirSim returns BGR; convert to RGB
                frame = frame[:, :, ::-1].copy()
        except Exception as e:
            print(f"  frame {i}: capture error ({e}), inserting black frame")
            frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)

        frames.append(frame)
        elapsed = time.monotonic() - t0
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

        if (i + 1) % args.fps == 0:
            print(f"  {i + 1}/{n_frames} frames captured ...")

    print(f"Writing {args.out} ...")
    imageio.mimwrite(args.out, frames, fps=args.fps, quality=8)
    print(f"Done. Saved {len(frames)} frames to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
