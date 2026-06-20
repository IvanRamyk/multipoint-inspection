# Cloud deployment (vast.ai / RunPod / any CUDA host)

Two independent targets. **Do them separately** — they have very different
difficulty and you almost never need both on the same box.

| Target | What | Difficulty on vast.ai | When |
|---|---|---|---|
| **A. DreamerV3 training (sheeprl)** | GPU training of the world-model agent | Easy | Phase 3 — the thing that's actually blocked on a slow laptop |
| **B. AirSim/UE5 photorealistic validation** | Headless Unreal + Cosys-AirSim | Hard (bare containers) | Phase 4 — only at the very end |

---

## A. DreamerV3 training on vast.ai (the easy win)

### 1. Get the code onto the box
Two options:

**(a) via git (recommended)** — push this repo to a private GitHub repo, then on the instance:
```bash
bash setup_vast.sh https://<token>@github.com/<you>/dreamer.git main
```

**(b) via rsync (no git host needed)** — from your Mac:
```bash
rsync -avz --exclude venv --exclude logs --exclude 'results' \
  -e "ssh -p <PORT>" ./ root@<HOST>:/workspace/dreamer/
ssh -p <PORT> root@<HOST> 'cd /workspace/dreamer && bash deploy/setup_vast.sh'
```

### 2. Pick the instance
- Image: any **PyTorch CUDA** template (e.g. `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`).
- GPU: a single **RTX 4090 / A5000 / A100** is plenty for the XS–S models. ~$0.3–0.7/hr.
- Disk: 20 GB is enough (checkpoints are ~105 MB each).

### 3. Train
```bash
# Sanity recipe on GPU (minutes, not hours):
python scripts/train_dreamer.py \
    fabric.accelerator=gpu fabric.precision=16-mixed \
    algo.total_steps=15000 metric.log_every=200 checkpoint.every=2000

# Real Phase-3 run (bigger model, more steps, parallel envs):
python scripts/train_dreamer.py \
    fabric.accelerator=gpu algo=dreamer_v3_S \
    algo.total_steps=2000000 env.num_envs=4 \
    metric.log_every=1000 checkpoint.every=50000
```
PyFlyt runs fully headless (PyBullet DIRECT + software depth) — no display needed.

### 4. Watch / fetch results
```bash
# TensorBoard (forward the port, or use vast's port mapping):
tensorboard --logdir logs/runs/dreamer_v3 --port 6006 --bind_all

# Pull checkpoints + events back to your Mac:
rsync -avz -e "ssh -p <PORT>" \
  root@<HOST>:/workspace/dreamer/logs/ ./logs/
```
Then eval / make route plots locally on the Mac with `scripts/eval_dreamer.py`.

### Or just use the Docker image
```bash
docker build -f deploy/Dockerfile.train -t dreamer-train .
docker run --gpus all -p 6006:6006 -v $PWD/logs:/workspace/logs dreamer-train \
  python scripts/train_dreamer.py fabric.accelerator=gpu algo.total_steps=15000
```

---

## B. AirSim / Cosys-AirSim (UE5) in the cloud — honest notes

This is the hard part and **vast.ai's bare containers are a poor fit**:

- UE needs a real **Vulkan/OpenGL + NVIDIA driver** stack and runs **headless**
  via `-RenderOffscreen` (often plus `Xvfb` / EGL). Many vast images lack the
  full GL stack, and getting offscreen UE rendering up is fiddly.
- Packaged UE environments are **large** (tens of GB) and vast disk is limited.
- The AirSim RPC server + the Python client must run **on the same box**
  (`AirSimBackend(ip="127.0.0.1")`), or expose port 41451 over SSH tunnel.

### Realistic options, best first
1. **Full GPU VM, not a bare container** — Lambda Cloud / GCP / AWS GPU VM with
   a desktop + NVIDIA drivers. Far more reliable for UE than vast containers.
2. **Cosys-AirSim packaged Linux environment** — download a prebuilt env binary,
   run headless:
   ```bash
   ./Blocks.sh -RenderOffscreen -opengl4 &     # or Vulkan; starts RPC on :41451
   python scripts/airsim_probe.py              # then our probe / rollout
   ```
   Put a proper `settings.json` (Multirotor + a `front` DepthPlanar 64×64 camera —
   see `envs/airsim_backend.py` docstring) at `~/Documents/AirSim/settings.json`.
3. **Community AirSim Docker image** with GPU + Xvfb — works but you inherit
   someone else's UE/driver pinning.

### Important: keep training and validation separate
Train the agent cheaply on a plain GPU box (Target A). Only spin up the heavy
UE/AirSim box for **final photorealistic validation** of an already-trained
checkpoint — don't pay for UE rendering during the millions of training steps.

The code is ready: set `backend: airsim` in a config YAML (or pass the env a
pre-built `AirSimBackend`), point it at the running AirSim, and the existing
`DroneInspectionEnv` works unchanged.
