"""Gymnasium wrapper that records training episodes to video periodically.

Wrap a :class:`~envs.tasks.drone_target_env.DroneTargetEnv` (optionally already
wrapped for sheeprl). Every ``record_every`` completed episodes it renders that
episode's drone-vs-target trajectory to an MP4/GIF, so you can watch the policy
evolve over the course of training (dumb at episode 1 → competent later).

The wrapper reads the finished episode's trajectory from the base env
(``unwrapped.positions`` / ``target_positions``), so it works no matter what
observation/action wrappers sit in between.
"""

from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import numpy as np

from eval.target_visualizer import animate_target_episode, animate_target_episode_3d


class EpisodeVideoRecorder(gym.Wrapper):
    """Render every Nth completed episode to a trajectory video.

    Args:
        env: The environment to wrap.
        record_every: Render one video every this many episodes (0 disables).
            Episode 1 is always rendered so you get an early "dumb agent" clip.
        out_dir: Directory for the video files.
        max_videos: Stop recording after this many videos (0 = unlimited).
        use_3d: Render the orbiting 3D view (else the 2D top-down + distance).
        fps: Video frame rate.
    """

    def __init__(
        self,
        env: gym.Env,
        record_every: int = 1000,
        out_dir: str = "results/training_videos",
        max_videos: int = 0,
        use_3d: bool = True,
        fps: int = 30,
        nest_under_run: bool = True,
    ) -> None:
        super().__init__(env)
        self.record_every = record_every
        self.out_dir = Path(out_dir)
        self.max_videos = max_videos
        self.use_3d = use_3d
        self.fps = fps
        self.nest_under_run = nest_under_run
        self._episode = 0
        self._recorded = 0
        self._ep_reward = 0.0
        # Output dir is resolved lazily on the first render, by which point
        # sheeprl's per-run log dir definitely exists (see _resolve_out_dir).
        self._out_dir: Path | None = None
        # Ownership: with async vector envs each env runs in its own process,
        # so several EpisodeVideoRecorders exist. The first to claim the lock
        # file records; the rest disable themselves, giving one clean stream.
        self._is_owner: bool | None = None

    def reset(self, **kwargs):
        self._ep_reward = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_reward += float(reward)
        if terminated or truncated:
            self._episode += 1
            if self._should_record():
                self._render_episode(caught=bool(info.get("is_success", False)))
                self._recorded += 1
        return obs, reward, terminated, truncated, info

    def _resolve_out_dir(self) -> Path:
        """Resolve (once) where videos go, nesting under the current run dir.

        When ``nest_under_run`` is set, videos are written to
        ``<sheeprl run dir>/videos/`` — i.e. alongside that run's checkpoints
        and TensorBoard events — so each training run keeps its own videos and
        ``deploy/fetch_results.sh`` pulls them automatically as part of logs/.
        Falls back to the plain ``out_dir`` if no run dir can be found (e.g.
        when the recorder is used outside a sheeprl run).
        """
        if self._out_dir is not None:
            return self._out_dir
        target = self.out_dir
        if self.nest_under_run:
            runs = Path("logs/runs")
            version_dirs = [p for p in runs.rglob("version_*") if p.is_dir()] if runs.exists() else []
            if version_dirs:
                run_dir = max(version_dirs, key=lambda p: p.stat().st_mtime)
                target = run_dir / "videos"
        target.mkdir(parents=True, exist_ok=True)
        self._out_dir = target
        return target

    def _claim_ownership(self) -> bool:
        """Atomically claim the single recorder slot for this run.

        Returns True for exactly one recorder across all env processes; the
        others get False and stop recording. Best-effort: on any OS error we
        default to recording rather than silently dropping all videos.
        """
        lock = self._resolve_out_dir() / ".recorder_owner"
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            return False
        except OSError:
            return True

    def _should_record(self) -> bool:
        if self.record_every <= 0:
            return False
        if self._is_owner is None:
            self._is_owner = self._claim_ownership()
        if not self._is_owner:
            self.record_every = 0  # disable further checks in this process
            return False
        if self.max_videos and self._recorded >= self.max_videos:
            return False
        return self._episode == 1 or self._episode % self.record_every == 0

    def _render_episode(self, caught: bool) -> None:
        base_env = self.env.unwrapped
        drone = np.asarray(base_env.positions)
        target = np.asarray(base_env.target_positions)
        if len(drone) < 2:
            return
        cfg = base_env.config
        tag = "CAUGHT" if caught else "miss"
        prefix = self._resolve_out_dir() / f"ep{self._episode:06d}_{tag}"
        suffix = f"(train ep {self._episode}, R={self._ep_reward:.0f})"
        render = animate_target_episode_3d if self.use_3d else animate_target_episode
        try:
            render(
                drone=drone,
                target=target,
                base=np.array(cfg.base_position),
                reach_distance=cfg.target_reach_distance,
                dome_size=cfg.flight_dome_size,
                caught=caught,
                dt=1.0 / cfg.agent_hz,
                save_path=f"{prefix}.mp4",
                fps=self.fps,
                title_suffix=suffix,
            )
        except Exception as exc:  # never let recording crash training
            print(f"[EpisodeVideoRecorder] render failed for episode {self._episode}: {exc}")
