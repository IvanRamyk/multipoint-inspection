"""Shared state access for the autonomous-experiment tooling.

Every tool under tools/ reads and writes the same small set of files. This module
owns the paths, the locking, and the atomic-write discipline so that several
subagents running concurrently cannot corrupt the run registry.

Nothing here talks to vast.ai or to sheeprl. Keep it dependency-free (stdlib +
PyYAML, which the project already requires) so the guard can run without the
project's venv.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent

STATE_DIR = REPO_ROOT / "experiments" / "state"
CAMPAIGNS_DIR = REPO_ROOT / "experiments" / "campaigns"

CURRENT_PATH = STATE_DIR / "current.json"
BUDGET_PATH = STATE_DIR / "budget.json"
INSTANCES_PATH = STATE_DIR / "instances.json"
KILLED_PATH = STATE_DIR / "KILLED"


def utcnow() -> str:
    """Timestamp for the record. Always UTC, always the same shape."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -- generic file helpers -----------------------------------------------------


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock keyed on ``path``.

    The lock lives in a sibling ``.lock`` file rather than on the target itself,
    so an atomic replace of the target cannot drop the lock out from under a
    concurrent holder.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON via a temp file + rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON, returning ``default`` for a missing or malformed file.

    Malformed is treated as missing on purpose: a half-written state file must
    never take down the orchestrator, which re-derives truth from vast.ai anyway.
    """
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


# -- kill switch and budget --------------------------------------------------


def is_killed() -> bool:
    return KILLED_PATH.exists()


def kill_reason() -> str:
    try:
        return KILLED_PATH.read_text().strip()
    except FileNotFoundError:
        return ""


def read_budget() -> dict:
    """Budget snapshot as written by the external guard.

    A missing file means the guard is not running. Callers must treat that as a
    refusal to spend, never as unlimited headroom.
    """
    return read_json(BUDGET_PATH, default={}) or {}


def headroom() -> float | None:
    """Remaining dollars, or None if the guard has not reported yet."""
    b = read_budget()
    if "cap" not in b or "spent" not in b:
        return None
    return float(b["cap"]) - float(b["spent"])


# -- config hashing ----------------------------------------------------------

# Overrides that identify a *seed* rather than an experiment cell. Excluded from
# the hash so that all seeds of one cell share a join key.
_SEED_PREFIXES = ("seed=",)


def config_hash(env_config: Path | str, overrides: list[str]) -> str:
    """Content hash identifying an experiment cell.

    Digests the env-config YAML *bytes* plus the sorted Hydra overrides with the
    seed removed. Hashing the file content rather than its path means a renamed
    but identical config still collides, which is what a sweep join key should do,
    and an edited config with the same name correctly does not.
    """
    h = hashlib.sha256()
    h.update(Path(env_config).read_bytes())
    h.update(b"\x00")
    for token in sorted(o for o in overrides if not o.startswith(_SEED_PREFIXES)):
        h.update(token.encode())
        h.update(b"\x00")
    return "sha256:" + h.hexdigest()[:32]


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -- run registry (one JSONL per campaign) -----------------------------------


def campaign_dir(campaign: str) -> Path:
    return CAMPAIGNS_DIR / campaign


def runs_path(campaign: str) -> Path:
    return campaign_dir(campaign) / "runs.jsonl"


def read_runs(campaign: str) -> list[dict]:
    """All run records for a campaign, in registration order.

    Skips unparseable lines rather than raising — a torn final line from a crash
    must not make the whole registry unreadable.
    """
    path = runs_path(campaign)
    if not path.exists():
        return []
    runs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return runs


def find_run(campaign: str, run_id: str) -> dict | None:
    for rec in read_runs(campaign):
        if rec.get("run_id") == run_id:
            return rec
    return None


def append_run(campaign: str, record: dict) -> None:
    """Register a run. Refuses to reuse an existing run_id."""
    path = runs_path(campaign)
    with locked(path):
        if any(r.get("run_id") == record["run_id"] for r in read_runs(campaign)):
            raise ValueError(f"run_id already registered: {record['run_id']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")


def update_run(campaign: str, run_id: str, **fields: Any) -> dict:
    """Patch fields on a registered run, rewriting the registry in place.

    JSONL is append-only by nature, so an update means a full rewrite. The files
    hold tens of lines, so this is cheap, and the lock makes it safe against the
    other subagents.
    """
    path = runs_path(campaign)
    with locked(path):
        runs = read_runs(campaign)
        updated = None
        for rec in runs:
            if rec.get("run_id") == run_id:
                rec.update(fields)
                updated = rec
                break
        if updated is None:
            raise KeyError(f"no such run: {run_id}")
        body = "".join(json.dumps(r) + "\n" for r in runs)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="runs", suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            fh.write(body)
        os.replace(tmp, path)
        return updated


# -- instance registry -------------------------------------------------------


def read_instances() -> list[dict]:
    return read_json(INSTANCES_PATH, default=[]) or []


def register_instance(record: dict) -> None:
    with locked(INSTANCES_PATH):
        instances = read_instances()
        instances = [i for i in instances if str(i.get("instance_id")) != str(record["instance_id"])]
        instances.append(record)
        write_json_atomic(INSTANCES_PATH, instances)


def drop_instance(instance_id: str | int) -> None:
    with locked(INSTANCES_PATH):
        instances = [i for i in read_instances() if str(i.get("instance_id")) != str(instance_id)]
        write_json_atomic(INSTANCES_PATH, instances)


# -- journal -----------------------------------------------------------------


def journal(campaign: str, message: str) -> None:
    """Append one timestamped line to the campaign's lab notebook.

    Append-only and never read back in full by the orchestrator — it is the
    human-facing history and the source of the final report.
    """
    path = campaign_dir(campaign) / "journal.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with locked(path):
        with open(path, "a") as fh:
            fh.write(f"- `{utcnow()}` {message}\n")
