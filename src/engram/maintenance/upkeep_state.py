"""Upkeep bookkeeping in ``maintenance_state.jsonl`` and the run lock.

The state file is append-only JSONL (git sync union-merges it across
machines); for each key the line with the latest ``at`` wins. Two kinds of
line exist:

- ``run``: when a full upkeep run last started, so a restarted server does
  not immediately re-run upkeep.
- ``consolidated``: per project scope, the snapshot time consolidation last
  read, the ``updated_at`` of every fact that run wrote (so its own writes do
  not count as new next time), and cluster seeds whose call failed (retried
  next run, up to ``MAX_SEED_ATTEMPTS``).

Writers hold ``upkeep.lock`` (see :func:`upkeep_lock`).
"""

from __future__ import annotations

import fcntl
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

STATE_FILE = "maintenance_state.jsonl"
LOCK_FILE = "upkeep.lock"
MAX_SEED_ATTEMPTS = 3
_COMPACT_LINES = 1000


class StateLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["run", "consolidated"]
    at: datetime
    project: str | None = None
    # Fact ID -> the updated_at upkeep's own write left it with.
    written: dict[str, datetime] = Field(default_factory=dict)
    # Seed fact ID -> failed consolidation attempts so far.
    retry_seeds: dict[str, int] = Field(default_factory=dict)


@dataclass
class ProjectState:
    consolidated_at: datetime
    written: dict[str, datetime] = field(default_factory=dict)
    retry_seeds: dict[str, int] = field(default_factory=dict)


@dataclass
class UpkeepState:
    last_run_at: datetime | None = None
    projects: dict[str | None, ProjectState] = field(default_factory=dict)


def _read_lines(path: Path) -> list[StateLine]:
    if not path.exists():
        return []
    lines: list[StateLine] = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            lines.append(StateLine.model_validate_json(raw))
        except ValidationError:
            logger.warning("Skipping unreadable line in %s", path)
    return lines


def _latest(lines: list[StateLine]) -> dict[tuple[str, str | None], StateLine]:
    latest: dict[tuple[str, str | None], StateLine] = {}
    for line in lines:
        key = (line.kind, line.project)
        if key not in latest or line.at >= latest[key].at:
            latest[key] = line
    return latest


def load_state(data_dir: Path) -> UpkeepState:
    state = UpkeepState()
    for (kind, project), line in _latest(_read_lines(data_dir / STATE_FILE)).items():
        if kind == "run":
            state.last_run_at = line.at
        else:
            state.projects[project] = ProjectState(
                consolidated_at=line.at,
                written=dict(line.written),
                retry_seeds=dict(line.retry_seeds),
            )
    return state


def append_state(data_dir: Path, lines: list[StateLine]) -> None:
    """Append state lines; rewrite to the latest line per key once long."""
    if not lines:
        return
    path = data_dir / STATE_FILE
    with path.open("a") as fh:
        fh.writelines(line.model_dump_json() + "\n" for line in lines)
    existing = _read_lines(path)
    if len(existing) <= _COMPACT_LINES:
        return
    fd, tmp = tempfile.mkstemp(
        dir=data_dir, prefix=".maintenance_state.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as fh:
            fh.writelines(
                line.model_dump_json() + "\n" for line in _latest(existing).values()
            )
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@contextmanager
def upkeep_lock(data_dir: Path) -> Iterator[bool]:
    """Non-blocking exclusive lock; yields False when another run holds it.

    ``flock`` locks belong to the open file description, so this excludes
    concurrent runs in other processes and in this process alike.
    """
    fh = (data_dir / LOCK_FILE).open("a")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()
