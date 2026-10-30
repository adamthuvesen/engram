"""Per-directory transaction exclusion, durable event intent, and event-log I/O."""

import fcntl
import json
import logging
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from engram.core.models import (
    EVENT_LOG_META_VERSION,
    EventLogMeta,
    FactEvent,
    decode_stored_event,
)

logger = logging.getLogger(__name__)
_THREAD_LOCKS: dict[Path, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
_STORE_LOCK_DEPTHS = threading.local()


class StoreIntegrityError(ValueError):
    """A persisted row cannot safely participate in an ordinary snapshot."""

    def __init__(self, path: Path, line: int, detail: str):
        self.path = path
        self.line = line
        super().__init__(
            f"{path}:{line}: {detail}. Run engram doctor to inspect this row. "
            "Restore the file or use the reported repair before continuing"
        )


def decode_event_log_meta(line: str) -> EventLogMeta:
    data = json.loads(line)
    if not isinstance(data, dict) or data.get("meta") != EVENT_LOG_META_VERSION:
        raise ValueError("Invalid event-log sentinel")
    return EventLogMeta.model_validate(data)


@dataclass
class EventLogScan:
    meta: EventLogMeta | None = None
    events: list[FactEvent] = field(default_factory=list)
    corrupt_lines: list[int] = field(default_factory=list)
    sentinel_line: int | None = None


def scan_event_log(path: Path, *, strict: bool = False) -> EventLogScan:
    """Decode one v1 log, including legacy sentinels retained by union merges."""
    result = EventLogScan()
    if not path.exists():
        return result
    first = True
    with path.open() as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                meta = decode_event_log_meta(line)
            except ValueError:
                meta = None
            if first:
                first = False
                if meta is None:
                    result.sentinel_line = lineno
                    if strict:
                        raise StoreIntegrityError(path, lineno, "invalid event-log sentinel")
            if meta is not None:
                if result.meta is None and result.sentinel_line is None:
                    result.meta = meta
                continue
            try:
                result.events.append(decode_stored_event(line))
            except ValueError as exc:
                if strict:
                    raise StoreIntegrityError(path, lineno, "invalid event record") from exc
                result.corrupt_lines.append(lineno)
    return result


def _thread_lock_for(path: Path) -> threading.RLock:
    """Return a process-local lock for the given filesystem lock path."""
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(path)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[path] = lock
        return lock


def _store_lock_depths() -> dict[Path, int]:
    depths = getattr(_STORE_LOCK_DEPTHS, "depths", None)
    if depths is None:
        depths = {}
        _STORE_LOCK_DEPTHS.depths = depths
    return depths


@contextmanager
def locked_write(path: Path) -> Iterator[None]:
    """Acquire an exclusive flock on <path>.lock before writing."""
    if sys.platform == "win32":
        logger.warning(
            "File locking not supported on Windows; concurrent writes are not serialized"
        )
        yield
        return
    lock_path = path.with_suffix(".lock")
    lock_fd = lock_path.open("a")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


@contextmanager
def locked_store(data_dir: Path) -> Iterator[None]:
    """Acquire the coarse store lock for multi-file operations."""
    data_dir = data_dir.resolve()
    lock_path = data_dir / "store.lock"
    thread_lock = _thread_lock_for(lock_path)
    with thread_lock:
        depths = _store_lock_depths()
        depth = depths.get(lock_path, 0)
        if depth > 0:
            depths[lock_path] = depth + 1
            try:
                yield
            finally:
                next_depth = depths[lock_path] - 1
                if next_depth:
                    depths[lock_path] = next_depth
                else:
                    depths.pop(lock_path, None)
            return

        depths[lock_path] = 1
        if sys.platform == "win32":
            try:
                logger.warning(
                    "File locking not supported on Windows; store transactions are not serialized"
                )
                yield
            finally:
                depths.pop(lock_path, None)
            return

        lock_fd = lock_path.open("a")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            depths.pop(lock_path, None)
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()


def ensure_trailing_newline(path: Path) -> None:
    """Ensure the file ends with a newline to prevent JSON concatenation."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as f:
        f.seek(-1, 2)
        if f.read(1) != b"\n":
            with path.open("ab") as af:
                af.write(b"\n")


class StoreTransactions:
    """Own local event intents and use one lock for every store access."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    @property
    def facts_path(self) -> Path:
        return self.data_dir / "facts.jsonl"

    @property
    def intent_path(self) -> Path:
        return self.data_dir / ".transaction-intent.json"

    def read_intent(self) -> list[FactEvent]:
        return [FactEvent.model_validate(row) for row in json.loads(self.intent_path.read_text())]

    def clear_intent(self) -> None:
        self.intent_path.unlink()
        self.sync_directory()

    def _is_event_log_format(self) -> bool:
        """Return True when ``facts.jsonl`` starts with an event-log meta sentinel."""
        if not self.facts_path.exists():
            return False
        with self.facts_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = decode_event_log_meta(line)
                except ValueError, ValidationError:
                    return False
                return payload.meta == EVENT_LOG_META_VERSION
        return False

    def _ensure_event_log_header(self) -> None:
        """Write the meta sentinel as the first line if the file is brand-new."""
        if self.facts_path.exists() and self.facts_path.stat().st_size > 0:
            return
        with self.facts_path.open("w") as f:
            f.write(EventLogMeta().model_dump_json(exclude_none=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def append_raw(self, events: list[FactEvent]) -> None:
        """Append one or more typed events to the event log under the write lock."""
        if not events:
            return
        with locked_store(self.data_dir):
            if (
                self.facts_path.exists()
                and self.facts_path.stat().st_size > 0
                and not self._is_event_log_format()
            ):
                raise ValueError(
                    f"{self.facts_path} is missing a valid event-log sentinel; "
                    "restore it from backup or move the invalid file aside."
                )
            existing_ids = {event.event_id for event in self.read_events()[1]}
            events = [event for event in events if event.event_id not in existing_ids]
            with locked_write(self.facts_path):
                self._ensure_event_log_header()
                ensure_trailing_newline(self.facts_path)
                with self.facts_path.open("a") as f:
                    for event in events:
                        f.write(event.model_dump_json() + "\n")
                    f.flush()
                    os.fsync(f.fileno())
        logger.info("Appended %d event(s) to event log", len(events))

    def sync_directory(self) -> None:
        fd = os.open(self.data_dir, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def write_intent(self, events: list[FactEvent]) -> None:
        temp = self.data_dir / ".transaction-intent.tmp"
        with temp.open("w") as stream:
            json.dump([event.model_dump(mode="json") for event in events], stream)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(self.intent_path)
        self.sync_directory()

    def read_events(
        self,
        *,
        strict: bool = False,
    ) -> tuple[EventLogMeta | None, list[FactEvent], int]:
        result = scan_event_log(self.facts_path, strict=strict)
        return result.meta, result.events, len(result.corrupt_lines)
