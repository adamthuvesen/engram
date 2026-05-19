"""JSONL-based knowledge store for facts."""

import asyncio
import fcntl
import logging
import os
import re
import sys
import tempfile
import threading
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import ParamSpec, TypeVar

from uuid import uuid4

from pydantic import ValidationError

from engram.config import get_settings
from engram.models import (
    CandidateStatus,
    EVENT_LOG_META_VERSION,
    EventLogMeta,
    EventType,
    Fact,
    FactCategory,
    FactEvent,
    IngestionRecord,
    MIN_ACTIVE_CONFIDENCE,
    MemoryCandidate,
    RecallRecord,
    StoreTransaction,
    TransactionStatus,
    materialize_events,
)

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_THREAD_LOCKS: dict[Path, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
_STORE_LOCK_DEPTHS = threading.local()
P = ParamSpec("P")
R = TypeVar("R")

# Lightweight suffix-stripping stemmer (no NLTK dependency)
_STEM_SUFFIXES = (
    "tion",
    "sion",
    "ment",
    "ness",
    "ible",
    "able",
    "ing",
    "ies",
    "ous",
    "ive",
    "ers",
    "ed",
    "ly",
    "es",
    "er",
    "al",
    "s",
)


def _stem(word: str) -> str:
    """Cheap suffix strip — good enough for prefilter scoring."""
    if len(word) <= 4:
        return word
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _content_hash(text: str) -> str:
    """Normalize and hash fact content for exact-match dedup."""
    return " ".join(text.lower().split())


def _is_expired_fact(fact: Fact, now: datetime) -> bool:
    return fact.expires_at is not None and fact.expires_at < now


def _is_active_fact(
    fact: Fact,
    now: datetime,
    *,
    include_stale: bool = False,
    min_confidence: float = MIN_ACTIVE_CONFIDENCE,
) -> bool:
    return (
        fact.confidence >= min_confidence
        and not _is_expired_fact(fact, now)
        and (include_stale or not fact.stale)
    )


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


def format_facts_for_llm(facts: list[Fact]) -> str:
    """Format facts as a numbered list for LLM context."""
    if not facts:
        return "(no facts stored)"
    lines = []
    for i, fact in enumerate(facts, 1):
        meta = f"[{fact.category.value}]"
        if fact.project:
            meta += f" [{fact.project}]"
        if fact.confidence < 1.0:
            meta += f" [confidence: {fact.confidence:.1f}]"
        if fact.source_ref:
            meta += f" [source: {fact.source_ref}]"
        if fact.supersedes:
            meta += f" [supersedes: {fact.supersedes}]"
        lines.append(f"{i}. {meta} {fact.content} (id: {fact.id})")
    return "\n".join(lines)


@contextmanager
def _locked_write(path: Path):
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
def _locked_store(data_dir: Path):
    """Acquire the coarse store lock for multi-file operations."""
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


class FactStore:
    """Read/write/filter operations on the JSONL fact store."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or get_settings().data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Cache: fact.id -> (updated_at_iso, unigrams, bigrams)
        self._tok_cache: dict[str, tuple[str, set[str], set[str]]] = {}
        self.recover_transactions()
        self._migrate_to_event_log_if_needed()

    @property
    def facts_path(self) -> Path:
        return self.data_dir / "facts.jsonl"

    @property
    def ingestion_log_path(self) -> Path:
        return self.data_dir / "ingestion_log.jsonl"

    @property
    def candidates_path(self) -> Path:
        return self.data_dir / "candidates.jsonl"

    @property
    def transaction_log_path(self) -> Path:
        return self.data_dir / "transactions.jsonl"

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
                    payload = EventLogMeta.model_validate_json(line)
                except (ValueError, ValidationError):
                    return False
                return payload.meta == EVENT_LOG_META_VERSION
        return False

    def _load_all_events(self) -> list[FactEvent]:
        """Read every ``FactEvent`` from ``facts.jsonl``, skipping the meta line."""
        if not self.facts_path.exists():
            return []
        events: list[FactEvent] = []
        first_data_line = True
        with self.facts_path.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                if first_data_line:
                    first_data_line = False
                    try:
                        EventLogMeta.model_validate_json(line)
                        continue
                    except (ValueError, ValidationError):
                        # Not a meta line — fall through and try to parse as event.
                        pass
                try:
                    events.append(FactEvent.model_validate_json(line))
                except (ValueError, ValidationError) as exc:
                    logger.warning("Skipping corrupt event at line %d: %s", lineno, exc)
        return events

    def load_facts(self) -> list[Fact]:
        """Load all facts from JSONL.

        Returns the same per-``fact_id`` view regardless of on-disk format. For
        event-log files, events are replayed into materialized facts; forgotten
        or superseded facts surface as ``confidence == 0`` Facts to preserve the
        existing "confidence-zero means inactive" contract used by callers.

        Corrupt lines are skipped with a warning.
        """
        if not self.facts_path.exists():
            return []
        if self._is_event_log_format():
            events = self._load_all_events()
            materialized = materialize_events(events)
            out: list[Fact] = []
            for fact, is_active in materialized.values():
                if not is_active and fact.confidence > 0.0:
                    fact = fact.model_copy(update={"confidence": 0.0})
                out.append(fact)
            return out
        facts = []
        with self.facts_path.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    facts.append(Fact.model_validate_json(line))
                except (ValueError, ValidationError) as exc:
                    logger.warning("Skipping corrupt fact at line %d: %s", lineno, exc)
        return facts

    def load_candidates(
        self,
        status: CandidateStatus | None = None,
        project: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryCandidate]:
        """Load memory candidates filtered by status and project."""
        if not self.candidates_path.exists():
            return []

        candidates = []
        with self.candidates_path.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidate = MemoryCandidate.model_validate_json(line)
                except (ValueError, ValidationError) as exc:
                    logger.warning(
                        "Skipping corrupt candidate at line %d: %s", lineno, exc
                    )
                    continue
                if status and candidate.status != status:
                    continue
                if project and candidate.project and candidate.project != project:
                    continue
                candidates.append(candidate)

        candidates.sort(key=lambda c: c.updated_at, reverse=True)
        if limit:
            candidates = candidates[:limit]
        return candidates

    def load_active_facts(
        self,
        category: FactCategory | None = None,
        project: str | None = None,
        include_global: bool = True,
        min_confidence: float = 0.1,
        limit: int | None = None,
        include_stale: bool = False,
    ) -> list[Fact]:
        """Load facts filtered by confidence, category, project, and stale status.

        Stale facts are excluded by default. Pass ``include_stale=True`` for
        inspection workflows that need to see the full set.
        """
        now = datetime.now(timezone.utc)
        facts = []
        for fact in self.load_facts():
            if not _is_active_fact(
                fact,
                now,
                include_stale=include_stale,
                min_confidence=min_confidence,
            ):
                continue
            if category and fact.category != category:
                continue
            if project:
                if fact.project != project and not (
                    include_global and fact.project is None
                ):
                    continue
            facts.append(fact)

        # Sort by recency
        facts.sort(key=lambda f: f.updated_at, reverse=True)
        if limit:
            facts = facts[:limit]
        return facts

    def prefilter_facts(
        self,
        query: str,
        project: str | None = None,
        limit: int | None = None,
    ) -> list[tuple[int, Fact]]:
        """Prefilter facts deterministically before agentic retrieval.

        Returns list of (score, Fact) tuples sorted by score descending.
        Score is included so the retriever can use it for tier selection.
        """
        facts = self.load_active_facts(project=project)
        if not facts:
            return []

        query_unigrams, query_bigrams = self._tokenize_extended(query)
        if not query_unigrams:
            return [(0, f) for f in (facts[:limit] if limit else facts)]

        now = datetime.now(timezone.utc)
        recency_cutoff = now - timedelta(days=7)

        scored: list[tuple[int, Fact]] = []
        for fact in facts:
            evidence_score = 0
            content_unigrams, content_bigrams = self._get_cached_tokens(fact)

            # Unigram overlap (stemmed)
            overlap = len(query_unigrams & content_unigrams)
            evidence_score += overlap * 5

            # Bigram overlap — catches "coding style" vs "coding_style"
            bigram_overlap = len(query_bigrams & content_bigrams)
            evidence_score += bigram_overlap * 3

            # Tag overlap (stem query tokens, compare to tags)
            tag_set = {_stem(t) for t in fact.tags}
            tag_overlap = len(query_unigrams & tag_set)
            evidence_score += tag_overlap * 4

            if fact.project and fact.project in query.lower():
                evidence_score += 6
            if any(token in fact.category.value for token in query_unigrams):
                evidence_score += 3

            score = evidence_score
            if fact.supersedes:
                score += 1
            if fact.expires_at:
                score += 1

            # Recency boost
            if fact.updated_at >= recency_cutoff:
                score += 2

            # Confidence boost
            if fact.confidence > 0.8:
                score += 1

            if evidence_score > 0:
                scored.append((score, fact))

        if not scored:
            return [(0, f) for f in (facts[:limit] if limit else facts)]

        scored.sort(
            key=lambda item: (item[0], item[1].updated_at.timestamp()),
            reverse=True,
        )
        if limit and len(scored) < limit:
            seen_ids = {fact.id for _, fact in scored}
            scored.extend((0, fact) for fact in facts if fact.id not in seen_ids)
        return scored[:limit] if limit else scored

    @property
    def pre_eventlog_backup_path(self) -> Path:
        return self.data_dir / "facts.jsonl.pre-eventlog"

    def _migrate_to_event_log_if_needed(self) -> dict | None:
        """One-shot migrate legacy ``facts.jsonl`` to event-log format.

        Returns a summary dict when migration ran, ``None`` when not needed.
        Idempotent: a file already in event-log format is left untouched.
        """
        with _locked_store(self.data_dir):
            if not self.facts_path.exists():
                return None
            if self.facts_path.stat().st_size == 0:
                return None
            if self._is_event_log_format():
                return None

            facts: list[Fact] = []
            with self.facts_path.open() as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        facts.append(Fact.model_validate_json(line))
                    except (ValueError, ValidationError) as exc:
                        logger.warning(
                            "Skipping corrupt fact at line %d during migration: %s",
                            lineno,
                            exc,
                        )

            # Backup original file before any writes.
            import shutil

            shutil.copy2(self.facts_path, self.pre_eventlog_backup_path)

            # Build event sequence: meta sentinel, then created (+ optional
            # forgotten / superseded) events for each legacy fact.
            events: list[FactEvent] = []
            forgotten_count = 0
            superseded_count = 0
            now = datetime.now(timezone.utc)
            for fact in facts:
                events.append(
                    FactEvent(
                        event_type=EventType.created,
                        fact_id=fact.id,
                        timestamp=fact.created_at,
                        actor="migration",
                        payload=fact.model_dump(),
                    )
                )
                if fact.supersedes:
                    superseded_count += 1
                    events.append(
                        FactEvent(
                            event_type=EventType.superseded,
                            fact_id=fact.supersedes,
                            timestamp=fact.updated_at,
                            actor="migration",
                            payload={"superseded_by": fact.id},
                        )
                    )
                if fact.confidence < MIN_ACTIVE_CONFIDENCE:
                    forgotten_count += 1
                    events.append(
                        FactEvent(
                            event_type=EventType.forgotten,
                            fact_id=fact.id,
                            timestamp=fact.updated_at,
                            actor="migration",
                            payload={"reason": "migrated_from_legacy_low_confidence"},
                        )
                    )

            meta = EventLogMeta(migrated_at=now)
            tmp_path: Path | None = None
            with _locked_write(self.facts_path):
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        dir=self.facts_path.parent,
                        prefix=self.facts_path.stem,
                        suffix=".tmp",
                        delete=False,
                    ) as f:
                        tmp_path = Path(f.name)
                        f.write(meta.model_dump_json() + "\n")
                        for event in events:
                            f.write(event.model_dump_json() + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, self.facts_path)
                    tmp_path = None
                finally:
                    if tmp_path and tmp_path.exists():
                        tmp_path.unlink()

            summary = {
                "migrated_facts": len(facts),
                "forgotten_events": forgotten_count,
                "superseded_events": superseded_count,
                "backup_path": str(self.pre_eventlog_backup_path),
            }
            try:
                self.log_ingestion(
                    IngestionRecord(
                        source="event_log_migration",
                        facts_updated=[fact.id for fact in facts],
                        agent_model="migration",
                    )
                )
            except Exception:
                logger.warning("Failed to log migration to ingestion log")
            logger.info(
                "Migrated %d legacy facts to event-log format (%d forgotten, %d superseded)",
                len(facts),
                forgotten_count,
                superseded_count,
            )
            return summary

    def _ensure_event_log_header(self) -> None:
        """Write the meta sentinel as the first line if the file is brand-new."""
        if self.facts_path.exists() and self.facts_path.stat().st_size > 0:
            return
        with self.facts_path.open("w") as f:
            f.write(EventLogMeta().model_dump_json() + "\n")
            f.flush()
            os.fsync(f.fileno())

    def append_events(self, events: list[FactEvent]) -> None:
        """Append one or more typed events to the event log under the write lock.

        Migrates the file to event-log format on demand if a legacy-format file
        materialized between ``__init__`` and this call.
        """
        if not events:
            return
        with _locked_store(self.data_dir):
            if (
                self.facts_path.exists()
                and self.facts_path.stat().st_size > 0
                and not self._is_event_log_format()
            ):
                self._migrate_to_event_log_if_needed()
            with _locked_write(self.facts_path):
                self._ensure_event_log_header()
                self._ensure_trailing_newline(self.facts_path)
                with self.facts_path.open("a") as f:
                    for event in events:
                        f.write(event.model_dump_json() + "\n")
                    f.flush()
                    os.fsync(f.fileno())
        logger.info("Appended %d event(s) to event log", len(events))

    def append_facts(self, facts: list[Fact]) -> None:
        """Append new facts to the store by emitting ``created`` events."""
        if not facts:
            return
        events = [
            FactEvent(
                event_type=EventType.created,
                fact_id=fact.id,
                timestamp=fact.created_at,
                payload=fact.model_dump(),
            )
            for fact in facts
        ]
        self.append_events(events)
        # Evict any stale token-cache entries for re-created ids.
        for fact in facts:
            self._tok_cache.pop(fact.id, None)
        logger.info("Appended %d facts to store", len(facts))

    def append_candidates(self, candidates: list[MemoryCandidate]) -> None:
        """Append memory candidates to the queue."""
        with _locked_store(self.data_dir):
            with _locked_write(self.candidates_path):
                self._ensure_trailing_newline(self.candidates_path)
                with self.candidates_path.open("a") as f:
                    for candidate in candidates:
                        f.write(candidate.model_dump_json() + "\n")
        logger.info("Appended %d memory candidate(s) to queue", len(candidates))

    def update_fact(self, fact_id: str, **updates) -> Fact | None:
        """Update a fact by appending an ``edited`` event to the log.

        Returns ``None`` when the fact is not found or is currently forgotten,
        unless the update explicitly sets ``confidence`` (forget/restore paths).
        Special-cases ``confidence == 0`` to emit a ``forgotten`` event instead
        of an edit, preserving the "confidence-zero means inactive" contract.
        """
        with _locked_store(self.data_dir):
            facts_by_id = {fact.id: fact for fact in self.load_facts()}
            existing = facts_by_id.get(fact_id)
            if existing is None:
                return None
            if existing.confidence == 0.0 and "confidence" not in updates:
                return None

            now = datetime.now(timezone.utc)

            # Pure forget path — emit a forgotten event, not an edit.
            if updates.get("confidence") == 0.0 and len(updates) == 1:
                self.append_events(
                    [
                        FactEvent(
                            event_type=EventType.forgotten,
                            fact_id=fact_id,
                            timestamp=now,
                            payload={},
                        )
                    ]
                )
                return existing.model_copy(
                    update={"confidence": 0.0, "updated_at": now}
                )

            self.append_events(
                [
                    FactEvent(
                        event_type=EventType.edited,
                        fact_id=fact_id,
                        timestamp=now,
                        payload=updates,
                    )
                ]
            )
            self._tok_cache.pop(fact_id, None)
            merged = {**existing.model_dump(), **updates, "updated_at": now}
            return Fact.model_validate(merged)

    def update_candidate(self, candidate_id: str, **updates) -> MemoryCandidate | None:
        """Update a candidate in-place by rewriting the candidate JSONL file."""
        with _locked_store(self.data_dir):
            candidates = self.load_candidates()
            updated = None
            for i, candidate in enumerate(candidates):
                if candidate.id == candidate_id:
                    for key, value in updates.items():
                        setattr(candidate, key, value)
                    candidate.updated_at = datetime.now(timezone.utc)
                    candidates[i] = candidate
                    updated = candidate
                    break

            if updated:
                self._rewrite(candidates, path=self.candidates_path)
            return updated

    def rename_project(self, old_project: str, new_project: str) -> int:
        """Bulk-rename the project field on facts (events) and candidates (rewrite)."""
        with _locked_store(self.data_dir):
            facts = self.load_facts()
            now = datetime.now(timezone.utc)
            events: list[FactEvent] = []
            for fact in facts:
                if fact.project == old_project and fact.confidence > 0.0:
                    events.append(
                        FactEvent(
                            event_type=EventType.edited,
                            fact_id=fact.id,
                            timestamp=now,
                            payload={"project": new_project},
                        )
                    )
            fact_count = len(events)
            if events:
                self.append_events(events)
                for event in events:
                    self._tok_cache.pop(event.fact_id, None)
                logger.info(
                    "Renamed project %s → %s for %d fact(s)",
                    old_project,
                    new_project,
                    fact_count,
                )

            # Candidates still use the legacy rewrite model — they are not on
            # the event-log path.
            candidates = self.load_candidates()
            cand_count = 0
            for i, candidate in enumerate(candidates):
                if candidate.project == old_project:
                    candidate.project = new_project
                    candidate.updated_at = now
                    candidates[i] = candidate
                    cand_count += 1
            if cand_count > 0:
                self._rewrite(candidates, path=self.candidates_path)
                logger.info(
                    "Renamed project %s → %s for %d candidate(s)",
                    old_project,
                    new_project,
                    cand_count,
                )

            return fact_count + cand_count

    def batch_update_facts(self, updates: dict[str, dict]) -> list[Fact]:
        """Apply multiple fact updates by appending a batch of events.

        Each ``updates[fact_id]`` dict may contain ``confidence`` to mark a
        fact forgotten (yields a ``forgotten`` event) or other editable fields
        (yields an ``edited`` event). Mixed updates that include a non-zero
        confidence plus other fields become a single ``edited`` event.

        Returns the list of (re-materialized) updated facts.
        """
        if not updates:
            return []

        with _locked_store(self.data_dir):
            facts_by_id = {fact.id: fact for fact in self.load_facts()}
            now = datetime.now(timezone.utc)
            events: list[FactEvent] = []
            updated_facts: list[Fact] = []

            for fact_id, field_updates in updates.items():
                existing = facts_by_id.get(fact_id)
                if existing is None:
                    continue

                if field_updates.get("confidence") == 0.0 and len(field_updates) == 1:
                    events.append(
                        FactEvent(
                            event_type=EventType.forgotten,
                            fact_id=fact_id,
                            timestamp=now,
                            payload={},
                        )
                    )
                    updated_facts.append(
                        existing.model_copy(
                            update={"confidence": 0.0, "updated_at": now}
                        )
                    )
                    continue

                events.append(
                    FactEvent(
                        event_type=EventType.edited,
                        fact_id=fact_id,
                        timestamp=now,
                        payload=field_updates,
                    )
                )
                merged = {
                    **existing.model_dump(),
                    **field_updates,
                    "updated_at": now,
                }
                updated_facts.append(Fact.model_validate(merged))

            if events:
                self.append_events(events)
                for event in events:
                    self._tok_cache.pop(event.fact_id, None)
                logger.info(
                    "Batch-updated %d facts via %d event(s)",
                    len(updated_facts),
                    len(events),
                )
            return updated_facts

    def batch_update_candidates(
        self, updates: dict[str, dict]
    ) -> list[MemoryCandidate]:
        """Apply multiple candidate updates in a single JSONL rewrite.

        Args:
            updates: Mapping of candidate_id -> field updates to apply.

        Returns:
            List of updated MemoryCandidate objects.
        """
        if not updates:
            return []

        with _locked_store(self.data_dir):
            candidates = self.load_candidates()
            now = datetime.now(timezone.utc)
            updated: list[MemoryCandidate] = []

            for i, candidate in enumerate(candidates):
                if candidate.id in updates:
                    for key, value in updates[candidate.id].items():
                        if key == "status":
                            value = CandidateStatus(value)
                        setattr(candidate, key, value)
                    candidate.updated_at = now
                    candidates[i] = candidate
                    updated.append(candidate)

            if updated:
                self._rewrite(candidates, path=self.candidates_path)
                logger.info(
                    "Batch-updated %d candidates in single rewrite", len(updated)
                )

            return updated

    def forget(self, fact_id: str, reason: str = "") -> Fact | None:
        """Soft-delete a fact by appending a ``forgotten`` event."""
        with _locked_store(self.data_dir):
            existing = next((f for f in self.load_facts() if f.id == fact_id), None)
            if existing is None or existing.confidence == 0.0:
                return None
            now = datetime.now(timezone.utc)
            self.append_events(
                [
                    FactEvent(
                        event_type=EventType.forgotten,
                        fact_id=fact_id,
                        timestamp=now,
                        payload={"reason": reason} if reason else {},
                    )
                ]
            )
            self._tok_cache.pop(fact_id, None)
            if reason:
                logger.info("Forgot fact %s: %s", fact_id, reason)
            return existing.model_copy(update={"confidence": 0.0, "updated_at": now})

    def restore_fact(self, fact_id: str) -> Fact | None:
        """Reverse a prior ``forget`` by appending a ``restored`` event.

        Returns the materialized fact on success, ``None`` when the fact is
        unknown or is not currently in the forgotten state.
        """
        with _locked_store(self.data_dir):
            events_by_id: dict[str, list[FactEvent]] = {}
            for event in self._load_all_events():
                events_by_id.setdefault(event.fact_id, []).append(event)
            fact_events = events_by_id.get(fact_id)
            if not fact_events:
                return None
            materialized = materialize_events(fact_events)
            entry = materialized.get(fact_id)
            if entry is None:
                return None
            fact, is_active = entry
            if is_active:
                # Nothing to restore.
                return None

            now = datetime.now(timezone.utc)
            self.append_events(
                [
                    FactEvent(
                        event_type=EventType.restored,
                        fact_id=fact_id,
                        timestamp=now,
                        payload={},
                    )
                ]
            )
            self._tok_cache.pop(fact_id, None)
            logger.info("Restored fact %s", fact_id)
            return fact.model_copy(update={"updated_at": now})

    def correct_fact(
        self,
        fact_id: str,
        new_content: str,
        *,
        category: FactCategory | None = None,
        tags: list[str] | None = None,
        project: str | None = None,
        reason: str = "",
    ) -> Fact | None:
        """Replace ``fact_id`` with a new active fact that supersedes it.

        The original fact is preserved (its confidence is reduced so it falls
        out of active recall) and a new fact with ``supersedes=fact_id`` is
        created. The new fact inherits the original's category/project/tags
        unless overrides are supplied.

        Returns the new fact, or ``None`` if the original fact was not found
        or was already forgotten.
        """
        with _locked_store(self.data_dir):
            facts_by_id = {fact.id: fact for fact in self.load_facts()}
            existing = facts_by_id.get(fact_id)
            if existing is None or existing.confidence == 0.0:
                return None

            now = datetime.now(timezone.utc)
            new_fact = Fact(
                id=uuid4().hex[:12],
                category=category or existing.category,
                content=new_content,
                source=existing.source,
                confidence=existing.confidence,
                created_at=now,
                updated_at=now,
                observed_at=now,
                project=project if project is not None else existing.project,
                tags=tags if tags is not None else list(existing.tags),
                supersedes=fact_id,
                evidence_kind=existing.evidence_kind,
                why_store=reason or existing.why_store,
            )
            self.append_events(
                [
                    FactEvent(
                        event_type=EventType.superseded,
                        fact_id=fact_id,
                        timestamp=now,
                        payload={"superseded_by": new_fact.id, "reason": reason},
                    ),
                    FactEvent(
                        event_type=EventType.created,
                        fact_id=new_fact.id,
                        timestamp=now,
                        payload=new_fact.model_dump(),
                    ),
                ]
            )
            self._tok_cache.pop(fact_id, None)

        self.log_ingestion(
            IngestionRecord(
                source="correct_fact",
                facts_created=[new_fact.id],
                facts_updated=[fact_id],
                agent_model="manual_correction",
            )
        )
        logger.info("Corrected fact %s -> %s: %s", fact_id, new_fact.id, reason)
        return new_fact

    def merge_facts(
        self,
        source_ids: list[str],
        merged_content: str,
        *,
        category: FactCategory | None = None,
        tags: list[str] | None = None,
        project: str | None = None,
        reason: str = "",
    ) -> tuple[Fact, list[str]] | None:
        """Consolidate multiple active facts into one new fact.

        Creates a single new active fact that supersedes the first source and
        marks every other source fact superseded as well (by reducing their
        confidence). Returns ``(new_fact, superseded_ids)`` on success.

        Returns ``None`` if fewer than two valid active source facts are
        provided or the source set is invalid.
        """
        if len(source_ids) < 2:
            return None

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique_ids = [sid for sid in source_ids if not (sid in seen or seen.add(sid))]
        if len(unique_ids) < 2:
            return None

        with _locked_store(self.data_dir):
            facts_by_id = {fact.id: fact for fact in self.load_facts()}
            sources: list[Fact] = []
            for sid in unique_ids:
                fact = facts_by_id.get(sid)
                if fact is None or fact.confidence == 0.0:
                    return None
                sources.append(fact)

            scopes = {fact.project for fact in sources}
            if len(scopes) > 1:
                return None

            primary = sources[0]
            now = datetime.now(timezone.utc)
            new_fact = Fact(
                id=uuid4().hex[:12],
                category=category or primary.category,
                content=merged_content,
                source=primary.source,
                confidence=max(f.confidence for f in sources),
                created_at=now,
                updated_at=now,
                observed_at=now,
                project=project if project is not None else primary.project,
                tags=tags if tags is not None else list(primary.tags),
                supersedes=primary.id,
                evidence_kind=primary.evidence_kind,
                why_store=reason or "merged",
            )
            events: list[FactEvent] = [
                FactEvent(
                    event_type=EventType.superseded,
                    fact_id=src.id,
                    timestamp=now,
                    payload={"superseded_by": new_fact.id, "reason": reason},
                )
                for src in sources
            ]
            events.append(
                FactEvent(
                    event_type=EventType.created,
                    fact_id=new_fact.id,
                    timestamp=now,
                    payload=new_fact.model_dump(),
                )
            )
            self.append_events(events)
            for src in sources:
                self._tok_cache.pop(src.id, None)

        self.log_ingestion(
            IngestionRecord(
                source="merge_facts",
                facts_created=[new_fact.id],
                facts_updated=list(unique_ids),
                agent_model="manual_merge",
            )
        )
        logger.info("Merged %d facts -> %s: %s", len(unique_ids), new_fact.id, reason)
        return new_fact, unique_ids

    def mark_stale(self, fact_id: str, reason: str = "") -> Fact | None:
        """Mark a fact stale by appending a ``stale`` event."""
        with _locked_store(self.data_dir):
            existing = next((f for f in self.load_facts() if f.id == fact_id), None)
            if existing is None or existing.confidence == 0.0:
                return None
            now = datetime.now(timezone.utc)
            self.append_events(
                [
                    FactEvent(
                        event_type=EventType.stale,
                        fact_id=fact_id,
                        timestamp=now,
                        payload={"reason": reason} if reason else {},
                    )
                ]
            )
            self._tok_cache.pop(fact_id, None)
            self.log_ingestion(
                IngestionRecord(
                    source="mark_stale",
                    facts_updated=[fact_id],
                    agent_model="manual_stale",
                )
            )
            logger.info("Marked fact %s stale: %s", fact_id, reason)
            return existing.model_copy(
                update={"stale": True, "stale_reason": reason, "updated_at": now}
            )

    def unmark_stale(self, fact_id: str) -> Fact | None:
        """Reverse a stale marking by appending an ``unstale`` event."""
        with _locked_store(self.data_dir):
            existing = next((f for f in self.load_facts() if f.id == fact_id), None)
            if existing is None or existing.confidence == 0.0:
                return None
            now = datetime.now(timezone.utc)
            self.append_events(
                [
                    FactEvent(
                        event_type=EventType.unstale,
                        fact_id=fact_id,
                        timestamp=now,
                        payload={},
                    )
                ]
            )
            self._tok_cache.pop(fact_id, None)
            return existing.model_copy(
                update={"stale": False, "stale_reason": "", "updated_at": now}
            )

    def approve_candidates(self, candidate_ids: list[str]) -> list[Fact]:
        """Approve candidates and promote them into active facts with recovery."""
        with _locked_store(self.data_dir):
            self._recover_transactions_locked()
            transaction = self._prepare_approval_transaction(candidate_ids)
            if not transaction:
                return []

            self._append_transaction(transaction)
            self._apply_approval_transaction(transaction)
            self._commit_transaction(transaction)
            return transaction.new_facts

    def reject_candidates(
        self, candidate_ids: list[str], reason: str = ""
    ) -> list[MemoryCandidate]:
        """Reject candidates without promoting them into active facts."""
        with _locked_store(self.data_dir):
            pending_candidates = {
                c.id: c for c in self.load_candidates(status=CandidateStatus.pending)
            }
            updates: dict[str, dict] = {}
            for candidate_id in candidate_ids:
                if candidate_id in pending_candidates:
                    updates[candidate_id] = {
                        "status": CandidateStatus.rejected,
                        "review_note": reason,
                    }
            return self.batch_update_candidates(updates)

    def recover_transactions(self) -> int:
        """Roll forward any prepared store transactions that lack a commit marker."""
        with _locked_store(self.data_dir):
            return self._recover_transactions_locked()

    def _recover_transactions_locked(self) -> int:
        recovered = 0
        for transaction in self._pending_transactions():
            logger.warning(
                "Recovering prepared %s transaction %s",
                transaction.type,
                transaction.id,
            )
            self._apply_transaction(transaction)
            self._commit_transaction(transaction)
            recovered += 1
        return recovered

    def _prepare_approval_transaction(
        self, candidate_ids: list[str]
    ) -> StoreTransaction | None:
        pending_candidates = {
            c.id: c for c in self.load_candidates(status=CandidateStatus.pending)
        }

        fact_updates: dict[str, dict] = {}
        candidate_updates: dict[str, dict] = {}
        new_facts: list[Fact] = []
        now = datetime.now(timezone.utc)
        seen_ids: set[str] = set()
        selected_ids: list[str] = []

        for candidate_id in candidate_ids:
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)

            candidate = pending_candidates.get(candidate_id)
            if not candidate:
                continue
            selected_ids.append(candidate_id)

            candidate_updates[candidate_id] = {"status": CandidateStatus.approved}

            if candidate.supersedes:
                fact_updates[candidate.supersedes] = {"confidence": 0.0}

            data = candidate.model_dump(exclude={"status", "review_note"})
            data.update(id=uuid4().hex[:12], created_at=now, updated_at=now)
            new_facts.append(Fact(**data))

        if not new_facts:
            return None

        return StoreTransaction(
            type="approve_candidates",
            status=TransactionStatus.prepared,
            candidate_ids=selected_ids,
            fact_updates=fact_updates,
            candidate_updates=candidate_updates,
            new_facts=new_facts,
            created_at=now,
        )

    def _apply_transaction(self, transaction: StoreTransaction) -> None:
        if transaction.type != "approve_candidates":
            logger.warning(
                "Skipping unsupported transaction %s of type %s",
                transaction.id,
                transaction.type,
            )
            return
        self._apply_approval_transaction(transaction)

    def _apply_approval_transaction(self, transaction: StoreTransaction) -> None:
        missing_facts = self._missing_facts(transaction.new_facts)
        if missing_facts:
            self.append_facts(missing_facts)
        if transaction.fact_updates:
            self.batch_update_facts(transaction.fact_updates)
        if transaction.candidate_updates:
            self.batch_update_candidates(transaction.candidate_updates)

    def _missing_facts(self, facts: list[Fact]) -> list[Fact]:
        existing_ids = {fact.id for fact in self.load_facts()}
        return [fact for fact in facts if fact.id not in existing_ids]

    def _commit_transaction(self, transaction: StoreTransaction) -> None:
        committed = transaction.model_copy(
            update={
                "status": TransactionStatus.committed,
                "committed_at": datetime.now(timezone.utc),
            }
        )
        self._append_transaction(committed)

    def _append_transaction(self, transaction: StoreTransaction) -> None:
        with _locked_write(self.transaction_log_path):
            self._ensure_trailing_newline(self.transaction_log_path)
            with self.transaction_log_path.open("a") as f:
                f.write(transaction.model_dump_json() + "\n")
                f.flush()
                os.fsync(f.fileno())

    def _load_transactions(self) -> list[StoreTransaction]:
        if not self.transaction_log_path.exists():
            return []

        transactions: list[StoreTransaction] = []
        with self.transaction_log_path.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    transactions.append(StoreTransaction.model_validate_json(line))
                except (ValueError, ValidationError) as exc:
                    logger.warning(
                        "Skipping corrupt transaction at line %d: %s",
                        lineno,
                        exc,
                    )
        return transactions

    def _pending_transactions(self) -> list[StoreTransaction]:
        prepared: dict[str, StoreTransaction] = {}
        committed_ids: set[str] = set()
        for transaction in self._load_transactions():
            if transaction.status == TransactionStatus.prepared:
                prepared[transaction.id] = transaction
            elif transaction.status == TransactionStatus.committed:
                committed_ids.add(transaction.id)
        return [
            transaction
            for transaction_id, transaction in prepared.items()
            if transaction_id not in committed_ids
        ]

    def log_ingestion(self, record: IngestionRecord) -> None:
        """Append an ingestion record to the audit log."""
        with _locked_write(self.ingestion_log_path):
            with self.ingestion_log_path.open("a") as f:
                f.write(record.model_dump_json() + "\n")

    def compact_event_log(self, *, keep_tombstones: bool = True) -> dict[str, int]:
        """Collapse the event log to a minimal representation.

        For every active ``fact_id``, emit exactly one ``created`` event that
        reflects the current materialized state. For forgotten/superseded
        facts: keep a ``created`` + ``forgotten`` (or ``superseded``) pair when
        ``keep_tombstones`` is true (audit-preserving — default), or drop them
        entirely when false.

        Compaction is the only mutation path that still rewrites the file. It
        holds the inter-process write lock for the full duration and writes
        via the existing atomic-replace path. A
        ``.engram-compaction-in-progress`` sentinel file is created for the
        duration so ``engram sync`` skips itself rather than racing the
        rewrite.

        Returns a small summary: events before, events after, fact counts.
        """
        from engram.sync import COMPACTION_SENTINEL_FILENAME

        sentinel = self.data_dir / COMPACTION_SENTINEL_FILENAME
        with _locked_store(self.data_dir):
            if not self._is_event_log_format():
                return {
                    "events_before": 0,
                    "events_after": 0,
                    "active_facts": 0,
                    "tombstones": 0,
                    "skipped": True,
                }

            events = self._load_all_events()
            events_before = len(events)
            materialized = materialize_events(events)

            now = datetime.now(timezone.utc)
            compacted: list[FactEvent] = []
            tombstones = 0
            for fact_id, (fact, is_active) in materialized.items():
                if not is_active:
                    if not keep_tombstones:
                        continue
                    compacted.append(
                        FactEvent(
                            event_type=EventType.created,
                            fact_id=fact_id,
                            timestamp=fact.created_at,
                            actor="compaction",
                            payload=fact.model_dump(),
                        )
                    )
                    compacted.append(
                        FactEvent(
                            event_type=EventType.forgotten,
                            fact_id=fact_id,
                            timestamp=fact.updated_at,
                            actor="compaction",
                            payload={"reason": "compacted_tombstone"},
                        )
                    )
                    tombstones += 1
                else:
                    compacted.append(
                        FactEvent(
                            event_type=EventType.created,
                            fact_id=fact_id,
                            timestamp=fact.created_at,
                            actor="compaction",
                            payload=fact.model_dump(),
                        )
                    )

            sentinel.write_text(f"compaction started at {now.isoformat()}\n")
            try:
                meta = EventLogMeta(migrated_at=now)
                tmp_path: Path | None = None
                with _locked_write(self.facts_path):
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="w",
                            dir=self.facts_path.parent,
                            prefix=self.facts_path.stem,
                            suffix=".tmp",
                            delete=False,
                        ) as fh:
                            tmp_path = Path(fh.name)
                            fh.write(meta.model_dump_json() + "\n")
                            for event in compacted:
                                fh.write(event.model_dump_json() + "\n")
                            fh.flush()
                            os.fsync(fh.fileno())
                        os.replace(tmp_path, self.facts_path)
                        tmp_path = None
                    finally:
                        if tmp_path and tmp_path.exists():
                            tmp_path.unlink()
            finally:
                sentinel.unlink(missing_ok=True)

            active_count = sum(1 for _, active in materialized.values() if active)
            logger.info(
                "Compacted event log: %d → %d events (%d active facts, %d tombstones)",
                events_before,
                len(compacted),
                active_count,
                tombstones,
            )
            return {
                "events_before": events_before,
                "events_after": len(compacted),
                "active_facts": active_count,
                "tombstones": tombstones,
                "skipped": False,
            }

    def purge(self) -> dict:
        """Remove forgotten and expired facts from the JSONL file.

        Returns counts of purged and retained facts.
        """
        with _locked_store(self.data_dir):
            facts = self.load_facts()
            now = datetime.now(timezone.utc)
            kept = []
            purged = 0
            for fact in facts:
                is_forgotten = fact.confidence < MIN_ACTIVE_CONFIDENCE
                is_expired = _is_expired_fact(fact, now)
                if is_forgotten or is_expired:
                    purged += 1
                else:
                    kept.append(fact)

            if purged > 0:
                self._rewrite(kept)
                logger.info("Purged %d facts (%d retained)", purged, len(kept))

            return {"purged": purged, "retained": len(kept)}

    def stats(self) -> dict:
        """Get store statistics."""
        facts = self.load_facts()
        now = datetime.now(timezone.utc)
        active = [f for f in facts if _is_active_fact(f, now)]
        forgotten = [f for f in facts if f.confidence < MIN_ACTIVE_CONFIDENCE]
        expired = [
            f
            for f in facts
            if f.confidence >= MIN_ACTIVE_CONFIDENCE and _is_expired_fact(f, now)
        ]
        by_category = Counter(f.category.value for f in active)
        by_project = Counter(f.project or "(none)" for f in active)

        return {
            "total_facts": len(facts),
            "active_facts": len(active),
            "forgotten_facts": len(forgotten),
            "expired_facts": len(expired),
            "by_category": dict(by_category.most_common()),
            "by_project": dict(by_project.most_common(10)),
            "storage_bytes": self.facts_path.stat().st_size
            if self.facts_path.exists()
            else 0,
            "pending_candidates": len(
                self.load_candidates(status=CandidateStatus.pending)
            ),
        }

    def format_candidates_for_review(self, candidates: list[MemoryCandidate]) -> str:
        """Format candidates as a numbered list for review."""
        if not candidates:
            return "(no candidates queued)"

        lines = []
        for i, candidate in enumerate(candidates, 1):
            meta = f"[{candidate.status.value}] [{candidate.category.value}]"
            if candidate.project:
                meta += f" [{candidate.project}]"
            if candidate.supersedes:
                meta += f" [supersedes: {candidate.supersedes}]"
            lines.append(
                f"{i}. {meta} {candidate.content} "
                f"(id: {candidate.id}, why: {candidate.why_store or 'useful future context'})"
            )
        return "\n".join(lines)

    @property
    def recall_log_path(self) -> Path:
        return self.data_dir / "recall_log.jsonl"

    def log_recall(self, record: RecallRecord) -> None:
        """Append a recall record to the recall log."""
        with _locked_write(self.recall_log_path):
            with self.recall_log_path.open("a") as f:
                f.write(record.model_dump_json() + "\n")

    def load_recall_log(self, limit: int | None = 500) -> list[RecallRecord]:
        """Load recent recall log entries."""
        if not self.recall_log_path.exists():
            return []
        records = []
        with self.recall_log_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(RecallRecord.model_validate_json(line))
                except Exception:
                    continue
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:limit] if limit is not None else records

    def repair(self) -> dict:
        """Recover from truncated/corrupt JSONL lines.

        For ``facts.jsonl`` in event-log format, validates each line as either
        the meta sentinel or a ``FactEvent``; corrupt lines are dropped via an
        atomic rewrite that preserves the sentinel. For legacy ``facts.jsonl``
        and for ``candidates.jsonl``, validates each line against the legacy
        record class. Returns counts of valid and corrupt lines found.
        """
        result: dict[str, int] = {}

        # facts.jsonl — format-aware
        if self.facts_path.exists():
            if self._is_event_log_format():
                meta: EventLogMeta | None = None
                events: list[FactEvent] = []
                corrupt = 0
                first_data_line = True
                with self.facts_path.open() as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        if first_data_line:
                            first_data_line = False
                            try:
                                meta = EventLogMeta.model_validate_json(line)
                                continue
                            except (ValueError, ValidationError):
                                pass
                        try:
                            events.append(FactEvent.model_validate_json(line))
                        except Exception:
                            corrupt += 1
                            logger.warning("Corrupt facts line dropped: %s", line[:80])
                if corrupt > 0:
                    meta = meta or EventLogMeta()
                    tmp_path: Path | None = None
                    with _locked_write(self.facts_path):
                        try:
                            with tempfile.NamedTemporaryFile(
                                mode="w",
                                dir=self.facts_path.parent,
                                prefix=self.facts_path.stem,
                                suffix=".tmp",
                                delete=False,
                            ) as f:
                                tmp_path = Path(f.name)
                                f.write(meta.model_dump_json() + "\n")
                                for event in events:
                                    f.write(event.model_dump_json() + "\n")
                                f.flush()
                                os.fsync(f.fileno())
                            os.replace(tmp_path, self.facts_path)
                            tmp_path = None
                        finally:
                            if tmp_path and tmp_path.exists():
                                tmp_path.unlink()
                # Treat unique active fact_ids as the "valid" count to preserve
                # the legacy semantic the caller checks.
                materialized = materialize_events(events)
                result["facts_valid"] = len(materialized)
                result["facts_corrupt"] = corrupt
            else:
                valid_facts: list[Fact] = []
                corrupt = 0
                with self.facts_path.open() as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            valid_facts.append(Fact.model_validate_json(line))
                        except Exception:
                            corrupt += 1
                            logger.warning("Corrupt facts line dropped: %s", line[:80])
                if corrupt > 0:
                    self._rewrite(valid_facts)
                result["facts_valid"] = len(valid_facts)
                result["facts_corrupt"] = corrupt
        else:
            result["facts_valid"] = 0
            result["facts_corrupt"] = 0

        # candidates.jsonl — unchanged legacy semantics
        if self.candidates_path.exists():
            valid_candidates: list[MemoryCandidate] = []
            corrupt = 0
            with self.candidates_path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        valid_candidates.append(
                            MemoryCandidate.model_validate_json(line)
                        )
                    except Exception:
                        corrupt += 1
                        logger.warning("Corrupt candidates line dropped: %s", line[:80])
            if corrupt > 0:
                self._rewrite(valid_candidates, path=self.candidates_path)
            result["candidates_valid"] = len(valid_candidates)
            result["candidates_corrupt"] = corrupt
        else:
            result["candidates_valid"] = 0
            result["candidates_corrupt"] = 0

        return result

    def _ensure_trailing_newline(self, path: Path) -> None:
        """Ensure the file ends with a newline to prevent JSON concatenation."""
        if not path.exists() or path.stat().st_size == 0:
            return
        with path.open("rb") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                with path.open("ab") as af:
                    af.write(b"\n")

    def _rewrite(
        self, records: list[Fact | MemoryCandidate], path: Path | None = None
    ) -> None:
        """Rewrite the entire JSONL file atomically with fsync and unique tmp path."""
        target_path = path or self.facts_path
        parent = target_path.parent
        name = target_path.stem
        tmp_path: Path | None = None
        with _locked_write(target_path):
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    dir=parent,
                    prefix=name,
                    suffix=".tmp",
                    delete=False,
                ) as f:
                    tmp_path = Path(f.name)
                    for record in records:
                        f.write(record.model_dump_json() + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, target_path)
                tmp_path = None  # Successfully replaced; don't clean up
            finally:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()

        # Evict cache entries for any facts removed by this rewrite
        if path is None or path == self.facts_path:
            kept_ids = {r.id for r in records if isinstance(r, Fact)}
            for stale_id in [k for k in self._tok_cache if k not in kept_ids]:
                del self._tok_cache[stale_id]

    def _get_cached_tokens(self, fact: Fact) -> tuple[set[str], set[str]]:
        """Return cached stemmed unigram/bigram sets, recomputing if stale."""
        updated_iso = fact.updated_at.isoformat()
        cached = self._tok_cache.get(fact.id)
        if cached and cached[0] == updated_iso:
            return cached[1], cached[2]
        unigrams, bigrams = self._tokenize_extended(fact.content)
        self._tok_cache[fact.id] = (updated_iso, unigrams, bigrams)
        return unigrams, bigrams

    def _tokenize_extended(self, text: str) -> tuple[set[str], set[str]]:
        """Tokenize text into stemmed unigrams and bigrams.

        Normalizes underscore/hyphen so "fact_store" and "fact-store" both
        produce the same tokens as "fact store".
        """
        normalized = text.lower().replace("_", " ").replace("-", " ")
        raw = _TOKEN_RE.findall(normalized)
        unigrams = {_stem(t) for t in raw}
        bigrams = {f"{raw[i]}_{raw[i + 1]}" for i in range(len(raw) - 1)}
        return unigrams, bigrams


class AsyncFactStore:
    """Async facade for running blocking JSONL store operations off the event loop."""

    def __init__(self, store: FactStore | None = None):
        self.sync_store = store or FactStore()

    @property
    def data_dir(self) -> Path:
        return self.sync_store.data_dir

    @property
    def facts_path(self) -> Path:
        return self.sync_store.facts_path

    @property
    def ingestion_log_path(self) -> Path:
        return self.sync_store.ingestion_log_path

    @property
    def candidates_path(self) -> Path:
        return self.sync_store.candidates_path

    @property
    def transaction_log_path(self) -> Path:
        return self.sync_store.transaction_log_path

    @property
    def recall_log_path(self) -> Path:
        return self.sync_store.recall_log_path

    async def _run(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        return await asyncio.to_thread(func, *args, **kwargs)

    async def load_facts(self) -> list[Fact]:
        return await self._run(self.sync_store.load_facts)

    async def load_candidates(
        self,
        status: CandidateStatus | None = None,
        project: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryCandidate]:
        return await self._run(self.sync_store.load_candidates, status, project, limit)

    async def load_active_facts(
        self,
        category: FactCategory | None = None,
        project: str | None = None,
        include_global: bool = True,
        min_confidence: float = 0.1,
        limit: int | None = None,
        include_stale: bool = False,
    ) -> list[Fact]:
        return await self._run(
            self.sync_store.load_active_facts,
            category,
            project,
            include_global,
            min_confidence,
            limit,
            include_stale,
        )

    async def prefilter_facts(
        self,
        query: str,
        project: str | None = None,
        limit: int | None = None,
    ) -> list[tuple[int, Fact]]:
        return await self._run(self.sync_store.prefilter_facts, query, project, limit)

    async def append_facts(self, facts: list[Fact]) -> None:
        await self._run(self.sync_store.append_facts, facts)

    async def append_candidates(self, candidates: list[MemoryCandidate]) -> None:
        await self._run(self.sync_store.append_candidates, candidates)

    async def update_fact(self, fact_id: str, **updates) -> Fact | None:
        return await self._run(self.sync_store.update_fact, fact_id, **updates)

    async def update_candidate(
        self, candidate_id: str, **updates
    ) -> MemoryCandidate | None:
        return await self._run(
            self.sync_store.update_candidate, candidate_id, **updates
        )

    async def rename_project(self, old_project: str, new_project: str) -> int:
        return await self._run(self.sync_store.rename_project, old_project, new_project)

    async def batch_update_facts(self, updates: dict[str, dict]) -> list[Fact]:
        return await self._run(self.sync_store.batch_update_facts, updates)

    async def batch_update_candidates(
        self, updates: dict[str, dict]
    ) -> list[MemoryCandidate]:
        return await self._run(self.sync_store.batch_update_candidates, updates)

    async def forget(self, fact_id: str, reason: str = "") -> Fact | None:
        return await self._run(self.sync_store.forget, fact_id, reason)

    async def correct_fact(
        self,
        fact_id: str,
        new_content: str,
        **kwargs,
    ) -> Fact | None:
        return await self._run(
            self.sync_store.correct_fact, fact_id, new_content, **kwargs
        )

    async def merge_facts(
        self,
        source_ids: list[str],
        merged_content: str,
        **kwargs,
    ) -> tuple[Fact, list[str]] | None:
        return await self._run(
            self.sync_store.merge_facts, source_ids, merged_content, **kwargs
        )

    async def mark_stale(self, fact_id: str, reason: str = "") -> Fact | None:
        return await self._run(self.sync_store.mark_stale, fact_id, reason)

    async def unmark_stale(self, fact_id: str) -> Fact | None:
        return await self._run(self.sync_store.unmark_stale, fact_id)

    async def approve_candidates(self, candidate_ids: list[str]) -> list[Fact]:
        return await self._run(self.sync_store.approve_candidates, candidate_ids)

    async def reject_candidates(
        self, candidate_ids: list[str], reason: str = ""
    ) -> list[MemoryCandidate]:
        return await self._run(self.sync_store.reject_candidates, candidate_ids, reason)

    async def recover_transactions(self) -> int:
        return await self._run(self.sync_store.recover_transactions)

    async def log_ingestion(self, record: IngestionRecord) -> None:
        await self._run(self.sync_store.log_ingestion, record)

    async def purge(self) -> dict:
        return await self._run(self.sync_store.purge)

    async def stats(self) -> dict:
        return await self._run(self.sync_store.stats)

    def format_candidates_for_review(self, candidates: list[MemoryCandidate]) -> str:
        return self.sync_store.format_candidates_for_review(candidates)

    async def log_recall(self, record: RecallRecord) -> None:
        await self._run(self.sync_store.log_recall, record)

    async def load_recall_log(self, limit: int | None = 500) -> list[RecallRecord]:
        return await self._run(self.sync_store.load_recall_log, limit)

    async def repair(self) -> dict:
        return await self._run(self.sync_store.repair)

    async def compact_event_log(
        self, *, keep_tombstones: bool = True
    ) -> dict[str, int]:
        return await self._run(
            self.sync_store.compact_event_log, keep_tombstones=keep_tombstones
        )
