"""JSONL-based knowledge store for facts."""

import logging
import os
import tempfile
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any, Concatenate
from uuid import uuid4

from engram.core.config import get_settings
from engram.core.fact_state import is_fact_eligible
from engram.core.models import (
    BRIEF_MEMORY_KEY,
    EDITABLE_FACT_FIELDS,
    MIN_ACTIVE_CONFIDENCE,
    CandidateStatus,
    EventLogMeta,
    EventType,
    Fact,
    FactCategory,
    FactEvent,
    FactLifecycle,
    MemoryCandidate,
    RecallRecord,
    StoreTransaction,
    TransactionStatus,
    materialize_events,
)
from engram.storage.candidates import (
    CandidateSnapshots,
    approval_snapshot,
    prepare_approval,
)
from engram.storage.changes import ApprovalResult, ChangeResult, ChangeSet
from engram.storage.compaction import compacted_event_log, compaction_created_event
from engram.storage.jsonl import load_jsonl_records
from engram.storage.search import TOKEN_RE, SearchHit, SearchIndex
from engram.storage.transactions import (
    StoreIntegrityError as StoreIntegrityError,
)
from engram.storage.transactions import (
    StoreTransactions,
    ensure_trailing_newline,
)
from engram.storage.transactions import (
    locked_store as _locked_store,
)
from engram.storage.transactions import (
    locked_write as _locked_write,
)

logger = logging.getLogger(__name__)


def _is_expired_fact(fact: Fact, now: datetime) -> bool:
    return fact.expires_at is not None and fact.expires_at < now


_is_active_fact = is_fact_eligible


def _normalized_project_text(value: str) -> str:
    return " ".join(TOKEN_RE.findall(value.lower().replace("_", " ").replace("-", " ")))


def _infer_project_from_query(facts: list[Fact], query: str) -> str | None:
    """Infer an explicit project scope when the query names exactly one project."""
    query_norm = f" {_normalized_project_text(query)} "
    matches = []
    for project in sorted({fact.project for fact in facts if fact.project}):
        project_norm = _normalized_project_text(project)
        if project_norm and f" {project_norm} " in query_norm:
            matches.append(project)
    return matches[0] if len(matches) == 1 else None


def _exclude_superseded_facts(facts: list[Fact]) -> list[Fact]:
    superseded_ids = {fact.supersedes for fact in facts if fact.supersedes}
    if not superseded_ids:
        return facts
    return [fact for fact in facts if fact.id not in superseded_ids]


def _unique_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item_id in ids:
        if item_id in seen:
            continue
        seen.add(item_id)
        unique.append(item_id)
    return unique


def _unique_strings(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _snapshot_read[**P, R](
    func: Callable[Concatenate[FactStore, P], R],
) -> Callable[Concatenate[FactStore, P], R]:
    @wraps(func)
    def locked(store: FactStore, /, *args: P.args, **kwargs: P.kwargs) -> R:
        if store.read_only:
            return func(store, *args, **kwargs)
        with _locked_store(store.data_dir):
            if not store.recovering and (
                store.intent_path.exists() or store.pending_transactions()
            ):
                raise RuntimeError("Interrupted store transaction requires recover_transactions()")
            return func(store, *args, **kwargs)

    return locked


class FactStore:
    def __init__(self, data_dir: Path | None = None, *, read_only: bool = False):
        self.data_dir = (data_dir or get_settings().data_dir).resolve()
        self.read_only = read_only
        self.transactions = StoreTransactions(self.data_dir)
        self.candidate_snapshots = CandidateSnapshots(self.data_dir)
        if not read_only:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        # Materialized-state cache keyed on the event log's (inode, size, mtime),
        # so repeated reads skip replaying an unchanged log.
        self._cache_lock = threading.Lock()
        self._facts_cache: tuple[tuple[int, int, int], list[Fact]] | None = None
        self._index_cache: tuple[tuple[int, int, int], SearchIndex] | None = None
        self.recovering = False
        self._pending_cache: tuple[tuple[int, int, int], list[StoreTransaction]] | None = None

    @classmethod
    def open_readonly(cls, data_dir: Path | None = None) -> FactStore:
        """Open diagnostic paths without creating files or recovering transactions."""
        return cls(data_dir, read_only=True)

    @property
    def facts_path(self) -> Path:
        return self.data_dir / "facts.jsonl"

    @property
    def candidates_path(self) -> Path:
        return self.data_dir / "candidates.jsonl"

    @property
    def transaction_log_path(self) -> Path:
        return self.data_dir / "transactions.jsonl"

    def _is_event_log_format(self) -> bool:
        return self.transactions.is_event_log_format()

    def _load_all_events(self) -> list[FactEvent]:
        """Read every ``FactEvent`` from ``facts.jsonl``, skipping the meta line."""
        if not self.facts_path.exists():
            return []
        _, events, _ = self._validated_event_log_records(strict=not self.read_only)
        return events

    def _log_signature(self) -> tuple[int, int, int] | None:
        try:
            stat = self.facts_path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_ino, stat.st_size, stat.st_mtime_ns)

    @_snapshot_read
    def load_facts(self) -> list[Fact]:
        """Load all facts from the event log.

        Events are replayed into materialized facts; forgotten or superseded
        facts surface as ``confidence == 0`` Facts to preserve the existing
        "confidence-zero means inactive" contract used by callers. Each read
        returns independent facts, including their nested collections.

        Corrupt rows raise StoreIntegrityError. Read-only diagnostics tolerate them.
        """
        signature = self._log_signature()
        if signature is None:
            return []
        cached = self._facts_cache
        if cached is not None and cached[0] == signature:
            return [fact.model_copy(deep=True) for fact in cached[1]]
        with self._cache_lock:
            cached = self._facts_cache
            if cached is not None and cached[0] == signature:
                return [fact.model_copy(deep=True) for fact in cached[1]]
            facts = self._replay_facts()
            self._facts_cache = (signature, facts)
            return [fact.model_copy(deep=True) for fact in facts]

    def _replay_facts(self) -> list[Fact]:
        if not self._is_event_log_format():
            if not self.read_only:
                raise StoreIntegrityError(self.facts_path, 1, "missing valid event-log sentinel")
            return []
        events = self._load_all_events()
        materialized = materialize_events(events)
        return [
            fact if active else fact.model_copy(update={"confidence": 0.0})
            for fact, active in materialized.values()
        ]

    @_snapshot_read
    def load_candidates(
        self,
        status: CandidateStatus | None = None,
        project: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryCandidate]:
        return self.candidate_snapshots.load(status, project, limit, strict=not self.read_only)

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
        now = datetime.now(UTC)
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
            if (
                project
                and fact.project != project
                and not (include_global and fact.project is None)
            ):
                continue
            facts.append(fact)

        # Sort by recency
        facts.sort(key=lambda f: f.updated_at, reverse=True)
        if limit:
            facts = facts[:limit]
        return facts

    @_snapshot_read
    def _search_index(self) -> SearchIndex:
        """BM25 index over recallable facts (active, current, not superseded).

        Project briefs are excluded: they summarize other cards and are served
        on their own, so they would crowd search results and ingest neighbors.
        """
        signature = self._log_signature()
        cached = self._index_cache
        if cached is not None and cached[0] == signature:
            return cached[1]
        facts = [
            fact
            for fact in _exclude_superseded_facts(self.load_active_facts())
            if fact.memory_key != BRIEF_MEMORY_KEY
        ]
        index = SearchIndex(facts)
        if signature is not None:
            self._index_cache = (signature, index)
        return index

    def search_index(self) -> SearchIndex:
        """Return a read-isolated index snapshot for maintenance consumers."""
        return deepcopy(self._search_index())

    def search_facts(
        self,
        query: str,
        project: str | None = None,
        limit: int | None = None,
        *,
        include_global: bool = True,
    ) -> list[SearchHit]:
        """Rank recallable facts for ``query`` within a project scope.

        With ``project`` set, the scope is that project plus global facts.
        Without it, a query that names exactly one known project is scoped to
        that project; otherwise every project is searched.
        """
        index = self._search_index()
        scope = project
        if scope is None:
            scope = _infer_project_from_query(index.facts, query)

        now = datetime.now(UTC)

        def in_scope(fact: Fact) -> bool:
            if not is_fact_eligible(fact, now):
                return False
            if scope is None:
                return True
            return fact.project == scope or (include_global and fact.project is None)

        return [
            SearchHit(hit.fact.model_copy(deep=True), hit.score, hit.coverage)
            for hit in index.search(query, accept=in_scope, limit=limit)
        ]

    def _append_events_raw(self, events: list[FactEvent]) -> None:
        self.transactions.append_raw(events)

    @property
    def intent_path(self) -> Path:
        return self.transactions.intent_path

    def _recover_intent(self) -> None:
        if not self.intent_path.exists():
            return
        events = self.transactions.read_intent()
        self._append_events_raw(events)
        self._validated_event_log_records(strict=True)
        self.transactions.clear_intent()

    def append_events(self, events: list[FactEvent]) -> None:
        """Durably prepare exact events, append missing IDs, then clear intent."""
        if self.read_only:
            raise RuntimeError("Read-only store cannot write")
        if not events:
            return
        with _locked_store(self.data_dir):
            self._recover_intent()
            events = [FactEvent.model_validate(event.model_dump()) for event in events]
            if self.facts_path.exists() and self.facts_path.stat().st_size:
                self._validated_event_log_records(strict=True)
            self.transactions.write_intent(events)
            self._recover_intent()

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
        logger.info("Appended %d facts to store", len(facts))

    def append_candidates(self, candidates: list[MemoryCandidate]) -> None:
        if self.read_only:
            raise RuntimeError("Read-only store cannot write")
        self.candidate_snapshots.append(candidates)

    def append_pending_candidates(self, candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
        if self.read_only:
            raise RuntimeError("Read-only store cannot write")
        return self.candidate_snapshots.append_pending(candidates)

    def _facts_by_id(self) -> dict[str, Fact]:
        return {fact.id: fact for fact in self.load_facts()}

    def _active_fact_by_id(self, fact_id: str) -> Fact | None:
        fact = self._facts_by_id().get(fact_id)
        if fact is None or fact.lifecycle is not FactLifecycle.active or fact.confidence == 0.0:
            return None
        return fact

    def _event_and_result_for_update(
        self, existing: Fact, updates: dict[str, Any], now: datetime
    ) -> tuple[FactEvent, Fact]:
        if updates.get("confidence") == 0.0 and len(updates) == 1:
            return (
                FactEvent(
                    event_type=EventType.forgotten,
                    fact_id=existing.id,
                    timestamp=now,
                    payload={},
                ),
                existing.model_copy(
                    update={
                        "lifecycle": FactLifecycle.forgotten,
                        "confidence": 0.0,
                        "updated_at": now,
                    }
                ),
            )

        return (
            FactEvent(
                event_type=EventType.edited,
                fact_id=existing.id,
                timestamp=now,
                payload=updates,
            ),
            Fact.model_validate({**existing.model_dump(), **updates, "updated_at": now}),
        )

    def _append_fact_event(
        self,
        event_type: EventType,
        fact_id: str,
        now: datetime,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.append_events(
            [
                FactEvent(
                    event_type=event_type,
                    fact_id=fact_id,
                    timestamp=now,
                    payload=payload or {},
                )
            ]
        )

    def update_fact(self, fact_id: str, **updates: Any) -> Fact | None:
        """Update a fact by appending an ``edited`` event to the log.

        Returns ``None`` when the fact is not found or is currently forgotten,
        unless the update explicitly sets ``confidence`` (forget/restore paths).
        Special-cases ``confidence == 0`` to emit a ``forgotten`` event instead
        of an edit, preserving the "confidence-zero means inactive" contract.
        """
        with _locked_store(self.data_dir):
            existing = self._facts_by_id().get(fact_id)
            if existing is None:
                return None
            if existing.lifecycle is not FactLifecycle.active or existing.confidence == 0.0:
                return None

            now = datetime.now(UTC)
            event, updated = self._event_and_result_for_update(existing, updates, now)
            self.append_events([event])
            return updated

    def update_candidate(self, candidate_id: str, **updates: Any) -> MemoryCandidate | None:
        """Append a validated snapshot for one candidate."""
        with _locked_store(self.data_dir):
            candidate = next(
                (row for row in self.load_candidates() if row.id == candidate_id), None
            )
            if candidate is None:
                return None
            updated = MemoryCandidate.model_validate(
                {
                    **candidate.model_dump(),
                    **updates,
                    "updated_at": datetime.now(UTC),
                }
            )
            if updated.id != candidate_id:
                raise ValueError("Candidate ID cannot change")
            self.append_candidates([updated])
            return updated

    def rename_project(self, old_project: str, new_project: str) -> int:
        """Rename project scopes through fact events and candidate snapshots."""
        with _locked_store(self.data_dir):
            facts = self.load_facts()
            now = datetime.now(UTC)
            events: list[FactEvent] = []
            for fact in facts:
                if fact.project == old_project and fact.lifecycle is FactLifecycle.active:
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
                logger.info(
                    "Renamed project %s → %s for %d fact(s)",
                    old_project,
                    new_project,
                    fact_count,
                )

            candidates = self.load_candidates()
            cand_count = 0
            for i, candidate in enumerate(candidates):
                if candidate.project == old_project:
                    candidate.project = new_project
                    candidate.updated_at = now
                    candidates[i] = candidate
                    cand_count += 1
            if cand_count > 0:
                self.append_candidates(
                    [candidate for candidate in candidates if candidate.project == new_project]
                )
                logger.info(
                    "Renamed project %s → %s for %d candidate(s)",
                    old_project,
                    new_project,
                    cand_count,
                )

            return fact_count + cand_count

    def batch_update_facts(self, updates: dict[str, dict[str, Any]]) -> list[Fact]:
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
            facts_by_id = self._facts_by_id()
            now = datetime.now(UTC)
            events: list[FactEvent] = []
            updated_facts: list[Fact] = []

            for fact_id, field_updates in updates.items():
                existing = facts_by_id.get(fact_id)
                if existing is None:
                    continue

                event, updated = self._event_and_result_for_update(existing, field_updates, now)
                events.append(event)
                updated_facts.append(updated)

            if events:
                self.append_events(events)
                logger.info(
                    "Batch-updated %d facts via %d event(s)",
                    len(updated_facts),
                    len(events),
                )
            return updated_facts

    def batch_update_candidates(self, updates: dict[str, dict[str, Any]]) -> list[MemoryCandidate]:
        """Append validated snapshots for multiple candidate updates.

        Args:
            updates: Mapping of candidate_id -> field updates to apply.

        Returns:
            List of updated MemoryCandidate objects.
        """
        if not updates:
            return []

        with _locked_store(self.data_dir):
            updated: list[MemoryCandidate] = []
            for candidate in self.load_candidates():
                if candidate.id in updates:
                    updated.append(
                        MemoryCandidate.model_validate(
                            {
                                **candidate.model_dump(),
                                **updates[candidate.id],
                                "updated_at": datetime.now(UTC),
                            }
                        )
                    )
            self.append_candidates(updated)
            return updated

    def forget(self, fact_id: str, reason: str = "") -> Fact | None:
        """Soft-delete a fact by appending a ``forgotten`` event."""
        with _locked_store(self.data_dir):
            existing = self._active_fact_by_id(fact_id)
            if existing is None:
                return None
            now = datetime.now(UTC)
            self._append_fact_event(
                EventType.forgotten,
                fact_id,
                now,
                {"reason": reason} if reason else {},
            )
            if reason:
                logger.info("Forgot fact %s: %s", fact_id, reason)
            return existing.model_copy(
                update={
                    "lifecycle": FactLifecycle.forgotten,
                    "confidence": 0.0,
                    "updated_at": now,
                }
            )

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
            fact, _is_active = entry
            if fact.lifecycle is not FactLifecycle.forgotten:
                # Only explicit forgetting can be restored.
                return None

            now = datetime.now(UTC)
            self._append_fact_event(EventType.restored, fact_id, now)
            logger.info("Restored fact %s", fact_id)
            return fact.model_copy(update={"updated_at": now, "lifecycle": FactLifecycle.active})

    def correct_fact(
        self,
        fact_id: str,
        new_content: str,
        *,
        category: FactCategory | None = None,
        tags: list[str] | None = None,
        memory_key: str | None = None,
        retrieval_hints: list[str] | None = None,
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
            existing = self._active_fact_by_id(fact_id)
            if existing is None:
                return None

            now = datetime.now(UTC)
            new_fact = Fact.model_validate(
                {
                    **existing.model_dump(),
                    "id": uuid4().hex[:12],
                    "category": category or existing.category,
                    "memory_key": memory_key if memory_key is not None else existing.memory_key,
                    "content": new_content,
                    "created_at": now,
                    "updated_at": now,
                    "observed_at": now,
                    "project": project if project is not None else existing.project,
                    "tags": tags if tags is not None else existing.tags,
                    "retrieval_hints": retrieval_hints
                    if retrieval_hints is not None
                    else existing.retrieval_hints,
                    "supersedes": fact_id,
                    "why_store": reason or existing.why_store,
                }
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

        logger.info("Corrected fact %s -> %s: %s", fact_id, new_fact.id, reason)
        return new_fact

    def _active_sources_for_merge(self, source_ids: list[str]) -> list[Fact] | None:
        facts_by_id = self._facts_by_id()
        sources: list[Fact] = []
        for source_id in source_ids:
            fact = facts_by_id.get(source_id)
            if fact is None or fact.lifecycle is not FactLifecycle.active or fact.confidence == 0.0:
                return None
            sources.append(fact)
        if len({fact.project for fact in sources}) > 1:
            return None
        return sources

    def _merged_fact(
        self,
        sources: list[Fact],
        merged_content: str,
        *,
        category: FactCategory | None,
        tags: list[str] | None,
        memory_key: str | None,
        retrieval_hints: list[str] | None,
        project: str | None,
        reason: str,
        now: datetime,
    ) -> Fact:
        primary = sources[0]
        return Fact(
            id=uuid4().hex[:12],
            category=category or primary.category,
            memory_key=memory_key if memory_key is not None else primary.memory_key,
            content=merged_content,
            source=primary.source,
            confidence=max(fact.confidence for fact in sources),
            created_at=now,
            updated_at=now,
            observed_at=now,
            first_observed_at=min(fact.oldest_claim_at for fact in sources),
            project=project if project is not None else primary.project,
            tags=tags if tags is not None else list(primary.tags),
            retrieval_hints=(
                _unique_strings(retrieval_hints)[:5]
                if retrieval_hints is not None
                else _unique_strings(hint for fact in sources for hint in fact.retrieval_hints)[:5]
            ),
            source_group_id=(
                primary.source_group_id
                if all(fact.source_group_id == primary.source_group_id for fact in sources)
                else None
            ),
            supersedes=primary.id,
            consolidates=_unique_strings(
                source_id for fact in sources for source_id in [fact.id, *fact.consolidates]
            ),
            evidence_kind=primary.evidence_kind,
            source_ref=primary.source_ref,
            why_store=reason or "merged",
        )

    def _merge_events(
        self,
        sources: list[Fact],
        new_fact: Fact,
        *,
        reason: str,
        now: datetime,
    ) -> list[FactEvent]:
        events = [
            FactEvent(
                event_type=EventType.superseded,
                fact_id=source.id,
                timestamp=now,
                payload={"superseded_by": new_fact.id, "reason": reason},
            )
            for source in sources
        ]
        events.append(
            FactEvent(
                event_type=EventType.created,
                fact_id=new_fact.id,
                timestamp=now,
                payload=new_fact.model_dump(),
            )
        )
        return events

    def merge_facts(
        self,
        source_ids: list[str],
        merged_content: str,
        *,
        category: FactCategory | None = None,
        tags: list[str] | None = None,
        memory_key: str | None = None,
        retrieval_hints: list[str] | None = None,
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

        unique_ids = _unique_ids(source_ids)
        if len(unique_ids) < 2:
            return None

        with _locked_store(self.data_dir):
            sources = self._active_sources_for_merge(unique_ids)
            if sources is None:
                return None

            now = datetime.now(UTC)
            new_fact = self._merged_fact(
                sources,
                merged_content,
                category=category,
                tags=tags,
                memory_key=memory_key,
                retrieval_hints=retrieval_hints,
                project=project,
                reason=reason,
                now=now,
            )
            events = self._merge_events(
                sources,
                new_fact,
                reason=reason,
                now=now,
            )
            self.append_events(events)

        logger.info("Merged %d facts -> %s: %s", len(unique_ids), new_fact.id, reason)
        return new_fact, unique_ids

    def apply_changes(self, changes: ChangeSet, *, all_or_nothing: bool = False) -> ChangeResult:
        """Append a batch of creations, supersessions, retirements, and edits.

        Every event lands in one append under the store lock, so readers never
        see a replacement without its retirement. Targets that are no longer
        active are skipped; with ``all_or_nothing`` any skip aborts the batch
        (upkeep uses this so a concurrently changed cluster is left alone).
        """
        changes.__post_init__()
        with _locked_store(self.data_dir):
            facts_by_id = self._facts_by_id()
            result = ChangeResult()
            new_ids = {fact.id for fact in changes.new_facts}
            if new_ids & facts_by_id.keys():
                raise ValueError("ChangeSet cannot recreate an existing fact ID")

            def active(fact_id: str) -> bool:
                fact = facts_by_id.get(fact_id)
                return (
                    fact is not None
                    and fact.lifecycle is FactLifecycle.active
                    and fact.confidence > 0.0
                )

            now = datetime.now(UTC)
            targets = [
                *changes.supersede,
                *changes.stale,
                *changes.unstale,
                *changes.edits,
                *changes.expected,
            ]
            for fact_id in targets:
                if not active(fact_id):
                    result.skipped[fact_id] = "not an active fact"
                    continue
                if (
                    changes.require_eligible
                    and fact_id in changes.expected
                    and not is_fact_eligible(facts_by_id[fact_id], now)
                ):
                    result.skipped[fact_id] = "not an eligible fact"
                    continue
                seen = changes.expected.get(fact_id)
                if seen is not None and facts_by_id[fact_id].updated_at != seen:
                    result.skipped[fact_id] = "changed since it was read"
            for old_id, new_id in changes.supersede.items():
                if new_id not in new_ids and not active(new_id):
                    result.skipped[old_id] = f"replacement {new_id} is not active"
            for fact_id, fields in changes.edits.items():
                bad = set(fields) - EDITABLE_FACT_FIELDS
                if bad:
                    result.skipped[fact_id] = f"non-editable fields: {sorted(bad)}"
            if all_or_nothing and result.skipped:
                return result

            events: list[FactEvent] = [
                FactEvent(
                    event_type=EventType.created,
                    fact_id=fact.id,
                    timestamp=now,
                    actor=changes.actor,
                    payload=fact.model_dump(),
                )
                for fact in changes.new_facts
            ]
            for old_id, new_id in changes.supersede.items():
                if old_id in result.skipped:
                    continue
                events.append(
                    FactEvent(
                        event_type=EventType.superseded,
                        fact_id=old_id,
                        timestamp=now,
                        actor=changes.actor,
                        payload={"superseded_by": new_id, "reason": changes.reason},
                    )
                )
                result.superseded[old_id] = new_id
            for fact_id, reason in changes.stale.items():
                if fact_id in result.skipped or fact_id in result.superseded:
                    continue
                events.append(
                    FactEvent(
                        event_type=EventType.stale,
                        fact_id=fact_id,
                        timestamp=now,
                        actor=changes.actor,
                        payload={"reason": reason},
                    )
                )
                result.staled[fact_id] = reason
            for fact_id in changes.unstale:
                if fact_id not in result.skipped:
                    events.append(
                        FactEvent(
                            event_type=EventType.unstale,
                            fact_id=fact_id,
                            timestamp=now,
                            actor=changes.actor,
                        )
                    )
                    result.unstaled.append(fact_id)
            for fact_id, fields in changes.edits.items():
                if fact_id in result.skipped or fact_id in result.superseded:
                    continue
                events.append(
                    FactEvent(
                        event_type=EventType.edited,
                        fact_id=fact_id,
                        timestamp=now,
                        actor=changes.actor,
                        payload=fields,
                    )
                )
                result.edited.append(fact_id)
            self.append_events(events)
            result.created = list(changes.new_facts)
        return result

    def mark_stale(self, fact_id: str, reason: str = "") -> Fact | None:
        """Mark a fact stale by appending a ``stale`` event."""
        with _locked_store(self.data_dir):
            existing = self._active_fact_by_id(fact_id)
            if existing is None:
                return None
            now = datetime.now(UTC)
            self._append_fact_event(
                EventType.stale,
                fact_id,
                now,
                {"reason": reason} if reason else {},
            )
            logger.info("Marked fact %s stale: %s", fact_id, reason)
            return existing.model_copy(
                update={"stale": True, "stale_reason": reason, "updated_at": now}
            )

    def unmark_stale(self, fact_id: str) -> Fact | None:
        """Reverse a stale marking by appending an ``unstale`` event."""
        with _locked_store(self.data_dir):
            existing = self._active_fact_by_id(fact_id)
            if existing is None:
                return None
            now = datetime.now(UTC)
            self._append_fact_event(EventType.unstale, fact_id, now)
            return existing.model_copy(
                update={"stale": False, "stale_reason": "", "updated_at": now}
            )

    def approve_candidates(self, candidate_ids: list[str]) -> list[Fact]:
        """Compatibility projection of the authoritative approval result."""
        return self.approve_candidates_result(candidate_ids).facts

    def approve_candidates_result(
        self,
        candidate_ids: list[str],
        *,
        edits: dict[str, str] | None = None,
    ) -> ApprovalResult:
        """Approve candidates and return their exact transaction-owned mapping."""
        if edits and (
            set(edits) - set(candidate_ids) or any(not text.strip() for text in edits.values())
        ):
            raise ValueError("Candidate edits require requested IDs and nonempty content")
        with _locked_store(self.data_dir):
            self.recover_transactions()
            result = ApprovalResult()
            if edits:
                pending_ids = {
                    row.id for row in self.load_candidates(status=CandidateStatus.pending)
                }
                result.invalid_edit_ids = [
                    candidate_id for candidate_id in edits if candidate_id not in pending_ids
                ]
                if result.invalid_edit_ids:
                    result.unresolved = dict.fromkeys(
                        dict.fromkeys(candidate_ids),
                        "Approval aborted because an edited candidate is missing or no longer pending",
                    )
                    return result
            transaction = self._prepare_approval_transaction(
                candidate_ids, edits=edits, unresolved=result.unresolved
            )
            if transaction is not None:
                self._append_transaction(transaction)
                self._apply_approval_transaction(transaction)
                self._commit_transaction(transaction)
                result.facts = transaction.new_facts
                result.edited = [
                    candidate_id
                    for candidate_id in transaction.candidate_ids
                    if candidate_id in (edits or {})
                ]
                result.approved = {
                    candidate_id: fact.id
                    for candidate_id, fact in zip(
                        transaction.candidate_ids, transaction.new_facts, strict=True
                    )
                }
            for candidate_id in dict.fromkeys(candidate_ids):
                if candidate_id not in result.approved:
                    result.unresolved.setdefault(
                        candidate_id,
                        "Candidate is missing, no longer pending, or its dependencies changed",
                    )
            return result

    def reject_candidates(
        self, candidate_ids: list[str], reason: str = ""
    ) -> list[MemoryCandidate]:
        """Reject candidates without promoting them into active facts."""
        with _locked_store(self.data_dir):
            pending_candidates = {
                c.id: c for c in self.load_candidates(status=CandidateStatus.pending)
            }
            updates: dict[str, dict[str, Any]] = {}
            for candidate_id in candidate_ids:
                if candidate_id in pending_candidates:
                    updates[candidate_id] = {
                        "status": CandidateStatus.rejected,
                        "review_note": reason,
                    }
            return self.batch_update_candidates(updates)

    def recover_transactions(self) -> int:
        """Roll forward any prepared store transactions that lack a commit marker."""
        if self.read_only:
            raise RuntimeError("Read-only store cannot recover transactions")
        with _locked_store(self.data_dir):
            self.recovering = True
            try:
                self._recover_intent()
                return self._recover_transactions_locked()
            finally:
                self.recovering = False

    def _recover_transactions_locked(self) -> int:
        recovered = 0
        for transaction in self.pending_transactions():
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
        self,
        candidate_ids: list[str],
        *,
        edits: dict[str, str] | None = None,
        unresolved: dict[str, str] | None = None,
    ) -> StoreTransaction | None:
        pending_candidates = {c.id: c for c in self.load_candidates(status=CandidateStatus.pending)}
        transaction = prepare_approval(
            candidate_ids,
            pending_candidates,
            self._facts_by_id(),
            edits=edits,
            unresolved=unresolved if unresolved is not None else {},
        )
        return None if transaction is None else self._approval_snapshot(transaction)

    def _apply_transaction(self, transaction: StoreTransaction) -> None:
        if transaction.type != "approve_candidates":
            logger.warning(
                "Unsupported transaction %s of type %s",
                transaction.id,
                transaction.type,
            )
            raise ValueError(f"Unsupported transaction type: {transaction.type}")
        self._apply_approval_transaction(transaction)

    def _approval_snapshot(self, transaction: StoreTransaction) -> StoreTransaction:
        return approval_snapshot(transaction, {row.id: row for row in self.load_candidates()})

    def _apply_approval_transaction(self, transaction: StoreTransaction) -> None:
        if not transaction.events and not transaction.candidates:
            transaction = self._approval_snapshot(transaction)
            self._append_transaction(transaction)
        self.append_events(transaction.events)
        existing = {
            row.model_dump_json()
            for row in load_jsonl_records(self.candidates_path, MemoryCandidate, "candidate")[0]
        }
        self.append_candidates(
            [row for row in transaction.candidates if row.model_dump_json() not in existing]
        )

    def _commit_transaction(self, transaction: StoreTransaction) -> None:
        committed = transaction.model_copy(
            update={
                "status": TransactionStatus.committed,
                "committed_at": datetime.now(UTC),
            }
        )
        self._append_transaction(committed)

    def _append_transaction(self, transaction: StoreTransaction) -> None:
        with _locked_write(self.transaction_log_path):
            ensure_trailing_newline(self.transaction_log_path)
            with self.transaction_log_path.open("a") as f:
                f.write(transaction.model_dump_json() + "\n")
                f.flush()
                os.fsync(f.fileno())

    def _load_transactions(self) -> list[StoreTransaction]:
        records, corrupt = load_jsonl_records(
            self.transaction_log_path, StoreTransaction, "transaction"
        )
        if corrupt:
            raise ValueError("Transaction journal contains invalid records and requires inspection")
        return records

    def pending_transactions(self) -> list[StoreTransaction]:
        if not self.transaction_log_path.exists():
            return []
        stat = self.transaction_log_path.stat()
        signature = stat.st_ino, stat.st_size, stat.st_mtime_ns
        if self._pending_cache is not None and self._pending_cache[0] == signature:
            return self._pending_cache[1]
        prepared: dict[str, StoreTransaction] = {}
        committed_ids: set[str] = set()
        for transaction in self._load_transactions():
            if transaction.status == TransactionStatus.prepared:
                prepared[transaction.id] = transaction
            elif transaction.status == TransactionStatus.committed:
                committed_ids.add(transaction.id)
        pending = [
            transaction
            for transaction_id, transaction in prepared.items()
            if transaction_id not in committed_ids
        ]

        self._pending_cache = (signature, pending)
        return pending

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
        from engram.storage.sync import COMPACTION_SENTINEL_FILENAME

        self._require_offline_rewrite()
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

            now = datetime.now(UTC)
            compacted, tombstones = compacted_event_log(
                materialized,
                keep_tombstones=keep_tombstones,
            )

            sentinel.write_text(f"compaction started at {now.isoformat()}\n")
            try:
                self._rewrite_event_log(EventLogMeta(created_at=now), compacted)
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

    def _require_offline_rewrite(self) -> None:
        if (self.data_dir / ".git").exists():
            raise ValueError(
                "Purge and compaction require a coordinated offline procedure for synced stores"
            )

    def purge(self) -> dict[str, Any]:
        """Remove forgotten and expired facts from the event log.

        Returns counts of purged and retained facts.
        """
        self._require_offline_rewrite()
        with _locked_store(self.data_dir):
            facts = self.load_facts()
            now = datetime.now(UTC)
            kept = []
            purged = 0
            for fact in facts:
                is_forgotten = (
                    fact.lifecycle is not FactLifecycle.active
                    or fact.confidence < MIN_ACTIVE_CONFIDENCE
                )
                is_expired = _is_expired_fact(fact, now)
                if is_forgotten or is_expired:
                    purged += 1
                else:
                    kept.append(fact)

            if purged > 0:
                self._rewrite_event_log(
                    EventLogMeta(created_at=now),
                    [compaction_created_event(fact.id, fact) for fact in kept],
                )
                logger.info("Purged %d facts (%d retained)", purged, len(kept))

            return {"purged": purged, "retained": len(kept)}

    def stats(self) -> dict[str, Any]:
        facts = self.load_facts()
        now = datetime.now(UTC)
        active = [f for f in facts if _is_active_fact(f, now)]
        forgotten = [
            f
            for f in facts
            if f.lifecycle is FactLifecycle.forgotten
            or (f.lifecycle is FactLifecycle.active and f.confidence < MIN_ACTIVE_CONFIDENCE)
        ]
        expired = [
            f for f in facts if f.confidence >= MIN_ACTIVE_CONFIDENCE and _is_expired_fact(f, now)
        ]
        by_category = Counter(f.category.value for f in active)
        by_project = Counter(f.project or "(none)" for f in active)

        return {
            "total_facts": len(facts),
            "active_facts": len(active),
            "forgotten_facts": len(forgotten),
            "superseded_facts": sum(f.lifecycle is FactLifecycle.superseded for f in facts),
            "expired_facts": len(expired),
            "by_category": dict(by_category.most_common()),
            "by_project": dict(by_project.most_common(10)),
            "storage_bytes": self.facts_path.stat().st_size if self.facts_path.exists() else 0,
            "pending_candidates": len(self.load_candidates(status=CandidateStatus.pending)),
        }

    def format_candidates_for_review(self, candidates: list[MemoryCandidate]) -> str:
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
        with (
            _locked_store(self.data_dir),
            _locked_write(self.recall_log_path),
            self.recall_log_path.open("a") as f,
        ):
            f.write(record.model_dump_json() + "\n")

    @_snapshot_read
    def load_recall_log(self, limit: int | None = 500) -> list[RecallRecord]:
        records, _ = load_jsonl_records(
            self.recall_log_path, RecallRecord, "recall log", strict=not self.read_only
        )
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:limit] if limit is not None else records

    def _validated_event_log_records(
        self, *, strict: bool = False
    ) -> tuple[EventLogMeta | None, list[FactEvent], int]:
        return self.transactions.read_events(strict=strict)

    def _rewrite_event_log(
        self,
        meta: EventLogMeta,
        events: list[FactEvent],
    ) -> None:
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
                    fh.write(meta.model_dump_json(exclude_none=True) + "\n")
                    for event in events:
                        fh.write(event.model_dump_json() + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                tmp_path.replace(self.facts_path)
                tmp_path = None
            finally:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()

    def repair(self) -> dict[str, Any]:
        """Explicitly repair corrupt rows under the shared transaction lock."""
        if self.read_only:
            raise RuntimeError("Read-only store cannot write")
        with _locked_store(self.data_dir):
            return self._repair_locked()

    def _repair_locked(self) -> dict[str, Any]:
        """Recover from truncated/corrupt JSONL lines.

        For ``facts.jsonl`` in event-log format, validates each line as either
        the meta sentinel or a ``FactEvent``; corrupt event lines are dropped
        via an atomic rewrite that preserves the sentinel. Files without a
        valid event-log sentinel are refused because their format cannot be
        inferred safely. Candidate records are validated independently.
        Returns counts of valid and corrupt lines found.
        """
        result: dict[str, int] = {}

        # facts.jsonl — format-aware
        if self.facts_path.exists():
            if self._is_event_log_format():
                meta, events, corrupt = self._validated_event_log_records()
                if corrupt > 0:
                    self._rewrite_event_log(meta or EventLogMeta(), events)
                materialized = materialize_events(events)
                result["facts_valid"] = len(materialized)
                result["facts_corrupt"] = corrupt
            else:
                raise ValueError(
                    f"Refusing to repair {self.facts_path}: file is missing a "
                    "valid event-log sentinel. Restore it from backup or move "
                    "the invalid file aside."
                )
        else:
            result["facts_valid"] = 0
            result["facts_corrupt"] = 0

        # candidates.jsonl
        if self.candidates_path.exists():
            valid_candidates, corrupt = load_jsonl_records(
                self.candidates_path, MemoryCandidate, "candidates"
            )
            if corrupt > 0:
                self._rewrite(valid_candidates, path=self.candidates_path)
            result["candidates_valid"] = len(valid_candidates)
            result["candidates_corrupt"] = corrupt
        else:
            result["candidates_valid"] = 0
            result["candidates_corrupt"] = 0

        return result

    def _rewrite(self, records: Sequence[Fact | MemoryCandidate], path: Path | None = None) -> None:
        """Rewrite a store JSONL file atomically with fsync and a unique temp path."""
        target_path = path or self.facts_path
        if target_path == self.facts_path:
            if any(not isinstance(record, Fact) for record in records):
                raise TypeError("facts.jsonl can only be rewritten from Fact records")
            facts = [record for record in records if isinstance(record, Fact)]
            self._rewrite_event_log(
                EventLogMeta(),
                [compaction_created_event(fact.id, fact) for fact in facts],
            )
            return

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
                tmp_path.replace(target_path)
                tmp_path = None  # Successfully replaced; don't clean up
            finally:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()
