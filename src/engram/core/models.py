"""Data models for Engram."""

import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

MIN_ACTIVE_CONFIDENCE = 0.1


_EVENT_ID_LOCK = threading.Lock()
_LAST_EVENT_MS: int = 0


def new_event_id() -> str:
    """Generate a monotonically-orderable event id.

    Format: ``{ms:016d}{rand:08x}`` — 16-digit ms timestamp (sortable to year ~2286)
    plus 8 hex chars from uuid4 for cross-process tie-breaking. Within a single
    process the ms portion is strictly monotonic (a slow clock or rapid calls
    cannot produce duplicates: the helper bumps to ``last + 1`` if needed).
    """
    global _LAST_EVENT_MS
    with _EVENT_ID_LOCK:
        now_ms = int(time.time() * 1000)
        if now_ms <= _LAST_EVENT_MS:
            now_ms = _LAST_EVENT_MS + 1
        _LAST_EVENT_MS = now_ms
    return f"{now_ms:016d}{uuid4().hex[:8]}"


class FactCategory(str, Enum):
    """Categories for structured knowledge extraction."""

    personal_info = "personal_info"
    preference = "preference"
    event = "event"
    decision = "decision"
    pitfall = "pitfall"
    convention = "convention"
    assistant_info = "assistant_info"
    project = "project"
    workflow = "workflow"
    correction = "correction"


class EvidenceKind(str, Enum):
    """Kinds of evidence backing a memory item."""

    conversation = "conversation"
    imported_memory = "imported_memory"
    user_statement = "user_statement"
    tool_output = "tool_output"
    file = "file"
    unknown = "unknown"


class CandidateStatus(str, Enum):
    """Lifecycle states for proposed memories."""

    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class TransactionStatus(str, Enum):
    """Lifecycle states for durable store transactions."""

    prepared = "prepared"
    committed = "committed"


class FactBase(BaseModel):
    """Shared fields between Fact and MemoryCandidate."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    category: FactCategory
    content: str
    source: str = "conversation"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    effective_at: datetime | None = None
    expires_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    project: str | None = None
    supersedes: str | None = None
    evidence_kind: EvidenceKind = EvidenceKind.unknown
    source_ref: str | None = None
    why_store: str = ""
    # Stale facts are inspectable but excluded from active recall and prefilter.
    # Set via the maintenance workflows (mark_stale).
    stale: bool = False
    stale_reason: str = ""


class Fact(FactBase):
    """Atomic unit of knowledge in the memory store."""

    pass


class MemoryCandidate(FactBase):
    """A proposed memory waiting for review."""

    status: CandidateStatus = CandidateStatus.pending
    review_note: str = ""


class IngestionRecord(BaseModel):
    """Audit trail for fact extraction."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    source: str
    facts_created: list[str] = Field(default_factory=list)
    facts_updated: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_model: str = ""


class RecallRecord(BaseModel):
    """Log entry for a recall query (quality observability)."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    query: str
    project: str | None = None
    tier: int
    prefilter_count: int
    latency_ms: float
    quality: str = ""  # high/medium/low/none — set by synthesis
    llm_calls: int | None = None
    input_tokens: int | None = None
    cached_tokens: int | None = None
    selector_version: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StoreTransaction(BaseModel):
    """Durable transaction journal entry for multi-file store operations."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    type: str
    status: TransactionStatus
    candidate_ids: list[str] = Field(default_factory=list)
    fact_updates: dict[str, dict] = Field(default_factory=dict)
    candidate_updates: dict[str, dict] = Field(default_factory=dict)
    new_facts: list[Fact] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    committed_at: datetime | None = None


# --- Event-log model ---------------------------------------------------------


EVENT_LOG_META_VERSION = "event-log-v1"

# Fields on Fact that are allowed in an ``edited`` event's payload. Restricting
# this prevents callers from mutating immutable fields (id, created_at) via
# edit events.
EDITABLE_FACT_FIELDS = frozenset(
    {
        "category",
        "content",
        "source",
        "confidence",
        "observed_at",
        "effective_at",
        "expires_at",
        "tags",
        "project",
        "supersedes",
        "evidence_kind",
        "source_ref",
        "why_store",
        "stale",
        "stale_reason",
    }
)


class EventType(str, Enum):
    """Event types appended to ``facts.jsonl``."""

    created = "created"
    edited = "edited"
    forgotten = "forgotten"
    restored = "restored"
    stale = "stale"
    unstale = "unstale"
    superseded = "superseded"


class FactEvent(BaseModel):
    """One typed mutation event for a fact in the append-only event log."""

    event_id: str = Field(default_factory=new_event_id)
    event_type: EventType
    fact_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor: str = "engram"
    payload: dict[str, Any] = Field(default_factory=dict)


class EventLogMeta(BaseModel):
    """First-line sentinel marking a ``facts.jsonl`` as event-log format."""

    model_config = ConfigDict(extra="forbid")

    meta: str = EVENT_LOG_META_VERSION
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _event_sort_key(event: FactEvent) -> tuple[datetime, str]:
    """Order events deterministically: timestamp first, event_id breaks ties."""
    return (event.timestamp, event.event_id)


def _ensure_single_fact_id(events: list[FactEvent]) -> None:
    fact_id = events[0].fact_id
    if any(event.fact_id != fact_id for event in events):
        raise ValueError(
            f"replay_fact received events for multiple fact_ids: {fact_id} + others"
        )


def _fact_with_event_timestamp(fact: Fact, event: FactEvent, **updates) -> Fact:
    return Fact.model_validate(
        {**fact.model_dump(), **updates, "updated_at": event.timestamp}
    )


def _fact_after_event(fact: Fact, event: FactEvent) -> Fact:
    if event.event_type is EventType.edited:
        updates = {
            key: value
            for key, value in event.payload.items()
            if key in EDITABLE_FACT_FIELDS
        }
        return _fact_with_event_timestamp(fact, event, **updates) if updates else fact
    if event.event_type is EventType.stale:
        return _fact_with_event_timestamp(
            fact,
            event,
            stale=True,
            stale_reason=event.payload.get("reason", ""),
        )
    if event.event_type is EventType.unstale:
        return _fact_with_event_timestamp(fact, event, stale=False, stale_reason="")
    if event.event_type in {
        EventType.forgotten,
        EventType.restored,
        EventType.superseded,
    }:
        return _fact_with_event_timestamp(fact, event)
    return fact


def _lifecycle_after_event(
    event_type: EventType, *, is_active: bool, is_forgotten: bool
) -> tuple[bool, bool]:
    if event_type is EventType.forgotten:
        return False, True
    if event_type is EventType.restored:
        return True, False
    if event_type is EventType.superseded:
        return False, is_forgotten
    return is_active, is_forgotten


def replay_fact(events: list[FactEvent]) -> tuple[Fact | None, bool]:
    """Replay events for a single ``fact_id`` to its current materialized state.

    Returns ``(fact, is_active)``:
      * ``fact`` is ``None`` if the event sequence does not begin with a
        ``created`` event (corrupt / unknown fact_id).
      * ``is_active`` reflects forgotten/restored lifecycle. Stale flag is
        encoded directly on the returned ``Fact``.

    Concurrent edits resolve by timestamp then ``event_id`` lexicographic tie-break.
    """
    if not events:
        return None, False

    ordered = sorted(events, key=_event_sort_key)
    _ensure_single_fact_id(ordered)

    fact: Fact | None = None
    is_active = True
    is_forgotten = False

    for event in ordered:
        if event.event_type is EventType.created:
            # Allow re-creation only as a no-op idempotent restore (e.g. during
            # compaction replay); the first created wins.
            if fact is None:
                fact = Fact(**event.payload)
            continue

        if fact is None:
            # Skip mutation events that arrive before created — defensive only.
            continue

        fact = _fact_after_event(fact, event)
        is_active, is_forgotten = _lifecycle_after_event(
            event.event_type,
            is_active=is_active,
            is_forgotten=is_forgotten,
        )

    if fact is None:
        return None, False
    if is_forgotten:
        is_active = False

    return fact, is_active


def materialize_events(events: list[FactEvent]) -> dict[str, tuple[Fact, bool]]:
    """Group events by fact_id and replay each group.

    Returns a mapping of fact_id → (fact, is_active). Facts whose first event
    is not a ``created`` event are skipped (defensive).
    """
    by_fact: dict[str, list[FactEvent]] = {}
    for event in events:
        by_fact.setdefault(event.fact_id, []).append(event)

    out: dict[str, tuple[Fact, bool]] = {}
    for fact_id, fact_events in by_fact.items():
        fact, is_active = replay_fact(fact_events)
        if fact is not None:
            out[fact_id] = (fact, is_active)
    return out
