"""Data models for Engram."""

import threading
import time
from copy import copy
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    model_validator,
)

MIN_ACTIVE_CONFIDENCE = 0.1

# memory_key of the per-project brief card that upkeep synthesizes. Briefs are
# served by ``recall_context(mode="brief")``, never as search results.
BRIEF_MEMORY_KEY = "project-brief"


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


class Durability(str, Enum):
    """How long a memory is expected to stay true.

    ``evergreen`` never ages (identity, lasting preferences). ``durable`` holds
    until something contradicts it (conventions, decisions, pitfalls).
    ``ephemeral`` describes in-flight state that decays within weeks and gets a
    default expiry.
    """

    evergreen = "evergreen"
    durable = "durable"
    ephemeral = "ephemeral"


class CandidateStatus(str, Enum):
    """Lifecycle states for proposed memories."""

    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    conflicted = "conflicted"


class TransactionStatus(str, Enum):
    """Lifecycle states for durable store transactions."""

    prepared = "prepared"
    committed = "committed"


class FactBase(BaseModel):
    """Shared fields between Fact and MemoryCandidate."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    category: FactCategory
    memory_key: str = ""
    content: str
    source: str = "conversation"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # When the oldest claim carried into this card was observed. Set when a
    # card replaces or merges older cards, so their claims keep their age
    # instead of taking the newest source's date. None means ``observed_at``.
    first_observed_at: datetime | None = None
    effective_at: datetime | None = None
    expires_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    retrieval_hints: list[str] = Field(default_factory=list)
    project: str | None = None
    source_group_id: str | None = None
    supersedes: str | None = None
    consolidates: list[str] = Field(default_factory=list)
    evidence_kind: EvidenceKind = EvidenceKind.unknown
    source_ref: str | None = None
    why_store: str = ""
    durability: Durability = Durability.durable
    # Repo-relative file paths or code symbols the memory depends on. Upkeep
    # checks them against the project's repository to catch code drift.
    anchors: list[str] = Field(default_factory=list)
    # Non-empty when upkeep found evidence the memory may be outdated (e.g.
    # some anchors vanished). Suspect facts stay recallable but are flagged.
    suspect_reason: str = ""
    # Stale facts are inspectable but excluded from active recall and prefilter.
    # Set via the maintenance workflows (mark_stale).
    stale: bool = False
    stale_reason: str = ""
    brief_source_revisions: dict[str, datetime] | None = None

    @property
    def oldest_claim_at(self) -> datetime:
        """When this card's oldest claim was observed."""
        return self.first_observed_at or self.observed_at


class FactLifecycle(str, Enum):
    active = "active"
    forgotten = "forgotten"
    superseded = "superseded"


class Fact(FactBase):
    """One coherent, independently maintainable memory card."""

    lifecycle: FactLifecycle = Field(default=FactLifecycle.active, exclude=True)
    superseded_by: str | None = Field(default=None, exclude=True)


_STORED_CATEGORY_NAMES: dict[str, str] = {
    "temporal": FactCategory.event.value,
    "update": FactCategory.correction.value,
}


def fact_from_stored_data(data: dict[str, Any]) -> Fact:
    """Load a Fact from persisted data written by any supported store version."""
    if isinstance(data.get("category"), str):
        category = data["category"]
        if category in _STORED_CATEGORY_NAMES:
            data = {**data, "category": _STORED_CATEGORY_NAMES[category]}
    return Fact.model_validate(data)


class MemoryCandidate(FactBase):
    """A proposed memory waiting for review."""

    status: CandidateStatus = CandidateStatus.pending
    review_note: str = ""
    # Active fact IDs this candidate replaces once approved.
    replaces: list[str] = Field(default_factory=list)
    replaces_expected: dict[str, datetime] | None = None


class StateLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["run", "consolidated"]
    at: datetime
    project: str | None = None
    # Fact ID -> the updated_at upkeep's own write left it with.
    written: dict[str, datetime] = Field(default_factory=dict)
    # Seed fact ID -> failed consolidation attempts so far.
    retry_seeds: dict[str, int] = Field(default_factory=dict)


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
    llm_attempts: int | None = None
    llm_completions: int | None = None
    llm_failures: int | None = None
    usage_unknown_calls: int | None = None
    output_tokens: int | None = None
    input_tokens: int | None = None
    cached_tokens: int | None = None
    selector_version: str | None = None
    mode: str | None = None
    # Fact IDs handed back to the caller (cards shown or answer citations).
    delivered_ids: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StoreTransaction(BaseModel):
    """Durable transaction journal entry for multi-file store operations."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    type: str
    status: TransactionStatus
    candidate_ids: list[str] = Field(default_factory=list)
    fact_updates: dict[str, dict[str, object]] = Field(default_factory=dict)
    candidate_updates: dict[str, dict[str, object]] = Field(default_factory=dict)
    new_facts: list[Fact] = Field(default_factory=list)
    # old fact id -> replacing fact id, applied as ``superseded`` events.
    supersessions: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    committed_at: datetime | None = None
    events: list[FactEvent] = Field(default_factory=list)
    candidates: list[MemoryCandidate] = Field(default_factory=list)


# --- Event-log model ---------------------------------------------------------


EVENT_LOG_META_VERSION: Literal["event-log-v1"] = "event-log-v1"

# Fields on Fact that are allowed in an ``edited`` event's payload. Restricting
# this prevents callers from mutating immutable fields (id, created_at) via
# edit events.
EDITABLE_FACT_FIELDS = frozenset(
    {
        "category",
        "memory_key",
        "content",
        "source",
        "confidence",
        "observed_at",
        "first_observed_at",
        "effective_at",
        "expires_at",
        "tags",
        "retrieval_hints",
        "project",
        "source_group_id",
        "supersedes",
        "evidence_kind",
        "source_ref",
        "why_store",
        "durability",
        "anchors",
        "suspect_reason",
        "consolidates",
        "stale",
        "stale_reason",
        "brief_source_revisions",
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


class ReasonPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = ""


class SupersededPayload(ReasonPayload):
    superseded_by: str


class EmptyPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


_edit_fields: dict[str, Any] = {}
for _name in EDITABLE_FACT_FIELDS:
    _field = copy(Fact.model_fields[_name])
    _field.default = None
    _field.default_factory = None
    _edit_fields[_name] = (_field.annotation, _field)
EditedPayload = create_model(
    "EditedPayload",
    __config__=ConfigDict(extra="forbid"),
    **_edit_fields,
)


class FactEvent(BaseModel):
    """One typed mutation event for a fact in the append-only event log."""

    event_id: str = Field(default_factory=new_event_id)
    event_type: EventType
    fact_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actor: str = "engram"
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self) -> FactEvent:
        if self.event_type is EventType.created:
            fact = fact_from_stored_data(self.payload)
            if fact.id != self.fact_id:
                raise ValueError("created payload id must match fact_id")
            self.payload = fact.model_dump(mode="json")
        elif self.event_type is EventType.edited:
            data = dict(self.payload)
            if isinstance(data.get("category"), str) and data["category"] in _STORED_CATEGORY_NAMES:
                data["category"] = _STORED_CATEGORY_NAMES[data["category"]]
            self.payload = EditedPayload.model_validate(data).model_dump(
                mode="json", exclude_unset=True
            )
        else:
            schema = (
                SupersededPayload
                if self.event_type is EventType.superseded
                else ReasonPayload
                if self.event_type in {EventType.forgotten, EventType.stale}
                else EmptyPayload
            )
            self.payload = schema.model_validate(self.payload).model_dump(mode="json")
        return self


def decode_stored_event(line: str) -> FactEvent:
    """Decode v1 rows, including fields legacy edit replay intentionally ignored."""
    import json

    data = json.loads(line)
    if not isinstance(data, dict):
        raise ValueError("Event row must be a JSON object")
    if data.get("event_type") == EventType.edited.value:
        if not isinstance(data.get("payload", {}), dict):
            raise ValueError("Edited event payload must be a JSON object")
        data["payload"] = {
            key: value
            for key, value in data.get("payload", {}).items()
            if key in EDITABLE_FACT_FIELDS
        }
    return FactEvent.model_validate(data)


class EventLogMeta(BaseModel):
    """First-line sentinel marking a ``facts.jsonl`` as event-log format."""

    model_config = ConfigDict(extra="forbid")

    meta: Literal["event-log-v1"] = EVENT_LOG_META_VERSION
    created_at: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("created_at", "migrated_at"),
    )


def _event_sort_key(event: FactEvent) -> tuple[datetime, str]:
    """Order events deterministically: timestamp first, event_id breaks ties."""
    return (event.timestamp, event.event_id)


def _ensure_single_fact_id(events: list[FactEvent]) -> None:
    fact_id = events[0].fact_id
    if any(event.fact_id != fact_id for event in events):
        raise ValueError(f"replay_fact received events for multiple fact_ids: {fact_id} + others")


def _fact_with_event_timestamp(fact: Fact, event: FactEvent, **updates: object) -> Fact:
    return fact_from_stored_data({**fact.model_dump(), **updates, "updated_at": event.timestamp})


def _fact_after_event(fact: Fact, event: FactEvent) -> Fact:
    if event.event_type is EventType.edited:
        updates = {
            key: value for key, value in event.payload.items() if key in EDITABLE_FACT_FIELDS
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
    is_forgotten = False
    superseded_by: str | None = None

    for event in ordered:
        if event.event_type is EventType.created:
            # Allow re-creation only as a no-op idempotent restore (e.g. during
            # compaction replay); the first created wins.
            if fact is None:
                fact = fact_from_stored_data(event.payload)
            continue

        if fact is None:
            # Skip mutation events that arrive before created — defensive only.
            continue

        fact = _fact_after_event(fact, event)
        if event.event_type is EventType.superseded:
            superseded_by = event.payload["superseded_by"]
        if event.event_type is EventType.forgotten:
            is_forgotten = True
        elif event.event_type is EventType.restored:
            is_forgotten = False

    if fact is None:
        return None, False

    lifecycle = (
        FactLifecycle.superseded
        if superseded_by is not None
        else FactLifecycle.forgotten
        if is_forgotten
        else FactLifecycle.active
    )
    fact = fact.model_copy(update={"lifecycle": lifecycle, "superseded_by": superseded_by})
    return fact, lifecycle is FactLifecycle.active


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
