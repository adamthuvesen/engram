"""Data models for Engram."""

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


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


# Migration map: legacy category value → canonical category
_CATEGORY_MIGRATION: dict[str, str] = {
    "temporal": "event",
    "update": "correction",
}


def migrate_category(raw: str) -> str:
    """Map legacy category names to their canonical equivalents."""
    return _CATEGORY_MIGRATION.get(raw, raw)


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

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_category(cls, data: object) -> object:
        if isinstance(data, dict) and "category" in data:
            raw = data["category"]
            # Only migrate string values; enum instances are already valid
            if isinstance(raw, str):
                data["category"] = migrate_category(raw)
        return data

    def model_copy(self, *, update: dict | None = None, **kwargs):  # type: ignore[override]
        """Override to migrate legacy category values passed via update."""
        if update and "category" in update:
            raw = update["category"]
            if isinstance(raw, str) and not isinstance(raw, FactCategory):
                update = {**update, "category": FactCategory(migrate_category(raw))}
        return super().model_copy(update=update, **kwargs)


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
