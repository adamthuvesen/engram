"""Typed payloads for application operations and their public projections."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from engram.core.models import CandidateStatus, Durability, FactCategory
from engram.core.provenance import RecallProvenance, SourceSummary, UsageSummary
from engram.core.structured_outputs import ExcludedClaim, UnresolvedDecision


class FactPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: str
    category: FactCategory
    project: str | None
    memory_key: str
    content: str
    confidence: float
    stale: bool
    stale_reason: str
    supersedes: str | None
    consolidates: list[str]
    tags: list[str]
    retrieval_hints: list[str]
    source_group_id: str | None
    durability: Durability
    anchors: list[str]
    suspect_reason: str
    expires_at: datetime | None
    observed_at: datetime
    first_observed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class CandidatePayload(FactPayload):
    status: CandidateStatus
    review_note: str
    why_store: str
    replaces: list[str]


class IngestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[FactPayload] = Field(default_factory=list)
    superseded: dict[str, str] = Field(default_factory=dict)
    retired: dict[str, str] = Field(default_factory=dict)
    duplicates: list[str] = Field(default_factory=list)
    excluded: list[ExcludedClaim] = Field(default_factory=list)
    unresolved: list[UnresolvedDecision] = Field(default_factory=list)


class SuggestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidatePayload]
    unresolved: list[UnresolvedDecision] = Field(default_factory=list)
    message: str | None = None


class RecallPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    mode: Literal["cards", "answer"]
    quality: str
    tier: int
    facts: list[SourceSummary]
    source_fact_ids: list[str]
    cited_fact_ids: list[str]
    usage: UsageSummary
    latency_ms: float
    provenance: RecallProvenance | None = None


class DoctorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report: dict[str, JsonValue]
    repair: dict[str, JsonValue] | None = None


class CandidateReviewPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[FactPayload] | None = None
    candidates: list[CandidatePayload] | None = None
    requested_ids: list[str]
    applied_ids: list[str]
    unresolved_ids: list[str]
    edited_ids: list[str] = Field(default_factory=list)
