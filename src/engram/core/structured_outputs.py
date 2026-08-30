"""Pydantic schemas for structured LLM responses."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from engram.core.models import FactCategory


class StructuredOutput(BaseModel):
    """Base class for schemas that define the LLM response contract."""

    model_config = ConfigDict(extra="forbid")


class ExtractedFact(StructuredOutput):
    """One coherent memory card proposed by the extraction prompt."""

    memory_key: str = Field(min_length=3, max_length=120)
    content: str
    category: FactCategory
    project: str | None = None
    tags: list[str] = Field(default_factory=list)
    retrieval_hints: list[str] = Field(min_length=1, max_length=5)
    covered_claims: list[str] = Field(min_length=1)
    why_store: str = Field(min_length=1)
    effective_at: datetime | None = None
    expires_at: datetime | None = None


class ExcludedClaim(StructuredOutput):
    """One source claim intentionally excluded from durable memory."""

    claim: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ExtractionResponse(StructuredOutput):
    """Response shape for memory-card extraction."""

    facts: list[ExtractedFact] = Field(default_factory=list)
    excluded_claims: list[ExcludedClaim] = Field(default_factory=list)


class DedupUpdate(StructuredOutput):
    """One candidate fact that supersedes an existing fact."""

    new_idx: StrictInt
    existing_id: str


class DedupResponse(StructuredOutput):
    """Response shape for deduplication classification."""

    new: list[StrictInt] = Field(default_factory=list)
    updates: list[DedupUpdate] = Field(default_factory=list)
    duplicates: list[StrictInt] = Field(default_factory=list)

    @field_validator("updates", mode="before")
    @classmethod
    def _drop_malformed_updates(cls, value: object) -> object:
        """Keep old per-entry tolerance for malformed update objects."""
        if not isinstance(value, list):
            return []
        return [
            item
            for item in value
            if isinstance(item, dict)
            and isinstance(item.get("new_idx"), int)
            and isinstance(item.get("existing_id"), str)
        ]
