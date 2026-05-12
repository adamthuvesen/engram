"""Pydantic schemas for structured LLM responses."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from engram.models import FactCategory


class StructuredOutput(BaseModel):
    """Base class for schemas that define the LLM response contract."""

    model_config = ConfigDict(extra="forbid")


class ExtractedFact(StructuredOutput):
    """One fact proposed by the extraction prompt."""

    content: str
    category: FactCategory
    tags: list[str] = Field(default_factory=list)
    why_store: str = ""
    effective_at: datetime | None = None
    expires_at: datetime | None = None


class ExtractionResponse(StructuredOutput):
    """Response shape for fact extraction."""

    facts: list[ExtractedFact] = Field(default_factory=list)


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


class SynthesisAction(StructuredOutput):
    """One maintenance decision for a stored fact."""

    fact_id: str
    action: str = "keep"
    reason: str | None = None
    new_content: str | None = None
    new_tags: list[str] | None = None
    merge_with: list[str] = Field(default_factory=list)
    merged_content: str | None = None
    merged_tags: list[str] | None = None
    merge_target: str | None = None


class SynthesisResponse(StructuredOutput):
    """Response shape for fact-store synthesis."""

    actions: list[SynthesisAction] = Field(default_factory=list)
