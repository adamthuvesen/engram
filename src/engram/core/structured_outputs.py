"""Pydantic schemas for structured LLM responses."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from engram.core.models import Durability, FactCategory


class StructuredOutput(BaseModel):
    """Base class for schemas that define the LLM response contract."""

    model_config = ConfigDict(extra="forbid")


class ExtractedFact(StructuredOutput):
    """One coherent memory card proposed by the ingest prompt."""

    memory_key: str = Field(min_length=3, max_length=120)
    content: str
    category: FactCategory
    project: str | None = None
    tags: list[str] = Field(default_factory=list)
    retrieval_hints: list[str] = Field(min_length=1, max_length=5)
    covered_claims: list[str] = Field(min_length=1)
    why_store: str = Field(min_length=1)
    durability: Durability = Durability.durable
    anchors: list[str] = Field(default_factory=list)
    expires_at: datetime | None = None
    # Existing card IDs this card updates, extends, merges, or contradicts.
    replaces: list[str] = Field(default_factory=list)
    # Existing card ID the input merely restates; the card is then skipped.
    duplicate_of: str | None = None


class RetiredCard(StructuredOutput):
    """An existing card the input shows is no longer true."""

    id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ExcludedClaim(StructuredOutput):
    """One source claim intentionally excluded from durable memory."""

    claim: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ExtractionResponse(StructuredOutput):
    """Response shape for extracting and reconciling memory cards in one call."""

    facts: list[ExtractedFact] = Field(default_factory=list)
    retire: list[RetiredCard] = Field(default_factory=list)
    excluded_claims: list[ExcludedClaim] = Field(default_factory=list)
