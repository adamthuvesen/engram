"""Recall provenance and trace data structures.

These models describe *how* a recall result was produced — what facts the
prefilter matched, which tier was selected, what facts ended up cited in the
final answer, what warnings apply (stale/superseded/forgotten/conflicting),
and what the call cost in time and tokens.

Provenance is computed from artifacts the retriever already produces; building
it must not add additional default LLM calls. Trace adds bounded prompt
excerpts and intermediate model output snippets — bounded by character counts
so the JSON output stays small enough for an agent to consume.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from engram.interfaces import EnvelopeWarning


# Default truncation limits for trace excerpts. Trace must be cheap by default
# so agents can call it without flooding their context window.
DEFAULT_PROMPT_EXCERPT_CHARS = 2000
DEFAULT_OUTPUT_EXCERPT_CHARS = 2000
DEFAULT_MAX_SOURCES = 25
DEFAULT_MAX_PREFILTER_MATCHES = 50


class PrefilterMatch(BaseModel):
    id: str
    score: int
    above_floor: bool


class SourceSummary(BaseModel):
    """Compact metadata for a fact used (or considered) in answering a query."""

    id: str
    project: str | None = None
    category: str
    confidence: float
    updated_at: datetime
    content_excerpt: str = ""
    score: int | None = None
    cited: bool = False
    superseded_by: str | None = None
    stale: bool = False
    forgotten: bool = False
    reason: str = ""


class TierDecision(BaseModel):
    tier: int
    rules: str
    relevant_count: int
    top_score: int | None = None
    gap_ratio: float | None = None
    cap_applied: bool = False


class UsageSummary(BaseModel):
    """LLM usage roll-up. Missing values are reported as None, not 0."""

    llm_calls: int | None = None
    input_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None
    cache_hit_ratio: float | None = None


class RecallProvenance(BaseModel):
    """Full provenance for a recall query.

    Built from artifacts the retriever already produces. No extra LLM calls
    are made to assemble this object.
    """

    query: str
    project: str | None = None
    tier: int
    quality: str = ""
    selected_decision: TierDecision
    prefilter_count: int
    prefilter_matches: list[PrefilterMatch] = Field(default_factory=list)
    source_fact_ids: list[str] = Field(default_factory=list)
    sources: list[SourceSummary] = Field(default_factory=list)
    cited_fact_ids: list[str] = Field(default_factory=list)
    warnings: list[EnvelopeWarning] = Field(default_factory=list)
    usage: UsageSummary = Field(default_factory=UsageSummary)
    latency_ms: float = 0.0


class LLMCallTrace(BaseModel):
    name: str
    system_excerpt: str = ""
    prompt_excerpt: str = ""
    output_excerpt: str = ""
    elapsed_ms: float | None = None
    input_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


class RecallTrace(BaseModel):
    """Bounded debug trace for a recall query.

    Includes the full provenance plus per-call prompt/output excerpts. Default
    excerpts are bounded; verbose output is opt-in via ``verbose=True`` at the
    caller, which raises the per-field char limit.
    """

    provenance: RecallProvenance
    calls: list[LLMCallTrace] = Field(default_factory=list)
    excerpt_chars: int = DEFAULT_PROMPT_EXCERPT_CHARS
    truncated: bool = False
    verbose: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


def excerpt(text: str | None, limit: int) -> tuple[str, bool]:
    """Truncate text to ``limit`` chars; return (excerpt, was_truncated)."""
    if not text:
        return "", False
    if len(text) <= limit:
        return text, False
    return text[:limit] + "…", True


__all__ = [
    "DEFAULT_MAX_PREFILTER_MATCHES",
    "DEFAULT_MAX_SOURCES",
    "DEFAULT_OUTPUT_EXCERPT_CHARS",
    "DEFAULT_PROMPT_EXCERPT_CHARS",
    "LLMCallTrace",
    "PrefilterMatch",
    "RecallProvenance",
    "RecallTrace",
    "SourceSummary",
    "TierDecision",
    "UsageSummary",
    "excerpt",
]
