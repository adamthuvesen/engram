"""Retriever: lexical memory cards by default, one LLM call when asked.

Recall ranks recallable facts with the BM25 search index, keeps the hits that
clear a deterministic relevance bar, and returns them as dated memory cards
with zero LLM calls. ``mode="answer"`` spends one LLM call to synthesize an
answer over those cards. When nothing clears the bar but the search still
found weak candidates, one LLM call picks the relevant ones (paraphrases and
synonyms the lexical pass cannot judge).
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field, JsonValue

from engram.core.config import Settings, get_settings
from engram.core.interfaces import EnvelopeWarning, WarningCode
from engram.core.models import Fact, RecallRecord
from engram.core.provenance import (
    DEFAULT_MAX_PREFILTER_MATCHES,
    DEFAULT_MAX_SOURCES,
    DEFAULT_OUTPUT_EXCERPT_CHARS,
    DEFAULT_PROMPT_EXCERPT_CHARS,
    LLMCallTrace,
    PrefilterMatch,
    RecallEvidence,
    RecallProvenance,
    RecallResult,
    RecallTrace,
    SourceSummary,
    TierDecision,
    UsageSummary,
    excerpt,
)
from engram.core.structured_outputs import StructuredOutput
from engram.llm import (
    Completion,
    ModelCompletion,
    ProviderResponseError,
    complete_model_with_usage,
    complete_with_usage,
    provider_available,
)
from engram.llm.client import response_format_for_model
from engram.storage.search import SearchHit
from engram.storage.store import (
    AsyncFactStore,
    FactStore,
    format_fact_line,
    format_facts_for_llm,
    run_sync,
)

logger = logging.getLogger(__name__)

RecallMode = Literal["cards", "answer"]

ANSWER_SYSTEM = """Answer the query using only the supplied stored facts.
Return structured claims and a quality rating: high, medium, low, or none.
Each claim has plain text and fact_ids copied exactly from the supplied IDs.
Do not put citation markers in claim text. The application renders citations.
Use an empty fact_ids list only to explain that the facts do not answer the query.
Prefer newer facts when they conflict and explain the conflict.
Mention uncertainty when relying on time-bound, expired, or unverified facts.
Keep the answer concise and complete."""

SELECT_SYSTEM = """You pick stored memory facts that are relevant to a query.
The facts matched the query only weakly by keywords, so judge meaning, not word
overlap: a fact is relevant when it answers or directly informs the query.
Return the IDs of the relevant facts, most relevant first, copied exactly from their
`id:` markers. Return an empty list when none are relevant."""

# Relevance bar for lexical hits. A hit counts when it covers enough of the
# query's IDF mass AND scores close enough to the best hit. Calibrated against
# ~1.8k real recall-log queries over a 5.6k-fact store: these values keep the
# typical card set at 1-10 facts with the right card first, and send roughly
# a third of queries (mostly vague keyword bags) to LLM selection instead.
MIN_COVERAGE = 0.3
RELATIVE_CUTOFF = 0.4
# Top-card coverage at or above this reports quality "high".
HIGH_QUALITY_COVERAGE = 0.6

# When nothing clears the relevance bar, at most this many top hits go to one
# LLM call (selection in cards mode, synthesis in answer mode).
ZERO_HIT_MAX_CANDIDATES = 30

# v4 = BM25 search with coverage/relative relevance and card output.
SELECTOR_VERSION = "v4"

NO_RELEVANT_MEMORIES = "No relevant memories found for this query."


class RecallSelection(StructuredOutput):
    """LLM response for zero-hit selection."""

    relevant_ids: list[str]


class AnswerClaim(StructuredOutput):
    text: str = Field(min_length=1)
    fact_ids: list[str]


class RecallAnswer(StructuredOutput):
    claims: list[AnswerClaim] = Field(min_length=1)
    quality: Literal["high", "medium", "low", "none"]


def _llm_available() -> bool:
    return provider_available()


def relevant_hits(hits: list[SearchHit]) -> list[SearchHit]:
    """Hits that clear the coverage floor and the relative score cutoff."""
    if not hits:
        return []
    floor = RELATIVE_CUTOFF * hits[0].score
    return [hit for hit in hits if hit.coverage >= MIN_COVERAGE and hit.score >= floor]


def _format_cards(facts: list[Fact], total: int, project: str | None) -> str:
    """Card output: a header line, then one dated fact line per card."""
    if not facts:
        return NO_RELEVANT_MEMORIES
    scope = f", project {project}" if project else ""
    lines = [f"Memory ({len(facts)} of {total} matches{scope}):"]
    lines.extend(f"- {format_fact_line(fact)}" for fact in facts)
    return "\n".join(lines)


def _cards_quality(cards: list[SearchHit]) -> str:
    if not cards:
        return "none"
    return "high" if cards[0].coverage >= HIGH_QUALITY_COVERAGE else "medium"


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def _provider_unavailable(message: str) -> EnvelopeWarning:
    return EnvelopeWarning(code=WarningCode.provider_unavailable, message=message)


def _delivery_warnings(delivered: list[Fact]) -> list[EnvelopeWarning]:
    """Advisories about the facts handed back to the caller.

    ``conflicting_facts`` fires when two delivered facts claim the same
    (project, memory_key) slot; ``suspect_fact`` when upkeep flagged a
    delivered fact as possibly outdated.
    """
    warnings: list[EnvelopeWarning] = []
    buckets: dict[tuple[str | None, str], list[str]] = {}
    for fact in delivered:
        if fact.memory_key:
            buckets.setdefault((fact.project, fact.memory_key), []).append(fact.id)
    for (project, memory_key), ids in buckets.items():
        if len(ids) >= 2:
            warnings.append(
                EnvelopeWarning(
                    code=WarningCode.conflicting_facts,
                    message=(
                        f"Multiple active facts share {project or '(global)'}/"
                        f"{memory_key}; verify they agree."
                    ),
                    ids=sorted(ids),
                    details={"project": project, "memory_key": memory_key},
                )
            )
    suspect: dict[str, JsonValue] = {
        fact.id: fact.suspect_reason for fact in delivered if fact.suspect_reason
    }
    if suspect:
        warnings.append(
            EnvelopeWarning(
                code=WarningCode.suspect_fact,
                message="Some returned facts may be outdated; verify before relying on them.",
                ids=sorted(suspect),
                details={"reasons": suspect},
            )
        )
    return warnings


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


@dataclass
class _Outcome:
    text: str
    quality: str
    selected: list[SearchHit]
    delivered: list[Fact]
    cited: list[Fact] = field(default_factory=list)
    calls: list[LLMCallTrace] = field(default_factory=list)
    truncated: bool = False
    warnings: list[EnvelopeWarning] = field(default_factory=list)


def _candidates_prompt(query: str, facts: list[Fact]) -> str:
    return f"QUERY: {query}\n\nSTORED FACTS:\n{format_facts_for_llm(facts)}"


def _record_completion(
    usage: UsageSummary, completion: Completion | ModelCompletion[RecallSelection]
) -> None:
    usage.llm_completions = (usage.llm_completions or 0) + 1
    usage.input_tokens = completion.input_tokens
    usage.cached_tokens = completion.cached_tokens
    usage.output_tokens = completion.output_tokens
    usage.usage_unknown_calls = int(
        completion.input_tokens is None or completion.output_tokens is None
    )
    usage.cache_hit_ratio = (
        completion.cached_tokens / completion.input_tokens
        if completion.input_tokens and completion.cached_tokens is not None
        else None
    )


def _validate_references(ids: list[str], candidate_ids: set[str]) -> list[str]:
    unknown = set(ids) - candidate_ids
    if unknown:
        raise ProviderResponseError("Provider referenced facts outside the supplied evidence.")
    return list(dict.fromkeys(ids))


async def _select_relevant(
    query: str,
    candidates: list[SearchHit],
    settings: Settings,
    usage: UsageSummary,
    *,
    with_trace: bool,
    excerpt_chars: int,
    output_chars: int,
) -> tuple[list[SearchHit], LLMCallTrace | None, bool]:
    prompt = _candidates_prompt(query, [hit.fact for hit in candidates])
    t_call = time.monotonic()
    completion = await asyncio.wait_for(
        complete_model_with_usage(
            prompt=prompt,
            system=SELECT_SYSTEM,
            response_model=RecallSelection,
            reasoning_effort=settings.recall_reasoning_effort,
        ),
        timeout=settings.retrieval_timeout,
    )
    _record_completion(usage, completion)
    selection = completion.value
    by_id = {hit.fact.id: hit for hit in candidates}
    selected = [by_id[fid] for fid in _validate_references(selection.relevant_ids, set(by_id))]
    if not with_trace:
        return selected, None, False
    trace, truncated = _call_trace(
        name="select",
        system=SELECT_SYSTEM,
        prompt=prompt,
        output=selection.model_dump_json(),
        elapsed_ms=(time.monotonic() - t_call) * 1000,
        excerpt_chars=excerpt_chars,
        output_chars=output_chars,
        input_tokens=completion.input_tokens,
        cached_tokens=completion.cached_tokens,
        output_tokens=completion.output_tokens,
    )
    return selected, trace, truncated


# Only citation syntax is recognized. Ordinary hashes, dates, and numbers stay text.
_INLINE_CITATION_RE = re.compile(
    r"\(\s*(?:id|facts?):[^)]*\)|\[\s*(?:facts?:[^]]*|[0-9a-f]{8,}(?:[\s,]+[0-9a-f]{8,})*)\s*\]",
    re.IGNORECASE,
)


def _render_answer(answer: RecallAnswer, facts: list[Fact]) -> tuple[str, list[Fact]]:
    by_id = {fact.id: fact for fact in facts}
    cited_ids: list[str] = []
    lines: list[str] = []
    for claim in answer.claims:
        if not claim.text.strip() or _INLINE_CITATION_RE.search(claim.text):
            raise ProviderResponseError(
                "Answer claim text must contain prose without citation markers."
            )
        refs = _validate_references(claim.fact_ids, set(by_id))
        if not refs and answer.quality != "none":
            raise ProviderResponseError("An answer claim has no supporting fact references.")
        cited_ids.extend(fid for fid in refs if fid not in cited_ids)
        citations = "".join(f" (id: {fid})" for fid in refs)
        lines.append(claim.text.strip() + citations)
    return "\n".join(lines), [by_id[fid] for fid in cited_ids]


async def _answer(
    query: str,
    candidates: list[SearchHit],
    settings: Settings,
    usage: UsageSummary,
    *,
    with_trace: bool,
    excerpt_chars: int,
    output_chars: int,
) -> _Outcome:
    facts = [hit.fact for hit in candidates]
    prompt = _candidates_prompt(query, facts)
    t_call = time.monotonic()
    completion = await asyncio.wait_for(
        complete_with_usage(
            prompt=prompt,
            system=ANSWER_SYSTEM,
            reasoning_effort=settings.recall_reasoning_effort,
            response_format=response_format_for_model(RecallAnswer),
        ),
        timeout=settings.retrieval_timeout,
    )
    _record_completion(usage, completion)
    answer = RecallAnswer.model_validate_json(completion.text)
    text, cited = _render_answer(answer, facts)
    outcome = _Outcome(text, answer.quality, candidates, cited, cited=cited)
    if with_trace:
        trace, outcome.truncated = _call_trace(
            name="answer",
            system=ANSWER_SYSTEM,
            prompt=prompt,
            output=completion.text,
            elapsed_ms=(time.monotonic() - t_call) * 1000,
            excerpt_chars=excerpt_chars,
            output_chars=output_chars,
            input_tokens=completion.input_tokens,
            cached_tokens=completion.cached_tokens,
            output_tokens=completion.output_tokens,
        )
        outcome.calls.append(trace)
    return outcome


def _call_trace(
    *,
    name: str,
    system: str,
    prompt: str,
    output: str,
    elapsed_ms: float,
    excerpt_chars: int,
    output_chars: int,
    input_tokens: int | None = None,
    cached_tokens: int | None = None,
    output_tokens: int | None = None,
    error: str | None = None,
) -> tuple[LLMCallTrace, bool]:
    prompt_excerpt, prompt_truncated = excerpt(prompt, excerpt_chars)
    output_excerpt, output_truncated = excerpt(output, output_chars)
    return (
        LLMCallTrace(
            name=name,
            system_excerpt=excerpt(system, excerpt_chars)[0],
            prompt_excerpt=prompt_excerpt,
            output_excerpt=output_excerpt,
            elapsed_ms=elapsed_ms,
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            error=error,
        ),
        prompt_truncated or output_truncated,
    )


# ---------------------------------------------------------------------------
# Public recall entry points
# ---------------------------------------------------------------------------


async def recall(
    query: str,
    project: str | None = None,
    store: FactStore | AsyncFactStore | None = None,
    mode: RecallMode = "cards",
) -> str:
    """Recall as plain text: memory cards by default, or an LLM answer."""
    text, _quality, _provenance, _trace = await recall_with_provenance(
        query, project=project, store=store, mode=mode
    )
    return text


async def recall_with_provenance(
    query: str,
    project: str | None = None,
    store: FactStore | AsyncFactStore | None = None,
    *,
    mode: RecallMode = "cards",
    with_trace: bool = False,
    verbose_trace: bool = False,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_prefilter_matches: int = DEFAULT_MAX_PREFILTER_MATCHES,
) -> tuple[str, str, RecallProvenance, RecallTrace | None]:
    """Compatibility tuple adapter over the authoritative typed result."""
    result = await recall_result(
        query,
        project,
        store,
        mode=mode,
        with_trace=with_trace,
        verbose_trace=verbose_trace,
        max_sources=max_sources,
        max_prefilter_matches=max_prefilter_matches,
    )
    return result.text, result.quality, result.provenance, result.trace


async def recall_result(
    query: str,
    project: str | None = None,
    store: FactStore | AsyncFactStore | None = None,
    *,
    mode: RecallMode = "cards",
    with_trace: bool = False,
    verbose_trace: bool = False,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_prefilter_matches: int = DEFAULT_MAX_PREFILTER_MATCHES,
    settings: Settings | None = None,
) -> RecallResult:
    """Recall bounded evidence, preserving the lexical selector's scores and rules.

    Provider failures return cards and an explicit warning. ``llm_calls`` keeps
    its historical meaning of successful recall calls. Attempts, received
    completions, failures, and missing usage have separate counters.
    """
    if mode not in ("cards", "answer"):
        raise ValueError("mode must be 'cards' or 'answer'.")
    if max_sources < 1 or max_prefilter_matches < 1:
        raise ValueError("Recall source and prefilter limits must be positive.")
    store = (
        store
        if isinstance(store, AsyncFactStore)
        else AsyncFactStore(store if store is not None else await run_sync(FactStore))
    )
    settings = settings or get_settings()
    t0 = time.monotonic()
    scale = 4 if verbose_trace else 1
    excerpt_chars = DEFAULT_PROMPT_EXCERPT_CHARS * scale
    output_chars = DEFAULT_OUTPUT_EXCERPT_CHARS * scale
    hits = await _search(
        store,
        query,
        project,
        limit=max(settings.max_facts_per_agent, ZERO_HIT_MAX_CANDIDATES),
    )
    relevant = relevant_hits(hits)
    zero_hit = not relevant and bool(hits)
    use_llm = bool(hits) and (mode == "answer" or zero_hit) and _llm_available()
    cards = relevant[:max_sources]
    outcome = _Outcome(
        _format_cards([hit.fact for hit in cards], len(relevant), project),
        _cards_quality(cards),
        relevant,
        [hit.fact for hit in cards],
    )
    usage = UsageSummary(
        llm_calls=0,
        llm_attempts=0,
        llm_completions=0,
        llm_failures=0,
        usage_unknown_calls=0,
    )
    if mode == "answer" and hits and not use_llm:
        outcome.warnings.append(
            _provider_unavailable("No LLM key configured; returned memory cards.")
        )
    if use_llm:
        pool = relevant or hits[:ZERO_HIT_MAX_CANDIDATES]
        if mode == "answer":
            pool = pool[: min(max_sources, settings.max_facts_per_agent)]
        usage.llm_attempts = 1
        t_call = time.monotonic()
        try:
            if mode == "answer":
                outcome = await _answer(
                    query,
                    pool,
                    settings,
                    usage,
                    with_trace=with_trace,
                    excerpt_chars=excerpt_chars,
                    output_chars=output_chars,
                )
            else:
                selected, trace, truncated = await _select_relevant(
                    query,
                    pool,
                    settings,
                    usage,
                    with_trace=with_trace,
                    excerpt_chars=excerpt_chars,
                    output_chars=output_chars,
                )
                shown = selected[:max_sources]
                outcome = _Outcome(
                    _format_cards([hit.fact for hit in shown], len(selected), project),
                    _cards_quality(shown),
                    selected,
                    [hit.fact for hit in shown],
                    calls=[trace] if trace else [],
                    truncated=truncated,
                )
            usage.llm_calls = 1
        except Exception as exc:
            usage.llm_failures = 1
            # A failed request may have consumed tokens without reporting usage.
            if not usage.llm_completions:
                usage.usage_unknown_calls = 1
            logger.warning("Recall LLM call failed: %s", exc)
            outcome.warnings.append(
                _provider_unavailable(
                    f"LLM recall failed ({type(exc).__name__}); returned memory cards."
                )
            )
            if with_trace:
                trace, truncated = _call_trace(
                    name="answer" if mode == "answer" else "select",
                    system=ANSWER_SYSTEM if mode == "answer" else SELECT_SYSTEM,
                    prompt=_candidates_prompt(query, [hit.fact for hit in pool]),
                    output="",
                    elapsed_ms=(time.monotonic() - t_call) * 1000,
                    excerpt_chars=excerpt_chars,
                    output_chars=output_chars,
                    error=type(exc).__name__,
                    input_tokens=usage.input_tokens,
                    cached_tokens=usage.cached_tokens,
                    output_tokens=usage.output_tokens,
                )
                outcome.calls.append(trace)
                outcome.truncated |= truncated
    latency_ms = (time.monotonic() - t0) * 1000
    tier = 1 if usage.llm_calls else 0
    evidence = {
        hit.fact.id: RecallEvidence(fact=hit.fact, score=hit.score, coverage=hit.coverage)
        for hit in hits
    }
    delivered = [evidence[fact.id] for fact in outcome.delivered]
    selected_evidence = [evidence[hit.fact.id] for hit in outcome.selected]
    cited = [evidence[fact.id] for fact in outcome.cited]
    provenance = _build_provenance(
        query=query,
        project=project,
        tier=tier,
        outcome=outcome,
        candidates=list(evidence.values()),
        selected=selected_evidence,
        delivered=delivered,
        relevant=relevant,
        zero_hit=zero_hit and use_llm,
        usage=usage,
        latency_ms=latency_ms,
        max_prefilter_matches=max_prefilter_matches,
        max_sources=max_sources,
    )
    trace_obj = (
        RecallTrace(
            provenance=provenance,
            calls=outcome.calls,
            excerpt_chars=excerpt_chars,
            truncated=outcome.truncated,
            verbose=verbose_trace,
        )
        if with_trace
        else None
    )
    result = RecallResult(
        text=outcome.text,
        quality=outcome.quality,
        candidates=list(evidence.values()),
        selected=selected_evidence,
        delivered=delivered,
        cited=cited,
        provenance=provenance,
        trace=trace_obj,
    )
    try:
        await _log_recall(
            store,
            RecallRecord(
                query=query,
                project=project,
                tier=tier,
                prefilter_count=len(result.candidates),
                latency_ms=latency_ms,
                quality=result.quality,
                llm_calls=usage.llm_calls,
                llm_attempts=usage.llm_attempts,
                llm_completions=usage.llm_completions,
                llm_failures=usage.llm_failures,
                usage_unknown_calls=usage.usage_unknown_calls,
                input_tokens=usage.input_tokens,
                cached_tokens=usage.cached_tokens,
                output_tokens=usage.output_tokens,
                selector_version=SELECTOR_VERSION,
                mode=mode,
                delivered_ids=[item.fact.id for item in result.delivered],
            ),
        )
    except Exception:
        logger.debug("Failed to log recall record", exc_info=True)
    return result


def _build_provenance(
    *,
    query: str,
    project: str | None,
    tier: int,
    outcome: _Outcome,
    candidates: list[RecallEvidence],
    selected: list[RecallEvidence],
    delivered: list[RecallEvidence],
    relevant: list[SearchHit],
    zero_hit: bool,
    usage: UsageSummary,
    latency_ms: float,
    max_prefilter_matches: int,
    max_sources: int,
) -> RecallProvenance:
    relevant_ids = {hit.fact.id for hit in relevant}
    delivered_ids = [item.fact.id for item in delivered]
    delivered_set = set(delivered_ids)
    ordered = [
        *delivered,
        *(item for item in selected if item.fact.id not in delivered_set),
    ]
    return RecallProvenance(
        query=query,
        project=project,
        tier=tier,
        quality=outcome.quality,
        selected_decision=TierDecision(
            tier=tier,
            rules=SELECTOR_VERSION,
            relevant_count=len(relevant),
            top_score=candidates[0].score if candidates else None,
            zero_hit_escalation=zero_hit,
        ),
        prefilter_count=len(candidates),
        prefilter_matches=[
            PrefilterMatch(
                id=item.fact.id,
                score=item.score,
                coverage=item.coverage,
                above_floor=item.fact.id in relevant_ids,
            )
            for item in candidates[:max_prefilter_matches]
        ],
        source_fact_ids=[item.fact.id for item in selected],
        candidate_fact_ids=[item.fact.id for item in candidates],
        selected_fact_ids=[item.fact.id for item in selected],
        delivered_fact_ids=delivered_ids,
        sources=[
            SourceSummary(
                id=item.fact.id,
                project=item.fact.project,
                category=item.fact.category.value,
                confidence=item.fact.confidence,
                updated_at=item.fact.updated_at,
                content_excerpt=excerpt(item.fact.content.strip(), 240)[0],
                score=item.score,
                cited=item.fact.id in delivered_set,
            )
            for item in ordered[:max_sources]
        ],
        cited_fact_ids=delivered_ids,
        warnings=[*outcome.warnings, *_delivery_warnings(outcome.delivered)],
        usage=usage,
        latency_ms=latency_ms,
    )


async def _search(
    store: FactStore | AsyncFactStore,
    query: str,
    project: str | None,
    *,
    limit: int,
) -> list[SearchHit]:
    if isinstance(store, AsyncFactStore):
        return await store.search_facts(query, project, limit=limit)
    return await AsyncFactStore(store).search_facts(query, project, limit=limit)


async def _log_recall(
    store: FactStore | AsyncFactStore,
    record: RecallRecord,
) -> None:
    if isinstance(store, AsyncFactStore):
        await store.log_recall(record)
    else:
        await AsyncFactStore(store).log_recall(record)


__all__ = [
    "ANSWER_SYSTEM",
    "MIN_COVERAGE",
    "RELATIVE_CUTOFF",
    "ZERO_HIT_MAX_CANDIDATES",
    "AnswerClaim",
    "RecallAnswer",
    "RecallMode",
    "RecallResult",
    "recall",
    "recall_result",
    "recall_with_provenance",
]
