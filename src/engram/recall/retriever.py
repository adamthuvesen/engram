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

from engram.core.config import Settings, ensure_openai_api_key, get_settings
from engram.core.interfaces import EnvelopeWarning, WarningCode
from engram.core.models import Fact, RecallRecord
from engram.core.provenance import (
    DEFAULT_MAX_PREFILTER_MATCHES,
    DEFAULT_MAX_SOURCES,
    DEFAULT_OUTPUT_EXCERPT_CHARS,
    DEFAULT_PROMPT_EXCERPT_CHARS,
    LLMCallTrace,
    PrefilterMatch,
    RecallProvenance,
    RecallTrace,
    SourceSummary,
    TierDecision,
    UsageSummary,
    excerpt,
)
from engram.core.structured_outputs import StructuredOutput
from engram.llm import complete_model, complete_with_usage
from engram.storage.search import SearchHit
from engram.storage.store import (
    AsyncFactStore,
    FactStore,
    format_fact_line,
    format_facts_for_llm,
)

logger = logging.getLogger(__name__)

RecallMode = Literal["cards", "answer"]

ANSWER_SYSTEM = """You are a memory search and synthesis agent. Given a query and stored
facts, answer the query from the facts that bear on it, as if briefing a colleague.
Keep it concise but complete.

- Every fact carries the date it was observed. When facts about the same thing
  conflict, prefer the newer one and say that it replaced the older one.
- Facts flagged `time-bound`, `expires ...`, or `unverified: ...` may be out of date;
  say so when you rely on one.
- Cite fact IDs for traceability: copy each ID exactly (12 hex characters) from its
  `id:` marker; never invent, merge, or truncate an ID. Omit a citation rather than
  guess one.
- If the facts do not answer the query, say so instead of guessing.

At the very end of your answer, on a new line, add a quality rating in the format:
[quality: high|medium|low|none]"""

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

# Fact-ID extraction from LLM responses. Fact IDs are 12-hex strings emitted in
# `(id: <hex>)` form by ``format_fact_line``.
_CITED_ID_RE = re.compile(r"\b([0-9a-f]{12})\b")

# ID-like hex runs in answer text. Runs shorter than 8 chars collide with
# ordinary English words made of hex letters ("decade", "deadbeef" aside).
_ID_LIKE_RE = re.compile(r"\b[0-9a-f]{8,}\b")

# Tidy-up rules for citation groups left ragged after invalid IDs are removed,
# e.g. "[facts: , ]", "([abc], , [def])", "[]". Applied to fixpoint.
_CITATION_TIDY_RULES = [
    (re.compile(r",\s*,"), ","),
    (re.compile(r"(?<=[\[(])\s*,\s*"), ""),
    (re.compile(r",\s*(?=[\])])"), ""),
    (re.compile(r"\s*\[\s*(?:facts?:\s*)?\]"), ""),
    (re.compile(r"\s*\(\s*(?:facts?:\s*)?\)"), ""),
]


class RecallSelection(StructuredOutput):
    """LLM response for zero-hit selection: IDs of the relevant candidates."""

    relevant_ids: list[str]


def _llm_available() -> bool:
    """True when an LLM provider key is set or loadable from the key cache."""
    return ensure_openai_api_key() is not None


def _relevant_hits(hits: list[SearchHit]) -> list[SearchHit]:
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


def _extract_quality(text: str) -> tuple[str, str]:
    """Extract [quality: ...] tag from synthesis output.

    Returns (clean_text, quality_level).
    """
    for level in ("high", "medium", "low", "none"):
        tag = f"[quality: {level}]"
        if tag in text:
            return text.replace(tag, "").strip(), level
    return text.strip(), ""


def _extract_cited_ids(text: str, candidate_ids: set[str]) -> list[str]:
    """Pull cited fact IDs out of an LLM response, preserving order.

    Only returns IDs that were actually in the prompt's fact set, so we never
    fabricate a citation from a hallucinated hex string.
    """
    if not text:
        return []
    seen: list[str] = []
    seen_set: set[str] = set()
    for match in _CITED_ID_RE.finditer(text):
        fact_id = match.group(1)
        if fact_id in candidate_ids and fact_id not in seen_set:
            seen.append(fact_id)
            seen_set.add(fact_id)
    return seen


def _scrub_invalid_citations(text: str, candidate_ids: set[str]) -> str:
    """Remove ID-like hex runs the model invented (not in the prompt's fact set).

    Provenance already filters citations through ``_extract_cited_ids``; this
    keeps the answer text consistent with it instead of shipping fabricated or
    mangled IDs (wrong length, merged digits) to the caller.
    """
    scrubbed = _ID_LIKE_RE.sub(
        lambda m: m.group(0) if m.group(0) in candidate_ids else "", text
    )
    if scrubbed == text:
        return text
    while True:
        before = scrubbed
        for pattern, replacement in _CITATION_TIDY_RULES:
            scrubbed = pattern.sub(replacement, scrubbed)
        if scrubbed == before:
            return scrubbed


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
    suspect = {
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
    """What one recall path produced, before provenance assembly."""

    text: str
    quality: str
    delivered: list[Fact]
    llm_calls: int = 0
    input_tokens: int | None = None
    cached_tokens: int | None = None
    calls: list[LLMCallTrace] = field(default_factory=list)
    truncated: bool = False
    warnings: list[EnvelopeWarning] = field(default_factory=list)


def _candidates_prompt(query: str, facts: list[Fact]) -> str:
    return f"QUERY: {query}\n\nSTORED FACTS:\n{format_facts_for_llm(facts)}"


async def _select_relevant(
    query: str,
    candidates: list[SearchHit],
    settings: Settings,
    *,
    with_trace: bool,
    excerpt_chars: int,
    output_chars: int,
) -> tuple[list[SearchHit], LLMCallTrace | None, bool]:
    """One structured LLM call choosing the relevant candidates by ID."""
    prompt = _candidates_prompt(query, [hit.fact for hit in candidates])
    t_call = time.monotonic()
    selection = await asyncio.wait_for(
        complete_model(
            prompt=prompt,
            system=SELECT_SYSTEM,
            response_model=RecallSelection,
            reasoning_effort=settings.recall_reasoning_effort,
        ),
        timeout=settings.retrieval_timeout,
    )
    elapsed_ms = (time.monotonic() - t_call) * 1000
    by_id = {hit.fact.id: hit for hit in candidates}
    selected = [
        by_id[fid] for fid in dict.fromkeys(selection.relevant_ids) if fid in by_id
    ]
    if not with_trace:
        return selected, None, False
    trace, truncated = _call_trace(
        name="select",
        system=SELECT_SYSTEM,
        prompt=prompt,
        output=selection.model_dump_json(),
        elapsed_ms=elapsed_ms,
        excerpt_chars=excerpt_chars,
        output_chars=output_chars,
    )
    return selected, trace, truncated


async def _answer(
    query: str,
    facts: list[Fact],
    settings: Settings,
    *,
    with_trace: bool,
    excerpt_chars: int,
    output_chars: int,
) -> _Outcome:
    """One LLM call synthesizing an answer over ``facts``."""
    prompt = _candidates_prompt(query, facts)
    t_call = time.monotonic()
    completion = await asyncio.wait_for(
        complete_with_usage(
            prompt=prompt,
            system=ANSWER_SYSTEM,
            reasoning_effort=settings.recall_reasoning_effort,
        ),
        timeout=settings.retrieval_timeout,
    )
    elapsed_ms = (time.monotonic() - t_call) * 1000
    answer, quality = _extract_quality(completion.text)
    candidate_ids = {fact.id for fact in facts}
    cited = set(_extract_cited_ids(completion.text, candidate_ids))
    outcome = _Outcome(
        _scrub_invalid_citations(answer, candidate_ids),
        quality,
        [fact for fact in facts if fact.id in cited],
        llm_calls=1,
        input_tokens=completion.input_tokens,
        cached_tokens=completion.cached_tokens,
    )
    if with_trace:
        trace, outcome.truncated = _call_trace(
            name="answer",
            system=ANSWER_SYSTEM,
            prompt=prompt,
            output=completion.text,
            elapsed_ms=elapsed_ms,
            excerpt_chars=excerpt_chars,
            output_chars=output_chars,
            input_tokens=completion.input_tokens,
            cached_tokens=completion.cached_tokens,
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
    """Recall returning ``(text, quality, provenance, trace_or_none)``.

    ``mode="cards"`` returns up to ``max_sources`` relevant facts with no LLM
    call; only a query whose hits all miss the relevance bar spends one LLM
    call to select among them. ``mode="answer"`` spends one LLM call to
    synthesize an answer. Provider failures degrade to cards (or to "no
    relevant memories") with a ``provider_unavailable`` warning, never raise.

    ``with_trace=True`` populates the ``RecallTrace`` with bounded prompt and
    output excerpts; ``verbose_trace=True`` widens those limits.
    """
    store = store or FactStore()
    settings = get_settings()
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
    relevant = _relevant_hits(hits)
    zero_hit = not relevant and bool(hits)
    use_llm = bool(hits) and (mode == "answer" or zero_hit) and _llm_available()
    cards = relevant[:max_sources]
    outcome = _Outcome(
        _format_cards([hit.fact for hit in cards], len(relevant), project),
        _cards_quality(cards),
        [hit.fact for hit in cards],
    )
    if mode == "answer" and hits and not use_llm:
        outcome.warnings.append(
            _provider_unavailable("No LLM key configured; returned memory cards.")
        )

    if use_llm:
        pool = relevant or hits[:ZERO_HIT_MAX_CANDIDATES]
        try:
            if mode == "answer":
                outcome = await _answer(
                    query,
                    [hit.fact for hit in pool[: settings.max_facts_per_agent]],
                    settings,
                    with_trace=with_trace,
                    excerpt_chars=excerpt_chars,
                    output_chars=output_chars,
                )
            else:
                selected, trace, truncated = await _select_relevant(
                    query,
                    pool,
                    settings,
                    with_trace=with_trace,
                    excerpt_chars=excerpt_chars,
                    output_chars=output_chars,
                )
                shown = selected[:max_sources]
                outcome = _Outcome(
                    _format_cards([hit.fact for hit in shown], len(selected), project),
                    _cards_quality(shown),
                    [hit.fact for hit in shown],
                    llm_calls=1,
                    calls=[trace] if trace else [],
                    truncated=truncated,
                )
        except Exception as exc:
            logger.warning("Recall LLM call failed: %s", exc, exc_info=True)
            outcome.warnings.append(
                _provider_unavailable(
                    f"LLM recall failed ({type(exc).__name__}); returned memory cards."
                )
            )

    latency_ms = (time.monotonic() - t0) * 1000
    tier = 1 if outcome.llm_calls else 0
    delivered_ids = [fact.id for fact in outcome.delivered]
    provenance = _build_provenance(
        query=query,
        project=project,
        tier=tier,
        outcome=outcome,
        hits=hits,
        relevant=relevant,
        zero_hit=zero_hit and use_llm,
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

    try:
        await _log_recall(
            store,
            RecallRecord(
                query=query,
                project=project,
                tier=tier,
                prefilter_count=len(hits),
                latency_ms=latency_ms,
                quality=outcome.quality,
                llm_calls=outcome.llm_calls,
                input_tokens=outcome.input_tokens,
                cached_tokens=outcome.cached_tokens,
                selector_version=SELECTOR_VERSION,
                mode=mode,
                delivered_ids=delivered_ids,
            ),
        )
    except Exception:
        logger.debug("Failed to log recall record", exc_info=True)

    logger.info(
        "recall mode=%s tier=%d hits=%d relevant=%d delivered=%d latency=%.0fms "
        "quality=%s calls=%d",
        mode,
        tier,
        len(hits),
        len(relevant),
        len(delivered_ids),
        latency_ms,
        outcome.quality,
        outcome.llm_calls,
    )
    return outcome.text, outcome.quality, provenance, trace_obj


def _build_provenance(
    *,
    query: str,
    project: str | None,
    tier: int,
    outcome: _Outcome,
    hits: list[SearchHit],
    relevant: list[SearchHit],
    zero_hit: bool,
    latency_ms: float,
    max_prefilter_matches: int,
    max_sources: int,
) -> RecallProvenance:
    relevant_ids = {hit.fact.id for hit in relevant}
    delivered_ids = [fact.id for fact in outcome.delivered]
    delivered_set = set(delivered_ids)
    # Delivered facts first so the source cap never drops what the caller got.
    ordered = sorted(hits, key=lambda hit: hit.fact.id not in delivered_set)
    input_tokens = outcome.input_tokens
    cached_tokens = outcome.cached_tokens
    return RecallProvenance(
        query=query,
        project=project,
        tier=tier,
        quality=outcome.quality,
        selected_decision=TierDecision(
            tier=tier,
            rules=SELECTOR_VERSION,
            relevant_count=len(relevant),
            top_score=hits[0].score if hits else None,
            zero_hit_escalation=zero_hit,
        ),
        prefilter_count=len(hits),
        prefilter_matches=[
            PrefilterMatch(
                id=hit.fact.id,
                score=hit.score,
                coverage=hit.coverage,
                above_floor=hit.fact.id in relevant_ids,
            )
            for hit in hits[:max_prefilter_matches]
        ],
        source_fact_ids=[hit.fact.id for hit in relevant],
        sources=[
            SourceSummary(
                id=hit.fact.id,
                project=hit.fact.project,
                category=hit.fact.category.value,
                confidence=hit.fact.confidence,
                updated_at=hit.fact.updated_at,
                content_excerpt=excerpt(hit.fact.content.strip(), 240)[0],
                score=hit.score,
                cited=hit.fact.id in delivered_set,
            )
            for hit in ordered[:max_sources]
        ],
        cited_fact_ids=delivered_ids,
        warnings=[*outcome.warnings, *_delivery_warnings(outcome.delivered)],
        usage=UsageSummary(
            llm_calls=outcome.llm_calls,
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            cache_hit_ratio=(
                cached_tokens / input_tokens
                if input_tokens and cached_tokens is not None
                else None
            ),
        ),
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
    return store.search_facts(query, project, limit=limit)


async def _log_recall(
    store: FactStore | AsyncFactStore,
    record: RecallRecord,
) -> None:
    if isinstance(store, AsyncFactStore):
        await store.log_recall(record)
    else:
        store.log_recall(record)


__all__ = [
    "ANSWER_SYSTEM",
    "MIN_COVERAGE",
    "RELATIVE_CUTOFF",
    "RecallMode",
    "ZERO_HIT_MAX_CANDIDATES",
    "recall",
    "recall_with_provenance",
]
