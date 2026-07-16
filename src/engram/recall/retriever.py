"""Retriever — tiered agentic memory retrieval with quality observability."""

import asyncio
import logging
import re
import time

from engram.core.config import ensure_openai_api_key, get_settings
from engram.core.interfaces import EnvelopeWarning, WarningCode
from engram.llm import Completion, complete_with_usage
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
from engram.storage.store import AsyncFactStore, FactStore, format_facts_for_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search system prompts
# ---------------------------------------------------------------------------

SINGLE_AGENT_SYSTEM = """You are a memory search and synthesis agent. Given a query and stored facts,
find the most relevant facts and produce a clear, concise answer.

1. Identify facts that directly answer the query
2. Note any useful context or background
3. Flag contradictions or stale information
4. Cite fact IDs for traceability — copy each ID exactly (12 hex characters) from its
   `id:` marker; never invent, merge, or truncate an ID. Omit a citation rather than guess one.

Write naturally, as if briefing someone. Keep it concise but complete.

At the very end of your answer, on a new line, add a quality rating in the format:
[quality: high|medium|low|none]"""

TIER2_SINGLE_SYSTEM = """You are a memory search and synthesis agent. Given a broad query and stored facts,
reason through the evidence from three perspectives before answering:

- Direct: facts that directly answer the query
- Contextual: useful background, related preferences, and connected facts
- Temporal: current vs. stale information, supersessions, contradictions, and timelines

Do the perspective work internally, then write only the final concise answer.
Cite fact IDs for traceability — copy each ID exactly (12 hex characters) from its `id:`
marker; never invent, merge, or truncate an ID. Omit a citation rather than guess one.
Prefer newer, non-expired facts when evidence conflicts.
If the evidence is weak or contradictory, say so instead of guessing.

At the very end of your answer, on a new line, add a quality rating in the format:
[quality: high|medium|low|none]"""

# ---------------------------------------------------------------------------
# Tier thresholds
# ---------------------------------------------------------------------------

# Minimum score to count as a "real" match (filters out recency/confidence noise)
RELEVANCE_FLOOR = 5

# Tier 0: trivial — very few matches, answer is obvious.
TIER_0_MAX_RELEVANT = 4
TIER_0_MIN_SCORE = 10

# Tier 1: focused — moderate matches with a concentrated top cluster.
TIER_1_MAX_RELEVANT = 20
TIER_1_MIN_TOP_SCORE = 8
TIER_1_MIN_GAP = 1.5  # top score must be ≥1.5x the 5th score

# Zero-hit escalation: when no fact clears RELEVANCE_FLOOR but the corpus is
# non-empty, the tier-1 LLM search runs over the top raw-scored candidates
# instead of answering "no relevant memories" without looking. Bounded so a
# large corpus doesn't blow up the prompt.
ZERO_HIT_MAX_CANDIDATES = 50

# v3 = the v2 thresholds plus zero-hit escalation, so recall_stats can split
# tier mixes recorded before and after escalation shipped.
SELECTOR_VERSION = "v3"

# Fact-ID extraction from LLM responses. Fact IDs are 12-hex strings emitted in
# `(id: <hex>)` form by ``format_facts_for_llm``.
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


def _tier_decision(
    tier: int,
    *,
    relevant_count: int,
    top_score: int | None,
    gap_ratio: float | None,
    cap_applied: bool = False,
) -> TierDecision:
    return TierDecision(
        tier=tier,
        rules=SELECTOR_VERSION,
        relevant_count=relevant_count,
        top_score=top_score,
        gap_ratio=gap_ratio,
        cap_applied=cap_applied,
    )


def _relevance_gap(relevant_scores: list[int]) -> float:
    top = relevant_scores[0]
    comparison = relevant_scores[-1] if len(relevant_scores) < 5 else relevant_scores[4]
    return top / comparison if comparison > 0 else float("inf")


def _reported_gap(gap: float) -> float | None:
    return gap if gap != float("inf") else None


def _select_tier_with_decision(
    scored_facts: list[tuple[int, Fact]],
    min_prefilter_for_tier2: int = 0,
) -> TierDecision:
    """Select retrieval tier and return the decision with its inputs.

    See :func:`_select_tier` for the tier semantics. This wrapper exists so
    provenance can capture the threshold inputs that drove the choice.
    """
    if not scored_facts:
        return _tier_decision(
            0,
            relevant_count=0,
            top_score=None,
            gap_ratio=None,
        )

    relevant = [s for s, _ in scored_facts if s >= RELEVANCE_FLOOR]

    if not relevant:
        return _tier_decision(
            0,
            relevant_count=0,
            top_score=scored_facts[0][0] if scored_facts else None,
            gap_ratio=None,
        )

    top = relevant[0]
    gap = _relevance_gap(relevant)

    if len(relevant) <= TIER_0_MAX_RELEVANT and top >= TIER_0_MIN_SCORE:
        return _tier_decision(
            0,
            relevant_count=len(relevant),
            top_score=top,
            gap_ratio=_reported_gap(gap),
        )

    if top >= 15 and gap >= TIER_1_MIN_GAP:
        return _tier_decision(
            1,
            relevant_count=len(relevant),
            top_score=top,
            gap_ratio=_reported_gap(gap),
        )

    chosen = 2
    cap_applied = False
    if min_prefilter_for_tier2 > 0:
        positive = sum(1 for s, _ in scored_facts if s > 0)
        if positive < min_prefilter_for_tier2:
            chosen = 1
            cap_applied = True

    return _tier_decision(
        chosen,
        relevant_count=len(relevant),
        top_score=top,
        gap_ratio=_reported_gap(gap),
        cap_applied=cap_applied,
    )


def _llm_available() -> bool:
    """True when an LLM provider key is set or loadable from the key cache."""
    return ensure_openai_api_key() is not None


def _escalate_zero_hit(
    decision: TierDecision,
    scored_facts: list[tuple[int, Fact]],
) -> TierDecision:
    """Escalate a zero-relevant tier-0 decision to the tier-1 LLM search.

    A paraphrased or synonym query can share zero tokens with a stored fact,
    so the keyword prefilter alone cannot rule out a match. When the corpus
    is non-empty and an LLM key is configured, the top raw-scored candidates
    (even below ``RELEVANCE_FLOOR``) go to tier 1 instead of hard-stopping at
    "no relevant memories". Without a key the decision is returned unchanged,
    keeping recall zero-LLM and crash-free.
    """
    if decision.tier != 0 or decision.relevant_count > 0 or not scored_facts:
        return decision
    if not _llm_available():
        return decision
    return TierDecision(
        tier=1,
        rules=SELECTOR_VERSION,
        relevant_count=0,
        top_score=scored_facts[0][0],
        gap_ratio=None,
        zero_hit_escalation=True,
    )


def _select_tier(
    scored_facts: list[tuple[int, Fact]],
    min_prefilter_for_tier2: int = 0,
) -> int:
    """Select retrieval tier based on score distribution shape.

    Uses both the count of relevant matches AND the gap ratio (how much
    the top results stand out from the pack) to decide:

    Tier 0: Few matches with a clear standout → direct return, no LLM.
    Tier 1: Focused matches with concentrated signal → one LLM call.
    Tier 2: Many matches or flat distribution → one broad LLM call.

    Zero relevant matches → Tier 0 (direct) here; ``recall_with_provenance``
    escalates that case to tier 1 via :func:`_escalate_zero_hit` when the
    corpus is non-empty and an LLM key is configured.

    When ``min_prefilter_for_tier2 > 0``, an additional cap applies: queries
    whose prefilter produced fewer than ``min_prefilter_for_tier2``
    strictly-positive-scored facts are downgraded from tier-2 to tier-1.
    Tier-0 decisions are never touched by the cap.
    """
    return _select_tier_with_decision(
        scored_facts, min_prefilter_for_tier2=min_prefilter_for_tier2
    ).tier


def _format_direct(scored_facts: list[tuple[int, Fact]], query: str) -> str:
    """Tier 0: Format high-confidence facts directly without LLM."""
    if not scored_facts:
        return "No memories stored yet. Use `remember` to add some."
    facts = [f for score, f in scored_facts if score >= RELEVANCE_FLOOR]
    if not facts:
        return "No relevant memories found for this query."

    lines = []
    for fact in facts[:10]:
        meta = f"[{fact.category.value}]"
        if fact.project:
            meta += f" [{fact.project}]"
        lines.append(f"- {meta} {fact.content} (id: {fact.id})")

    return "\n".join(lines)


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
# Provenance assembly helpers
# ---------------------------------------------------------------------------


def _content_excerpt(text: str, limit: int = 240) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _build_source_summaries(
    scored_facts: list[tuple[int, Fact]],
    cited_ids: set[str],
    max_sources: int = DEFAULT_MAX_SOURCES,
) -> list[SourceSummary]:
    """Build compact per-source summaries for provenance.

    Caps at ``max_sources``; cited facts always come first so we never drop
    citations to the truncation cap.
    """
    cited: list[SourceSummary] = []
    rest: list[SourceSummary] = []
    for score, fact in scored_facts:
        summary = SourceSummary(
            id=fact.id,
            project=fact.project,
            category=fact.category.value,
            confidence=fact.confidence,
            updated_at=fact.updated_at,
            content_excerpt=_content_excerpt(fact.content),
            score=score,
            cited=fact.id in cited_ids,
            superseded_by=None,
            stale=fact.stale,
            forgotten=fact.confidence == 0.0,
        )
        if summary.cited:
            cited.append(summary)
        else:
            rest.append(summary)
    return (cited + rest)[:max_sources]


def _build_warnings(
    scored_facts: list[tuple[int, Fact]],
    all_facts: list[Fact],
    cited_ids: set[str],
) -> list[EnvelopeWarning]:
    """Build provenance warnings from the prefilter and full fact set.

    Detects:
    - ``stale_fact``: facts marked stale that still appeared in matches.
    - ``superseded_fact``: matched facts whose IDs are in another active fact's
      ``supersedes`` chain.
    - ``forgotten_fact``: matched facts whose confidence is 0.
    - ``conflicting_facts``: two cited facts in the same project+category.
    """
    superseded_by = _superseded_by_map(all_facts)
    stale_ids, superseded_ids, forgotten_ids = _warning_id_groups(
        scored_facts,
        superseded_by,
    )
    warnings = _state_warnings(
        stale_ids,
        superseded_ids,
        forgotten_ids,
        superseded_by,
    )
    warnings.extend(_conflict_warnings(scored_facts, cited_ids))
    return warnings


def _superseded_by_map(facts: list[Fact]) -> dict[str, str]:
    return {fact.supersedes: fact.id for fact in facts if fact.supersedes}


def _warning_id_groups(
    scored_facts: list[tuple[int, Fact]],
    superseded_by: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    stale_ids: list[str] = []
    superseded_ids: list[str] = []
    forgotten_ids: list[str] = []
    for _, fact in scored_facts:
        if fact.stale:
            stale_ids.append(fact.id)
        if fact.id in superseded_by:
            superseded_ids.append(fact.id)
        if fact.supersedes:
            superseded_ids.append(fact.supersedes)
        if fact.confidence == 0.0:
            forgotten_ids.append(fact.id)
    return stale_ids, superseded_ids, forgotten_ids


def _state_warnings(
    stale_ids: list[str],
    superseded_ids: list[str],
    forgotten_ids: list[str],
    superseded_by: dict[str, str],
) -> list[EnvelopeWarning]:
    warnings: list[EnvelopeWarning] = []
    if stale_ids:
        warnings.append(
            EnvelopeWarning(
                code=WarningCode.stale_fact,
                message="Stale facts matched the query and were excluded from active recall.",
                ids=sorted(set(stale_ids)),
            )
        )
    if superseded_ids:
        warnings.append(
            EnvelopeWarning(
                code=WarningCode.superseded_fact,
                message="One or more matched facts have a newer active replacement.",
                ids=sorted(set(superseded_ids)),
                details={"superseded_by": superseded_by},
            )
        )
    if forgotten_ids:
        warnings.append(
            EnvelopeWarning(
                code=WarningCode.forgotten_fact,
                message="Forgotten facts appeared in the prefilter and were not used.",
                ids=sorted(set(forgotten_ids)),
            )
        )
    return warnings


def _conflict_warnings(
    scored_facts: list[tuple[int, Fact]],
    cited_ids: set[str],
) -> list[EnvelopeWarning]:
    cited_facts = [fact for _, fact in scored_facts if fact.id in cited_ids]
    buckets: dict[tuple[str | None, str], list[str]] = {}
    for fact in cited_facts:
        buckets.setdefault((fact.project, fact.category.value), []).append(fact.id)
    warnings: list[EnvelopeWarning] = []
    for (project, category), ids in buckets.items():
        if len(ids) >= 2:
            warnings.append(
                EnvelopeWarning(
                    code=WarningCode.conflicting_facts,
                    message=(
                        f"Multiple active facts in {project or '(global)'}/"
                        f"{category} were cited; verify they agree."
                    ),
                    ids=sorted(ids),
                    details={"project": project, "category": category},
                )
            )
    return warnings


def _usage_from_totals(totals: dict[str, int | None]) -> UsageSummary:
    input_tokens = totals.get("input_tokens")
    cached_tokens = totals.get("cached_tokens")
    ratio: float | None = None
    if input_tokens is not None and input_tokens > 0 and cached_tokens is not None:
        ratio = cached_tokens / input_tokens
    return UsageSummary(
        llm_calls=totals.get("llm_calls"),
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        output_tokens=totals.get("output_tokens"),
        cache_hit_ratio=ratio,
    )


def _completion_trace(
    *,
    name: str,
    system: str,
    prompt: str,
    completion: Completion,
    elapsed_ms: float,
    excerpt_chars: int,
    output_chars: int,
) -> tuple[LLMCallTrace, bool]:
    prompt_excerpt, prompt_truncated = excerpt(prompt, excerpt_chars)
    output_excerpt, output_truncated = excerpt(completion.text, output_chars)
    return (
        LLMCallTrace(
            name=name,
            system_excerpt=excerpt(system, excerpt_chars)[0],
            prompt_excerpt=prompt_excerpt,
            output_excerpt=output_excerpt,
            elapsed_ms=elapsed_ms,
            input_tokens=completion.input_tokens,
            cached_tokens=completion.cached_tokens,
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
) -> str:
    """Tiered agentic retrieval — text answer.

    This is the existing concise-answer entry point. Returns plain text so
    existing MCP/CLI callers see no behavior change.
    """
    answer, _quality, _provenance, _trace = await recall_with_provenance(
        query, project=project, store=store, with_trace=False
    )
    return answer


async def recall_with_provenance(
    query: str,
    project: str | None = None,
    store: FactStore | AsyncFactStore | None = None,
    *,
    with_trace: bool = False,
    verbose_trace: bool = False,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_prefilter_matches: int = DEFAULT_MAX_PREFILTER_MATCHES,
) -> tuple[str, str, RecallProvenance, RecallTrace | None]:
    """Tiered recall returning answer plus structured provenance.

    Returns ``(answer, quality, provenance, trace_or_none)``. Tiers 1 and 2
    each issue one LLM call; provenance is assembled from that call's output
    and from the deterministic prefilter, so enabling provenance does not add
    model work.

    ``with_trace=True`` populates the ``RecallTrace`` with bounded prompt
    and output excerpts. ``verbose_trace=True`` widens the per-field char
    limits for callers that opt in.
    """
    store = store or FactStore()
    settings = get_settings()
    t0 = time.monotonic()

    scored_facts = await _prefilter_facts(
        store,
        query=query,
        project=project,
        limit=settings.max_facts_per_agent,
    )

    decision = _select_tier_with_decision(
        scored_facts,
        min_prefilter_for_tier2=settings.tier2_min_prefilter_count,
    )
    decision = _escalate_zero_hit(decision, scored_facts)
    tier = decision.tier
    prefilter_count = len([s for s, _ in scored_facts if s > 0])
    llm_facts = (
        scored_facts[:ZERO_HIT_MAX_CANDIDATES]
        if decision.zero_hit_escalation
        else scored_facts
    )

    excerpt_chars = (
        DEFAULT_PROMPT_EXCERPT_CHARS * 4
        if verbose_trace
        else DEFAULT_PROMPT_EXCERPT_CHARS
    )
    output_chars = (
        DEFAULT_OUTPUT_EXCERPT_CHARS * 4
        if verbose_trace
        else DEFAULT_OUTPUT_EXCERPT_CHARS
    )

    (
        answer,
        quality,
        usage_totals,
        cited_ids,
        call_traces,
        truncated_any,
    ) = await _run_recall_tier(
        tier,
        llm_facts,
        query,
        settings,
        prefilter_count=prefilter_count,
        with_trace=with_trace,
        excerpt_chars=excerpt_chars,
        output_chars=output_chars,
    )

    latency_ms = (time.monotonic() - t0) * 1000

    provenance = await _build_recall_provenance(
        store,
        query=query,
        project=project,
        tier=tier,
        quality=quality,
        decision=decision,
        scored_facts=scored_facts,
        cited_ids=cited_ids,
        usage_totals=usage_totals,
        latency_ms=latency_ms,
        prefilter_count=prefilter_count,
        max_prefilter_matches=max_prefilter_matches,
        max_sources=max_sources,
    )
    trace_obj = _trace_for_recall(
        with_trace=with_trace,
        provenance=provenance,
        call_traces=call_traces,
        excerpt_chars=excerpt_chars,
        truncated_any=truncated_any,
        verbose_trace=verbose_trace,
    )

    await _record_recall_observation(
        store,
        query=query,
        project=project,
        tier=tier,
        prefilter_count=prefilter_count,
        latency_ms=latency_ms,
        quality=quality,
        usage_totals=usage_totals,
    )

    logger.info(
        "recall tier=%d prefilter=%d latency=%.0fms quality=%s calls=%s input=%s cached=%s",
        tier,
        prefilter_count,
        latency_ms,
        quality,
        usage_totals.get("llm_calls"),
        usage_totals.get("input_tokens"),
        usage_totals.get("cached_tokens"),
    )
    return answer, quality, provenance, trace_obj


async def _run_recall_tier(
    tier: int,
    scored_facts: list[tuple[int, Fact]],
    query: str,
    settings,
    *,
    prefilter_count: int,
    with_trace: bool,
    excerpt_chars: int,
    output_chars: int,
) -> tuple[str, str, dict[str, int | None], list[str], list[LLMCallTrace], bool]:
    if tier == 0:
        usage_totals: dict[str, int | None] = {
            "llm_calls": 0,
            "input_tokens": None,
            "cached_tokens": None,
            "output_tokens": None,
        }
        answer = _format_direct(scored_facts, query)
        if any(score >= RELEVANCE_FLOOR for score, _ in scored_facts):
            quality = "high"
            cited_ids = [
                fact.id for score, fact in scored_facts if score >= RELEVANCE_FLOOR
            ][:10]
        elif prefilter_count > 0:
            quality = "low"
            cited_ids = []
        else:
            quality = "none"
            cited_ids = []
        return answer, quality, usage_totals, cited_ids, [], False

    if tier == 1:
        return await _single_call_recall(
            scored_facts,
            query,
            settings,
            system=SINGLE_AGENT_SYSTEM,
            trace_name="single_agent",
            with_trace=with_trace,
            excerpt_chars=excerpt_chars,
            output_chars=output_chars,
        )

    return await _single_call_recall(
        scored_facts,
        query,
        settings,
        system=TIER2_SINGLE_SYSTEM,
        trace_name="tier2_single",
        prompt_suffix="\n\nAnswer the query using the stored facts.",
        with_trace=with_trace,
        excerpt_chars=excerpt_chars,
        output_chars=output_chars,
    )


async def _build_recall_provenance(
    store: FactStore | AsyncFactStore,
    *,
    query: str,
    project: str | None,
    tier: int,
    quality: str,
    decision: TierDecision,
    scored_facts: list[tuple[int, Fact]],
    cited_ids: list[str],
    usage_totals: dict[str, int | None],
    latency_ms: float,
    prefilter_count: int,
    max_prefilter_matches: int,
    max_sources: int,
) -> RecallProvenance:
    cited_set = set(cited_ids)
    prefilter_matches = [
        PrefilterMatch(id=fact.id, score=score, above_floor=score >= RELEVANCE_FLOOR)
        for score, fact in scored_facts[:max_prefilter_matches]
    ]
    all_facts = await _load_all_facts(store)
    return RecallProvenance(
        query=query,
        project=project,
        tier=tier,
        quality=quality,
        selected_decision=decision,
        prefilter_count=prefilter_count,
        prefilter_matches=prefilter_matches,
        source_fact_ids=[match.id for match in prefilter_matches if match.above_floor],
        sources=_build_source_summaries(
            scored_facts,
            cited_set,
            max_sources=max_sources,
        ),
        cited_fact_ids=list(cited_ids),
        warnings=_build_warnings(scored_facts, all_facts, cited_set),
        usage=_usage_from_totals(usage_totals),
        latency_ms=latency_ms,
    )


def _trace_for_recall(
    *,
    with_trace: bool,
    provenance: RecallProvenance,
    call_traces: list[LLMCallTrace],
    excerpt_chars: int,
    truncated_any: bool,
    verbose_trace: bool,
) -> RecallTrace | None:
    if not with_trace:
        return None
    return RecallTrace(
        provenance=provenance,
        calls=call_traces,
        excerpt_chars=excerpt_chars,
        truncated=truncated_any,
        verbose=verbose_trace,
    )


async def _record_recall_observation(
    store: FactStore | AsyncFactStore,
    *,
    query: str,
    project: str | None,
    tier: int,
    prefilter_count: int,
    latency_ms: float,
    quality: str,
    usage_totals: dict[str, int | None],
) -> None:
    try:
        await _log_recall(
            store,
            RecallRecord(
                query=query,
                project=project,
                tier=tier,
                prefilter_count=prefilter_count,
                latency_ms=latency_ms,
                quality=quality,
                llm_calls=usage_totals.get("llm_calls"),
                input_tokens=usage_totals.get("input_tokens"),
                cached_tokens=usage_totals.get("cached_tokens"),
                selector_version=SELECTOR_VERSION,
            ),
        )
    except Exception:
        logger.debug("Failed to log recall record", exc_info=True)


async def _prefilter_facts(
    store: FactStore | AsyncFactStore,
    query: str,
    project: str | None,
    limit: int,
) -> list[tuple[int, Fact]]:
    if isinstance(store, AsyncFactStore):
        return await store.prefilter_facts(query=query, project=project, limit=limit)
    return store.prefilter_facts(query=query, project=project, limit=limit)


async def _load_all_facts(store: FactStore | AsyncFactStore) -> list[Fact]:
    if isinstance(store, AsyncFactStore):
        return await store.load_facts()
    return store.load_facts()


async def _log_recall(
    store: FactStore | AsyncFactStore,
    record: RecallRecord,
) -> None:
    if isinstance(store, AsyncFactStore):
        await store.log_recall(record)
    else:
        store.log_recall(record)


def _accumulate(totals: dict[str, int | None], completion: Completion) -> None:
    """Fold a Completion's usage into the running totals dict in-place."""
    totals["llm_calls"] = (totals.get("llm_calls") or 0) + 1
    if completion.input_tokens is not None:
        totals["input_tokens"] = (
            totals.get("input_tokens") or 0
        ) + completion.input_tokens
    if completion.cached_tokens is not None:
        totals["cached_tokens"] = (
            totals.get("cached_tokens") or 0
        ) + completion.cached_tokens


async def _single_call_recall(
    scored_facts: list[tuple[int, Fact]],
    query: str,
    settings,
    *,
    system: str,
    trace_name: str,
    prompt_suffix: str = "",
    with_trace: bool = False,
    excerpt_chars: int = DEFAULT_PROMPT_EXCERPT_CHARS,
    output_chars: int = DEFAULT_OUTPUT_EXCERPT_CHARS,
) -> tuple[str, str, dict[str, int | None], list[str], list[LLMCallTrace], bool]:
    """One LLM call over the prefiltered facts. Handles tier 1 and tier 2.

    ``system`` and ``prompt_suffix`` differ per tier; ``trace_name`` labels the
    call in the trace. Since recall makes one call, there's no prompt prefix to
    cache across calls.
    """
    facts = [f for _, f in scored_facts]
    facts_text = format_facts_for_llm(facts)
    prompt = f"QUERY: {query}\n\nSTORED FACTS:\n{facts_text}{prompt_suffix}"

    totals: dict[str, int | None] = {
        "llm_calls": 0,
        "input_tokens": None,
        "cached_tokens": None,
        "output_tokens": None,
    }
    t_call = time.monotonic()
    completion = await asyncio.wait_for(
        complete_with_usage(prompt=prompt, system=system),
        timeout=settings.retrieval_timeout,
    )
    elapsed_ms = (time.monotonic() - t_call) * 1000
    _accumulate(totals, completion)
    answer, quality = _extract_quality(completion.text)

    candidate_ids = {f.id for f in facts}
    cited_ids = _extract_cited_ids(completion.text, candidate_ids)
    answer = _scrub_invalid_citations(answer, candidate_ids)

    traces: list[LLMCallTrace] = []
    truncated_any = False
    if with_trace:
        trace, truncated_any = _completion_trace(
            name=trace_name,
            system=system,
            prompt=prompt,
            completion=completion,
            elapsed_ms=elapsed_ms,
            excerpt_chars=excerpt_chars,
            output_chars=output_chars,
        )
        traces.append(trace)
    return answer, quality, totals, cited_ids, traces, truncated_any


__all__ = [
    "recall",
    "recall_with_provenance",
    "_extract_quality",
    "_extract_cited_ids",
    "_scrub_invalid_citations",
    "_format_direct",
    "_select_tier",
    "_select_tier_with_decision",
    "TIER2_SINGLE_SYSTEM",
]
