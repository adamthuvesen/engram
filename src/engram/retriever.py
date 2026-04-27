"""Retriever — tiered agentic memory retrieval with quality observability."""

import asyncio
import logging
import re
import time

from engram.config import get_settings
from engram.interfaces import EnvelopeWarning, WarningCode
from engram.llm import Completion, complete, complete_with_usage
from engram.models import Fact, RecallRecord
from engram.provenance import (
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
from engram.store import AsyncFactStore, FactStore, format_facts_for_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search system prompts
# ---------------------------------------------------------------------------

MULTI_LENS_SYSTEM = """You are a multi-lens memory search agent. You receive a query and a list of
stored facts. Reason about the facts from three distinct perspectives and return your findings
in one response with three clearly-labelled sections. Use these exact section headings, in this
order, each on its own line:

## DIRECT
Facts that directly answer the query. Be precise — only include facts that are clearly relevant.

## CONTEXTUAL
Facts that add useful background, connections between facts, or related preferences/patterns.

## TEMPORAL
Time- and state-aware observations: current vs. outdated, supersessions or contradictions,
timeline of relevant events, anything that looks stale or expired.

Under every heading, produce a numbered list. For each item include the fact ID (copy it from the
`id:` marker) and one short line of reasoning. If a section has nothing to report, still emit the
heading followed by a single line "(none)"."""

SYNTHESIS_SYSTEM = """You are a memory synthesis agent. You receive findings from specialized
search agents who searched a personal knowledge base.

Your job is to produce a clear, concise answer to the original query by:
1. Merging relevant findings from all agents
2. Resolving any contradictions (prefer newer/higher-confidence facts)
3. Flagging any uncertainty or stale information
4. Citing fact IDs for traceability

Format your response as a direct answer, not as a list of findings. Write naturally,
as if briefing someone. Keep it concise but complete.

Always prefer newer, non-expired facts when evidence conflicts.
Include source references when they are available.
If the evidence is weak or contradictory, say so instead of guessing.

At the very end of your answer, on a new line, add a quality rating in the format:
[quality: high|medium|low|none]
- high: strong, unambiguous evidence directly answers the query
- medium: partial evidence or some inference required
- low: weak or conflicting evidence
- none: no relevant evidence found"""

SINGLE_AGENT_SYSTEM = """You are a memory search and synthesis agent. Given a query and stored facts,
find the most relevant facts and produce a clear, concise answer.

1. Identify facts that directly answer the query
2. Note any useful context or background
3. Flag contradictions or stale information
4. Cite fact IDs for traceability

Write naturally, as if briefing someone. Keep it concise but complete.

At the very end of your answer, on a new line, add a quality rating in the format:
[quality: high|medium|low|none]"""

# ---------------------------------------------------------------------------
# Tier thresholds
# ---------------------------------------------------------------------------

# Minimum score to count as a "real" match (filters out recency/confidence noise)
RELEVANCE_FLOOR = 5

# Tier 0: trivial — very few matches, answer is obvious.
TIER_0_MAX_RELEVANT = 3
TIER_0_MIN_SCORE = 10

# Tier 1: focused — moderate matches with a concentrated top cluster.
TIER_1_MAX_RELEVANT = 20
TIER_1_MIN_TOP_SCORE = 8
TIER_1_MIN_GAP = 1.5  # top score must be ≥1.5x the 5th score

_unknown_tier_rules_warned: set[str] = set()

# Multi-lens response heading parser.
_MULTILENS_HEADING_RE = re.compile(
    r"^\s*##\s+(DIRECT|CONTEXTUAL|TEMPORAL)\s*$", re.MULTILINE
)

# Fact-ID extraction from LLM responses. Fact IDs are 12-hex strings emitted in
# `(id: <hex>)` form by ``format_facts_for_llm``.
_CITED_ID_RE = re.compile(r"\b([0-9a-f]{12})\b")


def _select_tier_with_decision(
    scored_facts: list[tuple[int, Fact]],
    rules: str = "v1",
    min_prefilter_for_tier2: int = 0,
) -> TierDecision:
    """Select retrieval tier and return the decision with its inputs.

    See :func:`_select_tier` for the tier semantics. This wrapper exists so
    provenance can capture the threshold inputs that drove the choice.
    """
    if not scored_facts:
        return TierDecision(
            tier=0,
            rules=rules,
            relevant_count=0,
            top_score=None,
            gap_ratio=None,
            cap_applied=False,
        )

    relevant = [s for s, _ in scored_facts if s >= RELEVANCE_FLOOR]

    if not relevant:
        return TierDecision(
            tier=0,
            rules=rules,
            relevant_count=0,
            top_score=scored_facts[0][0] if scored_facts else None,
            gap_ratio=None,
            cap_applied=False,
        )

    top = relevant[0]
    if len(relevant) < 5:
        gap = top / relevant[-1] if relevant[-1] > 0 else float("inf")
    else:
        gap = top / relevant[4] if relevant[4] > 0 else float("inf")

    if len(relevant) <= TIER_0_MAX_RELEVANT and top >= TIER_0_MIN_SCORE:
        return TierDecision(
            tier=0,
            rules=rules,
            relevant_count=len(relevant),
            top_score=top,
            gap_ratio=gap if gap != float("inf") else None,
            cap_applied=False,
        )

    if top >= 15 and gap >= TIER_1_MIN_GAP:
        return TierDecision(
            tier=1,
            rules=rules,
            relevant_count=len(relevant),
            top_score=top,
            gap_ratio=gap if gap != float("inf") else None,
            cap_applied=False,
        )

    chosen = 2
    cap_applied = False
    if rules == "v2" and min_prefilter_for_tier2 > 0:
        positive = sum(1 for s, _ in scored_facts if s > 0)
        if positive < min_prefilter_for_tier2:
            chosen = 1
            cap_applied = True

    return TierDecision(
        tier=chosen,
        rules=rules,
        relevant_count=len(relevant),
        top_score=top,
        gap_ratio=gap if gap != float("inf") else None,
        cap_applied=cap_applied,
    )


def _select_tier(
    scored_facts: list[tuple[int, Fact]],
    rules: str = "v1",
    min_prefilter_for_tier2: int = 0,
) -> int:
    """Select retrieval tier based on score distribution shape.

    Uses both the count of relevant matches AND the gap ratio (how much
    the top results stand out from the pack) to decide:

    Tier 0: Few matches with a clear standout → direct return, no LLM.
    Tier 1: Focused matches with concentrated signal → single-agent.
    Tier 2: Many matches or flat distribution → multi-lens search and synthesis.

    Zero relevant matches → Tier 0 (direct).

    When ``rules == "v2"`` and ``min_prefilter_for_tier2 > 0``, an additional cap
    applies: queries whose prefilter produced fewer than ``min_prefilter_for_tier2``
    strictly-positive-scored facts are downgraded from tier-2 to tier-1. Tier-0
    decisions are never touched by the cap.
    """
    return _select_tier_with_decision(
        scored_facts, rules=rules, min_prefilter_for_tier2=min_prefilter_for_tier2
    ).tier


# TODO(small-corpus-tier-cap-cleanup): after one release of clean v2 data in
# recall_log, remove the v1 branch, the `rules` parameter on `_select_tier`,
# and the `ENGRAM_TIER_RULES` setting. Keep the configurable threshold.
def _resolve_tier_rules(raw: str) -> str:
    """Map the tier_rules setting to a supported value, warning once on unknowns."""
    if raw in ("v1", "v2"):
        return raw
    if raw not in _unknown_tier_rules_warned:
        logger.warning("Unknown ENGRAM_TIER_RULES=%r; falling back to 'v2'", raw)
        _unknown_tier_rules_warned.add(raw)
    return "v2"


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


def _build_prefix(facts_text: str) -> str:
    """Build the stable, cacheable prompt prefix shared across tier-2 LLM calls.

    Layout matters: the prefix MUST come first in the final prompt, before any
    query-specific or perspective-specific text. OpenAI's implicit prompt cache
    and Anthropic's `cache_control` both benefit from a long, byte-identical
    leading run of tokens.
    """
    return f"STORED FACTS:\n{facts_text}\n\n"


def _parse_multilens_sections(text: str) -> dict[str, str]:
    """Split a multi-lens response into direct/contextual/temporal sections.

    Missing sections return empty strings. Malformed responses (no headings)
    put everything in `direct` so the synthesis step still has something to
    chew on.
    """
    sections = {"DIRECT": "", "CONTEXTUAL": "", "TEMPORAL": ""}
    matches = list(_MULTILENS_HEADING_RE.finditer(text))
    if not matches:
        logger.debug(
            "multilens response missing expected headings; using raw body as DIRECT"
        )
        sections["DIRECT"] = text.strip()
        return {k.lower(): v for k, v in sections.items()}

    for i, match in enumerate(matches):
        heading = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[heading] = text[start:end].strip()
    return {k.lower(): v for k, v in sections.items()}


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
            stale=getattr(fact, "stale", False),
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
    warnings: list[EnvelopeWarning] = []

    # Build lookup of supersession edges across all facts (active or not).
    superseded_by: dict[str, str] = {}
    for fact in all_facts:
        if fact.supersedes:
            superseded_by[fact.supersedes] = fact.id

    stale_ids: list[str] = []
    superseded_ids: list[str] = []
    forgotten_ids: list[str] = []
    for _, fact in scored_facts:
        if getattr(fact, "stale", False):
            stale_ids.append(fact.id)
        if fact.id in superseded_by:
            superseded_ids.append(fact.id)
        if fact.confidence == 0.0:
            forgotten_ids.append(fact.id)

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

    # Conflict detection: two or more cited facts share project+category.
    cited_facts = [f for _, f in scored_facts if f.id in cited_ids]
    buckets: dict[tuple[str | None, str], list[str]] = {}
    for fact in cited_facts:
        buckets.setdefault((fact.project, fact.category.value), []).append(fact.id)
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

    Returns ``(answer, quality, provenance, trace_or_none)``. The default
    tier-2 path still issues exactly two LLM calls (multi-lens search +
    synthesis); provenance is assembled from those calls' outputs and from
    the deterministic prefilter, so enabling provenance does not add model
    work.

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

    tier_rules = _resolve_tier_rules(settings.tier_rules)
    decision = _select_tier_with_decision(
        scored_facts,
        rules=tier_rules,
        min_prefilter_for_tier2=settings.tier2_min_prefilter_count,
    )
    tier = decision.tier
    prefilter_count = len([s for s, _ in scored_facts if s > 0])
    usage_totals: dict[str, int | None] = {
        "llm_calls": None,
        "input_tokens": None,
        "cached_tokens": None,
        "output_tokens": None,
    }

    call_traces: list[LLMCallTrace] = []
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

    cited_ids: list[str] = []

    truncated_any = False
    if tier == 0:
        answer = _format_direct(scored_facts, query)
        usage_totals["llm_calls"] = 0
        if any(s >= RELEVANCE_FLOOR for s, _ in scored_facts):
            quality = "high"
            cited_ids = [f.id for s, f in scored_facts if s >= RELEVANCE_FLOOR][:10]
        elif prefilter_count > 0:
            quality = "low"
        else:
            quality = "none"
    elif tier == 1:
        (
            answer,
            quality,
            usage_totals,
            cited_ids,
            call_traces,
            truncated_any,
        ) = await _single_agent_recall(
            scored_facts,
            query,
            settings,
            with_trace=with_trace,
            excerpt_chars=excerpt_chars,
            output_chars=output_chars,
        )
    else:
        (
            answer,
            quality,
            usage_totals,
            cited_ids,
            call_traces,
            truncated_any,
        ) = await _multilens_recall(
            scored_facts,
            query,
            settings,
            with_trace=with_trace,
            excerpt_chars=excerpt_chars,
            output_chars=output_chars,
        )

    latency_ms = (time.monotonic() - t0) * 1000

    # Build provenance from artifacts already computed.
    cited_set = set(cited_ids)
    prefilter_matches = [
        PrefilterMatch(id=fact.id, score=score, above_floor=score >= RELEVANCE_FLOOR)
        for score, fact in scored_facts[:max_prefilter_matches]
    ]
    sources = _build_source_summaries(scored_facts, cited_set, max_sources=max_sources)
    all_facts = await _load_all_facts(store)
    warnings = _build_warnings(scored_facts, all_facts, cited_set)
    usage = _usage_from_totals(usage_totals)

    provenance = RecallProvenance(
        query=query,
        project=project,
        tier=tier,
        quality=quality,
        selected_decision=decision,
        prefilter_count=prefilter_count,
        prefilter_matches=prefilter_matches,
        source_fact_ids=[m.id for m in prefilter_matches if m.above_floor],
        sources=sources,
        cited_fact_ids=list(cited_ids),
        warnings=warnings,
        usage=usage,
        latency_ms=latency_ms,
    )

    trace_obj: RecallTrace | None = None
    if with_trace:
        trace_obj = RecallTrace(
            provenance=provenance,
            calls=call_traces,
            excerpt_chars=excerpt_chars,
            truncated=truncated_any,
            verbose=verbose_trace,
        )

    # Log for observability (unchanged contract)
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
                selector_version=tier_rules,
            ),
        )
    except Exception:
        logger.debug("Failed to log recall record", exc_info=True)

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


async def _single_agent_recall(
    scored_facts: list[tuple[int, Fact]],
    query: str,
    settings,
    *,
    with_trace: bool = False,
    excerpt_chars: int = DEFAULT_PROMPT_EXCERPT_CHARS,
    output_chars: int = DEFAULT_OUTPUT_EXCERPT_CHARS,
) -> tuple[str, str, dict[str, int | None], list[str], list[LLMCallTrace], bool]:
    """Tier 1: Single agent synthesis.

    Note: this path intentionally does NOT yet adopt the stable-prefix layout.
    Since tier-1 only makes one LLM call per recall, there's no intra-query
    cache to win from. Cross-query prompt caching on an unchanged corpus is a
    later win, out of scope for this change.
    """
    facts = [f for _, f in scored_facts]
    facts_text = format_facts_for_llm(facts)
    prompt = f"QUERY: {query}\n\nSTORED FACTS:\n{facts_text}"

    totals: dict[str, int | None] = {
        "llm_calls": 0,
        "input_tokens": None,
        "cached_tokens": None,
        "output_tokens": None,
    }
    t_call = time.monotonic()
    completion = await asyncio.wait_for(
        complete_with_usage(prompt=prompt, system=SINGLE_AGENT_SYSTEM),
        timeout=settings.retrieval_timeout,
    )
    elapsed_ms = (time.monotonic() - t_call) * 1000
    _accumulate(totals, completion)
    answer, quality = _extract_quality(completion.text)

    candidate_ids = {f.id for f in facts}
    cited_ids = _extract_cited_ids(completion.text, candidate_ids)

    traces: list[LLMCallTrace] = []
    truncated_any = False
    if with_trace:
        prompt_excerpt, prompt_truncated = excerpt(prompt, excerpt_chars)
        output_excerpt, output_truncated = excerpt(completion.text, output_chars)
        traces.append(
            LLMCallTrace(
                name="single_agent",
                system_excerpt=excerpt(SINGLE_AGENT_SYSTEM, excerpt_chars)[0],
                prompt_excerpt=prompt_excerpt,
                output_excerpt=output_excerpt,
                elapsed_ms=elapsed_ms,
                input_tokens=completion.input_tokens,
                cached_tokens=completion.cached_tokens,
            )
        )
        truncated_any = prompt_truncated or output_truncated
    return answer, quality, totals, cited_ids, traces, truncated_any


async def _multilens_recall(
    scored_facts: list[tuple[int, Fact]],
    query: str,
    settings,
    *,
    with_trace: bool = False,
    excerpt_chars: int = DEFAULT_PROMPT_EXCERPT_CHARS,
    output_chars: int = DEFAULT_OUTPUT_EXCERPT_CHARS,
) -> tuple[str, str, dict[str, int | None], list[str], list[LLMCallTrace], bool]:
    """Tier 2 (default): multi-lens search + synthesis, two LLM calls total.

    Both calls share a byte-identical prefix (the formatted fact dump) so that
    OpenAI's implicit prefix cache and Anthropic's `cache_control` marker can
    skip re-processing the bulky part on the second call.
    """
    facts = [f for _, f in scored_facts]
    facts_text = format_facts_for_llm(facts)
    prefix = _build_prefix(facts_text)

    totals: dict[str, int | None] = {
        "llm_calls": 0,
        "input_tokens": None,
        "cached_tokens": None,
        "output_tokens": None,
    }

    search_prompt = (
        prefix + f"QUERY: {query}\n\n"
        "INSTRUCTIONS: Apply the three perspectives described in the system prompt "
        "and return one response with the `## DIRECT`, `## CONTEXTUAL`, and `## TEMPORAL` sections."
    )

    traces: list[LLMCallTrace] = []
    truncated_any = False
    candidate_ids = {f.id for f in facts}

    try:
        t_search = time.monotonic()
        search_completion = await asyncio.wait_for(
            complete_with_usage(
                prompt=search_prompt,
                system=MULTI_LENS_SYSTEM,
                cache_prefix=prefix,
            ),
            timeout=settings.retrieval_timeout,
        )
        search_elapsed = (time.monotonic() - t_search) * 1000
    except Exception as exc:
        logger.warning("multi-lens search call failed: %s", exc)
        if with_trace:
            traces.append(
                LLMCallTrace(
                    name="multilens_search",
                    system_excerpt=excerpt(MULTI_LENS_SYSTEM, excerpt_chars)[0],
                    prompt_excerpt=excerpt(search_prompt, excerpt_chars)[0],
                    output_excerpt="",
                    error=str(exc),
                )
            )
        raise

    _accumulate(totals, search_completion)
    sections = _parse_multilens_sections(search_completion.text)
    multilens_cited = _extract_cited_ids(search_completion.text, candidate_ids)

    if with_trace:
        prompt_excerpt, prompt_truncated = excerpt(search_prompt, excerpt_chars)
        output_excerpt, output_truncated = excerpt(search_completion.text, output_chars)
        traces.append(
            LLMCallTrace(
                name="multilens_search",
                system_excerpt=excerpt(MULTI_LENS_SYSTEM, excerpt_chars)[0],
                prompt_excerpt=prompt_excerpt,
                output_excerpt=output_excerpt,
                elapsed_ms=search_elapsed,
                input_tokens=search_completion.input_tokens,
                cached_tokens=search_completion.cached_tokens,
            )
        )
        truncated_any = truncated_any or prompt_truncated or output_truncated

    synthesis_prompt = (
        prefix
        + f"ORIGINAL QUERY: {query}\n\n"
        + "MULTI-LENS FINDINGS:\n"
        + f"## DIRECT\n{sections['direct'] or '(none)'}\n\n"
        + f"## CONTEXTUAL\n{sections['contextual'] or '(none)'}\n\n"
        + f"## TEMPORAL\n{sections['temporal'] or '(none)'}\n\n"
        + "Synthesize these findings into a clear, concise answer."
    )

    t_synth = time.monotonic()
    synthesis_completion = await asyncio.wait_for(
        complete_with_usage(
            prompt=synthesis_prompt,
            system=SYNTHESIS_SYSTEM,
            cache_prefix=prefix,
        ),
        timeout=settings.retrieval_timeout,
    )
    synth_elapsed = (time.monotonic() - t_synth) * 1000
    _accumulate(totals, synthesis_completion)
    answer, quality = _extract_quality(synthesis_completion.text)

    synthesis_cited = _extract_cited_ids(synthesis_completion.text, candidate_ids)
    # Synthesis is the final voice; prefer its citations, fall back to multilens.
    cited_ids = synthesis_cited or multilens_cited

    if with_trace:
        prompt_excerpt, prompt_truncated = excerpt(synthesis_prompt, excerpt_chars)
        output_excerpt, output_truncated = excerpt(
            synthesis_completion.text, output_chars
        )
        traces.append(
            LLMCallTrace(
                name="synthesis",
                system_excerpt=excerpt(SYNTHESIS_SYSTEM, excerpt_chars)[0],
                prompt_excerpt=prompt_excerpt,
                output_excerpt=output_excerpt,
                elapsed_ms=synth_elapsed,
                input_tokens=synthesis_completion.input_tokens,
                cached_tokens=synthesis_completion.cached_tokens,
            )
        )
        truncated_any = truncated_any or prompt_truncated or output_truncated

    return answer, quality, totals, cited_ids, traces, truncated_any


# Re-export `complete` at module level for external importers/mocks.
__all__ = [
    "recall",
    "recall_with_provenance",
    "complete",
    "_extract_quality",
    "_extract_cited_ids",
    "_format_direct",
    "_select_tier",
    "_select_tier_with_decision",
    "_build_prefix",
    "_parse_multilens_sections",
    "_resolve_tier_rules",
    "MULTI_LENS_SYSTEM",
    "SYNTHESIS_SYSTEM",
]
