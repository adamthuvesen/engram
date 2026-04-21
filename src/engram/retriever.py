"""Retriever — tiered agentic memory retrieval with quality observability."""

import asyncio
import logging
import re
import time

from engram.config import get_settings
from engram.llm import Completion, complete, complete_with_usage
from engram.models import Fact, RecallRecord
from engram.store import FactStore, format_facts_for_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent system prompts
# ---------------------------------------------------------------------------

AGENT_A_SYSTEM = """You are a DIRECT FACTS search agent. Given a query and a list of stored facts,
find the facts that directly answer the query. Be precise — only return facts that are
clearly relevant to what's being asked.

Return your findings as a numbered list of the most relevant facts, with brief reasoning
for why each is relevant. Include the fact IDs."""

AGENT_B_SYSTEM = """You are a CONTEXTUAL search agent. Given a query and a list of stored facts,
find facts that are indirectly related and add useful context. Look for:
- Facts that provide background to understand the answer
- Connections between multiple facts that aren't obvious
- Implications and related preferences/patterns

Return your findings as a numbered list with reasoning. Include fact IDs."""

AGENT_C_SYSTEM = """You are a TEMPORAL search agent. Given a query and a list of stored facts,
focus on time and state:
- What's the current state vs. what's outdated?
- Are there facts that supersede or contradict each other?
- What's the timeline of relevant events?
- Flag any facts that may be stale or expired

Return your findings as a numbered list, clearly marking anything time-sensitive. Include fact IDs."""

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

# Track unknown-pipeline warnings so we only log them once.
_unknown_pipeline_warned: set[str] = set()
_unknown_tier_rules_warned: set[str] = set()

# Multi-lens response heading parser.
_MULTILENS_HEADING_RE = re.compile(r"^\s*##\s+(DIRECT|CONTEXTUAL|TEMPORAL)\s*$", re.MULTILINE)


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
    Tier 2: Many matches or flat distribution → full 3-agent pipeline.

    Zero relevant matches → Tier 0 (direct).

    When `rules == "v2"` and `min_prefilter_for_tier2 > 0`, an additional cap
    applies: queries whose prefilter produced fewer than `min_prefilter_for_tier2`
    strictly-positive-scored facts are downgraded from tier-2 to tier-1. Tier-0
    decisions are never touched by the cap.
    """
    if not scored_facts:
        return 0

    relevant = [s for s, _ in scored_facts if s >= RELEVANCE_FLOOR]

    # No strong matches at all — nothing useful for any agent
    if not relevant:
        return 0

    top = relevant[0]

    # Gap ratio: how much the top score stands out from the pack.
    # Compare top vs 5th score. Fewer than 5 → compare to last available.
    if len(relevant) < 5:
        gap = top / relevant[-1] if relevant[-1] > 0 else float("inf")
    else:
        gap = top / relevant[4] if relevant[4] > 0 else float("inf")

    # Tier 0: few matches with strong scores — answer is obvious
    if len(relevant) <= TIER_0_MAX_RELEVANT and top >= TIER_0_MIN_SCORE:
        return 0

    # Tier 1: signal concentrated at top (gap-driven, count is secondary)
    # Even with many relevant matches, if the top cluster stands out clearly,
    # a single agent can synthesize the answer without 3 perspectives.
    # Requires both a strong top score (≥15) and a clear gap (≥1.5).
    if top >= 15 and gap >= TIER_1_MIN_GAP:
        return 1

    # Tier 2: flat distribution or weak top signal — needs multi-perspective
    chosen = 2

    # v2 small-corpus cap: if the prefilter returned too few positive-scoring
    # facts, multilens can't add useful lenses over a single-agent call.
    if rules == "v2" and min_prefilter_for_tier2 > 0 and chosen == 2:
        positive = sum(1 for s, _ in scored_facts if s > 0)
        if positive < min_prefilter_for_tier2:
            return 1
    return chosen


# TODO(small-corpus-tier-cap-cleanup): after one release of clean v2 data in
# recall_log, remove the v1 branch, the `rules` parameter on `_select_tier`,
# and the `ENGRAM_TIER_RULES` setting. Keep the configurable threshold.
def _resolve_tier_rules(raw: str) -> str:
    """Map the tier_rules setting to a supported value, warning once on unknowns."""
    if raw in ("v1", "v2"):
        return raw
    if raw not in _unknown_tier_rules_warned:
        logger.warning(
            "Unknown ENGRAM_TIER_RULES=%r; falling back to 'v2'", raw
        )
        _unknown_tier_rules_warned.add(raw)
    return "v2"


def _format_direct(scored_facts: list[tuple[int, Fact]], query: str) -> str:
    """Tier 0: Format high-confidence facts directly without LLM."""
    facts = [f for score, f in scored_facts if score > 0]
    if not facts:
        return "No memories stored yet. Use `remember` to add some."

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
        logger.debug("multilens response missing expected headings; using raw body as DIRECT")
        sections["DIRECT"] = text.strip()
        return {k.lower(): v for k, v in sections.items()}

    for i, match in enumerate(matches):
        heading = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[heading] = text[start:end].strip()
    return {k.lower(): v for k, v in sections.items()}


async def recall(
    query: str,
    project: str | None = None,
    store: FactStore | None = None,
    parallel: bool = True,
) -> str:
    """Tiered agentic retrieval with quality logging.

    Tier 0: Direct lookup — no LLM calls.
    Tier 1: Single agent synthesis.
    Tier 2: Multi-lens search + synthesis (default) or legacy 3-agent pipeline.
    """
    store = store or FactStore()
    settings = get_settings()
    t0 = time.monotonic()

    scored_facts = store.prefilter_facts(
        query=query,
        project=project,
        limit=settings.max_facts_per_agent,
    )

    tier_rules = _resolve_tier_rules(settings.tier_rules)
    tier = _select_tier(
        scored_facts,
        rules=tier_rules,
        min_prefilter_for_tier2=settings.tier2_min_prefilter_count,
    )
    prefilter_count = len([s for s, _ in scored_facts if s > 0])
    usage_totals: dict[str, int | None] = {
        "llm_calls": None,
        "input_tokens": None,
        "cached_tokens": None,
    }

    if tier == 0:
        answer = _format_direct(scored_facts, query)
        usage_totals["llm_calls"] = 0
        if any(s >= RELEVANCE_FLOOR for s, _ in scored_facts):
            quality = "high"
        elif prefilter_count > 0:
            quality = "low"
        else:
            quality = "none"
    elif tier == 1:
        answer, quality, usage_totals = await _single_agent_recall(
            scored_facts, query, settings
        )
    else:
        pipeline = _resolve_pipeline(settings.recall_pipeline)
        if pipeline == "legacy":
            answer, quality, usage_totals = await _full_agentic_recall(
                scored_facts, query, settings, parallel
            )
        else:
            answer, quality, usage_totals = await _multilens_recall(
                scored_facts, query, settings
            )

    latency_ms = (time.monotonic() - t0) * 1000

    # Log for observability
    try:
        store.log_recall(
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
            )
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
    return answer


def _resolve_pipeline(raw: str) -> str:
    """Map the recall_pipeline setting to a supported value, warning once on unknowns."""
    if raw in ("multilens", "legacy"):
        return raw
    if raw not in _unknown_pipeline_warned:
        logger.warning(
            "Unknown ENGRAM_RECALL_PIPELINE=%r; falling back to 'multilens'", raw
        )
        _unknown_pipeline_warned.add(raw)
    return "multilens"


def _accumulate(totals: dict[str, int | None], completion: Completion) -> None:
    """Fold a Completion's usage into the running totals dict in-place."""
    totals["llm_calls"] = (totals.get("llm_calls") or 0) + 1
    if completion.input_tokens is not None:
        totals["input_tokens"] = (totals.get("input_tokens") or 0) + completion.input_tokens
    if completion.cached_tokens is not None:
        totals["cached_tokens"] = (totals.get("cached_tokens") or 0) + completion.cached_tokens


async def _single_agent_recall(
    scored_facts: list[tuple[int, Fact]],
    query: str,
    settings,
) -> tuple[str, str, dict[str, int | None]]:
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
    }
    completion = await asyncio.wait_for(
        complete_with_usage(prompt=prompt, system=SINGLE_AGENT_SYSTEM),
        timeout=settings.retrieval_timeout,
    )
    _accumulate(totals, completion)
    answer, quality = _extract_quality(completion.text)
    return answer, quality, totals


async def _multilens_recall(
    scored_facts: list[tuple[int, Fact]],
    query: str,
    settings,
) -> tuple[str, str, dict[str, int | None]]:
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
    }

    search_prompt = (
        prefix + f"QUERY: {query}\n\n"
        "INSTRUCTIONS: Apply the three perspectives described in the system prompt "
        "and return one response with the `## DIRECT`, `## CONTEXTUAL`, and `## TEMPORAL` sections."
    )

    try:
        search_completion = await asyncio.wait_for(
            complete_with_usage(
                prompt=search_prompt,
                system=MULTI_LENS_SYSTEM,
                cache_prefix=prefix,
            ),
            timeout=settings.retrieval_timeout,
        )
    except Exception as exc:
        logger.warning("multi-lens search call failed: %s", exc)
        raise

    _accumulate(totals, search_completion)
    sections = _parse_multilens_sections(search_completion.text)

    synthesis_prompt = (
        prefix
        + f"ORIGINAL QUERY: {query}\n\n"
        + "MULTI-LENS FINDINGS:\n"
        + f"## DIRECT\n{sections['direct'] or '(none)'}\n\n"
        + f"## CONTEXTUAL\n{sections['contextual'] or '(none)'}\n\n"
        + f"## TEMPORAL\n{sections['temporal'] or '(none)'}\n\n"
        + "Synthesize these findings into a clear, concise answer."
    )

    synthesis_completion = await asyncio.wait_for(
        complete_with_usage(
            prompt=synthesis_prompt,
            system=SYNTHESIS_SYSTEM,
            cache_prefix=prefix,
        ),
        timeout=settings.retrieval_timeout,
    )
    _accumulate(totals, synthesis_completion)
    answer, quality = _extract_quality(synthesis_completion.text)
    return answer, quality, totals


async def _full_agentic_recall(
    scored_facts: list[tuple[int, Fact]],
    query: str,
    settings,
    parallel: bool,
) -> tuple[str, str, dict[str, int | None]]:
    """Legacy tier-2: three parallel search agents + synthesis (four LLM calls).

    Retained behind `ENGRAM_RECALL_PIPELINE=legacy` for a release so the
    multi-lens replacement can be A/B compared against it via recall_log before
    the legacy branch is removed.

    TODO(retrieval-fast-path-cleanup): remove this branch, the AGENT_{A,B,C}_SYSTEM
    constants, and the `recall_pipeline` setting after one release of clean
    recall_log quality data shows no regression vs the multi-lens pipeline.
    """
    facts = [f for _, f in scored_facts]
    facts_text = format_facts_for_llm(facts)
    prompt = f"QUERY: {query}\n\nSTORED FACTS:\n{facts_text}"

    totals: dict[str, int | None] = {
        "llm_calls": 0,
        "input_tokens": None,
        "cached_tokens": None,
    }

    run_parallel = parallel and settings.parallel_agents
    if run_parallel:
        raw_results = await asyncio.gather(
            asyncio.wait_for(
                complete_with_usage(prompt=prompt, system=AGENT_A_SYSTEM),
                timeout=settings.retrieval_timeout,
            ),
            asyncio.wait_for(
                complete_with_usage(prompt=prompt, system=AGENT_B_SYSTEM),
                timeout=settings.retrieval_timeout,
            ),
            asyncio.wait_for(
                complete_with_usage(prompt=prompt, system=AGENT_C_SYSTEM),
                timeout=settings.retrieval_timeout,
            ),
            return_exceptions=True,
        )
    else:
        raw_results = []
        for system_prompt in (AGENT_A_SYSTEM, AGENT_B_SYSTEM, AGENT_C_SYSTEM):
            try:
                result = await asyncio.wait_for(
                    complete_with_usage(prompt=prompt, system=system_prompt),
                    timeout=settings.retrieval_timeout,
                )
                raw_results.append(result)
            except Exception as exc:
                raw_results.append(exc)

    errors = []
    agent_outputs: list[str] = []
    labels = ("A", "B", "C")
    for label, result in zip(labels, raw_results):
        if isinstance(result, Exception):
            logger.warning("Recall agent %s failed: %s", label, result)
            errors.append(result)
            agent_outputs.append("")
        else:
            _accumulate(totals, result)
            agent_outputs.append(result.text)

    if all(isinstance(r, Exception) for r in raw_results):
        raise RuntimeError(f"All 3 recall agents failed: {[str(e) for e in errors]}")

    agent_a, agent_b, agent_c = agent_outputs

    synthesis_prompt = f"""ORIGINAL QUERY: {query}

PREFILTERED FACTS:
{facts_text}

AGENT A (Direct Facts):
{agent_a}

AGENT B (Context & Implications):
{agent_b}

AGENT C (Temporal & State):
{agent_c}

Synthesize these findings into a clear, concise answer."""

    synthesis = await asyncio.wait_for(
        complete_with_usage(prompt=synthesis_prompt, system=SYNTHESIS_SYSTEM),
        timeout=settings.retrieval_timeout,
    )
    _accumulate(totals, synthesis)
    answer, quality = _extract_quality(synthesis.text)
    return answer, quality, totals


# Re-export `complete` at module level for any legacy importers/mocks.
__all__ = [
    "recall",
    "complete",
    "_extract_quality",
    "_format_direct",
    "_select_tier",
    "_build_prefix",
    "_parse_multilens_sections",
    "_resolve_pipeline",
    "_resolve_tier_rules",
    "MULTI_LENS_SYSTEM",
    "SYNTHESIS_SYSTEM",
]
