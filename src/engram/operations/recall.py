"""Recall memories, briefs, traces, and recall statistics."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from engram.core.fact_format import format_fact_line
from engram.core.interfaces import (
    Envelope,
    EnvelopeMeta,
)
from engram.core.models import (
    BRIEF_MEMORY_KEY,
    RecallRecord,
)
from engram.core.operation_models import (
    RecallPayload,
)
from engram.core.provenance import DEFAULT_MAX_PREFILTER_MATCHES, DEFAULT_MAX_SOURCES
from engram.maintenance.upkeep import (
    project_brief,
)
from engram.operations.results import (
    OperationResult,
    exception_result,
    fact_payload,
    invalid_format_result,
    invalid_mode_result,
    invalid_positive_int_result,
    validation_result,
)
from engram.operations.stores import resolve_store, scope_project
from engram.recall.retriever import (
    RecallMode,
)
from engram.recall.retriever import (
    recall_result as _recall_result,
)
from engram.storage.async_store import AsyncFactStore
from engram.storage.store import FactStore


async def recall(
    query: str,
    *,
    project: str | None = None,
    mode: str = "cards",
    format: str = "text",
    with_provenance: bool = False,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_prefilter_matches: int = DEFAULT_MAX_PREFILTER_MATCHES,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    if format not in ("text", "json", ""):
        return invalid_format_result(format)
    if format == "":
        format = "text"
    if max_sources < 1:
        return invalid_positive_int_result("max_sources")
    if max_prefilter_matches < 1:
        return invalid_positive_int_result("max_prefilter_matches")
    if mode not in RECALL_MODES:
        return invalid_mode_result(mode, RECALL_MODES)
    if not query.strip():
        return validation_result("query must not be empty")

    store_obj = await resolve_store(store)
    result = await _recall_result(
        query,
        project=await scope_project(store_obj, project),
        store=store_obj,
        mode=_recall_mode(mode),
        max_sources=max_sources,
        max_prefilter_matches=max_prefilter_matches,
    )
    provenance = result.provenance
    delivered = {item.fact.id for item in result.delivered}
    payload = RecallPayload(
        answer=result.text,
        mode=_recall_mode(mode),
        quality=result.quality,
        tier=provenance.tier,
        facts=[source for source in provenance.sources if source.id in delivered],
        source_fact_ids=provenance.source_fact_ids,
        cited_fact_ids=provenance.cited_fact_ids,
        usage=provenance.usage,
        latency_ms=provenance.latency_ms,
        provenance=provenance if with_provenance else None,
    )
    total = max(len(result.selected), provenance.selected_decision.relevant_count)
    meta = EnvelopeMeta(
        limit=max_sources,
        returned=len(result.delivered),
        total=total,
        truncated=total > len(result.delivered),
        truncation_reason="max_sources" if total > max_sources else None,
    )
    return OperationResult(
        envelope=Envelope.success(
            data=payload.model_dump(mode="json", exclude_none=True),
            warnings=provenance.warnings,
            meta=meta,
        ),
        text=result.text,
    )


async def recall_trace(
    query: str,
    *,
    project: str | None = None,
    verbose: bool = False,
    mode: str = "cards",
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_prefilter_matches: int = DEFAULT_MAX_PREFILTER_MATCHES,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    if max_sources < 1:
        return invalid_positive_int_result("max_sources")
    if max_prefilter_matches < 1:
        return invalid_positive_int_result("max_prefilter_matches")

    if mode not in RECALL_MODES:
        return invalid_mode_result(mode, RECALL_MODES)
    if not query.strip():
        return validation_result("query must not be empty")
    store_obj = await resolve_store(store)
    try:
        result = await _recall_result(
            query,
            project=await scope_project(store_obj, project),
            store=store_obj,
            mode=_recall_mode(mode),
            with_trace=True,
            verbose_trace=verbose,
            max_sources=max_sources,
            max_prefilter_matches=max_prefilter_matches,
        )
    except Exception as exc:
        return exception_result(exc)

    answer, quality, provenance, trace = (
        result.text,
        result.quality,
        result.provenance,
        result.trace,
    )
    data = {
        "query": query,
        "answer": answer,
        "quality": quality,
        "tier": provenance.tier,
        "trace": trace.model_dump(mode="json") if trace else None,
    }
    truncated = bool(trace and trace.truncated)
    meta = EnvelopeMeta(
        limit=max_sources,
        returned=len(provenance.sources),
        total=provenance.prefilter_count,
        truncated=truncated,
        truncation_reason="excerpt_truncated" if truncated else None,
    )
    text = (
        f"Tier {provenance.tier} | quality={quality} | "
        f"prefilter={provenance.prefilter_count} | "
        f"calls={provenance.usage.llm_calls or 0}\n"
        f"Answer:\n{answer}"
    )
    return OperationResult(
        envelope=Envelope.success(data=data, warnings=provenance.warnings, meta=meta),
        text=text,
    )


RECALL_MODES = ("cards", "answer")


RECALL_CONTEXT_MODES = ("cards", "prompt", "answer", "brief")


def _recall_mode(mode: str) -> RecallMode:
    return "answer" if mode == "answer" else "cards"


async def recall_context(
    query: str = "",
    *,
    project: str | None = None,
    mode: str = "cards",
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    """Context for an agent: dated cards, a synthesized answer, or a brief.

    ``prompt`` is the older name for ``cards``. ``brief`` returns the
    project's synthesized brief (session-start orientation).
    """
    if mode not in RECALL_CONTEXT_MODES:
        return invalid_mode_result(mode, RECALL_CONTEXT_MODES)
    if mode == "brief" and not project:
        return validation_result("mode 'brief' requires a project")
    if mode != "brief" and not query.strip():
        return validation_result("query must not be empty unless mode is 'brief'")
    store_obj = await resolve_store(store)
    scope = await scope_project(store_obj, project)

    if mode == "brief":
        return await _project_brief_result(store_obj, scope)

    result = await _recall_result(query, project=scope, store=store_obj, mode=_recall_mode(mode))
    answer, provenance = result.text, result.provenance
    return OperationResult(
        envelope=Envelope.success(
            data={
                "answer": answer,
                "mode": mode,
                "fact_ids": provenance.cited_fact_ids,
            },
            warnings=provenance.warnings,
        ),
        text=answer,
    )


async def _project_brief_result(store: AsyncFactStore, project: str | None) -> OperationResult:
    if project is None:
        return validation_result("mode 'brief' requires a project")
    brief = await project_brief(store, project)
    if brief is not None:
        text = f"# {project} brief ({brief.observed_at.date().isoformat()})\n\n{brief.content}"
        return OperationResult(
            envelope=Envelope.success(data={"mode": "brief", "brief": fact_payload(brief)}),
            text=text,
        )

    # No brief synthesized yet: fall back to the project's newest durable cards.
    facts = [
        fact
        for fact in await store.load_active_facts(project=project, include_global=False)
        if fact.durability.value != "ephemeral" and fact.memory_key != BRIEF_MEMORY_KEY
    ][:BRIEF_FALLBACK_CARDS]
    if not facts:
        text = f"No memories stored for project {project!r} yet."
        return OperationResult(
            envelope=Envelope.success(data={"mode": "brief", "facts": []}),
            text=text,
        )
    lines = [f"# {project}: no brief yet — {len(facts)} most recent cards"]
    lines.extend(f"- {format_fact_line(fact)}" for fact in facts)
    return OperationResult(
        envelope=Envelope.success(
            data={"mode": "brief", "facts": [fact_payload(f) for f in facts]}
        ),
        text="\n".join(lines),
    )


BRIEF_FALLBACK_CARDS = 15


def _recall_stats_payload(records: list[RecallRecord]) -> dict[str, Any]:
    total = len(records)
    tier_counts = Counter(r.tier for r in records)
    quality_counts = Counter(r.quality for r in records if r.quality)
    latencies = [r.latency_ms for r in records]
    avg_latency = sum(latencies) / total if total else 0.0

    tier_latency: dict[int, list[float]] = {}
    for r in records:
        tier_latency.setdefault(r.tier, []).append(r.latency_ms)

    llm_call_totals = [r.llm_calls for r in records if r.llm_calls is not None]
    total_llm_calls = sum(llm_call_totals)
    input_totals = [r.input_tokens for r in records if r.input_tokens is not None]
    cached_totals = [r.cached_tokens for r in records if r.cached_tokens is not None]
    sum_input = sum(input_totals)
    sum_cached = sum(cached_totals)
    cache_hit_ratio = (sum_cached / sum_input) if sum_input > 0 else None
    selector_versions = sorted({r.selector_version for r in records if r.selector_version})

    return {
        "total_queries": total,
        "selector_versions": selector_versions,
        "avg_latency_ms": avg_latency,
        "by_tier": {
            str(tier): {
                "count": count,
                "percent": count / total * 100 if total else 0.0,
                "avg_latency_ms": sum(tier_latency[tier]) / count,
            }
            for tier, count in sorted(tier_counts.items())
        },
        "by_quality": {
            quality: {
                "count": count,
                "percent": count / total * 100 if total else 0.0,
            }
            for quality, count in sorted(quality_counts.items())
        },
        "token_usage": {
            "llm_calls": total_llm_calls if llm_call_totals else None,
            "input_tokens": sum_input if input_totals else None,
            "cached_input_tokens": sum_cached if cached_totals else None,
            "cache_hit_ratio": cache_hit_ratio,
        },
    }


def _format_recall_summary(records: list[RecallRecord], heading: str) -> list[str]:
    payload = _recall_stats_payload(records)
    total = payload["total_queries"]
    avg_latency = payload["avg_latency_ms"]

    lines = [
        f"{heading}",
        f"**Total queries:** {total}",
        f"**Avg latency:** {avg_latency:.0f}ms\n",
        "## By Tier",
    ]
    for tier, stats in payload["by_tier"].items():
        count = stats["count"]
        avg = stats["avg_latency_ms"]
        pct = stats["percent"]
        lines.append(f"- Tier {tier}: {count} ({pct:.0f}%) — avg {avg:.0f}ms")

    if payload["by_quality"]:
        lines.append("\n## By Quality")
        for quality in ("high", "medium", "low", "none"):
            if quality in payload["by_quality"]:
                count = payload["by_quality"][quality]["count"]
                pct = payload["by_quality"][quality]["percent"]
                lines.append(f"- {quality}: {count} ({pct:.0f}%)")

    token_usage = payload["token_usage"]
    lines.append("\n## Token Usage")
    llm_calls = token_usage["llm_calls"]
    lines.append(f"- LLM calls (reported): {llm_calls if llm_calls is not None else '-'}")
    lines.append(
        f"- Input tokens: {token_usage['input_tokens']:,}"
        if token_usage["input_tokens"] is not None
        else "- Input tokens: -"
    )
    lines.append(
        f"- Cached input tokens: {token_usage['cached_input_tokens']:,}"
        if token_usage["cached_input_tokens"] is not None
        else "- Cached input tokens: -"
    )
    if token_usage["cache_hit_ratio"] is not None:
        lines.append(f"- Cache hit ratio: {token_usage['cache_hit_ratio'] * 100:.1f}%")
    else:
        lines.append("- Cache hit ratio: -")
    return lines


def _parse_since(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("since must be an ISO-8601 datetime or date") from exc
    if parsed and parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _recall_stats_meta(limit: int, total: int, returned: int) -> EnvelopeMeta:
    truncated = total > returned
    return EnvelopeMeta(
        limit=limit,
        requested_limit=limit,
        returned=returned,
        total=total,
        truncated=truncated,
        truncation_reason="limit" if truncated else None,
    )


def _empty_recall_stats_result(
    *,
    limit: int,
    total: int,
    include_records: bool,
) -> OperationResult:
    text = "No recall data yet."
    data = {**_recall_stats_payload([]), "message": text}
    if include_records:
        data["records"] = []
    return OperationResult(
        envelope=Envelope.success(
            data=data,
            meta=_recall_stats_meta(limit, total=total, returned=0),
        ),
        text=text,
    )


def _recall_stats_lines(records: list[RecallRecord], payload: dict[str, Any]) -> list[str]:
    lines = _format_recall_summary(records, "# Recall Stats\n")
    selector_versions = payload["selector_versions"]
    if len(selector_versions) >= 2:
        for version in selector_versions:
            subset = [r for r in records if r.selector_version == version]
            lines.append("")
            lines.extend(_format_recall_summary(subset, heading=f"# Selector {version}\n"))
    elif selector_versions:
        lines.insert(1, f"*Selector version:* `{selector_versions[0]}`")
    return lines


async def recall_stats(
    *,
    limit: int = 500,
    since: str | datetime | None = None,
    include_records: bool = False,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    if limit < 1:
        return invalid_positive_int_result("limit")

    try:
        since_dt = _parse_since(since)
    except ValueError as exc:
        return validation_result(str(exc), details={"parameter": "since"})

    all_records = await (await resolve_store(store)).load_recall_log(limit=None)
    filtered_records = [
        record for record in all_records if since_dt is None or record.timestamp >= since_dt
    ]
    records = filtered_records[:limit]
    if not records:
        return _empty_recall_stats_result(
            limit=limit,
            total=len(filtered_records),
            include_records=include_records,
        )

    payload = _recall_stats_payload(records)
    lines = _recall_stats_lines(records, payload)

    if include_records:
        payload["records"] = [record.model_dump(mode="json") for record in records]

    return OperationResult(
        envelope=Envelope.success(
            data=payload,
            meta=_recall_stats_meta(
                limit,
                total=len(filtered_records),
                returned=len(records),
            ),
        ),
        text="\n".join(lines),
    )
