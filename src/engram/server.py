"""Engram MCP Server — structured, cross-project memory for coding agents."""

import logging
from collections import Counter

from fastmcp import FastMCP

from engram.config import configure_logging, get_settings
from engram.doctor import check_provider, repair_store, run_doctor
from engram.importer import import_claude_code_memories
from engram.interfaces import (
    Envelope,
    EnvelopeMeta,
    not_found_error,
    provider_error,
    validation_error,
)
from engram.models import CandidateStatus, FactCategory, IngestionRecord
from engram.observer import extract_facts, suggest_memories as _suggest_memories
from engram.provenance import DEFAULT_MAX_PREFILTER_MATCHES, DEFAULT_MAX_SOURCES
from engram.retriever import (
    RELEVANCE_FLOOR,
    recall as _recall,
    recall_with_provenance as _recall_with_provenance,
)
from engram.store import AsyncFactStore, FactStore, format_facts_for_llm
from engram.synthesizer import format_synthesis_result, synthesize as _synthesize

logger = logging.getLogger(__name__)

INSTRUCTIONS = """Engram — structured, cross-project memory for coding agents.

A cross-project, structured memory system that uses LLM-powered retrieval
instead of vector search. Facts are extracted, categorized, and stored as
structured knowledge. Tiered retrieval uses deterministic fast paths first,
then a multi-lens search-and-synthesis path for complex queries.

Tools:
- remember: Store new memories (extracts structured facts from natural language)
- suggest_memories: Propose memories for review instead of storing them immediately
- list_candidates: Browse pending/reviewed memory suggestions
- approve_candidates: Promote reviewed suggestions into active memory
- reject_candidates: Dismiss bad suggestions with an audit trail
- recall: Search memory with tiered multi-lens retrieval
- recall_context: Recall as answer or compact prompt block
- forget: Remove a fact from memory (soft delete)
- edit_fact: Edit a fact's content, category, tags, or project in place
- inspect: Browse stored facts for transparency
- import_memories: Bootstrap from existing Claude Code memory files
- memory_stats: View memory system statistics
- recall_stats: View recall quality and performance statistics
- purge: Permanently remove forgotten and expired facts from storage
- rename_project: Bulk-rename facts and candidates from one project to another
- synthesize: Consolidate and clean up memory (deduplicate, merge, rewrite, prune)
"""

mcp = FastMCP("engram", instructions=INSTRUCTIONS)
_store: AsyncFactStore | FactStore  # initialized in main()


def _async_store() -> AsyncFactStore:
    """Return the async store facade, wrapping raw test stores when needed."""
    if isinstance(_store, AsyncFactStore):
        return _store
    return AsyncFactStore(_store)


@mcp.tool()
async def remember(
    content: str,
    source: str = "conversation",
    project: str | None = None,
) -> str:
    """Store a new memory. Extracts structured facts from natural language input.

    Use this to save important context: user preferences, decisions made,
    project details, corrections, or anything worth remembering across sessions.

    Args:
        content: Natural language description of what to remember.
        source: Where this memory comes from (default: "conversation").
        project: Optional project name to associate with this memory.
    """
    store = _async_store()
    facts = await extract_facts(content, source=source, project=project, store=store)

    if not facts:
        return "No structured facts could be extracted from the input."

    lines = [f"Stored {len(facts)} fact(s):\n"]
    for fact in facts:
        lines.append(f"- [{fact.category.value}] {fact.content} (id: {fact.id})")
    return "\n".join(lines)


@mcp.tool()
async def suggest_memories(
    content: str,
    source: str = "conversation",
    project: str | None = None,
) -> str:
    """Propose memories for review without storing them as active facts.

    Use this after user corrections, reusable workflow discoveries,
    architectural decisions, or durable preferences.
    """
    store = _async_store()
    candidates = await _suggest_memories(
        content, source=source, project=project, store=store
    )
    if not candidates:
        return "No memory candidates were proposed from the input."

    lines = [f"Queued {len(candidates)} memory candidate(s):\n"]
    for candidate in candidates:
        lines.append(
            f"- [{candidate.category.value}] {candidate.content} "
            f"(id: {candidate.id}, why: {candidate.why_store or 'useful future context'})"
        )
    return "\n".join(lines)


@mcp.tool()
async def list_candidates(
    status: str = "pending",
    project: str | None = None,
    search: str | None = None,
    limit: int = 50,
) -> str:
    """Browse queued memory candidates for review.

    Args:
        status: Filter by status (pending, approved, rejected).
        project: Filter by project name.
        search: Optional text search across candidate content.
        limit: Maximum results to return.
    """
    try:
        candidate_status = CandidateStatus(status)
    except ValueError:
        valid = ", ".join(item.value for item in CandidateStatus)
        return f"Unsupported status: {status}. Use one of: {valid}."
    store = _async_store()
    candidates = await store.load_candidates(
        status=candidate_status,
        project=project,
        limit=None if search else limit,
    )

    if search:
        search_lower = search.lower()
        candidates = [c for c in candidates if search_lower in c.content.lower()]
        candidates = candidates[:limit]

    if not candidates:
        return "No memory candidates found matching the criteria."
    return store.format_candidates_for_review(candidates)


@mcp.tool()
async def approve_candidates(
    candidate_ids: list[str],
    edits: dict[str, str] | None = None,
) -> str:
    """Promote reviewed candidates into active memory.

    Args:
        candidate_ids: List of candidate IDs to approve.
        edits: Optional dict mapping candidate_id -> new content to apply before promoting.
    """
    store = _async_store()
    # Apply edits before approval
    if edits:
        for cid, new_content in edits.items():
            await store.update_candidate(cid, content=new_content)

    facts = await store.approve_candidates(candidate_ids)
    if not facts:
        return "No matching candidates were approved."

    lines = [f"Approved {len(facts)} candidate(s):\n"]
    for fact in facts:
        lines.append(f"- [{fact.category.value}] {fact.content} (id: {fact.id})")
    return "\n".join(lines)


@mcp.tool()
async def reject_candidates(candidate_ids: list[str], reason: str = "") -> str:
    """Reject proposed candidates without storing them as active memory."""
    store = _async_store()
    rejected = await store.reject_candidates(candidate_ids, reason=reason)
    if not rejected:
        return "No matching candidates were rejected."

    lines = [f"Rejected {len(rejected)} candidate(s):\n"]
    for candidate in rejected:
        lines.append(
            f"- [{candidate.category.value}] {candidate.content} (id: {candidate.id})"
        )
    return "\n".join(lines)


@mcp.tool()
async def recall(
    query: str,
    project: str | None = None,
    format: str = "text",
    with_provenance: bool = False,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_prefilter_matches: int = DEFAULT_MAX_PREFILTER_MATCHES,
) -> str:
    """Search memory using tiered agentic retrieval.

    Automatically selects the fastest strategy based on query complexity:
    - Simple lookups: instant (no LLM calls)
    - Moderate queries: single-agent synthesis
    - Complex queries: multi-lens search plus synthesis

    Args:
        query: What you want to know (natural language).
        project: Optional project filter.
        format: ``text`` (default, concise prose) or ``json`` (structured envelope
            with answer, sources, warnings, and usage). Existing callers see no
            behavior change.
        with_provenance: When True (and ``format='json'``), include full
            provenance (prefilter matches, source summaries, tier decision).
            Ignored when ``format='text'``.
        max_sources: Cap on source summaries returned in JSON mode.
        max_prefilter_matches: Cap on prefilter matches returned in JSON mode.
    """
    if format not in ("text", "json"):
        if format == "":
            format = "text"
        else:
            return Envelope.failure(
                validation_error(
                    f"Unsupported format: {format}. Use 'text' or 'json'.",
                    details={"parameter": "format", "value": format},
                )
            ).to_json()

    if format == "text":
        return await _recall(query, project=project, store=_async_store())

    answer, quality, provenance, _ = await _recall_with_provenance(
        query,
        project=project,
        store=_async_store(),
        max_sources=max_sources,
        max_prefilter_matches=max_prefilter_matches,
    )

    data: dict = {
        "answer": answer,
        "quality": quality,
        "tier": provenance.tier,
        "source_fact_ids": provenance.source_fact_ids,
        "cited_fact_ids": provenance.cited_fact_ids,
        "usage": provenance.usage.model_dump(),
        "latency_ms": provenance.latency_ms,
    }
    if with_provenance:
        data["provenance"] = provenance.model_dump(mode="json")

    truncated = (
        len(provenance.sources) >= max_sources
        or len(provenance.prefilter_matches) >= max_prefilter_matches
    )
    meta = EnvelopeMeta(
        limit=max_sources,
        returned=len(provenance.sources),
        total=provenance.prefilter_count,
        truncated=truncated,
        truncation_reason="max_sources_or_max_matches" if truncated else None,
    )
    return Envelope.success(
        data=data, warnings=provenance.warnings, meta=meta
    ).to_json()


@mcp.tool()
async def recall_trace(
    query: str,
    project: str | None = None,
    verbose: bool = False,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_prefilter_matches: int = DEFAULT_MAX_PREFILTER_MATCHES,
) -> str:
    """Run recall and return a structured trace for debugging.

    The trace includes the full provenance object plus bounded prompt and
    output excerpts for each LLM call, timing, and reported token usage.
    Verbose mode raises the per-field char limit but still bounds output.

    Always returns a JSON envelope. Provider failures are reported as
    structured errors rather than raised exceptions.

    Args:
        query: What you want to trace.
        project: Optional project filter.
        verbose: Increase the per-field excerpt limit.
        max_sources: Cap on source summaries.
        max_prefilter_matches: Cap on prefilter matches.
    """
    try:
        answer, quality, provenance, trace = await _recall_with_provenance(
            query,
            project=project,
            store=_async_store(),
            with_trace=True,
            verbose_trace=verbose,
            max_sources=max_sources,
            max_prefilter_matches=max_prefilter_matches,
        )
    except Exception as exc:  # provider/timeout/etc.
        return Envelope.failure(
            provider_error(
                f"Recall trace failed: {exc}",
                details={"exception_type": type(exc).__name__},
            )
        ).to_json()

    data = {
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
    return Envelope.success(
        data=data, warnings=provenance.warnings, meta=meta
    ).to_json()


@mcp.tool()
async def recall_context(
    query: str,
    project: str | None = None,
    mode: str = "answer",
) -> str:
    """Recall either a natural-language answer or a compact prompt block.

    mode:
    - answer: regular answer synthesis
    - prompt: compact fact block for injecting into another agent's prompt
    """
    if mode == "answer":
        return await _recall(query, project=project, store=_async_store())

    if mode != "prompt":
        return f"Unsupported mode: {mode}. Use 'answer' or 'prompt'."

    settings = get_settings()
    store = _async_store()
    scored = await store.prefilter_facts(
        query=query, project=project, limit=settings.max_facts_per_agent
    )
    facts = [f for score, f in scored if score >= RELEVANCE_FLOOR]
    if not facts:
        return "No relevant memories found for this query."

    return "# Memory Context\n\n" + format_facts_for_llm(facts)


@mcp.tool()
async def forget(fact_id: str, reason: str = "") -> str:
    """Mark a fact as forgotten (soft delete). Does not physically remove it.

    Use when a stored fact is wrong or no longer relevant.

    Args:
        fact_id: The ID of the fact to forget.
        reason: Optional reason for forgetting.
    """
    store = _async_store()
    fact = await store.forget(fact_id, reason)
    if fact:
        return f"Forgotten: [{fact.category.value}] {fact.content}"
    return f"Fact not found: {fact_id}"


@mcp.tool()
async def edit_fact(
    fact_id: str,
    content: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    project: str | None = None,
) -> str:
    """Edit a fact in place without losing its ID, timestamps, or supersession chain.

    Args:
        fact_id: The fact to edit.
        content: New content text (optional).
        category: New category (optional).
        tags: New tags list (optional).
        project: New project scope (optional).
    """
    updates: dict = {}
    if content is not None:
        updates["content"] = content
    if category is not None:
        try:
            updates["category"] = FactCategory(category)
        except ValueError:
            valid = ", ".join(c.value for c in FactCategory)
            return f"Invalid category: {category}. Use one of: {valid}"
    if tags is not None:
        updates["tags"] = tags
    if project is not None:
        updates["project"] = project

    if not updates:
        return "No changes specified."

    # Check if the fact exists and whether it's forgotten before updating
    store = _async_store()
    all_facts = await store.load_facts()
    existing = next((f for f in all_facts if f.id == fact_id), None)
    if existing is None:
        return f"Fact not found: {fact_id}"
    if existing.confidence == 0.0:
        return (
            f"Fact {fact_id} is forgotten and cannot be edited. "
            "Restore it first by approving a candidate that supersedes it."
        )

    fact = await store.update_fact(fact_id, **updates)
    if not fact:
        return f"Fact not found: {fact_id}"

    await store.log_ingestion(
        IngestionRecord(
            source="edit",
            facts_updated=[fact_id],
            agent_model="manual_edit",
        )
    )

    return f"Updated: [{fact.category.value}] {fact.content} (id: {fact.id})"


@mcp.tool()
async def inspect(
    category: str | None = None,
    project: str | None = None,
    limit: int = 50,
    include_stale: bool = False,
    format: str = "text",
) -> str:
    """Browse stored facts for transparency and debugging.

    Args:
        category: Filter by category.
        project: Filter by project name.
        limit: Maximum number of facts to return.
        include_stale: When True, include facts that have been marked stale.
        format: ``text`` (default) or ``json`` (structured envelope).
    """
    try:
        cat = FactCategory(category) if category else None
    except ValueError:
        valid = ", ".join(c.value for c in FactCategory)
        if format == "json":
            return Envelope.failure(
                validation_error(
                    f"Invalid category: {category}",
                    details={"valid": list(FactCategory.__members__.keys())},
                )
            ).to_json()
        return f"Invalid category: {category}. Use one of: {valid}"
    store = _async_store()
    facts = await store.load_active_facts(
        category=cat,
        project=project,
        limit=limit,
        include_stale=include_stale,
    )

    if format == "json":
        data = [
            {
                "id": f.id,
                "category": f.category.value,
                "project": f.project,
                "content": f.content,
                "confidence": f.confidence,
                "stale": f.stale,
                "stale_reason": f.stale_reason,
                "supersedes": f.supersedes,
                "tags": f.tags,
                "created_at": f.created_at.isoformat(),
                "updated_at": f.updated_at.isoformat(),
            }
            for f in facts
        ]
        meta = EnvelopeMeta(
            limit=limit,
            returned=len(data),
            truncated=len(data) >= limit,
            truncation_reason="default_limit" if len(data) >= limit else None,
        )
        return Envelope.success(data=data, meta=meta).to_json()

    if not facts:
        return "No facts found matching the criteria."

    lines = [f"Found {len(facts)} fact(s):\n"]
    for fact in facts:
        meta_str = f"[{fact.category.value}]"
        if fact.project:
            meta_str += f" [{fact.project}]"
        if fact.stale:
            meta_str += " [stale]"
        created = fact.created_at.strftime("%Y-%m-%d")
        lines.append(f"- {meta_str} {fact.content} (id: {fact.id}, created: {created})")
    return "\n".join(lines)


@mcp.tool()
async def correct_memory(
    fact_id: str,
    new_content: str,
    category: str | None = None,
    tags: list[str] | None = None,
    project: str | None = None,
    reason: str = "",
) -> str:
    """Replace ``fact_id`` with a new active fact that supersedes it.

    The original fact is preserved (audit trail) and the new fact records
    ``supersedes=fact_id``. Returns a JSON envelope with the new fact ID and
    superseded fact ID, or a ``not_found`` error if the fact doesn't exist.
    """
    try:
        cat = FactCategory(category) if category else None
    except ValueError:
        return Envelope.failure(
            validation_error(
                f"Invalid category: {category}",
                details={"valid": list(FactCategory.__members__.keys())},
            )
        ).to_json()

    store = _async_store()
    new_fact = await store.correct_fact(
        fact_id,
        new_content,
        category=cat,
        tags=tags,
        project=project,
        reason=reason,
    )
    if new_fact is None:
        return Envelope.failure(
            not_found_error(
                f"Active fact {fact_id} not found",
                ids=[fact_id],
            )
        ).to_json()

    return Envelope.success(
        data={
            "new_fact_id": new_fact.id,
            "superseded_fact_id": fact_id,
            "category": new_fact.category.value,
            "project": new_fact.project,
            "content": new_fact.content,
        }
    ).to_json()


@mcp.tool()
async def merge_memories(
    source_ids: list[str],
    merged_content: str,
    category: str | None = None,
    tags: list[str] | None = None,
    project: str | None = None,
    reason: str = "",
) -> str:
    """Consolidate two or more active facts into one new fact.

    The new fact supersedes the first source; every other source has its
    confidence reduced so it falls out of active recall. Returns the new
    fact ID and the list of superseded fact IDs.
    """
    try:
        cat = FactCategory(category) if category else None
    except ValueError:
        return Envelope.failure(
            validation_error(
                f"Invalid category: {category}",
                details={"valid": list(FactCategory.__members__.keys())},
            )
        ).to_json()

    if len(source_ids) < 2:
        return Envelope.failure(
            validation_error(
                "merge_memories requires at least two source IDs",
                details={"received": len(source_ids)},
            )
        ).to_json()

    store = _async_store()
    result = await store.merge_facts(
        source_ids,
        merged_content,
        category=cat,
        tags=tags,
        project=project,
        reason=reason,
    )
    if result is None:
        return Envelope.failure(
            validation_error(
                "Could not merge — sources missing, forgotten, or in different projects.",
                ids=source_ids,
            )
        ).to_json()

    new_fact, superseded_ids = result
    return Envelope.success(
        data={
            "new_fact_id": new_fact.id,
            "superseded_fact_ids": superseded_ids,
            "category": new_fact.category.value,
            "project": new_fact.project,
            "content": new_fact.content,
        }
    ).to_json()


@mcp.tool()
async def mark_stale(fact_id: str, reason: str = "") -> str:
    """Exclude a fact from active recall while preserving it for inspection.

    Stale facts remain visible via ``inspect(include_stale=True)`` and keep
    their supersession history. Use ``correct_memory`` instead if you have a
    replacement fact to record.
    """
    store = _async_store()
    fact = await store.mark_stale(fact_id, reason)
    if fact is None:
        return Envelope.failure(
            not_found_error(f"Fact {fact_id} not found", ids=[fact_id])
        ).to_json()
    return Envelope.success(
        data={
            "fact_id": fact.id,
            "stale": True,
            "stale_reason": fact.stale_reason,
        }
    ).to_json()


@mcp.tool()
async def unmark_stale(fact_id: str) -> str:
    """Reverse a previous ``mark_stale`` call so the fact is recall-eligible again."""
    store = _async_store()
    fact = await store.unmark_stale(fact_id)
    if fact is None:
        return Envelope.failure(
            not_found_error(f"Fact {fact_id} not found", ids=[fact_id])
        ).to_json()
    return Envelope.success(data={"fact_id": fact.id, "stale": False}).to_json()


@mcp.tool()
async def import_memories(source: str = "claude_code") -> str:
    """Import existing memories from Claude Code's per-project memory files.

    One-time bootstrap to seed the knowledge store from ~/.claude/projects/*/memory/.

    Args:
        source: Import source. Currently only "claude_code" is supported.
    """
    if source != "claude_code":
        return f"Unsupported import source: {source}. Use 'claude_code'."

    result = await import_claude_code_memories(store=_async_store())

    if "error" in result:
        return result["error"]
    if result.get("message") and result.get("total_facts", 0) == 0:
        return result["message"]

    lines = [
        f"Imported {result['total_facts']} facts from {result['imported_files']} files:\n"
    ]
    for detail in result.get("details", []):
        lines.append(
            f"- {detail['file']} ({detail['project']}): {detail['facts_extracted']} facts"
        )
    return "\n".join(lines)


@mcp.tool()
async def purge() -> str:
    """Permanently remove forgotten and expired facts from storage.

    Use this to reclaim space after bulk forget operations or when
    expired facts have accumulated.
    """
    store = _async_store()
    result = await store.purge()
    if result["purged"] == 0:
        return "Nothing to purge — store is clean."
    return f"Purged {result['purged']} facts ({result['retained']} retained)."


@mcp.tool()
async def rename_project(old_project: str, new_project: str) -> str:
    """Bulk-rename facts and candidates from one project to another.

    Args:
        old_project: Current project name to match.
        new_project: New project name to set.
    """
    store = _async_store()
    count = await store.rename_project(old_project, new_project)
    if count == 0:
        return f"No facts or candidates found with project '{old_project}'."
    return f"Renamed {count} record(s) from project '{old_project}' → '{new_project}'."


@mcp.tool()
async def synthesize(
    project: str | None = None,
    dry_run: bool = True,
) -> str:
    """Consolidate and clean up stored memories using LLM analysis.

    Removes duplicates, merges related facts, clears stale entries,
    and improves clarity. Run with dry_run=True first to preview changes.

    Args:
        project: Optional project filter. If None, processes all projects.
        dry_run: If True (default), preview changes without applying them.
    """
    result = await _synthesize(project=project, dry_run=dry_run, store=_async_store())
    return format_synthesis_result(result, dry_run=dry_run)


@mcp.tool()
async def doctor(
    check_provider_flag: bool = False,
    repair: bool = False,
    repair_jsonl: bool = False,
    recover_transactions: bool = False,
) -> str:
    """Run a structured health check on the memory store.

    Read-only by default. Pass ``repair=True`` together with one of the
    repair-specific flags to run safe recovery actions.

    Args:
        check_provider_flag: When True, verify the LLM provider with a
            minimal call. Reported separately from storage health.
        repair: Master switch for opt-in repairs.
        repair_jsonl: When ``repair=True``, drop unparseable JSONL lines.
        recover_transactions: When ``repair=True``, roll forward any prepared
            transactions.
    """
    store = _async_store()

    provider_issue = None
    if check_provider_flag:
        provider_issue = await check_provider()

    report = run_doctor(
        store,
        check_provider_flag=check_provider_flag,
        provider_issue=provider_issue,
    )

    repair_summary = None
    if repair and (repair_jsonl or recover_transactions):
        repair_summary = repair_store(
            store,
            repair_jsonl=repair_jsonl,
            recover_transactions=recover_transactions,
        )

    return Envelope.success(
        data={
            "report": report.model_dump(mode="json"),
            "repair": repair_summary,
        }
    ).to_json()


@mcp.tool()
async def memory_stats() -> str:
    """Show memory system statistics: fact counts, storage size, category breakdown."""
    store = _async_store()
    stats = await store.stats()

    lines = [
        "# Engram Stats\n",
        f"**Total facts:** {stats['total_facts']}",
        f"**Active facts:** {stats['active_facts']}",
        f"**Forgotten facts:** {stats['forgotten_facts']}",
        f"**Expired facts:** {stats['expired_facts']}",
        f"**Pending candidates:** {stats['pending_candidates']}",
        f"**Storage size:** {stats['storage_bytes']:,} bytes\n",
        "## By Category",
    ]
    for cat, count in stats["by_category"].items():
        lines.append(f"- {cat}: {count}")

    if stats["by_project"]:
        lines.append("\n## By Project")
        for proj, count in stats["by_project"].items():
            lines.append(f"- {proj}: {count}")

    return "\n".join(lines)


def _format_recall_summary(records, heading: str) -> list[str]:
    """Render the tier/quality/token breakdown for a slice of recall records.

    Shared between the aggregate view and per-selector-version sub-sections so
    both layouts stay consistent.
    """
    total = len(records)
    tier_counts = Counter(r.tier for r in records)
    quality_counts = Counter(r.quality for r in records if r.quality)
    latencies = [r.latency_ms for r in records]
    avg_latency = sum(latencies) / total

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

    lines = [
        f"{heading}",
        f"**Total queries:** {total}",
        f"**Avg latency:** {avg_latency:.0f}ms\n",
        "## By Tier",
    ]
    for tier in sorted(tier_counts):
        count = tier_counts[tier]
        avg = sum(tier_latency[tier]) / count
        pct = count / total * 100
        lines.append(f"- Tier {tier}: {count} ({pct:.0f}%) — avg {avg:.0f}ms")

    if quality_counts:
        lines.append("\n## By Quality")
        for quality in ("high", "medium", "low", "none"):
            if quality in quality_counts:
                count = quality_counts[quality]
                pct = count / total * 100
                lines.append(f"- {quality}: {count} ({pct:.0f}%)")

    lines.append("\n## Token Usage")
    lines.append(
        f"- LLM calls (reported): {total_llm_calls if llm_call_totals else '-'}"
    )
    lines.append(
        f"- Input tokens: {sum_input:,}" if input_totals else "- Input tokens: -"
    )
    lines.append(
        f"- Cached input tokens: {sum_cached:,}"
        if cached_totals
        else "- Cached input tokens: -"
    )
    if cache_hit_ratio is not None:
        lines.append(f"- Cache hit ratio: {cache_hit_ratio * 100:.1f}%")
    else:
        lines.append("- Cache hit ratio: -")
    return lines


@mcp.tool()
async def recall_stats() -> str:
    """Show recall quality and performance statistics from the recall log."""
    store = _async_store()
    records = await store.load_recall_log(limit=500)
    if not records:
        return "No recall data yet."

    lines = _format_recall_summary(records, "# Recall Stats\n")

    # Split by selector version when the log contains more than one. Otherwise
    # note the single version inline so operators can see which rules produced
    # the numbers above.
    selector_versions = sorted(
        {r.selector_version for r in records if r.selector_version}
    )
    if len(selector_versions) >= 2:
        for version in selector_versions:
            subset = [r for r in records if r.selector_version == version]
            lines.append("")
            lines.extend(
                _format_recall_summary(subset, heading=f"# Selector {version}\n")
            )
    elif selector_versions:
        lines.insert(1, f"*Selector version:* `{selector_versions[0]}`")

    return "\n".join(lines)


def main():
    """Entry point for the MCP server."""
    global _store
    configure_logging()
    _store = AsyncFactStore()
    mcp.run()


if __name__ == "__main__":
    main()
