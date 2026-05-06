"""Engram MCP Server — structured, cross-project memory for coding agents."""

from fastmcp import FastMCP

from engram.config import configure_logging
from engram.operations import (
    async_store,
    approve_candidates as op_approve_candidates,
    correct_memory as op_correct_memory,
    doctor as op_doctor,
    edit_fact as op_edit_fact,
    forget as op_forget,
    import_memories as op_import_memories,
    inspect as op_inspect,
    list_candidates as op_list_candidates,
    mark_stale as op_mark_stale,
    memory_stats as op_memory_stats,
    merge_memories as op_merge_memories,
    purge as op_purge,
    recall as op_recall,
    recall_context as op_recall_context,
    recall_stats as op_recall_stats,
    recall_trace as op_recall_trace,
    reject_candidates as op_reject_candidates,
    remember as op_remember,
    rename_project as op_rename_project,
    suggest_memories as op_suggest_memories,
    synthesize as op_synthesize,
    unmark_stale as op_unmark_stale,
)
from engram.provenance import DEFAULT_MAX_PREFILTER_MATCHES, DEFAULT_MAX_SOURCES
from engram.store import AsyncFactStore, FactStore

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
_store: AsyncFactStore | FactStore | None = None  # set in main(); tests monkeypatch.


def _async_store() -> AsyncFactStore:
    if _store is None:
        raise RuntimeError(
            "engram.server._store is not initialized — call server.main() or "
            "assign server._store before invoking MCP tools."
        )
    return async_store(_store)


def _render(result, *, format: str = "text") -> str:
    return result.render(as_json=format == "json")


@mcp.tool()
async def remember(
    content: str,
    source: str = "conversation",
    project: str | None = None,
    format: str = "text",
) -> str:
    """Store a new memory. Extracts structured facts from natural language input."""
    result = await op_remember(
        content,
        source=source,
        project=project,
        store=_async_store(),
    )
    return _render(result, format=format)


@mcp.tool()
async def suggest_memories(
    content: str,
    source: str = "conversation",
    project: str | None = None,
    format: str = "text",
) -> str:
    """Propose memories for review without storing them as active facts."""
    result = await op_suggest_memories(
        content,
        source=source,
        project=project,
        store=_async_store(),
    )
    return _render(result, format=format)


@mcp.tool()
async def list_candidates(
    status: str = "pending",
    project: str | None = None,
    search: str | None = None,
    limit: int = 50,
    format: str = "text",
) -> str:
    """Browse queued memory candidates for review."""
    result = await op_list_candidates(
        status=status,
        project=project,
        search=search,
        limit=limit,
        store=_async_store(),
    )
    return _render(result, format=format)


@mcp.tool()
async def approve_candidates(
    candidate_ids: list[str],
    edits: dict[str, str] | None = None,
    format: str = "text",
) -> str:
    """Promote reviewed candidates into active memory."""
    result = await op_approve_candidates(
        candidate_ids,
        edits=edits,
        store=_async_store(),
    )
    return _render(result, format=format)


@mcp.tool()
async def reject_candidates(
    candidate_ids: list[str],
    reason: str = "",
    format: str = "text",
) -> str:
    """Reject proposed candidates without storing them as active memory."""
    result = await op_reject_candidates(
        candidate_ids,
        reason=reason,
        store=_async_store(),
    )
    return _render(result, format=format)


@mcp.tool()
async def recall(
    query: str,
    project: str | None = None,
    format: str = "text",
    with_provenance: bool = False,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_prefilter_matches: int = DEFAULT_MAX_PREFILTER_MATCHES,
) -> str:
    """Search memory using tiered agentic retrieval."""
    result = await op_recall(
        query,
        project=project,
        format=format,
        with_provenance=with_provenance,
        max_sources=max_sources,
        max_prefilter_matches=max_prefilter_matches,
        store=_async_store(),
    )
    return result.envelope.to_json() if result.exit_code else _render(result, format=format)


@mcp.tool()
async def recall_trace(
    query: str,
    project: str | None = None,
    verbose: bool = False,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_prefilter_matches: int = DEFAULT_MAX_PREFILTER_MATCHES,
) -> str:
    """Run recall and return a structured trace for debugging."""
    result = await op_recall_trace(
        query,
        project=project,
        verbose=verbose,
        max_sources=max_sources,
        max_prefilter_matches=max_prefilter_matches,
        store=_async_store(),
    )
    return result.envelope.to_json()


@mcp.tool()
async def recall_context(
    query: str,
    project: str | None = None,
    mode: str = "answer",
    format: str = "text",
) -> str:
    """Recall either a natural-language answer or a compact prompt block."""
    result = await op_recall_context(
        query,
        project=project,
        mode=mode,
        store=_async_store(),
    )
    return _render(result, format=format)


@mcp.tool()
async def forget(fact_id: str, reason: str = "", format: str = "text") -> str:
    """Mark a fact as forgotten (soft delete)."""
    result = await op_forget(fact_id, reason=reason, store=_async_store())
    return _render(result, format=format)


@mcp.tool()
async def edit_fact(
    fact_id: str,
    content: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    project: str | None = None,
    format: str = "text",
) -> str:
    """Edit a fact in place without losing its ID, timestamps, or supersession chain."""
    result = await op_edit_fact(
        fact_id,
        content=content,
        category=category,
        tags=tags,
        project=project,
        store=_async_store(),
    )
    return _render(result, format=format)


@mcp.tool()
async def inspect(
    category: str | None = None,
    project: str | None = None,
    limit: int = 50,
    include_stale: bool = False,
    format: str = "text",
) -> str:
    """Browse stored facts for transparency and debugging."""
    result = await op_inspect(
        category=category,
        project=project,
        limit=limit,
        include_stale=include_stale,
        store=_async_store(),
    )
    return _render(result, format=format)


@mcp.tool()
async def correct_memory(
    fact_id: str,
    new_content: str,
    category: str | None = None,
    tags: list[str] | None = None,
    project: str | None = None,
    reason: str = "",
) -> str:
    """Replace ``fact_id`` with a new active fact that supersedes it."""
    result = await op_correct_memory(
        fact_id,
        new_content,
        category=category,
        tags=tags,
        project=project,
        reason=reason,
        store=_async_store(),
    )
    return result.envelope.to_json()


@mcp.tool()
async def merge_memories(
    source_ids: list[str],
    merged_content: str,
    category: str | None = None,
    tags: list[str] | None = None,
    project: str | None = None,
    reason: str = "",
) -> str:
    """Consolidate two or more active facts into one new fact."""
    result = await op_merge_memories(
        source_ids,
        merged_content,
        category=category,
        tags=tags,
        project=project,
        reason=reason,
        store=_async_store(),
    )
    return result.envelope.to_json()


@mcp.tool()
async def mark_stale(fact_id: str, reason: str = "") -> str:
    """Exclude a fact from active recall while preserving it for inspection."""
    result = await op_mark_stale(fact_id, reason=reason, store=_async_store())
    return result.envelope.to_json()


@mcp.tool()
async def unmark_stale(fact_id: str) -> str:
    """Reverse a previous ``mark_stale`` call so the fact is recall-eligible again."""
    result = await op_unmark_stale(fact_id, store=_async_store())
    return result.envelope.to_json()


@mcp.tool()
async def import_memories(source: str = "claude_code", format: str = "text") -> str:
    """Import existing memories from Claude Code's per-project memory files."""
    result = await op_import_memories(source=source, store=_async_store())
    return _render(result, format=format)


@mcp.tool()
async def purge(format: str = "text") -> str:
    """Permanently remove forgotten and expired facts from storage."""
    result = await op_purge(store=_async_store())
    return _render(result, format=format)


@mcp.tool()
async def rename_project(
    old_project: str,
    new_project: str,
    format: str = "text",
) -> str:
    """Bulk-rename facts and candidates from one project to another."""
    result = await op_rename_project(
        old_project,
        new_project,
        store=_async_store(),
    )
    return _render(result, format=format)


@mcp.tool()
async def synthesize(
    project: str | None = None,
    dry_run: bool = True,
    format: str = "text",
) -> str:
    """Consolidate and clean up stored memories using LLM analysis."""
    result = await op_synthesize(
        project=project,
        dry_run=dry_run,
        store=_async_store(),
    )
    return _render(result, format=format)


@mcp.tool()
async def doctor(
    check_provider_flag: bool = False,
    repair: bool = False,
    repair_jsonl: bool = False,
    recover_transactions: bool = False,
) -> str:
    """Run a structured health check on the memory store."""
    result = await op_doctor(
        check_provider_flag=check_provider_flag,
        repair=repair,
        repair_jsonl=repair_jsonl,
        recover_transactions=recover_transactions,
        store=_async_store(),
    )
    return result.envelope.to_json()


@mcp.tool()
async def memory_stats(format: str = "text") -> str:
    """Show memory system statistics: fact counts, storage size, category breakdown."""
    result = await op_memory_stats(store=_async_store())
    return _render(result, format=format)


@mcp.tool()
async def recall_stats(format: str = "text") -> str:
    """Show recall quality and performance statistics from the recall log."""
    result = await op_recall_stats(store=_async_store())
    return _render(result, format=format)


def main():
    """Entry point for the MCP server."""
    global _store
    configure_logging()
    _store = AsyncFactStore()
    mcp.run()


if __name__ == "__main__":
    main()
