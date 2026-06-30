"""Engram MCP Server — structured, cross-project memory for coding agents."""

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent

from engram.core.config import configure_logging, get_settings
from engram.core.interfaces import Envelope, validation_error
from engram.operations import (
    EXIT_VALIDATION,
    OperationResult,
    async_store,
    approve_candidates as op_approve_candidates,
    audit_memories as op_audit_memories,
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
    sync as op_sync,
    synthesize as op_synthesize,
    unmark_stale as op_unmark_stale,
)
from engram.core.provenance import DEFAULT_MAX_PREFILTER_MATCHES, DEFAULT_MAX_SOURCES
from engram.storage.store import AsyncFactStore, FactStore

INSTRUCTIONS = """Engram — structured, cross-project memory for coding agents.

A cross-project, structured memory system that uses LLM-powered retrieval
instead of vector search. Facts are extracted, categorized, and stored as
structured knowledge. Tiered retrieval uses deterministic fast paths first,
then a multi-lens search-and-synthesis path for complex queries.

Tools:
- remember: Store new memories (extracts structured facts from natural language)
- suggest_memories: Propose memories for review without storing them immediately
- list_candidates: Browse pending/reviewed memory suggestions
- approve_candidates: Promote reviewed suggestions into active memory
- reject_candidates: Dismiss candidates with an audit trail
- recall: Search memory using tiered retrieval
- recall_context: Recall as answer or compact prompt block
- forget: Remove a fact from active memory
- edit_fact: Edit a fact in place
- inspect: Browse stored facts
- import_memories: Bootstrap from Claude Code memory files
- memory_stats: View memory system statistics
- recall_stats: View recall quality and performance statistics
- purge: Permanently remove forgotten and expired facts
- rename_project: Bulk-rename facts and candidates
- synthesize: Consolidate and clean up memory
- audit_memories: Suggest duplicate, stale, and contradictory memory cleanup
"""


def _render(result: OperationResult, *, format: str = "text") -> str:
    return result.render(as_json=format == "json")


def _resolve_recall_max_sources(
    max_sources: int,
    limit: int | None,
) -> int:
    """Map optional ``limit`` alias to ``max_sources`` for recall tools."""
    if limit is None:
        return max_sources
    if max_sources != DEFAULT_MAX_SOURCES and max_sources != limit:
        raise ValueError("limit and max_sources conflict")
    return limit


def _recall_limit_conflict_result() -> OperationResult:
    message = (
        "Cannot pass limit and max_sources with different values; "
        "use one or set them equal."
    )
    return OperationResult(
        envelope=Envelope.failure(
            validation_error(message, details={"parameter": "limit"})
        ),
        text=message,
        exit_code=EXIT_VALIDATION,
    )


def _tool_result(
    result: OperationResult,
    *,
    format: str = "text",
    force_json_text: bool = False,
) -> ToolResult:
    text = (
        result.envelope.to_json() if force_json_text else _render(result, format=format)
    )
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=result.envelope.model_dump(mode="json"),
    )


logger = logging.getLogger(__name__)
StoreGetter = Callable[[], AsyncFactStore]


def _store_getter(store: FactStore | AsyncFactStore | None) -> StoreGetter:
    store_ref: FactStore | AsyncFactStore | None = store

    def get_store() -> AsyncFactStore:
        nonlocal store_ref
        if store_ref is None:
            store_ref = AsyncFactStore()
        return async_store(store_ref)

    return get_store


def _make_lifespan(get_store: StoreGetter):
    @asynccontextmanager
    async def lifespan(_app: FastMCP):
        settings = get_settings()
        auto_sync_task: asyncio.Task | None = None
        if settings.sync_enabled:
            from engram.storage.sync import auto_sync_loop

            data_dir = get_store().data_dir
            auto_sync_task = asyncio.create_task(
                auto_sync_loop(
                    data_dir,
                    interval=settings.sync_interval,
                    timeout=settings.sync_timeout,
                )
            )
            logger.info(
                "engram-sync auto-loop scheduled (interval=%.1fs)",
                settings.sync_interval,
            )
        try:
            yield
        finally:
            if auto_sync_task is not None:
                auto_sync_task.cancel()
                try:
                    await auto_sync_task
                except (asyncio.CancelledError, Exception):
                    pass
                # Final sync attempt on shutdown — best-effort.
                from engram.storage.sync import run_final_sync

                await run_final_sync(
                    get_store().data_dir, timeout=settings.sync_timeout
                )

    return lifespan


def create_mcp(store: FactStore | AsyncFactStore | None = None) -> FastMCP:
    """Create an Engram MCP server with an isolated, injectable store."""
    get_store = _store_getter(store)
    app = FastMCP(
        "engram",
        instructions=INSTRUCTIONS,
        lifespan=_make_lifespan(get_store),
    )
    _register_capture_tools(app, get_store)
    _register_candidate_tools(app, get_store)
    _register_recall_tools(app, get_store)
    _register_fact_edit_tools(app, get_store)
    _register_fact_correction_tools(app, get_store)
    _register_maintenance_tools(app, get_store)
    _register_reporting_tools(app, get_store)
    return app


def _register_capture_tools(app: FastMCP, get_store: StoreGetter) -> None:
    @app.tool()
    async def remember(
        content: str,
        source: str = "conversation",
        project: str | None = None,
        format: str = "text",
    ) -> ToolResult:
        """Store a new memory. Extracts structured facts from natural language input."""
        result = await op_remember(
            content,
            source=source,
            project=project,
            store=get_store(),
        )
        return _tool_result(result, format=format)

    @app.tool()
    async def suggest_memories(
        content: str,
        source: str = "conversation",
        project: str | None = None,
        format: str = "text",
    ) -> ToolResult:
        """Propose memories for review without storing them as active facts."""
        result = await op_suggest_memories(
            content,
            source=source,
            project=project,
            store=get_store(),
        )
        return _tool_result(result, format=format)


def _register_candidate_tools(app: FastMCP, get_store: StoreGetter) -> None:
    @app.tool()
    async def list_candidates(
        status: str = "pending",
        project: str | None = None,
        search: str | None = None,
        limit: int = 50,
        format: str = "text",
    ) -> ToolResult:
        """Browse queued memory candidates for review."""
        result = await op_list_candidates(
            status=status,
            project=project,
            search=search,
            limit=limit,
            store=get_store(),
        )
        return _tool_result(result, format=format)

    @app.tool()
    async def approve_candidates(
        candidate_ids: list[str],
        edits: dict[str, str] | None = None,
        format: str = "text",
    ) -> ToolResult:
        """Promote reviewed candidates into active memory."""
        result = await op_approve_candidates(
            candidate_ids,
            edits=edits,
            store=get_store(),
        )
        return _tool_result(result, format=format)

    @app.tool()
    async def reject_candidates(
        candidate_ids: list[str],
        reason: str = "",
        format: str = "text",
    ) -> ToolResult:
        """Reject proposed candidates without storing them as active memory."""
        result = await op_reject_candidates(
            candidate_ids,
            reason=reason,
            store=get_store(),
        )
        return _tool_result(result, format=format)


def _register_recall_tools(app: FastMCP, get_store: StoreGetter) -> None:
    @app.tool()
    async def recall(
        query: str,
        project: str | None = None,
        format: str = "text",
        with_provenance: bool = False,
        max_sources: int = DEFAULT_MAX_SOURCES,
        max_prefilter_matches: int = DEFAULT_MAX_PREFILTER_MATCHES,
        limit: int | None = None,
    ) -> ToolResult:
        """Search memory using tiered agentic retrieval.

        ``limit`` is an alias for ``max_sources`` (for consistency with inspect
        and list tools). Do not pass both with different values.
        """
        try:
            resolved_max_sources = _resolve_recall_max_sources(max_sources, limit)
        except ValueError:
            result = _recall_limit_conflict_result()
            return _tool_result(
                result, format=format, force_json_text=bool(result.exit_code)
            )
        result = await op_recall(
            query,
            project=project,
            format=format,
            with_provenance=with_provenance,
            max_sources=resolved_max_sources,
            max_prefilter_matches=max_prefilter_matches,
            store=get_store(),
        )
        return _tool_result(
            result, format=format, force_json_text=bool(result.exit_code)
        )

    @app.tool()
    async def recall_trace(
        query: str,
        project: str | None = None,
        verbose: bool = False,
        max_sources: int = DEFAULT_MAX_SOURCES,
        max_prefilter_matches: int = DEFAULT_MAX_PREFILTER_MATCHES,
        limit: int | None = None,
    ) -> ToolResult:
        """Run recall and return a structured trace for debugging.

        ``limit`` is an alias for ``max_sources``. Do not pass both with
        different values.
        """
        try:
            resolved_max_sources = _resolve_recall_max_sources(max_sources, limit)
        except ValueError:
            return _tool_result(_recall_limit_conflict_result(), format="json")
        result = await op_recall_trace(
            query,
            project=project,
            verbose=verbose,
            max_sources=resolved_max_sources,
            max_prefilter_matches=max_prefilter_matches,
            store=get_store(),
        )
        return _tool_result(result, format="json")

    @app.tool()
    async def recall_context(
        query: str,
        project: str | None = None,
        mode: str = "answer",
        format: str = "text",
    ) -> ToolResult:
        """Recall either a natural-language answer or a compact prompt block."""
        result = await op_recall_context(
            query,
            project=project,
            mode=mode,
            store=get_store(),
        )
        return _tool_result(result, format=format)


def _register_fact_edit_tools(app: FastMCP, get_store: StoreGetter) -> None:
    @app.tool()
    async def forget(
        fact_id: str,
        reason: str = "",
        format: str = "text",
    ) -> ToolResult:
        """Mark a fact as forgotten (soft delete)."""
        result = await op_forget(fact_id, reason=reason, store=get_store())
        return _tool_result(result, format=format)

    @app.tool()
    async def edit_fact(
        fact_id: str,
        content: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        project: str | None = None,
        format: str = "text",
    ) -> ToolResult:
        """Edit a fact in place without losing its ID or supersession chain."""
        result = await op_edit_fact(
            fact_id,
            content=content,
            category=category,
            tags=tags,
            project=project,
            store=get_store(),
        )
        return _tool_result(result, format=format)

    @app.tool()
    async def inspect(
        category: str | None = None,
        project: str | None = None,
        limit: int = 50,
        include_stale: bool = False,
        format: str = "text",
    ) -> ToolResult:
        """Browse stored facts for transparency and debugging."""
        result = await op_inspect(
            category=category,
            project=project,
            limit=limit,
            include_stale=include_stale,
            store=get_store(),
        )
        return _tool_result(result, format=format)


def _register_fact_correction_tools(app: FastMCP, get_store: StoreGetter) -> None:
    @app.tool()
    async def correct_memory(
        fact_id: str,
        new_content: str,
        category: str | None = None,
        tags: list[str] | None = None,
        project: str | None = None,
        reason: str = "",
    ) -> ToolResult:
        """Replace ``fact_id`` with a new active fact that supersedes it."""
        result = await op_correct_memory(
            fact_id,
            new_content,
            category=category,
            tags=tags,
            project=project,
            reason=reason,
            store=get_store(),
        )
        return _tool_result(result, format="json")

    @app.tool()
    async def merge_memories(
        source_ids: list[str],
        merged_content: str,
        category: str | None = None,
        tags: list[str] | None = None,
        project: str | None = None,
        reason: str = "",
    ) -> ToolResult:
        """Consolidate two or more active facts into one new fact."""
        result = await op_merge_memories(
            source_ids,
            merged_content,
            category=category,
            tags=tags,
            project=project,
            reason=reason,
            store=get_store(),
        )
        return _tool_result(result, format="json")

    @app.tool()
    async def mark_stale(fact_id: str, reason: str = "") -> ToolResult:
        """Exclude a fact from active recall while preserving it for inspection."""
        result = await op_mark_stale(fact_id, reason=reason, store=get_store())
        return _tool_result(result, format="json")

    @app.tool()
    async def unmark_stale(fact_id: str) -> ToolResult:
        """Reverse a previous ``mark_stale`` call so the fact is recall-eligible."""
        result = await op_unmark_stale(fact_id, store=get_store())
        return _tool_result(result, format="json")


def _register_maintenance_tools(app: FastMCP, get_store: StoreGetter) -> None:
    @app.tool()
    async def import_memories(
        source: str = "claude_code",
        format: str = "text",
    ) -> ToolResult:
        """Import existing memories from Claude Code's per-project memory files."""
        result = await op_import_memories(source=source, store=get_store())
        return _tool_result(result, format=format)

    @app.tool()
    async def purge(format: str = "text") -> ToolResult:
        """Permanently remove forgotten and expired facts from storage."""
        result = await op_purge(store=get_store())
        return _tool_result(result, format=format)

    @app.tool()
    async def rename_project(
        old_project: str,
        new_project: str,
        format: str = "text",
    ) -> ToolResult:
        """Bulk-rename facts and candidates from one project to another."""
        result = await op_rename_project(
            old_project,
            new_project,
            store=get_store(),
        )
        return _tool_result(result, format=format)

    @app.tool()
    async def synthesize(
        project: str | None = None,
        dry_run: bool = True,
        format: str = "text",
    ) -> ToolResult:
        """Consolidate and clean up stored memories using LLM analysis."""
        result = await op_synthesize(
            project=project,
            dry_run=dry_run,
            store=get_store(),
        )
        return _tool_result(result, format=format)

    @app.tool()
    async def audit_memories(
        project: str | None = None,
        format: str = "text",
    ) -> ToolResult:
        """Suggest duplicate, stale, and contradictory memory cleanup.

        This is read-only: it emits reviewable suggestions and example commands
        but does not mutate the memory store.
        """
        result = await op_audit_memories(project=project, store=get_store())
        return _tool_result(result, format=format)

    @app.tool()
    async def doctor(
        check_provider_flag: bool = False,
        repair: bool = False,
        repair_jsonl: bool = False,
        recover_transactions: bool = False,
        repair_orphaned_supersessions: bool = False,
    ) -> ToolResult:
        """Run a structured health check on the memory store."""
        result = await op_doctor(
            check_provider_flag=check_provider_flag,
            repair=(
                repair
                or repair_jsonl
                or recover_transactions
                or repair_orphaned_supersessions
            ),
            repair_jsonl=repair_jsonl,
            recover_transactions=recover_transactions,
            repair_orphaned_supersessions=repair_orphaned_supersessions,
            store=get_store(),
        )
        return _tool_result(result, format="json")


def _register_reporting_tools(app: FastMCP, get_store: StoreGetter) -> None:
    @app.tool()
    async def memory_stats(format: str = "text") -> ToolResult:
        """Show memory system statistics: fact counts, storage size, category breakdown."""
        result = await op_memory_stats(store=get_store())
        return _tool_result(result, format=format)

    @app.tool()
    async def recall_stats(
        format: str = "text",
        limit: int = 500,
        since: str | None = None,
        include_records: bool = False,
    ) -> ToolResult:
        """Show recall quality and performance statistics from the recall log."""
        result = await op_recall_stats(
            limit=limit,
            since=since,
            include_records=include_records,
            store=get_store(),
        )
        return _tool_result(result, format=format)

    @app.tool()
    async def sync(
        format: str = "text",
        timeout: float = 30.0,
    ) -> ToolResult:
        """Git-backed sync of the data directory against its configured remote.

        Runs ``git fetch`` + ``git pull --rebase`` + ``git push`` against the
        Engram data directory. Requires the data directory to be a git
        repository with at least one remote — see CLAUDE.md for setup.
        """
        result = await op_sync(timeout=timeout, store=get_store())
        return _tool_result(result, format=format)


mcp = create_mcp()


def main(mcp_factory: Callable[[], FastMCP] = create_mcp) -> None:
    """Entry point for the MCP server."""
    configure_logging()
    mcp_factory().run()


if __name__ == "__main__":
    main()
