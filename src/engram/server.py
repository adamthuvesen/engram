"""Engram MCP Server — structured, cross-project memory for coding agents."""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import wraps
from inspect import signature
from typing import Literal, ParamSpec

from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from mcp.types import TextContent

from engram.core.config import (
    DEFAULT_HTTP_HOST,
    DEFAULT_HTTP_PORT,
    configure_logging,
)
from engram.core.interfaces import Envelope, validation_error
from engram.core.provenance import DEFAULT_MAX_PREFILTER_MATCHES, DEFAULT_MAX_SOURCES
from engram.operations import (
    EXIT_VALIDATION,
    OperationResult,
    exception_result,
    invalid_format_result,
)
from engram.operations import (
    approve_candidates as op_approve_candidates,
)
from engram.operations import (
    audit_memories as op_audit_memories,
)
from engram.operations import (
    correct_memory as op_correct_memory,
)
from engram.operations import (
    doctor as op_doctor,
)
from engram.operations import (
    edit_fact as op_edit_fact,
)
from engram.operations import (
    forget as op_forget,
)
from engram.operations import (
    import_memories as op_import_memories,
)
from engram.operations import (
    inspect as op_inspect,
)
from engram.operations import (
    list_candidates as op_list_candidates,
)
from engram.operations import (
    mark_stale as op_mark_stale,
)
from engram.operations import (
    memory_stats as op_memory_stats,
)
from engram.operations import (
    merge_memories as op_merge_memories,
)
from engram.operations import (
    purge as op_purge,
)
from engram.operations import (
    recall as op_recall,
)
from engram.operations import (
    recall_context as op_recall_context,
)
from engram.operations import (
    recall_stats as op_recall_stats,
)
from engram.operations import (
    recall_trace as op_recall_trace,
)
from engram.operations import (
    reject_candidates as op_reject_candidates,
)
from engram.operations import (
    remember as op_remember,
)
from engram.operations import (
    rename_project as op_rename_project,
)
from engram.operations import (
    suggest_memories as op_suggest_memories,
)
from engram.operations import (
    sync as op_sync,
)
from engram.operations import (
    unmark_stale as op_unmark_stale,
)
from engram.operations import (
    upkeep as op_upkeep,
)
from engram.runtime import Runtime
from engram.storage.store import AsyncFactStore, FactStore

INSTRUCTIONS = """Engram — durable, cross-project memory for coding agents.

Pass `project` as your working directory path (or the repo name) on every
call; Engram maps paths and worktrees to the repository's project scope.

Read:
- Session start: recall_context(mode="brief", project=<cwd>) for the project's
  synthesized brief.
- Before deciding something the user may have an opinion on, or when stuck:
  recall(query, project=<cwd>). Returns dated memory cards in milliseconds
  (no LLM). Use mode="answer" only when you need a synthesized answer.
- Cards flagged "unverified" or "time-bound" may be outdated: check the code.

Write:
- remember(content, project=<cwd>) after learning something durable: a user
  preference or correction, a decision and its rationale, a pitfall and its
  fix, a convention, a workflow. Plain language is fine; Engram extracts
  cards, merges them with what it already knows, and retires what the new
  information contradicts. Skip progress, test results, and branch/PR
  bookkeeping.
- When a recalled card is wrong: correct_memory(fact_id, new_content); when
  it is no longer true: mark_stale(fact_id, reason).

Engram keeps itself current: background upkeep merges fragmented cards,
splits overloaded ones, retires junk, verifies file anchors against the
repository, and refreshes project briefs.
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
    message = "Cannot pass limit and max_sources with different values; use one or set them equal."
    return OperationResult(
        envelope=Envelope.failure(validation_error(message, details={"parameter": "limit"})),
        text=message,
        exit_code=EXIT_VALIDATION,
    )


def _tool_result(
    result: OperationResult,
    *,
    format: str = "text",
    force_json_text: bool = False,
) -> ToolResult:
    text = result.envelope.to_json() if force_json_text else _render(result, format=format)
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=result.envelope.model_dump(mode="json"),
        is_error=result.is_error,
    )


logger = logging.getLogger(__name__)
StoreGetter = Runtime


def _make_lifespan(
    runtime: Runtime,
) -> Callable[[FastMCP], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(_app: FastMCP) -> AsyncIterator[None]:
        with runtime.bind():
            try:
                await runtime.start_background()
                yield
            finally:
                await runtime.aclose()

    return lifespan


P = ParamSpec("P")


def _tool(
    app: FastMCP, runtime: Runtime
) -> Callable[[Callable[P, Awaitable[ToolResult]]], Callable[P, Awaitable[ToolResult]]]:
    """Validate rendering before dispatch and bind request dependencies."""

    def register(
        function: Callable[P, Awaitable[ToolResult]],
    ) -> Callable[P, Awaitable[ToolResult]]:
        call_signature = signature(function)

        @wraps(function)
        async def invoke(*args: P.args, **kwargs: P.kwargs) -> ToolResult:
            arguments = call_signature.bind(*args, **kwargs)
            output_format = arguments.arguments.get("format", "text")
            if output_format not in ("text", "json", ""):
                return _tool_result(invalid_format_result(output_format), format="json")
            try:
                with runtime.bind():
                    return await function(*args, **kwargs)
            except Exception as exc:
                return _tool_result(exception_result(exc), format=output_format)

        app.tool()(invoke)
        return invoke

    return register


def create_mcp(
    store: FactStore | AsyncFactStore | None = None, *, runtime: Runtime | None = None
) -> FastMCP:
    """Create an Engram MCP server with an isolated, injectable store."""
    if runtime is not None and store is not None:
        raise ValueError("Pass store or runtime, not both")
    get_store = runtime or Runtime(store=store)
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
    @_tool(app, get_store)
    async def remember(
        content: str,
        source: str = "conversation",
        project: str | None = None,
        format: str = "text",
    ) -> ToolResult:
        """Save durable knowledge from plain language.

        One call extracts memory cards, merges them with related existing
        cards (updating or replacing them), and retires cards the input shows
        are no longer true. Pass ``project`` as your working directory.
        """
        result = await op_remember(
            content,
            source=source,
            project=project,
        )
        return _tool_result(result, format=format)

    @_tool(app, get_store)
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
        )
        return _tool_result(result, format=format)


def _register_candidate_tools(app: FastMCP, get_store: StoreGetter) -> None:
    @_tool(app, get_store)
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
        )
        return _tool_result(result, format=format)

    @_tool(app, get_store)
    async def approve_candidates(
        candidate_ids: list[str],
        edits: dict[str, str] | None = None,
        format: str = "text",
    ) -> ToolResult:
        """Promote reviewed candidates into active memory."""
        result = await op_approve_candidates(
            candidate_ids,
            edits=edits,
        )
        return _tool_result(result, format=format)

    @_tool(app, get_store)
    async def reject_candidates(
        candidate_ids: list[str],
        reason: str = "",
        format: str = "text",
    ) -> ToolResult:
        """Reject proposed candidates without storing them as active memory."""
        result = await op_reject_candidates(
            candidate_ids,
            reason=reason,
        )
        return _tool_result(result, format=format)


def _register_recall_tools(app: FastMCP, get_store: StoreGetter) -> None:
    @_tool(app, get_store)
    async def recall(
        query: str,
        project: str | None = None,
        mode: str = "cards",
        format: str = "text",
        with_provenance: bool = False,
        max_sources: int = DEFAULT_MAX_SOURCES,
        max_prefilter_matches: int = DEFAULT_MAX_PREFILTER_MATCHES,
        limit: int | None = None,
    ) -> ToolResult:
        """Find memories relevant to ``query``.

        ``mode="cards"`` (default) returns ranked, dated memory cards without
        an LLM call. ``mode="answer"`` synthesizes an answer with one LLM call.
        Pass ``project`` as your working directory. ``limit`` is an alias for
        ``max_sources``; do not pass both with different values.
        """
        try:
            resolved_max_sources = _resolve_recall_max_sources(max_sources, limit)
        except ValueError:
            result = _recall_limit_conflict_result()
            return _tool_result(result, format=format, force_json_text=bool(result.exit_code))
        result = await op_recall(
            query,
            project=project,
            mode=mode,
            format=format,
            with_provenance=with_provenance,
            max_sources=resolved_max_sources,
            max_prefilter_matches=max_prefilter_matches,
        )
        return _tool_result(result, format=format, force_json_text=bool(result.exit_code))

    @_tool(app, get_store)
    async def recall_trace(
        query: str,
        project: str | None = None,
        mode: str = "cards",
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
            mode=mode,
            verbose=verbose,
            max_sources=resolved_max_sources,
            max_prefilter_matches=max_prefilter_matches,
        )
        return _tool_result(result, format="json")

    @_tool(app, get_store)
    async def recall_context(
        query: str = "",
        project: str | None = None,
        mode: str = "cards",
        format: str = "text",
    ) -> ToolResult:
        """Context for the current task.

        ``mode``: ``brief`` (the project's synthesized brief; no query needed —
        call at session start), ``cards`` (dated memory cards, no LLM),
        ``answer`` (one LLM call). ``prompt`` is an alias for ``cards``.
        """
        result = await op_recall_context(
            query,
            project=project,
            mode=mode,
        )
        return _tool_result(result, format=format)


def _register_fact_edit_tools(app: FastMCP, get_store: StoreGetter) -> None:
    @_tool(app, get_store)
    async def forget(
        fact_id: str,
        reason: str = "",
        format: str = "text",
    ) -> ToolResult:
        """Mark a fact as forgotten (soft delete)."""
        result = await op_forget(fact_id, reason=reason)
        return _tool_result(result, format=format)

    @_tool(app, get_store)
    async def edit_fact(
        fact_id: str,
        content: str | None = None,
        category: str | None = None,
        memory_key: str | None = None,
        tags: list[str] | None = None,
        retrieval_hints: list[str] | None = None,
        project: str | None = None,
        format: str = "text",
    ) -> ToolResult:
        """Edit a fact in place without losing its ID or supersession chain."""
        result = await op_edit_fact(
            fact_id,
            content=content,
            category=category,
            memory_key=memory_key,
            tags=tags,
            retrieval_hints=retrieval_hints,
            project=project,
        )
        return _tool_result(result, format=format)

    @_tool(app, get_store)
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
        )
        return _tool_result(result, format=format)


def _register_fact_correction_tools(app: FastMCP, get_store: StoreGetter) -> None:
    @_tool(app, get_store)
    async def correct_memory(
        fact_id: str,
        new_content: str,
        category: str | None = None,
        tags: list[str] | None = None,
        memory_key: str | None = None,
        retrieval_hints: list[str] | None = None,
        project: str | None = None,
        reason: str = "",
    ) -> ToolResult:
        """Replace ``fact_id`` with a new active fact that supersedes it."""
        result = await op_correct_memory(
            fact_id,
            new_content,
            category=category,
            tags=tags,
            memory_key=memory_key,
            retrieval_hints=retrieval_hints,
            project=project,
            reason=reason,
        )
        return _tool_result(result, format="json")

    @_tool(app, get_store)
    async def merge_memories(
        source_ids: list[str],
        merged_content: str,
        category: str | None = None,
        tags: list[str] | None = None,
        memory_key: str | None = None,
        retrieval_hints: list[str] | None = None,
        project: str | None = None,
        reason: str = "",
    ) -> ToolResult:
        """Consolidate two or more active facts into one new fact."""
        result = await op_merge_memories(
            source_ids,
            merged_content,
            category=category,
            tags=tags,
            memory_key=memory_key,
            retrieval_hints=retrieval_hints,
            project=project,
            reason=reason,
        )
        return _tool_result(result, format="json")

    @_tool(app, get_store)
    async def mark_stale(fact_id: str, reason: str = "") -> ToolResult:
        """Exclude a fact from active recall while preserving it for inspection."""
        result = await op_mark_stale(fact_id, reason=reason)
        return _tool_result(result, format="json")

    @_tool(app, get_store)
    async def unmark_stale(fact_id: str) -> ToolResult:
        """Reverse a previous ``mark_stale`` call so the fact is recall-eligible."""
        result = await op_unmark_stale(fact_id)
        return _tool_result(result, format="json")


def _register_maintenance_tools(app: FastMCP, get_store: StoreGetter) -> None:
    @_tool(app, get_store)
    async def import_memories(
        source: str = "claude_code",
        format: str = "text",
    ) -> ToolResult:
        """Import existing memories from Claude Code's per-project memory files."""
        result = await op_import_memories(source=source)
        return _tool_result(result, format=format)

    @_tool(app, get_store)
    async def purge(format: str = "text") -> ToolResult:
        """Permanently remove forgotten and expired facts from storage."""
        result = await op_purge()
        return _tool_result(result, format=format)

    @_tool(app, get_store)
    async def rename_project(
        old_project: str,
        new_project: str,
        format: str = "text",
    ) -> ToolResult:
        """Bulk-rename facts and candidates from one project to another."""
        result = await op_rename_project(
            old_project,
            new_project,
        )
        return _tool_result(result, format=format)

    @_tool(app, get_store)
    async def audit_memories(
        project: str | None = None,
        format: str = "text",
    ) -> ToolResult:
        """Suggest duplicate, stale, and contradictory memory cleanup.

        This is read-only: it emits reviewable suggestions and example commands
        but does not mutate the memory store.
        """
        result = await op_audit_memories(project=project)
        return _tool_result(result, format=format)

    @_tool(app, get_store)
    async def upkeep(
        project: str | None = None,
        steps: list[str] | None = None,
        dry_run: bool = False,
        full: bool = False,
        format: str = "text",
    ) -> ToolResult:
        """Run memory upkeep now (it also runs in the background).

        Steps: ``projects`` (canonical scopes), ``verify`` (check file/symbol
        anchors against the repo; flag or retire drifted cards),
        ``consolidate`` (merge fragmented cards, split overloaded ones,
        retire junk; LLM),
        ``briefs`` (refresh project briefs; LLM). Changes are applied to the
        append-only log; ``dry_run`` reports without writing.
        """
        result = await op_upkeep(
            project=project,
            steps=steps,
            dry_run=dry_run,
            full=full,
        )
        return _tool_result(result, format=format)

    @_tool(app, get_store)
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
                repair or repair_jsonl or recover_transactions or repair_orphaned_supersessions
            ),
            repair_jsonl=repair_jsonl,
            recover_transactions=recover_transactions,
            repair_orphaned_supersessions=repair_orphaned_supersessions,
        )
        return _tool_result(result, format="json")


def _register_reporting_tools(app: FastMCP, get_store: StoreGetter) -> None:
    @_tool(app, get_store)
    async def memory_stats(format: str = "text") -> ToolResult:
        """Show memory system statistics: fact counts, storage size, category breakdown."""
        result = await op_memory_stats()
        return _tool_result(result, format=format)

    @_tool(app, get_store)
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
        )
        return _tool_result(result, format=format)

    @_tool(app, get_store)
    async def sync(
        format: str = "text",
        timeout: float = 30.0,
    ) -> ToolResult:
        """Git-backed sync of the data directory against its configured remote.

        Runs ``git fetch`` + ``git pull --rebase`` + ``git push`` against the
        Engram data directory. Requires the data directory to be a git
        repository with at least one remote — see docs/sync.md for setup.
        """
        result = await op_sync(timeout=timeout)
        return _tool_result(result, format=format)


mcp = create_mcp()


HTTP_PATH = "/mcp"


def main(
    mcp_factory: Callable[[], FastMCP] = create_mcp,
    *,
    transport: Literal["stdio", "http"] = "stdio",
    host: str = DEFAULT_HTTP_HOST,
    port: int = DEFAULT_HTTP_PORT,
) -> None:
    """Entry point for the MCP server."""
    configure_logging()
    if transport == "stdio":
        mcp_factory().run()
        return
    # Stateless: no server-side MCP session, so clients survive a daemon restart
    # without holding a session id the new process has never seen. FastMCP
    # leaves Host/Origin checks off by default; "auto" blocks DNS-rebinding
    # pages from reaching the unauthenticated write tools on a loopback bind.
    mcp_factory().run(
        transport="http",
        host=host,
        port=port,
        path=HTTP_PATH,
        stateless_http=True,
        host_origin_protection="auto",
    )


if __name__ == "__main__":
    main()
