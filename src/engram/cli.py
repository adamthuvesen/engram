"""Agent-first CLI commands for Engram."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable
from typing import Sequence

from engram.config import configure_logging
from engram.interfaces import Envelope, storage_error, validation_error
from engram.operations import (
    EXIT_DOCTOR_ERROR,
    EXIT_NOT_FOUND,
    EXIT_OK,
    EXIT_RUNTIME,
    EXIT_VALIDATION,
    OperationResult,
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

CommandHandler = Callable[[argparse.Namespace], Awaitable[OperationResult]]


CANONICAL_COMMANDS = frozenset(
    {
        "remember",
        "suggest-memories",
        "list-candidates",
        "approve-candidates",
        "reject-candidates",
        "recall",
        "recall-trace",
        "recall-context",
        "forget",
        "edit-fact",
        "inspect",
        "correct-memory",
        "merge-memories",
        "mark-stale",
        "unmark-stale",
        "import-memories",
        "purge",
        "rename-project",
        "synthesize",
        "doctor",
        "memory-stats",
        "recall-stats",
    }
)

ALIASES = {
    "trace": "recall-trace",
    "correct": "correct-memory",
    "merge": "merge-memories",
    "stale": "mark-stale",
    "unstale": "unmark-stale",
}

CLI_SUBCOMMANDS = CANONICAL_COMMANDS | frozenset(ALIASES)


def _emit(result: OperationResult, *, as_json: bool, out=None) -> int:
    target = out if out is not None else sys.stdout
    target.write(result.render(as_json=as_json) + "\n")
    target.flush()
    return result.exit_code


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Emit JSON envelope output")


def _parse_edits(values: list[str] | None) -> dict[str, str] | None:
    if not values:
        return None
    edits: dict[str, str] = {}
    invalid: list[str] = []
    for value in values:
        candidate_id, sep, content = value.partition("=")
        if not sep or not candidate_id or not content:
            invalid.append(value)
            continue
        edits[candidate_id] = content
    if invalid:
        raise ValueError(
            "--edit values must use id=new content format; invalid: "
            + ", ".join(invalid)
        )
    return edits


async def cmd_remember(args: argparse.Namespace) -> OperationResult:
    return await op_remember(args.content, source=args.source, project=args.project)


async def cmd_suggest_memories(args: argparse.Namespace) -> OperationResult:
    return await op_suggest_memories(
        args.content,
        source=args.source,
        project=args.project,
    )


async def cmd_list_candidates(args: argparse.Namespace) -> OperationResult:
    return await op_list_candidates(
        status=args.status,
        project=args.project,
        search=args.search,
        limit=args.limit,
    )


async def cmd_approve_candidates(args: argparse.Namespace) -> OperationResult:
    try:
        edits = _parse_edits(args.edit)
    except ValueError as exc:
        return OperationResult(
            envelope=Envelope.failure(validation_error(str(exc))),
            text=str(exc),
            exit_code=EXIT_VALIDATION,
        )
    return await op_approve_candidates(args.candidate_ids, edits=edits)


async def cmd_reject_candidates(args: argparse.Namespace) -> OperationResult:
    return await op_reject_candidates(args.candidate_ids, reason=args.reason)


async def cmd_recall(args: argparse.Namespace) -> OperationResult:
    return await op_recall(
        args.query,
        project=args.project,
        format="json" if args.json else "text",
        with_provenance=args.with_provenance,
        max_sources=args.max_sources,
        max_prefilter_matches=args.max_prefilter_matches,
    )


async def cmd_recall_trace(args: argparse.Namespace) -> OperationResult:
    return await op_recall_trace(
        args.query,
        project=args.project,
        verbose=args.verbose,
        max_sources=args.max_sources,
        max_prefilter_matches=args.max_prefilter_matches,
    )


async def cmd_recall_context(args: argparse.Namespace) -> OperationResult:
    return await op_recall_context(
        args.query,
        project=args.project,
        mode=args.mode,
    )


async def cmd_forget(args: argparse.Namespace) -> OperationResult:
    return await op_forget(args.fact_id, reason=args.reason)


async def cmd_edit_fact(args: argparse.Namespace) -> OperationResult:
    return await op_edit_fact(
        args.fact_id,
        content=args.content,
        category=args.category,
        tags=args.tags,
        project=args.project,
    )


async def cmd_inspect(args: argparse.Namespace) -> OperationResult:
    return await op_inspect(
        category=args.category,
        project=args.project,
        limit=args.limit,
        include_stale=args.include_stale,
    )


async def cmd_correct_memory(args: argparse.Namespace) -> OperationResult:
    return await op_correct_memory(
        args.fact_id,
        args.content,
        category=args.category,
        tags=args.tags,
        project=args.project,
        reason=args.reason or "",
    )


async def cmd_merge_memories(args: argparse.Namespace) -> OperationResult:
    return await op_merge_memories(
        args.source_ids,
        args.content,
        category=args.category,
        tags=args.tags,
        project=args.project,
        reason=args.reason or "",
    )


async def cmd_mark_stale(args: argparse.Namespace) -> OperationResult:
    return await op_mark_stale(args.fact_id, reason=args.reason or "")


async def cmd_unmark_stale(args: argparse.Namespace) -> OperationResult:
    return await op_unmark_stale(args.fact_id)


async def cmd_import_memories(args: argparse.Namespace) -> OperationResult:
    return await op_import_memories(source=args.source)


async def cmd_purge(args: argparse.Namespace) -> OperationResult:
    return await op_purge()


async def cmd_rename_project(args: argparse.Namespace) -> OperationResult:
    return await op_rename_project(args.old_project, args.new_project)


async def cmd_synthesize(args: argparse.Namespace) -> OperationResult:
    return await op_synthesize(project=args.project, dry_run=not args.apply)


async def cmd_doctor(args: argparse.Namespace) -> OperationResult:
    return await op_doctor(
        check_provider_flag=args.check_provider,
        repair=args.repair or args.repair_jsonl or args.recover_transactions,
        repair_jsonl=args.repair_jsonl,
        recover_transactions=args.recover_transactions,
    )


async def cmd_memory_stats(args: argparse.Namespace) -> OperationResult:
    return await op_memory_stats()


async def cmd_recall_stats(args: argparse.Namespace) -> OperationResult:
    return await op_recall_stats()


HANDLERS: dict[str, CommandHandler] = {
    "remember": cmd_remember,
    "suggest-memories": cmd_suggest_memories,
    "list-candidates": cmd_list_candidates,
    "approve-candidates": cmd_approve_candidates,
    "reject-candidates": cmd_reject_candidates,
    "recall": cmd_recall,
    "recall-trace": cmd_recall_trace,
    "recall-context": cmd_recall_context,
    "forget": cmd_forget,
    "edit-fact": cmd_edit_fact,
    "inspect": cmd_inspect,
    "correct-memory": cmd_correct_memory,
    "merge-memories": cmd_merge_memories,
    "mark-stale": cmd_mark_stale,
    "unmark-stale": cmd_unmark_stale,
    "import-memories": cmd_import_memories,
    "purge": cmd_purge,
    "rename-project": cmd_rename_project,
    "synthesize": cmd_synthesize,
    "doctor": cmd_doctor,
    "memory-stats": cmd_memory_stats,
    "recall-stats": cmd_recall_stats,
    "trace": cmd_recall_trace,
    "correct": cmd_correct_memory,
    "merge": cmd_merge_memories,
    "stale": cmd_mark_stale,
    "unstale": cmd_unmark_stale,
}


def _add_remember_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("content")
    parser.add_argument("--source", default="conversation")
    parser.add_argument("--project", default=None)
    _add_json_flag(parser)


def _add_trace_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query")
    parser.add_argument("--project", default=None)
    parser.add_argument(
        "--limit",
        "--max-sources",
        dest="max_sources",
        type=int,
        default=DEFAULT_MAX_SOURCES,
    )
    parser.add_argument(
        "--max-prefilter-matches",
        type=int,
        default=DEFAULT_MAX_PREFILTER_MATCHES,
    )
    parser.add_argument("--verbose", action="store_true")
    _add_json_flag(parser)


def _add_correct_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("fact_id")
    parser.add_argument("content")
    parser.add_argument("--category", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--reason", default=None)
    _add_json_flag(parser)


def _add_merge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source_ids", nargs="+")
    parser.add_argument("--content", required=True)
    parser.add_argument("--category", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--reason", default=None)
    _add_json_flag(parser)


def _add_stale_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("fact_id")
    parser.add_argument("--reason", default=None)
    _add_json_flag(parser)


def _add_unstale_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("fact_id")
    _add_json_flag(parser)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engram", description="Engram CLI")
    _add_json_flag(parser)
    sub = parser.add_subparsers(dest="cmd")

    _add_remember_args(sub.add_parser("remember", help="Store memories"))
    _add_remember_args(
        sub.add_parser("suggest-memories", help="Queue memory candidates")
    )

    p_list = sub.add_parser("list-candidates", help="Browse memory candidates")
    p_list.add_argument("--status", default="pending")
    p_list.add_argument("--project", default=None)
    p_list.add_argument("--search", default=None)
    p_list.add_argument("--limit", type=int, default=50)
    _add_json_flag(p_list)

    p_approve = sub.add_parser("approve-candidates", help="Approve candidates")
    p_approve.add_argument("candidate_ids", nargs="+")
    p_approve.add_argument("--edit", action="append", default=None)
    _add_json_flag(p_approve)

    p_reject = sub.add_parser("reject-candidates", help="Reject candidates")
    p_reject.add_argument("candidate_ids", nargs="+")
    p_reject.add_argument("--reason", default="")
    _add_json_flag(p_reject)

    p_recall = sub.add_parser("recall", help="Recall memory")
    p_recall.add_argument("query")
    p_recall.add_argument("--project", default=None)
    p_recall.add_argument("--with-provenance", action="store_true")
    p_recall.add_argument("--max-sources", type=int, default=DEFAULT_MAX_SOURCES)
    p_recall.add_argument(
        "--max-prefilter-matches",
        type=int,
        default=DEFAULT_MAX_PREFILTER_MATCHES,
    )
    _add_json_flag(p_recall)

    _add_trace_args(sub.add_parser("recall-trace", help="Run recall debug trace"))
    _add_trace_args(sub.add_parser("trace", help="Alias for recall-trace"))

    p_context = sub.add_parser("recall-context", help="Recall as answer or prompt")
    p_context.add_argument("query")
    p_context.add_argument("--project", default=None)
    p_context.add_argument("--mode", default="answer")
    _add_json_flag(p_context)

    p_forget = sub.add_parser("forget", help="Soft-delete a fact")
    p_forget.add_argument("fact_id")
    p_forget.add_argument("--reason", default="")
    _add_json_flag(p_forget)

    p_edit = sub.add_parser("edit-fact", help="Edit a fact in place")
    p_edit.add_argument("fact_id")
    p_edit.add_argument("--content", default=None)
    p_edit.add_argument("--category", default=None)
    p_edit.add_argument("--tags", nargs="*", default=None)
    p_edit.add_argument("--project", default=None)
    _add_json_flag(p_edit)

    p_inspect = sub.add_parser("inspect", help="List active facts")
    p_inspect.add_argument("--category", default=None)
    p_inspect.add_argument("--project", default=None)
    p_inspect.add_argument("--limit", type=int, default=50)
    p_inspect.add_argument("--include-stale", action="store_true")
    _add_json_flag(p_inspect)

    _add_correct_args(
        sub.add_parser("correct-memory", help="Replace a fact with a superseding one")
    )
    _add_correct_args(sub.add_parser("correct", help="Alias for correct-memory"))

    _add_merge_args(sub.add_parser("merge-memories", help="Merge facts"))
    _add_merge_args(sub.add_parser("merge", help="Alias for merge-memories"))

    _add_stale_args(sub.add_parser("mark-stale", help="Mark a fact stale"))
    _add_stale_args(sub.add_parser("stale", help="Alias for mark-stale"))
    _add_unstale_args(sub.add_parser("unmark-stale", help="Clear stale marker"))
    _add_unstale_args(sub.add_parser("unstale", help="Alias for unmark-stale"))

    p_import = sub.add_parser("import-memories", help="Import memories")
    p_import.add_argument("--source", default="claude_code")
    _add_json_flag(p_import)

    _add_json_flag(sub.add_parser("purge", help="Purge forgotten/expired facts"))

    p_rename = sub.add_parser("rename-project", help="Rename a project scope")
    p_rename.add_argument("old_project")
    p_rename.add_argument("new_project")
    _add_json_flag(p_rename)

    p_synthesize = sub.add_parser("synthesize", help="Consolidate facts")
    p_synthesize.add_argument("--project", default=None)
    p_synthesize.add_argument("--apply", action="store_true")
    _add_json_flag(p_synthesize)

    p_doctor = sub.add_parser("doctor", help="Health check the memory store")
    p_doctor.add_argument("--check-provider", action="store_true")
    p_doctor.add_argument("--repair", action="store_true")
    p_doctor.add_argument("--repair-jsonl", action="store_true")
    p_doctor.add_argument("--recover-transactions", action="store_true")
    _add_json_flag(p_doctor)

    _add_json_flag(sub.add_parser("memory-stats", help="Show memory stats"))
    _add_json_flag(sub.add_parser("recall-stats", help="Show recall stats"))
    return parser


def _normalize_argv(argv: Sequence[str] | None) -> list[str] | None:
    if argv is None:
        return None
    argv = list(argv)
    if any(token in ("help", "--help", "-h") for token in argv):
        return ["--help"]
    if len(argv) >= 2 and argv[0] == "--json":
        return [argv[1], *argv[2:], "--json"]
    return argv


def run(argv: Sequence[str] | None = None) -> int:
    """Run a CLI invocation and return its exit code."""
    parser = _build_parser()
    normalized = _normalize_argv(argv)
    if normalized in (["--help"], ["-h"]):
        parser.print_help()
        return EXIT_OK
    args = parser.parse_args(normalized)
    if args.cmd is None:
        parser.print_help()
        return EXIT_OK

    handler = HANDLERS[args.cmd]
    try:
        result = asyncio.run(handler(args))
    except Exception as exc:
        env = Envelope.failure(
            storage_error(
                f"Unexpected error: {exc}",
                details={"exception_type": type(exc).__name__},
            )
        )
        result = OperationResult(envelope=env, text=env.to_json(), exit_code=EXIT_RUNTIME)
    return _emit(result, as_json=args.json)


def is_cli_invocation(argv: Sequence[str]) -> bool:
    """Return True when argv should run the CLI instead of the MCP server.

    Bare ``engram`` (no args) starts the MCP stdio server; anything else is
    treated as a CLI invocation so typos surface as argparse errors instead of
    silently launching a long-running stdio process.
    """
    return bool(argv)


def main_dispatch(argv: Sequence[str] | None = None) -> int:
    """Pick between CLI and MCP server based on argv."""
    if argv is None:
        argv = sys.argv[1:]
    if is_cli_invocation(argv):
        configure_logging()
        return run(argv)
    from engram.server import main as server_main

    server_main()
    return EXIT_OK


__all__ = [
    "ALIASES",
    "CANONICAL_COMMANDS",
    "CLI_SUBCOMMANDS",
    "EXIT_DOCTOR_ERROR",
    "EXIT_NOT_FOUND",
    "EXIT_OK",
    "EXIT_RUNTIME",
    "EXIT_VALIDATION",
    "HANDLERS",
    "is_cli_invocation",
    "main_dispatch",
    "run",
]
