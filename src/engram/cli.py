"""Agent-first CLI commands for Engram.

The default ``engram`` entrypoint starts the MCP server (current behavior).
When the first argument is a recognized subcommand, the CLI dispatcher runs
instead. All subcommands accept ``--json`` and use stable structured envelopes
for machine consumption; non-JSON output stays human-friendly.

Subcommands:
    trace <query>           — recall trace with bounded prompt/output excerpts
    doctor                  — health check; ``--repair`` plus a repair flag
    correct <id> <content>  — replace a fact with a superseding new fact
    merge <id> <id> ... <content> — consolidate two or more facts
    stale <id>              — mark a fact stale
    unstale <id>            — reverse a stale marking
    inspect                 — list active facts (optionally include stale)

Exit codes:
    0  success or non-fatal warnings
    1  validation error
    2  not found
    3  storage or provider error
    4  doctor reports an error-level issue
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from typing import Sequence

from engram.config import configure_logging
from engram.doctor import check_provider, repair_store, run_doctor
from engram.interfaces import (
    Envelope,
    EnvelopeMeta,
    not_found_error,
    provider_error,
    storage_error,
    validation_error,
)
from engram.models import FactCategory
from engram.provenance import DEFAULT_MAX_PREFILTER_MATCHES, DEFAULT_MAX_SOURCES
from engram.retriever import recall_with_provenance
from engram.store import AsyncFactStore, FactStore

EXIT_OK = 0
EXIT_VALIDATION = 1
EXIT_NOT_FOUND = 2
EXIT_RUNTIME = 3
EXIT_DOCTOR_ERROR = 4


@dataclass
class CliResult:
    envelope: Envelope
    exit_code: int
    text: str = ""


def _emit(result: CliResult, *, as_json: bool, out=None) -> int:
    target = out if out is not None else sys.stdout
    if as_json:
        target.write(result.envelope.to_json() + "\n")
    else:
        target.write((result.text or result.envelope.to_json()) + "\n")
    target.flush()
    return result.exit_code


def _store() -> AsyncFactStore:
    return AsyncFactStore(FactStore())


def _category(value: str | None) -> FactCategory | None:
    if value is None:
        return None
    try:
        return FactCategory(value)
    except ValueError:
        return None


async def cmd_trace(args: argparse.Namespace) -> CliResult:
    try:
        answer, quality, provenance, trace = await recall_with_provenance(
            args.query,
            project=args.project,
            store=_store(),
            with_trace=True,
            verbose_trace=args.verbose,
            max_sources=args.limit,
            max_prefilter_matches=DEFAULT_MAX_PREFILTER_MATCHES,
        )
    except Exception as exc:
        env = Envelope.failure(
            provider_error(
                f"Recall trace failed: {exc}",
                details={"exception_type": type(exc).__name__},
            )
        )
        return CliResult(envelope=env, exit_code=EXIT_RUNTIME)

    data = {
        "query": args.query,
        "answer": answer,
        "quality": quality,
        "tier": provenance.tier,
        "trace": trace.model_dump(mode="json") if trace else None,
    }
    truncated = bool(trace and trace.truncated)
    meta = EnvelopeMeta(
        limit=args.limit,
        returned=len(provenance.sources),
        total=provenance.prefilter_count,
        truncated=truncated,
        truncation_reason="excerpt_truncated" if truncated else None,
    )
    env = Envelope.success(data=data, warnings=provenance.warnings, meta=meta)
    text = (
        f"Tier {provenance.tier} | quality={quality} | "
        f"prefilter={provenance.prefilter_count} | "
        f"calls={provenance.usage.llm_calls or 0}\n"
        f"Answer:\n{answer}"
    )
    return CliResult(envelope=env, exit_code=EXIT_OK, text=text)


async def cmd_doctor(args: argparse.Namespace) -> CliResult:
    store = _store()
    provider_issue = None
    if args.check_provider:
        provider_issue = await check_provider()
    report = run_doctor(
        store,
        check_provider_flag=args.check_provider,
        provider_issue=provider_issue,
    )

    repair_summary = None
    if args.repair and (args.repair_jsonl or args.recover_transactions):
        repair_summary = repair_store(
            store,
            repair_jsonl=args.repair_jsonl,
            recover_transactions=args.recover_transactions,
        )

    env = Envelope.success(
        data={"report": report.model_dump(mode="json"), "repair": repair_summary}
    )

    exit_code = EXIT_DOCTOR_ERROR if report.status == "error" else EXIT_OK

    text_lines = [f"Doctor: {report.status}"]
    for issue in report.issues:
        text_lines.append(
            f"  [{issue.severity}] {issue.category}/{issue.code}: {issue.message}"
        )
        if issue.repair:
            text_lines.append(f"    repair: {issue.repair}")
    return CliResult(envelope=env, exit_code=exit_code, text="\n".join(text_lines))


async def cmd_correct(args: argparse.Namespace) -> CliResult:
    cat = _category(args.category)
    if args.category and cat is None:
        env = Envelope.failure(
            validation_error(
                f"Invalid category: {args.category}",
                details={"valid": list(FactCategory.__members__.keys())},
            )
        )
        return CliResult(envelope=env, exit_code=EXIT_VALIDATION)

    new_fact = await _store().correct_fact(
        args.fact_id,
        args.content,
        category=cat,
        tags=args.tags,
        project=args.project,
        reason=args.reason or "",
    )
    if new_fact is None:
        env = Envelope.failure(
            not_found_error(
                f"Active fact {args.fact_id} not found",
                ids=[args.fact_id],
            )
        )
        return CliResult(envelope=env, exit_code=EXIT_NOT_FOUND)

    env = Envelope.success(
        data={
            "new_fact_id": new_fact.id,
            "superseded_fact_id": args.fact_id,
            "category": new_fact.category.value,
            "project": new_fact.project,
            "content": new_fact.content,
        }
    )
    text = (
        f"Corrected {args.fact_id} -> {new_fact.id}\n"
        f"  [{new_fact.category.value}] {new_fact.content}"
    )
    return CliResult(envelope=env, exit_code=EXIT_OK, text=text)


async def cmd_merge(args: argparse.Namespace) -> CliResult:
    if len(args.source_ids) < 2:
        env = Envelope.failure(
            validation_error(
                "merge requires at least two source IDs",
                details={"received": len(args.source_ids)},
            )
        )
        return CliResult(envelope=env, exit_code=EXIT_VALIDATION)

    cat = _category(args.category)
    result = await _store().merge_facts(
        args.source_ids,
        args.content,
        category=cat,
        tags=args.tags,
        project=args.project,
        reason=args.reason or "",
    )
    if result is None:
        env = Envelope.failure(
            validation_error(
                "Could not merge — sources missing, forgotten, or in different projects.",
                ids=args.source_ids,
            )
        )
        return CliResult(envelope=env, exit_code=EXIT_VALIDATION)

    new_fact, superseded = result
    env = Envelope.success(
        data={
            "new_fact_id": new_fact.id,
            "superseded_fact_ids": superseded,
            "category": new_fact.category.value,
            "project": new_fact.project,
            "content": new_fact.content,
        }
    )
    text = f"Merged {len(superseded)} facts -> {new_fact.id}"
    return CliResult(envelope=env, exit_code=EXIT_OK, text=text)


async def cmd_stale(args: argparse.Namespace) -> CliResult:
    fact = await _store().mark_stale(args.fact_id, args.reason or "")
    if fact is None:
        env = Envelope.failure(
            not_found_error(
                f"Fact {args.fact_id} not found",
                ids=[args.fact_id],
            )
        )
        return CliResult(envelope=env, exit_code=EXIT_NOT_FOUND)
    env = Envelope.success(
        data={
            "fact_id": fact.id,
            "stale": True,
            "stale_reason": fact.stale_reason,
        }
    )
    return CliResult(envelope=env, exit_code=EXIT_OK, text=f"Marked {fact.id} stale.")


async def cmd_unstale(args: argparse.Namespace) -> CliResult:
    fact = await _store().unmark_stale(args.fact_id)
    if fact is None:
        env = Envelope.failure(
            not_found_error(f"Fact {args.fact_id} not found", ids=[args.fact_id])
        )
        return CliResult(envelope=env, exit_code=EXIT_NOT_FOUND)
    env = Envelope.success(data={"fact_id": fact.id, "stale": False})
    return CliResult(
        envelope=env, exit_code=EXIT_OK, text=f"Cleared stale on {fact.id}."
    )


async def cmd_inspect(args: argparse.Namespace) -> CliResult:
    cat = _category(args.category)
    if args.category and cat is None:
        env = Envelope.failure(
            validation_error(
                f"Invalid category: {args.category}",
                details={"valid": list(FactCategory.__members__.keys())},
            )
        )
        return CliResult(envelope=env, exit_code=EXIT_VALIDATION)

    facts = await _store().load_active_facts(
        category=cat,
        project=args.project,
        limit=args.limit,
        include_stale=args.include_stale,
    )
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
    truncated = len(data) >= args.limit
    meta = EnvelopeMeta(
        limit=args.limit,
        returned=len(data),
        truncated=truncated,
        truncation_reason="default_limit" if truncated else None,
    )
    env = Envelope.success(data=data, meta=meta)

    if not facts:
        text = "No facts found matching the criteria."
    else:
        text_lines = [f"Found {len(facts)} fact(s):"]
        for fact in facts:
            stale_marker = " [stale]" if fact.stale else ""
            text_lines.append(
                f"  [{fact.category.value}]{stale_marker} {fact.content} (id: {fact.id})"
            )
        text = "\n".join(text_lines)
    return CliResult(envelope=env, exit_code=EXIT_OK, text=text)


def _add_json_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="Emit JSON envelope output")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engram", description="Engram CLI")
    _add_json_flag(parser)
    sub = parser.add_subparsers(dest="cmd")

    p_trace = sub.add_parser("trace", help="Run recall and emit a debug trace")
    p_trace.add_argument("query")
    p_trace.add_argument("--project", default=None)
    p_trace.add_argument("--limit", type=int, default=DEFAULT_MAX_SOURCES)
    p_trace.add_argument("--verbose", action="store_true")
    _add_json_flag(p_trace)

    p_doctor = sub.add_parser("doctor", help="Health check the memory store")
    p_doctor.add_argument("--check-provider", action="store_true")
    p_doctor.add_argument("--repair", action="store_true")
    p_doctor.add_argument("--repair-jsonl", action="store_true")
    p_doctor.add_argument("--recover-transactions", action="store_true")
    _add_json_flag(p_doctor)

    p_correct = sub.add_parser(
        "correct", help="Replace a fact with a superseding new one"
    )
    p_correct.add_argument("fact_id")
    p_correct.add_argument("content")
    p_correct.add_argument("--category", default=None)
    p_correct.add_argument("--project", default=None)
    p_correct.add_argument("--tags", nargs="*", default=None)
    p_correct.add_argument("--reason", default=None)
    _add_json_flag(p_correct)

    p_merge = sub.add_parser("merge", help="Consolidate multiple facts into one")
    p_merge.add_argument("source_ids", nargs="+")
    p_merge.add_argument("--content", required=True)
    p_merge.add_argument("--category", default=None)
    p_merge.add_argument("--project", default=None)
    p_merge.add_argument("--tags", nargs="*", default=None)
    p_merge.add_argument("--reason", default=None)
    _add_json_flag(p_merge)

    p_stale = sub.add_parser("stale", help="Mark a fact stale")
    p_stale.add_argument("fact_id")
    p_stale.add_argument("--reason", default=None)
    _add_json_flag(p_stale)

    p_unstale = sub.add_parser("unstale", help="Reverse a stale marking")
    p_unstale.add_argument("fact_id")
    _add_json_flag(p_unstale)

    p_inspect = sub.add_parser("inspect", help="List active facts")
    p_inspect.add_argument("--category", default=None)
    p_inspect.add_argument("--project", default=None)
    p_inspect.add_argument("--limit", type=int, default=50)
    p_inspect.add_argument("--include-stale", action="store_true")
    _add_json_flag(p_inspect)

    return parser


_HANDLERS = {
    "trace": cmd_trace,
    "doctor": cmd_doctor,
    "correct": cmd_correct,
    "merge": cmd_merge,
    "stale": cmd_stale,
    "unstale": cmd_unstale,
    "inspect": cmd_inspect,
}

CLI_SUBCOMMANDS = frozenset(_HANDLERS)


def run(argv: Sequence[str] | None = None) -> int:
    """Entry point — returns the exit code for the CLI invocation."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return EXIT_OK

    handler = _HANDLERS[args.cmd]
    try:
        result = asyncio.run(handler(args))
    except Exception as exc:
        env = Envelope.failure(
            storage_error(
                f"Unexpected error: {exc}",
                details={"exception_type": type(exc).__name__},
            )
        )
        result = CliResult(envelope=env, exit_code=EXIT_RUNTIME)
    return _emit(result, as_json=args.json)


def is_cli_invocation(argv: Sequence[str]) -> bool:
    """Return True when the first argument is a known CLI subcommand."""
    return bool(argv) and argv[0] in CLI_SUBCOMMANDS


def main_dispatch(argv: Sequence[str] | None = None) -> int:
    """Pick between CLI and MCP server based on argv.

    Used by the `engram` console script — when the user types `engram trace ...`
    the CLI runs; bare `engram` continues to start the MCP server.
    """
    if argv is None:
        argv = sys.argv[1:]
    if is_cli_invocation(argv):
        configure_logging()
        return run(argv)
    from engram.server import main as server_main

    server_main()
    return 0


__all__ = [
    "CLI_SUBCOMMANDS",
    "CliResult",
    "EXIT_DOCTOR_ERROR",
    "EXIT_NOT_FOUND",
    "EXIT_OK",
    "EXIT_RUNTIME",
    "EXIT_VALIDATION",
    "is_cli_invocation",
    "main_dispatch",
    "run",
]
