"""Audit, doctor, statistics, upkeep, and sync operations."""

from __future__ import annotations

from math import isfinite
from typing import Any

from engram.core.config import get_settings
from engram.core.interfaces import (
    Envelope,
    EnvelopeError,
    EnvelopeStatus,
    ErrorCode,
    storage_error,
)
from engram.core.operation_models import (
    DoctorPayload,
)
from engram.core.projects import canonical_project
from engram.maintenance.doctor import check_provider, repair_store, run_doctor
from engram.maintenance.memory_audit import (
    audit_memory_store as _audit_memory_store,
)
from engram.maintenance.memory_audit import (
    format_audit_result,
)
from engram.maintenance.upkeep import (
    ALL_STEPS,
    UpkeepStep,
    format_upkeep_report,
    run_upkeep,
)
from engram.operations.results import (
    EXIT_DOCTOR_ERROR,
    EXIT_OK,
    EXIT_RUNTIME,
    OperationResult,
    failure_result,
    validation_result,
)
from engram.operations.stores import resolve_store, scope_project
from engram.runtime import current_runtime
from engram.storage.async_store import AsyncFactStore, run_sync
from engram.storage.store import FactStore
from engram.storage.sync import (
    DEFAULT_GIT_TIMEOUT_SECONDS,
    SyncError,
)
from engram.storage.sync import (
    sync as _sync,
)


async def audit_memories(
    *,
    project: str | None = None,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    result = await _audit_memory_store(
        project=await run_sync(canonical_project, project),
        store=await resolve_store(store),
    )
    data = result.model_dump(mode="json")
    data.update(
        {
            "duplicate_groups": result.duplicate_groups,
            "stale_facts": result.stale_facts,
            "contradiction_groups": result.contradiction_groups,
        }
    )
    return OperationResult(
        envelope=Envelope.success(data=data),
        text=format_audit_result(result),
    )


async def doctor(
    *,
    check_provider_flag: bool = False,
    repair: bool = False,
    repair_jsonl: bool = False,
    recover_transactions: bool = False,
    repair_orphaned_supersessions: bool = False,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    provider_issue = await check_provider() if check_provider_flag else None
    repair_requested = (
        repair or repair_jsonl or recover_transactions or repair_orphaned_supersessions
    )
    runtime = current_runtime()
    data_dir = (
        store.data_dir
        if store is not None
        else runtime.settings.data_dir
        if runtime
        else get_settings().data_dir
    )
    readonly = await run_sync(FactStore.open_readonly, data_dir)
    report = await run_sync(
        run_doctor,
        readonly,
        check_provider_flag=check_provider_flag,
        provider_issue=provider_issue,
    )
    repair_summary = None
    if repair_requested and (repair_jsonl or recover_transactions or repair_orphaned_supersessions):
        # Repair must inspect damaged storage before any recovery or write.
        write_store = store or await run_sync(FactStore, data_dir)
        repair_summary = await run_sync(
            repair_store,
            write_store,
            repair_jsonl=repair_jsonl,
            recover_transactions=recover_transactions,
            repair_orphaned_supersessions=repair_orphaned_supersessions,
        )
    text_lines = [f"Doctor: {report.status}"]
    for issue in report.issues:
        text_lines.append(f"  [{issue.severity}] {issue.category}/{issue.code}: {issue.message}")
        if issue.repair:
            text_lines.append(f"    repair: {issue.repair}")
    return OperationResult(
        envelope=Envelope.success(
            data=DoctorPayload(
                report=report.model_dump(mode="json"), repair=repair_summary
            ).model_dump(mode="json")
        ),
        text="\n".join(text_lines),
        exit_code=EXIT_DOCTOR_ERROR if report.status == "error" else EXIT_OK,
    )


async def memory_stats(
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    stats = await (await resolve_store(store)).stats()
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

    return OperationResult(
        envelope=Envelope.success(data=stats),
        text="\n".join(lines),
    )


async def upkeep(
    *,
    project: str | None = None,
    steps: list[str] | None = None,
    dry_run: bool = False,
    full: bool = False,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    valid = [step.value for step in ALL_STEPS]
    requested = steps or valid
    unknown = [step for step in requested if step not in valid]
    if unknown:
        return validation_result(
            f"Unknown upkeep step(s): {', '.join(unknown)}. Use: {', '.join(valid)}.",
            details={"parameter": "steps", "valid": valid},
        )
    store_obj = await resolve_store(store)
    report = await run_upkeep(
        store_obj,
        project=await scope_project(store_obj, project),
        steps=[UpkeepStep(step) for step in requested],
        dry_run=dry_run,
        full=full,
    )
    return OperationResult(
        envelope=Envelope(
            status=EnvelopeStatus.partial if report.partial else EnvelopeStatus.ok,
            data=report.model_dump(mode="json"),
            errors=[
                EnvelopeError(code=ErrorCode.conflict, message=reason, ids=[fact_id])
                for step in report.steps
                for fact_id, reason in step.conflicts.items()
            ],
        ),
        text=format_upkeep_report(report),
        exit_code=EXIT_RUNTIME if report.partial else EXIT_OK,
    )


async def sync(
    *,
    timeout: float = DEFAULT_GIT_TIMEOUT_SECONDS,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    """Run git-backed sync against the configured remote.

    Returns a structured ``OperationResult`` envelope. Sync failures surface as
    non-zero exit codes with the underlying git stderr preserved verbatim.
    """
    if not isfinite(timeout) or timeout <= 0:
        return validation_result("timeout must be greater than zero")
    data_dir = (await resolve_store(store)).data_dir

    try:
        result = await run_sync(_sync, data_dir, timeout=timeout)
    except SyncError as exc:
        details: dict[str, Any] = {"sync_error_code": exc.code}
        if exc.git_stderr:
            details["git_stderr"] = exc.git_stderr
        return failure_result(storage_error(exc.message, details=details), text=exc.message)

    if result.get("status") == "skipped":
        text = f"Sync skipped: {result.get('reason', 'unknown')}."
    else:
        text = (
            f"Sync OK — pulled {result['pulled_commits']} commit(s), "
            f"pushed {result['pushed_commits']} commit(s) "
            f"({result['remote']}/{result['branch']}) in {result['took_ms']} ms."
        )
    return OperationResult(envelope=Envelope.success(data=result), text=text)
