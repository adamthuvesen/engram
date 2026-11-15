"""Store memories from text and imports."""

from __future__ import annotations

from engram.core.interfaces import (
    Envelope,
    EnvelopeError,
    EnvelopeStatus,
    ErrorCode,
    provider_error,
    storage_error,
)
from engram.core.operation_models import (
    CandidatePayload,
    FactPayload,
    IngestPayload,
    SuggestPayload,
)
from engram.extraction.importer import import_claude_code_memories
from engram.extraction.observer import IngestResult, ingest
from engram.operations.results import (
    EXIT_OK,
    EXIT_RUNTIME,
    OperationResult,
    exception_result,
    failure_result,
    outcome_envelope,
    validation_result,
)
from engram.operations.stores import resolve_store, scope_project
from engram.storage.async_store import AsyncFactStore
from engram.storage.store import FactStore


def _ingest_errors(result: IngestResult) -> list[EnvelopeError]:
    return [
        *(provider_error(message) for message in result.errors),
        *(
            EnvelopeError(
                code=ErrorCode.conflict,
                message=item.reason,
                ids=[item.fact_id],
                details={"action": item.action},
            )
            for item in result.unresolved
        ),
    ]


def _ingest_lines(result: IngestResult) -> list[str]:
    lines: list[str] = []
    replaced_by: dict[str, list[str]] = {}
    for old_id, new_id in result.superseded.items():
        replaced_by.setdefault(new_id, []).append(old_id)
    if result.created:
        lines.append(f"Stored {len(result.created)} memory card(s):")
        for fact in result.created:
            line = f"- [{fact.category.value}] {fact.content} (id: {fact.id})"
            if fact.id in replaced_by:
                line += f" — replaces {', '.join(replaced_by[fact.id])}"
            lines.append(line)
    if result.retired:
        lines.append("Retired as no longer true:")
        lines.extend(f"- {fact_id}: {reason}" for fact_id, reason in result.retired.items())
    if result.duplicates:
        lines.append(f"Already known: {', '.join(result.duplicates)}")
    lines.extend(
        f"Unapplied {item.action} {item.fact_id}: {item.reason}" for item in result.unresolved
    )
    lines.extend(f"Error: {message}" for message in result.errors)
    if not lines:
        lines.append("Nothing new worth storing in the input.")
    if result.excluded:
        lines.append(f"Skipped {len(result.excluded)} non-durable claim(s).")
    return lines


async def remember(
    content: str,
    *,
    source: str = "conversation",
    project: str | None = None,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    if not content.strip():
        return validation_result("content must not be empty")
    store_obj = await resolve_store(store)
    try:
        result = await ingest(
            content,
            source=source,
            project=await scope_project(store_obj, project),
            store=store_obj,
        )
    except Exception as exc:
        return exception_result(exc)

    data = IngestPayload(
        facts=[FactPayload.model_validate(fact) for fact in result.created],
        superseded=result.superseded,
        retired=result.retired,
        duplicates=result.duplicates,
        excluded=result.excluded,
        unresolved=result.unresolved,
    ).model_dump(mode="json")
    errors = _ingest_errors(result)
    return OperationResult(
        envelope=outcome_envelope(
            data,
            errors,
            applied=bool(result.created or result.superseded or result.retired),
        ),
        text="\n".join(_ingest_lines(result)),
        exit_code=EXIT_RUNTIME if errors else EXIT_OK,
    )


async def suggest_memories(
    content: str,
    *,
    source: str = "conversation",
    project: str | None = None,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    if not content.strip():
        return validation_result("content must not be empty")
    store_obj = await resolve_store(store)
    try:
        result = await ingest(
            content,
            source=source,
            project=await scope_project(store_obj, project),
            store=store_obj,
            queue_for_review=True,
        )
    except Exception as exc:
        return exception_result(exc)

    candidates = result.candidates
    lines = (
        [f"Queued {len(candidates)} memory candidate(s):"]
        if candidates
        else ["No memory candidates were proposed from the input."]
    )
    for candidate in candidates:
        why = candidate.why_store or "useful future context"
        line = (
            f"- [{candidate.category.value}] {candidate.content} (id: {candidate.id}, why: {why})"
        )
        if candidate.replaces:
            line += f" replaces {', '.join(candidate.replaces)}"
        lines.append(line)
    lines.extend(
        f"Unapplied {item.action} {item.fact_id}: {item.reason}" for item in result.unresolved
    )
    lines.extend(f"Error: {message}" for message in result.errors)
    payload = SuggestPayload(
        candidates=[CandidatePayload.model_validate(candidate) for candidate in candidates],
        unresolved=result.unresolved,
        message=lines[0] if not candidates else None,
    )
    errors = _ingest_errors(result)
    return OperationResult(
        envelope=outcome_envelope(
            payload.model_dump(mode="json", exclude_none=True),
            errors,
            applied=bool(candidates),
        ),
        text="\n".join(lines),
        exit_code=EXIT_RUNTIME if errors else EXIT_OK,
    )


async def import_memories(
    *,
    source: str = "claude_code",
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    if source != "claude_code":
        return validation_result(
            f"Unsupported import source: {source}. Use 'claude_code'.",
            text=f"Unsupported import source: {source}. Use 'claude_code'.",
            details={"parameter": "source", "value": source},
        )

    result = await import_claude_code_memories(store=await resolve_store(store))
    if "error" in result:
        return failure_result(storage_error(result["error"]), text=result["error"])
    if result.get("message") and result.get("total_facts", 0) == 0:
        return OperationResult(
            envelope=Envelope.success(data=result),
            text=result["message"],
        )

    lines = [f"Imported {result['total_facts']} facts from {result['imported_files']} files:\n"]
    for detail in result.get("details", []):
        if detail.get("error"):
            lines.append(f"- {detail['file']}: error: {detail['error']}")
        else:
            lines.append(
                f"- {detail['file']} ({detail['project']}): {detail['facts_extracted']} facts"
            )
            lines.extend(
                f"  Unapplied {item['action']} {item['fact_id']}: {item['reason']}"
                for item in detail.get("unresolved", [])
            )
    incomplete = result.get("status") == "partial"
    return OperationResult(
        envelope=Envelope(
            status=EnvelopeStatus.partial if incomplete else EnvelopeStatus.ok,
            data=result,
        ),
        text="\n".join(lines),
        exit_code=EXIT_RUNTIME if incomplete else EXIT_OK,
    )
