"""Review pending memory candidates."""

from __future__ import annotations

from engram.core.interfaces import (
    Envelope,
    EnvelopeError,
    EnvelopeMeta,
    ErrorCode,
)
from engram.core.models import (
    CandidateStatus,
)
from engram.core.operation_models import (
    CandidatePayload,
    CandidateReviewPayload,
    FactPayload,
)
from engram.core.projects import canonical_project
from engram.operations.results import (
    OperationResult,
    candidate_payload,
    invalid_positive_int_result,
    missing_result,
    outcome_envelope,
    validation_result,
)
from engram.operations.stores import resolve_store
from engram.storage.async_store import AsyncFactStore, run_sync
from engram.storage.store import FactStore


async def list_candidates(
    *,
    status: str = "pending",
    project: str | None = None,
    search: str | None = None,
    limit: int = 50,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    if limit < 1:
        return invalid_positive_int_result("limit")

    try:
        candidate_status = CandidateStatus(status)
    except ValueError:
        valid = ", ".join(item.value for item in CandidateStatus)
        return validation_result(
            f"Unsupported status: {status}. Use one of: {valid}.",
            text=f"Unsupported status: {status}. Use one of: {valid}.",
            details={"valid": [item.value for item in CandidateStatus]},
        )

    store_obj = await resolve_store(store)
    project = await run_sync(canonical_project, project)
    all_candidates = await store_obj.load_candidates(
        status=candidate_status,
        project=project,
        limit=None,
    )

    if search:
        search_lower = search.lower()
        all_candidates = [c for c in all_candidates if search_lower in c.content.lower()]
    candidates = all_candidates[:limit]
    total = len(all_candidates)
    truncated = total > limit

    meta = EnvelopeMeta(
        limit=limit,
        returned=len(candidates),
        total=total,
        truncated=truncated,
        truncation_reason="default_limit" if truncated else None,
    )
    if not candidates:
        text = "No memory candidates found matching the criteria."
    else:
        text = store_obj.format_candidates_for_review(candidates)
    return OperationResult(
        envelope=Envelope.success(
            data={"candidates": [candidate_payload(c) for c in candidates]},
            meta=meta,
        ),
        text=text,
    )


async def approve_candidates(
    candidate_ids: list[str],
    *,
    edits: dict[str, str] | None = None,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    if not candidate_ids or any(not candidate_id.strip() for candidate_id in candidate_ids):
        return validation_result("candidate_ids must contain nonempty IDs")
    if edits and any(not content.strip() for content in edits.values()):
        return validation_result("candidate edit content must not be empty")
    if edits:
        stray = [eid for eid in edits if eid not in candidate_ids]
        if stray:
            return validation_result(
                "--edit ids must also appear in candidate_ids",
                text=f"--edit ids not in approve list: {', '.join(stray)}",
                ids=stray,
                details={"stray_edit_ids": stray},
            )
    store_obj = await resolve_store(store)
    approval = await store_obj.approve_candidates_result(candidate_ids, edits=edits)
    if approval.invalid_edit_ids:
        return missing_result("Candidate(s) not pending for --edit", ids=approval.invalid_edit_ids)
    facts = approval.facts
    if not facts:
        text = "No matching candidates were approved."
    else:
        lines = [f"Approved {len(facts)} candidate(s):\n"]
        for fact in facts:
            lines.append(f"- [{fact.category.value}] {fact.content} (id: {fact.id})")
        text = "\n".join(lines)

    applied_ids = list(approval.approved)
    unresolved_ids = list(approval.unresolved)
    errors = [
        EnvelopeError(code=ErrorCode.conflict, message=reason, ids=[candidate_id])
        for candidate_id, reason in approval.unresolved.items()
    ]
    if unresolved_ids:
        text += "\nUnapplied: " + ", ".join(unresolved_ids)
    payload = CandidateReviewPayload(
        facts=[FactPayload.model_validate(fact) for fact in facts],
        requested_ids=candidate_ids,
        applied_ids=applied_ids,
        unresolved_ids=unresolved_ids,
        edited_ids=approval.edited,
    )
    return OperationResult(
        envelope=outcome_envelope(
            payload.model_dump(mode="json", exclude_none=True),
            errors,
            applied=bool(facts or approval.edited),
        ),
        text=text,
    )


async def reject_candidates(
    candidate_ids: list[str],
    *,
    reason: str = "",
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    if not candidate_ids or any(not candidate_id.strip() for candidate_id in candidate_ids):
        return validation_result("candidate_ids must contain nonempty IDs")
    rejected = await (await resolve_store(store)).reject_candidates(candidate_ids, reason=reason)
    if not rejected:
        text = "No matching candidates were rejected."
    else:
        lines = [f"Rejected {len(rejected)} candidate(s):\n"]
        for candidate in rejected:
            lines.append(f"- [{candidate.category.value}] {candidate.content} (id: {candidate.id})")
        text = "\n".join(lines)
    applied_ids = [candidate.id for candidate in rejected]
    unresolved_ids = [
        candidate_id
        for candidate_id in dict.fromkeys(candidate_ids)
        if candidate_id not in applied_ids
    ]
    errors = [
        EnvelopeError(
            code=ErrorCode.conflict,
            message="Candidate is missing or no longer pending",
            ids=[candidate_id],
        )
        for candidate_id in unresolved_ids
    ]
    if unresolved_ids:
        text += "\nUnapplied: " + ", ".join(unresolved_ids)
    payload = CandidateReviewPayload(
        candidates=[CandidatePayload.model_validate(candidate) for candidate in rejected],
        requested_ids=candidate_ids,
        applied_ids=applied_ids,
        unresolved_ids=unresolved_ids,
    )
    return OperationResult(
        envelope=outcome_envelope(
            payload.model_dump(mode="json", exclude_none=True),
            errors,
            applied=bool(rejected),
        ),
        text=text,
    )
