"""Shared application operations for Engram's MCP and CLI adapters."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from engram.core.config import get_settings
from engram.maintenance.doctor import check_provider, repair_store, run_doctor
from engram.extraction.importer import import_claude_code_memories
from engram.core.interfaces import (
    Envelope,
    EnvelopeError,
    EnvelopeMeta,
    not_found_error,
    provider_error,
    storage_error,
    validation_error,
)
from engram.maintenance.memory_audit import (
    audit_memory_store as _audit_memory_store,
    format_audit_result,
)
from engram.core.models import (
    CandidateStatus,
    FactBase,
    FactCategory,
    MemoryCandidate,
    RecallRecord,
)
from engram.extraction.observer import (
    extract_facts,
    suggest_memories as _suggest_memories,
)
from engram.core.provenance import DEFAULT_MAX_PREFILTER_MATCHES, DEFAULT_MAX_SOURCES
from engram.recall.retriever import (
    RELEVANCE_FLOOR,
    recall as _recall,
    recall_with_provenance as _recall_with_provenance,
)
from engram.storage.store import AsyncFactStore, FactStore, format_facts_for_llm
from engram.storage.sync import (
    DEFAULT_GIT_TIMEOUT_SECONDS,
    SyncError,
    sync as _sync,
)

EXIT_OK = 0
EXIT_VALIDATION = 1
EXIT_NOT_FOUND = 2
EXIT_RUNTIME = 3
EXIT_DOCTOR_ERROR = 4


@dataclass
class OperationResult:
    """Result shared by CLI and MCP adapters."""

    envelope: Envelope
    text: str
    exit_code: int = EXIT_OK

    def render(self, *, as_json: bool) -> str:
        return self.envelope.to_json() if as_json else self.text


def failure_result(
    error: EnvelopeError, text: str, exit_code: int = EXIT_RUNTIME
) -> OperationResult:
    return OperationResult(
        envelope=Envelope.failure(error),
        text=text,
        exit_code=exit_code,
    )


def validation_result(
    message: str,
    *,
    text: str | None = None,
    ids: list[str] | None = None,
    details: dict | None = None,
) -> OperationResult:
    return failure_result(
        validation_error(message, ids=ids, details=details),
        text or message,
        EXIT_VALIDATION,
    )


def missing_result(
    message: str, *, text: str | None = None, ids: list[str] | None = None
) -> OperationResult:
    return failure_result(
        not_found_error(message, ids=ids),
        text or message,
        EXIT_NOT_FOUND,
    )


def async_store(store: FactStore | AsyncFactStore | None = None) -> AsyncFactStore:
    if store is None:
        return AsyncFactStore()
    if isinstance(store, AsyncFactStore):
        return store
    return AsyncFactStore(store)


def category_from_value(value: str | None) -> FactCategory | None:
    if value is None:
        return None
    try:
        return FactCategory(value)
    except ValueError:
        return None


def invalid_category_result(value: str) -> OperationResult:
    valid = ", ".join(c.value for c in FactCategory)
    return validation_result(
        f"Invalid category: {value}",
        text=f"Invalid category: {value}. Use one of: {valid}",
        details={"valid": [c.value for c in FactCategory]},
    )


def invalid_format_result(value: str) -> OperationResult:
    return validation_result(
        f"Unsupported format: {value}. Use 'text' or 'json'.",
        text=f"Unsupported format: {value}. Use 'text' or 'json'.",
        details={"parameter": "format", "value": value},
    )


def invalid_positive_int_result(parameter: str) -> OperationResult:
    message = f"{parameter} must be greater than zero"
    return validation_result(message, details={"parameter": parameter})


def fact_payload(fact: FactBase) -> dict:
    return {
        "id": fact.id,
        "category": fact.category.value,
        "project": fact.project,
        "content": fact.content,
        "confidence": fact.confidence,
        "stale": fact.stale,
        "stale_reason": fact.stale_reason,
        "supersedes": fact.supersedes,
        "tags": fact.tags,
        "created_at": fact.created_at.isoformat(),
        "updated_at": fact.updated_at.isoformat(),
    }


def candidate_payload(candidate: MemoryCandidate) -> dict:
    data = fact_payload(candidate)
    data.update(
        status=candidate.status.value,
        review_note=candidate.review_note,
        why_store=candidate.why_store,
    )
    return data


async def remember(
    content: str,
    *,
    source: str = "conversation",
    project: str | None = None,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    try:
        facts = await extract_facts(
            content,
            source=source,
            project=project,
            store=async_store(store),
        )
    except Exception as exc:
        return failure_result(
            provider_error(
                f"Memory extraction failed: {exc}",
                details={"exception_type": type(exc).__name__},
            ),
            text=f"Memory extraction failed: {exc}",
        )

    if not facts:
        text = "No structured facts could be extracted from the input."
        return OperationResult(
            envelope=Envelope.success(data={"facts": [], "message": text}),
            text=text,
        )

    lines = [f"Stored {len(facts)} fact(s):\n"]
    for fact in facts:
        lines.append(f"- [{fact.category.value}] {fact.content} (id: {fact.id})")
    return OperationResult(
        envelope=Envelope.success(data={"facts": [fact_payload(f) for f in facts]}),
        text="\n".join(lines),
    )


async def suggest_memories(
    content: str,
    *,
    source: str = "conversation",
    project: str | None = None,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    try:
        candidates = await _suggest_memories(
            content,
            source=source,
            project=project,
            store=async_store(store),
        )
    except Exception as exc:
        return failure_result(
            provider_error(
                f"Memory suggestion failed: {exc}",
                details={"exception_type": type(exc).__name__},
            ),
            text=f"Memory suggestion failed: {exc}",
        )

    if not candidates:
        text = "No memory candidates were proposed from the input."
        return OperationResult(
            envelope=Envelope.success(data={"candidates": [], "message": text}),
            text=text,
        )

    lines = [f"Queued {len(candidates)} memory candidate(s):\n"]
    for candidate in candidates:
        why = candidate.why_store or "useful future context"
        lines.append(
            f"- [{candidate.category.value}] {candidate.content} "
            f"(id: {candidate.id}, why: {why})"
        )
    return OperationResult(
        envelope=Envelope.success(
            data={"candidates": [candidate_payload(c) for c in candidates]}
        ),
        text="\n".join(lines),
    )


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

    store_obj = async_store(store)
    all_candidates = await store_obj.load_candidates(
        status=candidate_status,
        project=project,
        limit=None,
    )

    if search:
        search_lower = search.lower()
        all_candidates = [
            c for c in all_candidates if search_lower in c.content.lower()
        ]
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
    store_obj = async_store(store)
    if edits:
        stray = [eid for eid in edits if eid not in candidate_ids]
        if stray:
            return validation_result(
                "--edit ids must also appear in candidate_ids",
                text=f"--edit ids not in approve list: {', '.join(stray)}",
                ids=stray,
                details={"stray_edit_ids": stray},
            )
        # Edits rewrite candidates.jsonl outside the approval transaction; a
        # crash mid-loop leaves edits applied but candidates un-approved (safe
        # to retry — approval is idempotent on edited content).
        missing: list[str] = []
        for candidate_id, content in edits.items():
            updated = await store_obj.update_candidate(candidate_id, content=content)
            if updated is None:
                missing.append(candidate_id)
        if missing:
            return missing_result(
                "Candidate(s) not found for --edit",
                text=f"Candidate(s) not found: {', '.join(missing)}",
                ids=missing,
            )

    facts = await store_obj.approve_candidates(candidate_ids)
    if not facts:
        text = "No matching candidates were approved."
    else:
        lines = [f"Approved {len(facts)} candidate(s):\n"]
        for fact in facts:
            lines.append(f"- [{fact.category.value}] {fact.content} (id: {fact.id})")
        text = "\n".join(lines)

    return OperationResult(
        envelope=Envelope.success(data={"facts": [fact_payload(f) for f in facts]}),
        text=text,
    )


async def reject_candidates(
    candidate_ids: list[str],
    *,
    reason: str = "",
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    rejected = await async_store(store).reject_candidates(candidate_ids, reason=reason)
    if not rejected:
        text = "No matching candidates were rejected."
    else:
        lines = [f"Rejected {len(rejected)} candidate(s):\n"]
        for candidate in rejected:
            lines.append(
                f"- [{candidate.category.value}] {candidate.content} "
                f"(id: {candidate.id})"
            )
        text = "\n".join(lines)
    return OperationResult(
        envelope=Envelope.success(
            data={"candidates": [candidate_payload(c) for c in rejected]}
        ),
        text=text,
    )


async def recall(
    query: str,
    *,
    project: str | None = None,
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

    store_obj = async_store(store)
    answer, quality, provenance, _ = await _recall_with_provenance(
        query,
        project=project,
        store=store_obj,
        max_sources=max_sources,
        max_prefilter_matches=max_prefilter_matches,
    )
    if format == "text":
        return OperationResult(
            envelope=Envelope.success(data={"answer": answer}),
            text=answer,
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
    return OperationResult(
        envelope=Envelope.success(data=data, warnings=provenance.warnings, meta=meta),
        text=answer,
    )


async def recall_trace(
    query: str,
    *,
    project: str | None = None,
    verbose: bool = False,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_prefilter_matches: int = DEFAULT_MAX_PREFILTER_MATCHES,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    if max_sources < 1:
        return invalid_positive_int_result("max_sources")
    if max_prefilter_matches < 1:
        return invalid_positive_int_result("max_prefilter_matches")

    try:
        answer, quality, provenance, trace = await _recall_with_provenance(
            query,
            project=project,
            store=async_store(store),
            with_trace=True,
            verbose_trace=verbose,
            max_sources=max_sources,
            max_prefilter_matches=max_prefilter_matches,
        )
    except Exception as exc:
        return failure_result(
            provider_error(
                f"Recall trace failed: {exc}",
                details={"exception_type": type(exc).__name__},
            ),
            text=f"Recall trace failed: {exc}",
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


async def recall_context(
    query: str,
    *,
    project: str | None = None,
    mode: str = "answer",
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    store_obj = async_store(store)
    if mode == "answer":
        answer = await _recall(query, project=project, store=store_obj)
        return OperationResult(
            envelope=Envelope.success(data={"answer": answer, "mode": mode}),
            text=answer,
        )

    if mode != "prompt":
        return validation_result(
            f"Unsupported mode: {mode}. Use 'answer' or 'prompt'.",
            text=f"Unsupported mode: {mode}. Use 'answer' or 'prompt'.",
            details={"parameter": "mode", "value": mode},
        )

    settings = get_settings()
    scored = await store_obj.prefilter_facts(
        query=query,
        project=project,
        limit=settings.max_facts_per_agent,
    )
    facts = [f for score, f in scored if score >= RELEVANCE_FLOOR]
    if not facts:
        text = "No relevant memories found for this query."
        return OperationResult(
            envelope=Envelope.success(
                data={"facts": [], "mode": mode, "message": text}
            ),
            text=text,
        )

    text = "# Memory Context\n\n" + format_facts_for_llm(facts)
    return OperationResult(
        envelope=Envelope.success(
            data={"facts": [fact_payload(f) for f in facts], "mode": mode}
        ),
        text=text,
    )


async def forget(
    fact_id: str,
    *,
    reason: str = "",
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    fact = await async_store(store).forget(fact_id, reason)
    if fact is None:
        return missing_result(
            f"Fact {fact_id} not found",
            text=f"Fact not found: {fact_id}",
            ids=[fact_id],
        )
    return OperationResult(
        envelope=Envelope.success(data={"fact": fact_payload(fact), "forgotten": True}),
        text=f"Forgotten: [{fact.category.value}] {fact.content}",
    )


async def edit_fact(
    fact_id: str,
    *,
    content: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    project: str | None = None,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    updates: dict = {}
    if content is not None:
        updates["content"] = content
    if category is not None:
        cat = category_from_value(category)
        if cat is None:
            return invalid_category_result(category)
        updates["category"] = cat
    if tags is not None:
        updates["tags"] = tags
    if project is not None:
        updates["project"] = project

    if not updates:
        return validation_result("No changes specified.")

    store_obj = async_store(store)
    all_facts = await store_obj.load_facts()
    existing = next((f for f in all_facts if f.id == fact_id), None)
    if existing is None:
        return missing_result(
            f"Fact {fact_id} not found",
            text=f"Fact not found: {fact_id}",
            ids=[fact_id],
        )
    if existing.confidence == 0.0:
        text = (
            f"Fact {fact_id} is forgotten and cannot be edited. "
            "Restore it first by approving a candidate that supersedes it."
        )
        return validation_result(
            text,
            text=text,
            ids=[fact_id],
            details={"forgotten": True},
        )

    fact = await store_obj.update_fact(fact_id, **updates)
    if fact is None:
        return missing_result(
            f"Fact {fact_id} not found",
            text=f"Fact not found: {fact_id}",
            ids=[fact_id],
        )

    return OperationResult(
        envelope=Envelope.success(data={"fact": fact_payload(fact)}),
        text=f"Updated: [{fact.category.value}] {fact.content} (id: {fact.id})",
    )


async def inspect(
    *,
    category: str | None = None,
    project: str | None = None,
    limit: int = 50,
    include_stale: bool = False,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    if limit < 1:
        return invalid_positive_int_result("limit")

    cat = category_from_value(category)
    if category and cat is None:
        return invalid_category_result(category)

    facts = await async_store(store).load_active_facts(
        category=cat,
        project=project,
        limit=None,
        include_stale=include_stale,
    )
    visible_facts = facts[:limit]
    data = [fact_payload(f) for f in visible_facts]
    truncated = len(facts) > limit
    meta = EnvelopeMeta(
        limit=limit,
        returned=len(data),
        total=len(facts),
        truncated=truncated,
        truncation_reason="default_limit" if truncated else None,
    )
    if not visible_facts:
        text = "No facts found matching the criteria."
    else:
        lines = [f"Found {len(visible_facts)} fact(s):"]
        for fact in visible_facts:
            meta_str = f"[{fact.category.value}]"
            if fact.project:
                meta_str += f" [{fact.project}]"
            if fact.stale:
                meta_str += " [stale]"
            created = fact.created_at.strftime("%Y-%m-%d")
            lines.append(
                f"- {meta_str} {fact.content} (id: {fact.id}, created: {created})"
            )
        text = "\n".join(lines)
    return OperationResult(envelope=Envelope.success(data=data, meta=meta), text=text)


async def correct_memory(
    fact_id: str,
    new_content: str,
    *,
    category: str | None = None,
    tags: list[str] | None = None,
    project: str | None = None,
    reason: str = "",
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    cat = category_from_value(category)
    if category and cat is None:
        return invalid_category_result(category)

    new_fact = await async_store(store).correct_fact(
        fact_id,
        new_content,
        category=cat,
        tags=tags,
        project=project,
        reason=reason,
    )
    if new_fact is None:
        return missing_result(
            f"Active fact {fact_id} not found",
            text=f"Active fact {fact_id} not found",
            ids=[fact_id],
        )

    data = {
        "new_fact_id": new_fact.id,
        "superseded_fact_id": fact_id,
        "category": new_fact.category.value,
        "project": new_fact.project,
        "content": new_fact.content,
    }
    return OperationResult(
        envelope=Envelope.success(data=data),
        text=(
            f"Corrected {fact_id} -> {new_fact.id}\n"
            f"  [{new_fact.category.value}] {new_fact.content}"
        ),
    )


async def merge_memories(
    source_ids: list[str],
    merged_content: str,
    *,
    category: str | None = None,
    tags: list[str] | None = None,
    project: str | None = None,
    reason: str = "",
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    if len(source_ids) < 2:
        return validation_result(
            "merge_memories requires at least two source IDs",
            text="merge_memories requires at least two source IDs",
            details={"received": len(source_ids)},
        )

    cat = category_from_value(category)
    if category and cat is None:
        return invalid_category_result(category)

    result = await async_store(store).merge_facts(
        source_ids,
        merged_content,
        category=cat,
        tags=tags,
        project=project,
        reason=reason,
    )
    if result is None:
        return validation_result(
            "Could not merge — sources missing, forgotten, or in different projects.",
            text="Could not merge — sources missing, forgotten, or in different projects.",
            ids=source_ids,
        )

    new_fact, superseded_ids = result
    return OperationResult(
        envelope=Envelope.success(
            data={
                "new_fact_id": new_fact.id,
                "superseded_fact_ids": superseded_ids,
                "category": new_fact.category.value,
                "project": new_fact.project,
                "content": new_fact.content,
            }
        ),
        text=f"Merged {len(superseded_ids)} facts -> {new_fact.id}",
    )


async def mark_stale(
    fact_id: str,
    *,
    reason: str = "",
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    fact = await async_store(store).mark_stale(fact_id, reason)
    if fact is None:
        return missing_result(
            f"Fact {fact_id} not found",
            text=f"Fact {fact_id} not found",
            ids=[fact_id],
        )
    return OperationResult(
        envelope=Envelope.success(
            data={
                "fact_id": fact.id,
                "stale": True,
                "stale_reason": fact.stale_reason,
            }
        ),
        text=f"Marked {fact.id} stale.",
    )


async def unmark_stale(
    fact_id: str,
    *,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    fact = await async_store(store).unmark_stale(fact_id)
    if fact is None:
        return missing_result(
            f"Fact {fact_id} not found",
            text=f"Fact {fact_id} not found",
            ids=[fact_id],
        )
    return OperationResult(
        envelope=Envelope.success(data={"fact_id": fact.id, "stale": False}),
        text=f"Cleared stale on {fact.id}.",
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

    result = await import_claude_code_memories(store=async_store(store))
    if "error" in result:
        return failure_result(storage_error(result["error"]), text=result["error"])
    if result.get("message") and result.get("total_facts", 0) == 0:
        return OperationResult(
            envelope=Envelope.success(data=result),
            text=result["message"],
        )

    lines = [
        f"Imported {result['total_facts']} facts from {result['imported_files']} files:\n"
    ]
    for detail in result.get("details", []):
        lines.append(
            f"- {detail['file']} ({detail['project']}): {detail['facts_extracted']} facts"
        )
    return OperationResult(
        envelope=Envelope.success(data=result),
        text="\n".join(lines),
    )


async def purge(store: FactStore | AsyncFactStore | None = None) -> OperationResult:
    result = await async_store(store).purge()
    if result["purged"] == 0:
        text = "Nothing to purge — store is clean."
    else:
        text = f"Purged {result['purged']} facts ({result['retained']} retained)."
    return OperationResult(envelope=Envelope.success(data=result), text=text)


async def rename_project(
    old_project: str,
    new_project: str,
    *,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    count = await async_store(store).rename_project(old_project, new_project)
    if count == 0:
        text = f"No facts or candidates found with project '{old_project}'."
    else:
        text = (
            f"Renamed {count} record(s) from project '{old_project}' → '{new_project}'."
        )
    return OperationResult(
        envelope=Envelope.success(
            data={
                "old_project": old_project,
                "new_project": new_project,
                "renamed": count,
            }
        ),
        text=text,
    )


async def audit_memories(
    *,
    project: str | None = None,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    result = await _audit_memory_store(project=project, store=async_store(store))
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
    store_obj = async_store(store)
    provider_issue = None
    if check_provider_flag:
        provider_issue = await check_provider()
    report = run_doctor(
        store_obj,
        check_provider_flag=check_provider_flag,
        provider_issue=provider_issue,
    )

    repair_summary = None
    repair_requested = (
        repair or repair_jsonl or recover_transactions or repair_orphaned_supersessions
    )
    if repair_requested and (
        repair_jsonl or recover_transactions or repair_orphaned_supersessions
    ):
        repair_summary = repair_store(
            store_obj,
            repair_jsonl=repair_jsonl,
            recover_transactions=recover_transactions,
            repair_orphaned_supersessions=repair_orphaned_supersessions,
        )

    text_lines = [f"Doctor: {report.status}"]
    for issue in report.issues:
        text_lines.append(
            f"  [{issue.severity}] {issue.category}/{issue.code}: {issue.message}"
        )
        if issue.repair:
            text_lines.append(f"    repair: {issue.repair}")
    return OperationResult(
        envelope=Envelope.success(
            data={"report": report.model_dump(mode="json"), "repair": repair_summary}
        ),
        text="\n".join(text_lines),
        exit_code=EXIT_DOCTOR_ERROR if report.status == "error" else EXIT_OK,
    )


async def memory_stats(
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    stats = await async_store(store).stats()
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


def _recall_stats_payload(records: list[RecallRecord]) -> dict:
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
    selector_versions = sorted(
        {r.selector_version for r in records if r.selector_version}
    )

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
    lines.append(
        f"- LLM calls (reported): {llm_calls if llm_calls is not None else '-'}"
    )
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
        return parsed.replace(tzinfo=timezone.utc)
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


def _recall_stats_lines(records: list[RecallRecord], payload: dict) -> list[str]:
    lines = _format_recall_summary(records, "# Recall Stats\n")
    selector_versions = payload["selector_versions"]
    if len(selector_versions) >= 2:
        for version in selector_versions:
            subset = [r for r in records if r.selector_version == version]
            lines.append("")
            lines.extend(
                _format_recall_summary(subset, heading=f"# Selector {version}\n")
            )
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

    all_records = await async_store(store).load_recall_log(limit=None)
    filtered_records = [
        record
        for record in all_records
        if since_dt is None or record.timestamp >= since_dt
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


async def sync(
    *,
    timeout: float = DEFAULT_GIT_TIMEOUT_SECONDS,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    """Run git-backed sync against the configured remote.

    Returns a structured ``OperationResult`` envelope. Sync failures surface as
    non-zero exit codes with the underlying git stderr preserved verbatim.
    """
    import asyncio as _asyncio

    data_dir = async_store(store).data_dir

    try:
        result = await _asyncio.to_thread(_sync, data_dir, timeout=timeout)
    except SyncError as exc:
        details: dict = {"sync_error_code": exc.code}
        if exc.git_stderr:
            details["git_stderr"] = exc.git_stderr
        return failure_result(
            storage_error(exc.message, details=details), text=exc.message
        )

    if result.get("status") == "skipped":
        text = f"Sync skipped: {result.get('reason', 'unknown')}."
    else:
        text = (
            f"Sync OK — pulled {result['pulled_commits']} commit(s), "
            f"pushed {result['pushed_commits']} commit(s) "
            f"({result['remote']}/{result['branch']}) in {result['took_ms']} ms."
        )
    return OperationResult(envelope=Envelope.success(data=result), text=text)


__all__ = [
    "EXIT_DOCTOR_ERROR",
    "EXIT_NOT_FOUND",
    "EXIT_OK",
    "EXIT_RUNTIME",
    "EXIT_VALIDATION",
    "OperationResult",
    "approve_candidates",
    "audit_memories",
    "async_store",
    "correct_memory",
    "doctor",
    "edit_fact",
    "forget",
    "import_memories",
    "inspect",
    "list_candidates",
    "mark_stale",
    "memory_stats",
    "merge_memories",
    "purge",
    "recall",
    "recall_context",
    "recall_stats",
    "recall_trace",
    "reject_candidates",
    "remember",
    "rename_project",
    "suggest_memories",
    "sync",
    "unmark_stale",
]
