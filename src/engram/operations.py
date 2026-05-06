"""Shared application operations for Engram's MCP and CLI adapters."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from engram.config import get_settings
from engram.doctor import check_provider, repair_store, run_doctor
from engram.importer import import_claude_code_memories
from engram.interfaces import (
    Envelope,
    EnvelopeMeta,
    not_found_error,
    provider_error,
    storage_error,
    validation_error,
)
from engram.models import (
    CandidateStatus,
    Fact,
    FactCategory,
    IngestionRecord,
    MemoryCandidate,
    RecallRecord,
)
from engram.observer import extract_facts, suggest_memories as _suggest_memories
from engram.provenance import DEFAULT_MAX_PREFILTER_MATCHES, DEFAULT_MAX_SOURCES
from engram.retriever import (
    RELEVANCE_FLOOR,
    recall as _recall,
    recall_with_provenance as _recall_with_provenance,
)
from engram.store import AsyncFactStore, FactStore, format_facts_for_llm
from engram.synthesizer import format_synthesis_result, synthesize as _synthesize

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
    env = Envelope.failure(
        validation_error(
            f"Invalid category: {value}",
            details={"valid": [c.value for c in FactCategory]},
        )
    )
    valid = ", ".join(c.value for c in FactCategory)
    return OperationResult(
        envelope=env,
        text=f"Invalid category: {value}. Use one of: {valid}",
        exit_code=EXIT_VALIDATION,
    )


def invalid_format_result(value: str) -> OperationResult:
    env = Envelope.failure(
        validation_error(
            f"Unsupported format: {value}. Use 'text' or 'json'.",
            details={"parameter": "format", "value": value},
        )
    )
    return OperationResult(
        envelope=env,
        text=f"Unsupported format: {value}. Use 'text' or 'json'.",
        exit_code=EXIT_VALIDATION,
    )


def fact_payload(fact: Fact) -> dict:
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
        return OperationResult(
            envelope=Envelope.failure(
                provider_error(
                    f"Memory extraction failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                )
            ),
            text=f"Memory extraction failed: {exc}",
            exit_code=EXIT_RUNTIME,
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
        return OperationResult(
            envelope=Envelope.failure(
                provider_error(
                    f"Memory suggestion failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                )
            ),
            text=f"Memory suggestion failed: {exc}",
            exit_code=EXIT_RUNTIME,
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
    try:
        candidate_status = CandidateStatus(status)
    except ValueError:
        valid = ", ".join(item.value for item in CandidateStatus)
        env = Envelope.failure(
            validation_error(
                f"Unsupported status: {status}. Use one of: {valid}.",
                details={"valid": [item.value for item in CandidateStatus]},
            )
        )
        return OperationResult(
            envelope=env,
            text=f"Unsupported status: {status}. Use one of: {valid}.",
            exit_code=EXIT_VALIDATION,
        )

    store_obj = async_store(store)
    candidates = await store_obj.load_candidates(
        status=candidate_status,
        project=project,
        limit=None if search else limit,
    )

    total_before_search = len(candidates)
    if search:
        search_lower = search.lower()
        candidates = [c for c in candidates if search_lower in c.content.lower()]
        candidates = candidates[:limit]

    meta = EnvelopeMeta(
        limit=limit,
        returned=len(candidates),
        total=total_before_search,
        truncated=len(candidates) >= limit,
        truncation_reason="default_limit" if len(candidates) >= limit else None,
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
            return OperationResult(
                envelope=Envelope.failure(
                    validation_error(
                        "--edit ids must also appear in candidate_ids",
                        ids=stray,
                        details={"stray_edit_ids": stray},
                    )
                ),
                text=f"--edit ids not in approve list: {', '.join(stray)}",
                exit_code=EXIT_VALIDATION,
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
            return OperationResult(
                envelope=Envelope.failure(
                    not_found_error(
                        "Candidate(s) not found for --edit",
                        ids=missing,
                    )
                ),
                text=f"Candidate(s) not found: {', '.join(missing)}",
                exit_code=EXIT_NOT_FOUND,
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
        return OperationResult(
            envelope=Envelope.failure(
                provider_error(
                    f"Recall trace failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                )
            ),
            text=f"Recall trace failed: {exc}",
            exit_code=EXIT_RUNTIME,
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
        env = Envelope.failure(
            validation_error(
                f"Unsupported mode: {mode}. Use 'answer' or 'prompt'.",
                details={"parameter": "mode", "value": mode},
            )
        )
        return OperationResult(
            envelope=env,
            text=f"Unsupported mode: {mode}. Use 'answer' or 'prompt'.",
            exit_code=EXIT_VALIDATION,
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
            envelope=Envelope.success(data={"facts": [], "mode": mode, "message": text}),
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
        return OperationResult(
            envelope=Envelope.failure(
                not_found_error(f"Fact {fact_id} not found", ids=[fact_id])
            ),
            text=f"Fact not found: {fact_id}",
            exit_code=EXIT_NOT_FOUND,
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
        env = Envelope.failure(validation_error("No changes specified."))
        return OperationResult(
            envelope=env,
            text="No changes specified.",
            exit_code=EXIT_VALIDATION,
        )

    store_obj = async_store(store)
    all_facts = await store_obj.load_facts()
    existing = next((f for f in all_facts if f.id == fact_id), None)
    if existing is None:
        return OperationResult(
            envelope=Envelope.failure(
                not_found_error(f"Fact {fact_id} not found", ids=[fact_id])
            ),
            text=f"Fact not found: {fact_id}",
            exit_code=EXIT_NOT_FOUND,
        )
    if existing.confidence == 0.0:
        text = (
            f"Fact {fact_id} is forgotten and cannot be edited. "
            "Restore it first by approving a candidate that supersedes it."
        )
        return OperationResult(
            envelope=Envelope.failure(
                validation_error(text, ids=[fact_id], details={"forgotten": True})
            ),
            text=text,
            exit_code=EXIT_VALIDATION,
        )

    fact = await store_obj.update_fact(fact_id, **updates)
    if fact is None:
        return OperationResult(
            envelope=Envelope.failure(
                not_found_error(f"Fact {fact_id} not found", ids=[fact_id])
            ),
            text=f"Fact not found: {fact_id}",
            exit_code=EXIT_NOT_FOUND,
        )

    await store_obj.log_ingestion(
        IngestionRecord(
            source="edit",
            facts_updated=[fact_id],
            agent_model="manual_edit",
        )
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
    cat = category_from_value(category)
    if category and cat is None:
        return invalid_category_result(category)

    facts = await async_store(store).load_active_facts(
        category=cat,
        project=project,
        limit=limit,
        include_stale=include_stale,
    )
    data = [fact_payload(f) for f in facts]
    meta = EnvelopeMeta(
        limit=limit,
        returned=len(data),
        truncated=len(data) >= limit,
        truncation_reason="default_limit" if len(data) >= limit else None,
    )
    if not facts:
        text = "No facts found matching the criteria."
    else:
        lines = [f"Found {len(facts)} fact(s):"]
        for fact in facts:
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
        return OperationResult(
            envelope=Envelope.failure(
                not_found_error(f"Active fact {fact_id} not found", ids=[fact_id])
            ),
            text=f"Active fact {fact_id} not found",
            exit_code=EXIT_NOT_FOUND,
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
        env = Envelope.failure(
            validation_error(
                "merge_memories requires at least two source IDs",
                details={"received": len(source_ids)},
            )
        )
        return OperationResult(
            envelope=env,
            text="merge_memories requires at least two source IDs",
            exit_code=EXIT_VALIDATION,
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
        env = Envelope.failure(
            validation_error(
                "Could not merge — sources missing, forgotten, or in different projects.",
                ids=source_ids,
            )
        )
        return OperationResult(
            envelope=env,
            text="Could not merge — sources missing, forgotten, or in different projects.",
            exit_code=EXIT_VALIDATION,
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
        return OperationResult(
            envelope=Envelope.failure(
                not_found_error(f"Fact {fact_id} not found", ids=[fact_id])
            ),
            text=f"Fact {fact_id} not found",
            exit_code=EXIT_NOT_FOUND,
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
        return OperationResult(
            envelope=Envelope.failure(
                not_found_error(f"Fact {fact_id} not found", ids=[fact_id])
            ),
            text=f"Fact {fact_id} not found",
            exit_code=EXIT_NOT_FOUND,
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
        env = Envelope.failure(
            validation_error(
                f"Unsupported import source: {source}. Use 'claude_code'.",
                details={"parameter": "source", "value": source},
            )
        )
        return OperationResult(
            envelope=env,
            text=f"Unsupported import source: {source}. Use 'claude_code'.",
            exit_code=EXIT_VALIDATION,
        )

    result = await import_claude_code_memories(store=async_store(store))
    if "error" in result:
        return OperationResult(
            envelope=Envelope.failure(storage_error(result["error"])),
            text=result["error"],
            exit_code=EXIT_RUNTIME,
        )
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
        text = f"Renamed {count} record(s) from project '{old_project}' → '{new_project}'."
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


async def synthesize(
    *,
    project: str | None = None,
    dry_run: bool = True,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    result = await _synthesize(project=project, dry_run=dry_run, store=async_store(store))
    data = {
        "total_analyzed": result.total_analyzed,
        "kept": result.kept,
        "removed": result.removed,
        "rewritten": result.rewritten,
        "merged_groups": result.merged_groups,
        "merged_sources": result.merged_sources,
        "errors": result.errors,
        "details": result.details,
        "dry_run": dry_run,
    }
    return OperationResult(
        envelope=Envelope.success(data=data),
        text=format_synthesis_result(result, dry_run=dry_run),
    )


async def doctor(
    *,
    check_provider_flag: bool = False,
    repair: bool = False,
    repair_jsonl: bool = False,
    recover_transactions: bool = False,
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
    if repair and (repair_jsonl or recover_transactions):
        repair_summary = repair_store(
            store_obj,
            repair_jsonl=repair_jsonl,
            recover_transactions=recover_transactions,
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

    return OperationResult(envelope=Envelope.success(data=stats), text="\n".join(lines))


def _format_recall_summary(records: list[RecallRecord], heading: str) -> list[str]:
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


async def recall_stats(
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    records = await async_store(store).load_recall_log(limit=500)
    if not records:
        text = "No recall data yet."
        return OperationResult(
            envelope=Envelope.success(data={"records": [], "message": text}),
            text=text,
        )

    lines = _format_recall_summary(records, "# Recall Stats\n")
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

    data = {
        "total_queries": len(records),
        "selector_versions": selector_versions,
        "records": [record.model_dump(mode="json") for record in records],
    }
    return OperationResult(envelope=Envelope.success(data=data), text="\n".join(lines))


__all__ = [
    "EXIT_DOCTOR_ERROR",
    "EXIT_NOT_FOUND",
    "EXIT_OK",
    "EXIT_RUNTIME",
    "EXIT_VALIDATION",
    "OperationResult",
    "approve_candidates",
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
    "synthesize",
    "unmark_stale",
]
