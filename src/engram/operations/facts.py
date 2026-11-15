"""Inspect and change stored facts."""

from __future__ import annotations

from typing import Any

from engram.core.interfaces import (
    Envelope,
    EnvelopeMeta,
)
from engram.core.projects import canonical_project
from engram.operations.results import (
    OperationResult,
    category_from_value,
    fact_payload,
    invalid_category_result,
    invalid_positive_int_result,
    missing_result,
    validation_result,
)
from engram.operations.stores import resolve_store
from engram.storage.async_store import AsyncFactStore, run_sync
from engram.storage.store import FactStore


async def forget(
    fact_id: str,
    *,
    reason: str = "",
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    fact = await (await resolve_store(store)).forget(fact_id, reason)
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
    memory_key: str | None = None,
    tags: list[str] | None = None,
    retrieval_hints: list[str] | None = None,
    project: str | None = None,
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    updates: dict[str, Any] = {}
    if content is not None:
        if not content.strip():
            return validation_result("content must not be empty")
        updates["content"] = content
    if category is not None:
        cat = category_from_value(category)
        if cat is None:
            return invalid_category_result(category)
        updates["category"] = cat
    if memory_key is not None:
        updates["memory_key"] = memory_key
    if tags is not None:
        updates["tags"] = tags
    if retrieval_hints is not None:
        updates["retrieval_hints"] = retrieval_hints
    if project is not None:
        updates["project"] = await run_sync(canonical_project, project)

    if not updates:
        return validation_result("No changes specified.")

    store_obj = await resolve_store(store)
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

    project = await run_sync(canonical_project, project)
    facts = await (await resolve_store(store)).load_active_facts(
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
            lines.append(f"- {meta_str} {fact.content} (id: {fact.id}, created: {created})")
        text = "\n".join(lines)
    return OperationResult(envelope=Envelope.success(data=data, meta=meta), text=text)


async def correct_memory(
    fact_id: str,
    new_content: str,
    *,
    category: str | None = None,
    tags: list[str] | None = None,
    memory_key: str | None = None,
    retrieval_hints: list[str] | None = None,
    project: str | None = None,
    reason: str = "",
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    if not new_content.strip():
        return validation_result("content must not be empty")

    cat = category_from_value(category)
    if category and cat is None:
        return invalid_category_result(category)

    project = await run_sync(canonical_project, project)
    new_fact = await (await resolve_store(store)).correct_fact(
        fact_id,
        new_content,
        category=cat,
        tags=tags,
        memory_key=memory_key,
        retrieval_hints=retrieval_hints,
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
        "memory_key": new_fact.memory_key,
        "retrieval_hints": new_fact.retrieval_hints,
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
    memory_key: str | None = None,
    retrieval_hints: list[str] | None = None,
    project: str | None = None,
    reason: str = "",
    store: FactStore | AsyncFactStore | None = None,
) -> OperationResult:
    if not merged_content.strip():
        return validation_result("content must not be empty")

    if len(source_ids) < 2:
        return validation_result(
            "merge_memories requires at least two source IDs",
            text="merge_memories requires at least two source IDs",
            details={"received": len(source_ids)},
        )

    cat = category_from_value(category)
    if category and cat is None:
        return invalid_category_result(category)

    project = await run_sync(canonical_project, project)
    result = await (await resolve_store(store)).merge_facts(
        source_ids,
        merged_content,
        category=cat,
        tags=tags,
        memory_key=memory_key,
        retrieval_hints=retrieval_hints,
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
                "memory_key": new_fact.memory_key,
                "retrieval_hints": new_fact.retrieval_hints,
                "consolidates": new_fact.consolidates,
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
    fact = await (await resolve_store(store)).mark_stale(fact_id, reason)
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
    fact = await (await resolve_store(store)).unmark_stale(fact_id)
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


async def purge(store: FactStore | AsyncFactStore | None = None) -> OperationResult:
    result = await (await resolve_store(store)).purge()
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
    if not old_project.strip() or not new_project.strip():
        return validation_result("project names must not be empty")
    count = await (await resolve_store(store)).rename_project(old_project, new_project)
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
