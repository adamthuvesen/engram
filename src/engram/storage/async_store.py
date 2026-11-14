"""Async facade that runs blocking store operations off the event loop."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from engram.core.models import CandidateStatus, Fact, FactCategory, MemoryCandidate, RecallRecord
from engram.storage.changes import ApprovalResult, ChangeResult, ChangeSet
from engram.storage.search import SearchHit, SearchIndex
from engram.storage.store import FactStore

P = ParamSpec("P")
R = TypeVar("R")


async def run_sync[**P, R](func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Wait for an in-flight thread before propagating caller cancellation."""
    worker = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()
        raise


class AsyncFactStore:
    """Async facade for running blocking JSONL store operations off the event loop."""

    def __init__(self, store: FactStore | None = None):
        self.sync_store = store or FactStore()

    @property
    def data_dir(self) -> Path:
        return self.sync_store.data_dir

    @property
    def facts_path(self) -> Path:
        return self.sync_store.facts_path

    @property
    def candidates_path(self) -> Path:
        return self.sync_store.candidates_path

    @property
    def transaction_log_path(self) -> Path:
        return self.sync_store.transaction_log_path

    @property
    def recall_log_path(self) -> Path:
        return self.sync_store.recall_log_path

    async def _run(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        return await run_sync(func, *args, **kwargs)

    async def load_facts(self) -> list[Fact]:
        return await self._run(self.sync_store.load_facts)

    async def load_candidates(
        self,
        status: CandidateStatus | None = None,
        project: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryCandidate]:
        return await self._run(self.sync_store.load_candidates, status, project, limit)

    async def load_active_facts(
        self,
        category: FactCategory | None = None,
        project: str | None = None,
        include_global: bool = True,
        min_confidence: float = 0.1,
        limit: int | None = None,
        include_stale: bool = False,
    ) -> list[Fact]:
        return await self._run(
            self.sync_store.load_active_facts,
            category,
            project,
            include_global,
            min_confidence,
            limit,
            include_stale,
        )

    async def search_facts(
        self,
        query: str,
        project: str | None = None,
        limit: int | None = None,
        *,
        include_global: bool = True,
    ) -> list[SearchHit]:
        return await self._run(
            self.sync_store.search_facts,
            query,
            project,
            limit,
            include_global=include_global,
        )

    async def search_index(self) -> SearchIndex:
        return await self._run(self.sync_store.search_index)

    async def apply_changes(
        self, changes: ChangeSet, *, all_or_nothing: bool = False
    ) -> ChangeResult:
        return await self._run(
            self.sync_store.apply_changes, changes, all_or_nothing=all_or_nothing
        )

    async def append_facts(self, facts: list[Fact]) -> None:
        await self._run(self.sync_store.append_facts, facts)

    async def recover_transactions(self) -> int:
        return await self._run(self.sync_store.recover_transactions)

    async def append_pending_candidates(
        self, candidates: list[MemoryCandidate]
    ) -> list[MemoryCandidate]:
        return await self._run(self.sync_store.append_pending_candidates, candidates)

    async def append_candidates(self, candidates: list[MemoryCandidate]) -> None:
        await self._run(self.sync_store.append_candidates, candidates)

    async def update_fact(self, fact_id: str, **updates: Any) -> Fact | None:
        return await self._run(self.sync_store.update_fact, fact_id, **updates)

    async def batch_update_candidates(
        self, updates: dict[str, dict[str, Any]]
    ) -> list[MemoryCandidate]:
        return await self._run(self.sync_store.batch_update_candidates, updates)

    async def update_candidate(self, candidate_id: str, **updates: Any) -> MemoryCandidate | None:
        return await self._run(self.sync_store.update_candidate, candidate_id, **updates)

    async def rename_project(self, old_project: str, new_project: str) -> int:
        return await self._run(self.sync_store.rename_project, old_project, new_project)

    async def batch_update_facts(self, updates: dict[str, dict[str, Any]]) -> list[Fact]:
        return await self._run(self.sync_store.batch_update_facts, updates)

    async def forget(self, fact_id: str, reason: str = "") -> Fact | None:
        return await self._run(self.sync_store.forget, fact_id, reason)

    async def correct_fact(
        self,
        fact_id: str,
        new_content: str,
        **kwargs: Any,
    ) -> Fact | None:
        return await self._run(self.sync_store.correct_fact, fact_id, new_content, **kwargs)

    async def merge_facts(
        self,
        source_ids: list[str],
        merged_content: str,
        **kwargs: Any,
    ) -> tuple[Fact, list[str]] | None:
        return await self._run(self.sync_store.merge_facts, source_ids, merged_content, **kwargs)

    async def mark_stale(self, fact_id: str, reason: str = "") -> Fact | None:
        return await self._run(self.sync_store.mark_stale, fact_id, reason)

    async def unmark_stale(self, fact_id: str) -> Fact | None:
        return await self._run(self.sync_store.unmark_stale, fact_id)

    async def approve_candidates(self, candidate_ids: list[str]) -> list[Fact]:
        return await self._run(self.sync_store.approve_candidates, candidate_ids)

    async def approve_candidates_result(
        self,
        candidate_ids: list[str],
        *,
        edits: dict[str, str] | None = None,
    ) -> ApprovalResult:
        return await self._run(
            self.sync_store.approve_candidates_result,
            candidate_ids,
            edits=edits,
        )

    async def reject_candidates(
        self, candidate_ids: list[str], reason: str = ""
    ) -> list[MemoryCandidate]:
        return await self._run(self.sync_store.reject_candidates, candidate_ids, reason)

    async def purge(self) -> dict[str, Any]:
        return await self._run(self.sync_store.purge)

    async def stats(self) -> dict[str, Any]:
        return await self._run(self.sync_store.stats)

    def format_candidates_for_review(self, candidates: list[MemoryCandidate]) -> str:
        return self.sync_store.format_candidates_for_review(candidates)

    async def log_recall(self, record: RecallRecord) -> None:
        await self._run(self.sync_store.log_recall, record)

    async def load_recall_log(self, limit: int | None = 500) -> list[RecallRecord]:
        return await self._run(self.sync_store.load_recall_log, limit)
