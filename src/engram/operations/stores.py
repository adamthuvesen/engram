"""Resolve the store and project scope that an operation runs against."""

from __future__ import annotations

from engram.core.projects import register_project
from engram.runtime import current_runtime
from engram.storage.async_store import AsyncFactStore, run_sync
from engram.storage.store import FactStore


def async_store(store: FactStore | AsyncFactStore) -> AsyncFactStore:
    return store if isinstance(store, AsyncFactStore) else AsyncFactStore(store)


async def resolve_store(
    store: FactStore | AsyncFactStore | None = None,
) -> AsyncFactStore:
    if store is not None:
        return async_store(store)
    runtime = current_runtime()
    if runtime is not None:
        return await runtime.store()
    result = AsyncFactStore(await run_sync(FactStore))
    await result.recover_transactions()
    return result


async def scope_project(store: AsyncFactStore, project: str | None) -> str | None:
    """Canonicalize a caller's project (name or working directory path).

    A path also records its repository root so upkeep can verify anchors.
    """
    if project is None:
        return None
    return await run_sync(register_project, store.data_dir, project)
