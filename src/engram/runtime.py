"""Application dependencies and task ownership for one CLI or MCP lifetime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from engram.core.config import Settings, get_settings, settings_scope
from engram.llm import CompletionClient, OpenAICompletionClient, completion_client
from engram.storage.store import AsyncFactStore, FactStore, run_sync

logger = logging.getLogger(__name__)
_CURRENT_RUNTIME: ContextVar[Runtime | None] = ContextVar("engram_runtime", default=None)


class Runtime:
    """Own the store, provider, and background tasks until explicit shutdown."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        store: FactStore | AsyncFactStore | None = None,
        provider: CompletionClient | None = None,
    ):
        self.settings = settings or get_settings()
        if store is not None:
            if settings is not None and settings.data_dir.resolve() != store.data_dir.resolve():
                raise ValueError("settings.data_dir must match the injected store")
            self.settings = self.settings.model_copy(update={"data_dir": store.data_dir})
        self.provider = provider or OpenAICompletionClient()
        self._store = (
            store
            if isinstance(store, AsyncFactStore)
            else AsyncFactStore(store)
            if store is not None
            else None
        )
        self._store_lock = asyncio.Lock()
        self._recovered = False
        self._tasks: list[asyncio.Task[None]] = []
        self._sync_started = False
        self._closed = False

    @contextmanager
    def bind(self) -> Generator[Runtime]:
        token = _CURRENT_RUNTIME.set(self)
        try:
            with settings_scope(self.settings), completion_client(self.provider):
                yield self
        finally:
            _CURRENT_RUNTIME.reset(token)

    async def store(self, *, recover: bool = True) -> AsyncFactStore:
        if self._closed:
            raise RuntimeError("Engram runtime is closed")
        async with self._store_lock:
            if self._store is None:
                self._store = AsyncFactStore(await run_sync(FactStore, self.settings.data_dir))
            if recover and not self._recovered:
                await self._store.recover_transactions()
                self._recovered = True
            return self._store

    async def start_background(self) -> None:
        with self.bind():
            if self.settings.sync_enabled:
                from engram.storage.sync import auto_sync_loop

                self._tasks.append(
                    asyncio.create_task(
                        auto_sync_loop(
                            self.settings.data_dir,
                            interval=self.settings.sync_interval,
                            timeout=self.settings.sync_timeout,
                        )
                    )
                )
                self._sync_started = True
            if self.settings.maintenance_enabled:
                self._tasks.append(asyncio.create_task(self._upkeep()))

    async def _upkeep(self) -> None:
        from engram.maintenance.upkeep import upkeep_loop

        await upkeep_loop(
            self.store,
            data_dir=self.settings.data_dir,
            interval=self.settings.maintenance_interval,
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        try:
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            if self._sync_started:
                from engram.storage.sync import run_final_sync

                await run_final_sync(self.settings.data_dir, timeout=self.settings.sync_timeout)
        finally:
            try:
                await self.provider.aclose()
            finally:
                self._closed = True


def current_runtime() -> Runtime | None:
    return _CURRENT_RUNTIME.get()
