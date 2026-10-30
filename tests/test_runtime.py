"""Application lifetime and adapter failure behavior without network access."""

import asyncio
import threading
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from engram import cli, operations, server
from engram.core.config import Settings, get_settings
from engram.core.models import Fact, FactCategory, MemoryCandidate
from engram.core.structured_outputs import UnresolvedDecision
from engram.extraction.observer import IngestResult
from engram.llm.client import Completion, ModelCompletion
from engram.runtime import Runtime
from engram.storage.store import FactStore
from tests.mcp_helpers import structured_content, text_of


class DisabledProvider:
    available = False

    def __init__(self):
        self.closed = False

    async def complete(self, *args: Any, **kwargs: Any) -> Completion:
        raise AssertionError("a disabled provider must never be called")

    async def complete_model(self, *args: Any, **kwargs: Any) -> ModelCompletion[Any]:
        raise AssertionError("a disabled provider must never be called")

    async def aclose(self):
        self.closed = True


def settings(data_dir: Path, **overrides) -> Settings:
    return Settings(data_dir=data_dir, maintenance_enabled=False, sync_enabled=False, **overrides)


def test_runtime_constructs_one_store_off_event_loop(tmp_path, monkeypatch):
    provider = DisabledProvider()
    main_thread = threading.get_ident()
    construction_threads = []

    def construct(data_dir):
        construction_threads.append(threading.get_ident())
        return FactStore(data_dir)

    monkeypatch.setattr("engram.runtime.FactStore", construct)

    async def exercise():
        runtime = Runtime(settings=settings(tmp_path / "data"), provider=provider)
        with runtime.bind():
            stores = await asyncio.gather(*(runtime.store() for _ in range(8)))
            assert all(store is stores[0] for store in stores)
        await runtime.aclose()

    asyncio.run(exercise())
    assert len(construction_threads) == 1 and construction_threads[0] != main_thread
    assert provider.closed


def test_runtime_keeps_unused_dependencies_lazy(tmp_path):
    provider = DisabledProvider()
    runtime = Runtime(settings=settings(tmp_path / "unused"), provider=provider)

    async def exercise():
        await runtime.start_background()
        await runtime.aclose()

    asyncio.run(exercise())
    assert not runtime.settings.data_dir.exists()
    assert provider.closed


def test_provider_closes_when_final_sync_fails(tmp_path, monkeypatch):
    provider = DisabledProvider()
    runtime = Runtime(
        settings=Settings(data_dir=tmp_path, maintenance_enabled=False, sync_enabled=True),
        provider=provider,
    )

    async def fail_sync(*args, **kwargs):
        raise OSError("sync failed")

    monkeypatch.setattr("engram.storage.sync.run_final_sync", fail_sync)

    async def exercise():
        await runtime.start_background()
        await runtime.aclose()

    with pytest.raises(OSError, match="sync failed"):
        asyncio.run(exercise())
    assert provider.closed


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("remember", {"content": "A durable convention"}),
        ("forget", {"fact_id": "missing"}),
        ("purge", {}),
        ("approve_candidates", {"candidate_ids": ["missing"]}),
        ("import_memories", {}),
    ],
)
def test_mcp_invalid_format_never_opens_storage(tmp_path, tool, arguments):
    runtime = Runtime(settings=settings(tmp_path / "data"), provider=DisabledProvider())
    app = server.create_mcp(runtime=runtime)
    result = asyncio.run(app.call_tool(tool, {**arguments, "format": "yaml"}))
    assert result.is_error
    assert structured_content(result)["errors"][0]["code"] == "validation_error"
    assert not runtime.settings.data_dir.exists()
    asyncio.run(runtime.aclose())


def test_readonly_doctor_does_not_create_directory(tmp_path):
    runtime = Runtime(settings=settings(tmp_path / "missing"), provider=DisabledProvider())

    async def exercise():
        with runtime.bind():
            result = await operations.doctor()
        await runtime.aclose()
        return result

    result = asyncio.run(exercise())
    assert result.exit_code == operations.EXIT_DOCTOR_ERROR
    assert result.envelope.status.value == "ok"
    assert not result.is_error
    assert not runtime.settings.data_dir.exists()


@pytest.mark.parametrize(
    "field",
    [
        "max_facts_per_agent",
        "retrieval_timeout",
        "ephemeral_ttl_days",
        "maintenance_interval",
        "maintenance_concurrency",
        "sync_interval",
        "sync_timeout",
    ],
)
def test_invalid_runtime_settings_fail_before_storage(tmp_path, field):
    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path / "missing", **{field: 0})
    assert not (tmp_path / "missing").exists()


def test_invalid_settings_cli_json_has_no_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ENGRAM_DATA_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("ENGRAM_MAINTENANCE_INTERVAL", "0")
    get_settings.cache_clear()
    try:
        assert cli.main_dispatch(["inspect", "--json"]) == operations.EXIT_VALIDATION
        output = capsys.readouterr()
        assert '"validation_error"' in output.out
        assert "Traceback" not in output.err
        assert not (tmp_path / "missing").exists()
    finally:
        get_settings.cache_clear()


def test_ingest_unresolved_is_partial_in_mcp(tmp_path, monkeypatch):
    store = FactStore(tmp_path)
    created = Fact(category=FactCategory.project, content="New information")

    async def fake_ingest(*args, **kwargs):
        return IngestResult(
            created=[created],
            unresolved=[UnresolvedDecision(fact_id="old", action="replace", reason="changed")],
        )

    monkeypatch.setattr(operations, "ingest", fake_ingest)
    app = server.create_mcp(store)
    result = asyncio.run(
        app.call_tool("remember", {"content": "New information", "format": "json"})
    )
    assert result.is_error
    assert structured_content(result)["status"] == "partial"
    assert structured_content(result)["data"]["unresolved"][0]["fact_id"] == "old"


@pytest.mark.parametrize("operation", [operations.approve_candidates, operations.reject_candidates])
def test_candidate_subset_is_visible_as_partial(tmp_path, operation):
    store = FactStore(tmp_path)
    store.append_candidates(
        [MemoryCandidate(id="valid", category=FactCategory.project, content="Useful knowledge")]
    )
    result = asyncio.run(operation(["valid", "missing"], store=store))
    assert result.envelope.status.value == "partial"
    assert result.exit_code == operations.EXIT_RUNTIME
    assert result.envelope.data["applied_ids"] == ["valid"]
    assert result.envelope.data["unresolved_ids"] == ["missing"]


def test_candidate_edit_missing_id_aborts_before_editing(tmp_path):
    store = FactStore(tmp_path)
    store.append_candidates(
        [MemoryCandidate(id="valid", category=FactCategory.project, content="Original")]
    )
    result = asyncio.run(
        operations.approve_candidates(
            ["valid", "missing"],
            edits={"valid": "Edited", "missing": "Missing"},
            store=store,
        )
    )
    assert result.exit_code == operations.EXIT_NOT_FOUND
    assert store.load_candidates()[0].content == "Original"
    assert not store.load_facts()


def test_recall_text_keeps_requested_structured_provenance(tmp_path):
    store = FactStore(tmp_path)
    store.append_facts(
        [Fact(category=FactCategory.project, content="Zagblort xylophone ownership")]
    )
    runtime = Runtime(settings=settings(tmp_path), store=store, provider=DisabledProvider())
    app = server.create_mcp(runtime=runtime)
    result = asyncio.run(
        app.call_tool(
            "recall",
            {"query": "Zagblort xylophone", "format": "text", "with_provenance": True},
        )
    )
    assert not text_of(result.content[0]).startswith("{")
    assert structured_content(result)["data"]["provenance"]["delivered_fact_ids"]
    assert structured_content(result)["meta"]["returned"] == 1
    asyncio.run(runtime.aclose())


@pytest.mark.parametrize(
    "exception,code",
    [
        (OSError("disk"), "storage_error"),
        (RuntimeError("bug"), "internal_error"),
        (ValueError("input"), "validation_error"),
        (TimeoutError("late"), "timeout"),
    ],
)
def test_exception_classification_uses_actual_source(exception, code):
    result = operations.exception_result(exception)
    assert result.envelope.errors[0].code.value == code
    assert result.is_error


def test_cli_human_failures_go_to_stderr(capsys):
    code = cli.run(["recall", "query", "--max-sources", "0"])
    output = capsys.readouterr()
    assert code == operations.EXIT_VALIDATION
    assert not output.out
    assert "max_sources" in output.err


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["inspect", "--json"], 0),
        (["doctor", "--json"], 4),
        (["recall", "query", "--max-sources", "0", "--json"], 1),
        (["sync", "--json"], 0),
    ],
)
def test_cli_does_not_own_background_final_sync(tmp_path, monkeypatch, capsys, arguments, expected):
    monkeypatch.setenv("ENGRAM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ENGRAM_SYNC_ENABLED", "true")
    final_calls = []

    async def final_sync(*args, **kwargs):
        final_calls.append(True)

    monkeypatch.setattr("engram.storage.sync.run_final_sync", final_sync)
    monkeypatch.setattr(
        operations,
        "_sync",
        lambda *args, **kwargs: {"status": "skipped", "reason": "offline fixture"},
    )
    get_settings.cache_clear()
    try:
        assert cli.run(arguments) == expected
        assert not final_calls
        assert '"status"' in capsys.readouterr().out
    finally:
        get_settings.cache_clear()


def test_upkeep_skipped_runs_keep_nonzero_recurring_wait(tmp_path, monkeypatch):
    from datetime import datetime

    from engram.maintenance import upkeep
    from engram.maintenance.upkeep_report import StepReport, UpkeepReport, UpkeepStep

    waits = []
    opened = []
    store = operations.async_store(FactStore(tmp_path))

    async def dependency():
        opened.append(len(waits))
        return store

    async def sleep(delay):
        waits.append(delay)
        if len(waits) == 3:
            raise asyncio.CancelledError

    async def skipped(*args, **kwargs):
        return UpkeepReport(
            started_at=datetime.now(UTC),
            steps=[StepReport(step=UpkeepStep.projects, skipped="another run owns the lock")],
        )

    monkeypatch.setattr(upkeep.asyncio, "sleep", sleep)
    monkeypatch.setattr(upkeep, "run_upkeep", skipped)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(upkeep.upkeep_loop(dependency, data_dir=tmp_path, interval=3600))
    assert waits == [120, 120, 120]
    assert opened == [1, 2]


def test_corrupt_recall_log_is_storage_failure_for_stats(tmp_path, monkeypatch, capsys):
    from engram.core.models import RecallRecord
    from engram.storage.store import StoreIntegrityError

    store = FactStore(tmp_path)
    store.log_recall(RecallRecord(query="valid", tier=0, prefilter_count=1, latency_ms=1))
    with store.recall_log_path.open("a") as stream:
        stream.write("broken JSON\n")
    original = store.recall_log_path.read_bytes()
    with pytest.raises(StoreIntegrityError, match=r"recall_log\.jsonl:2"):
        store.load_recall_log()
    assert len(FactStore.open_readonly(tmp_path).load_recall_log()) == 1
    monkeypatch.setenv("ENGRAM_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        assert cli.run(["recall-stats", "--json"]) == operations.EXIT_RUNTIME
        assert '"storage_error"' in capsys.readouterr().out
        runtime = Runtime(settings=settings(tmp_path), store=store, provider=DisabledProvider())
        result = asyncio.run(
            server.create_mcp(runtime=runtime).call_tool("recall_stats", {"format": "json"})
        )
        assert result.is_error
        assert structured_content(result)["errors"][0]["code"] == "storage_error"
        asyncio.run(runtime.aclose())
    finally:
        get_settings.cache_clear()
    assert store.recall_log_path.read_bytes() == original
