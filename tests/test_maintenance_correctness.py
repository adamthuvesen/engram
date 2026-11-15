"""Behavioral checks for unavailable evidence and derived brief validity."""

import asyncio
import subprocess
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from engram.core.models import Fact, FactCategory
from engram.extraction.importer import _parse_memory_file, import_claude_code_memories
from engram.maintenance import briefs, verify
from engram.maintenance.doctor import inspect_data_dir
from engram.maintenance.upkeep_report import StepReport, UpkeepStep
from engram.storage.async_store import AsyncFactStore
from engram.storage.store import FactStore

SOME_TIME = datetime(2026, 6, 11, 12, 0, tzinfo=UTC)


def fact(index: int, **fields) -> Fact:
    return Fact(
        id=f"source{index}",
        category=FactCategory.project,
        content=f"Card {index}",
        project="example",
        **fields,
    )


def refresh(store: AsyncFactStore) -> StepReport:
    report = StepReport(step=UpkeepStep.briefs)
    asyncio.run(
        briefs.run_briefs(
            store,
            project=None,
            report=report,
            dry_run=False,
            now=SOME_TIME,
        )
    )
    return report


@pytest.fixture
def brief_store(tmp_path, monkeypatch):
    async def provider(*args, **kwargs):
        return briefs.BriefResponse(content="Current orientation", retrieval_hints=["overview"])

    monkeypatch.setattr(briefs, "complete_model", provider)
    store = AsyncFactStore(FactStore(tmp_path / "data"))
    store.sync_store.append_facts([fact(i) for i in range(6)])
    refresh(store)
    return store


@pytest.mark.parametrize("mutation", ["forget", "stale", "move", "expiry", "purge"])
def test_brief_rejects_changed_source_membership(brief_store, mutation):
    store = brief_store
    assert asyncio.run(briefs.project_brief(store, "example")) is not None
    if mutation == "forget":
        store.sync_store.forget("source0")
    elif mutation == "stale":
        store.sync_store.mark_stale("source0", "invalid")
    elif mutation == "move":
        store.sync_store.update_fact("source0", project="elsewhere")
    elif mutation == "expiry":
        store.sync_store.update_fact("source0", expires_at=SOME_TIME - timedelta(seconds=1))
    else:
        store.sync_store.forget("source0")
        store.sync_store.purge()
    assert asyncio.run(briefs.project_brief(store, "example")) is None
    report = refresh(store)
    assert len(report.fact_ids["created"]) == 1
    assert asyncio.run(briefs.project_brief(store, "example")) is not None


def test_brief_retires_below_threshold_and_legacy_falls_back(brief_store):
    store = brief_store
    for index in (0, 1):
        store.sync_store.forget(f"source{index}")
    assert asyncio.run(briefs.project_brief(store, "example")) is None
    report = refresh(store)
    assert len(report.fact_ids["staled"]) == 1
    store.sync_store.append_facts([fact(9, memory_key="project-brief")])
    assert asyncio.run(briefs.project_brief(store, "example")) is None


@pytest.mark.parametrize("command", ["for-each-ref", "ls-tree"])
def test_git_failure_cannot_retire_missing_anchor(tmp_path, monkeypatch, command):
    def git(root, *args):
        if args[0] == command:
            return subprocess.CompletedProcess(args, 128, "", "failed")
        output = "tree-id\n" if args[0] == "for-each-ref" else ""
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(verify, "_run_git", git)
    verdict = verify.check_fact(fact(0, anchors=["src/gone.py"]), verify.RepoCheck(tmp_path))
    assert verdict is not None and verdict.unavailable == ["src/gone.py"]
    report = StepReport(step=UpkeepStep.verify)
    assert not verify.plan_verification([verdict], report).changes.stale


def test_unknown_anchor_prevents_retirement_from_known_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(verify.RepoCheck, "status", lambda *args: "missing")
    verdict = verify.check_fact(
        fact(0, anchors=["src/gone.py", "../outside.py"]), verify.RepoCheck(tmp_path)
    )
    assert verdict is not None and verdict.unavailable == ["../outside.py"]
    report = StepReport(step=UpkeepStep.verify)
    assert not verify.plan_verification([verdict], report).changes.stale


def test_anchor_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("sensitive")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "escape.py").symlink_to(outside)
    assert verify.normalize_anchor("../outside.py") is None
    assert verify.RepoCheck(root).path_status("escape.py") == "unavailable"


@pytest.mark.parametrize("yaml", ["- value", "scalar", "42", "false"])
def test_import_rejects_nonmapping_frontmatter(tmp_path, yaml):
    path = tmp_path / "memory.md"
    path.write_text(f"---\n{yaml}\n---\nbody")
    with pytest.raises(ValueError, match="YAML mapping"):
        _parse_memory_file(path)


def test_import_returns_per_file_failure(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    memory = projects / "example" / "memory"
    memory.mkdir(parents=True)
    (memory / "invalid.md").write_text("---\n- list\n---\nbody")
    monkeypatch.setattr(
        "engram.extraction.importer.get_settings",
        lambda: SimpleNamespace(claude_projects_dir=projects),
    )
    result = asyncio.run(import_claude_code_memories(FactStore(tmp_path / "data")))
    assert result["imported_files"] == 0
    assert result["failed_files"] == 1
    assert result["details"][0]["status"] == "error"


def test_raw_doctor_never_creates_missing_data_directory(tmp_path):
    missing = tmp_path / "missing"
    report = inspect_data_dir(missing)
    assert report.status == "error"
    assert not missing.exists()


def test_brief_source_provenance_survives_replica(tmp_path, brief_store):
    replica = AsyncFactStore(FactStore(tmp_path / "replica"))
    replica.facts_path.write_bytes(brief_store.facts_path.read_bytes())
    assert asyncio.run(briefs.project_brief(replica, "example")) is not None
    replica.sync_store.update_fact("source0", content="Changed information")
    assert asyncio.run(briefs.project_brief(replica, "example")) is None


def test_brief_source_change_during_generation_is_conflict(brief_store, monkeypatch):
    store = brief_store
    store.sync_store.update_fact("source0", content="New information")

    async def provider(*args, **kwargs):
        store.sync_store.update_fact("source0", tags=["concurrent"])
        return briefs.BriefResponse(content="Outdated orientation", retrieval_hints=["overview"])

    monkeypatch.setattr(briefs, "complete_model", provider)
    report = refresh(store)
    assert report.conflicts == {"source0": "changed since it was read"}
    assert not report.fact_ids.get("created")
    assert asyncio.run(briefs.project_brief(store, "example")) is None


def test_blank_provider_brief_does_not_replace_existing(brief_store, monkeypatch):
    store = brief_store
    store.sync_store.update_fact("source0", content="New information")
    before = store.facts_path.read_bytes()

    async def provider(*args, **kwargs):
        return briefs.BriefResponse(content="   ", retrieval_hints=["overview"])

    monkeypatch.setattr(briefs, "complete_model", provider)
    report = refresh(store)
    assert report.errors
    assert not report.fact_ids
    assert store.facts_path.read_bytes() == before


def test_restored_fact_clears_verifier_suspect_atomically(tmp_path, monkeypatch):
    store = FactStore(tmp_path / "data")
    store.append_facts(
        [fact(0, anchors=["src/present.py"], suspect_reason="missing: src/present.py")]
    )
    store.mark_stale("source0", "anchors missing: src/present.py")
    stale = store.load_facts()[0]
    verdict = verify.AnchorVerdict(
        fact=stale, present=["src/present.py"], on_branch=[], missing=[], derived=False
    )
    report = StepReport(step=UpkeepStep.verify)
    plan = verify.plan_verification([verdict], report)
    assert plan.changes.unstale == [stale.id]
    assert plan.changes.edits[stale.id] == {"suspect_reason": ""}
    store.update_fact(stale.id, tags=["concurrent"])
    result = store.apply_changes(plan.changes, all_or_nothing=True)
    assert result.skipped
    assert store.load_facts()[0].stale
    assert store.load_facts()[0].suspect_reason
    current = store.load_facts()[0]
    verdict = verify.AnchorVerdict(
        fact=current,
        present=["src/present.py"],
        on_branch=[],
        missing=[],
        derived=False,
    )
    store.apply_changes(verify.plan_verification([verdict], report).changes, all_or_nothing=True)
    assert not store.load_facts()[0].stale
    assert store.load_facts()[0].suspect_reason == ""


def test_project_registry_serializes_process_updates(tmp_path):
    import multiprocessing
    from pathlib import Path

    from engram.core.projects import load_project_roots, record_project_root

    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=record_project_root,
            args=(tmp_path, f"project{i}", Path(f"/repo{i}")),
        )
        for i in range(12)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert load_project_roots(tmp_path) == {f"project{i}": Path(f"/repo{i}") for i in range(12)}


@pytest.mark.parametrize("writer", ["recall", "upkeep_state"])
def test_sync_store_lock_blocks_all_synced_log_writers(tmp_path, writer):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from engram.core.models import RecallRecord
    from engram.maintenance.upkeep_state import StateLine, append_state
    from engram.storage.transactions import locked_store

    store = FactStore(tmp_path)
    started = threading.Event()
    completed = threading.Event()

    def append():
        started.set()
        if writer == "recall":
            store.log_recall(RecallRecord(query="offline", tier=0, latency_ms=1, prefilter_count=0))
        else:
            append_state(tmp_path, [StateLine(kind="run", at=SOME_TIME)])
        completed.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with locked_store(tmp_path):
            future = pool.submit(append)
            assert started.wait(timeout=2)
            assert not completed.wait(timeout=0.1)
        future.result(timeout=2)
    assert completed.is_set()
