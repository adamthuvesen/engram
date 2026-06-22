"""Tests for memory correction, merge, and stale-marking workflows."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from engram import server
from engram.core.models import Fact, FactCategory
from engram.storage.store import AsyncFactStore, FactStore


def _make_store() -> FactStore:
    return FactStore(data_dir=Path(tempfile.mkdtemp()))


def _setup_server(monkeypatch) -> FactStore:
    store = _make_store()
    monkeypatch.setattr(
        server,
        "mcp",
        server.create_mcp(AsyncFactStore(store)),
        raising=False,
    )
    return store


async def _call(tool_name: str, **kwargs):
    return await server.mcp._call_tool_mcp(tool_name, kwargs)


def _structured(result):
    return result[1]


# ---------------------------------------------------------------------------
# correct_fact
# ---------------------------------------------------------------------------


def test_correct_fact_creates_replacement_with_supersedes():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="oldaaaaaaaaa",
                category=FactCategory.preference,
                content="prefers vim",
            )
        ]
    )

    new = store.correct_fact("oldaaaaaaaaa", "prefers neovim", reason="user updated")
    assert new is not None
    assert new.supersedes == "oldaaaaaaaaa"
    assert new.content == "prefers neovim"

    # Old fact is dropped from active recall via reduced confidence.
    active_ids = [f.id for f in store.load_active_facts()]
    assert "oldaaaaaaaaa" not in active_ids
    assert new.id in active_ids


def test_correct_fact_returns_none_for_missing():
    store = _make_store()
    assert store.correct_fact("doesnotexist", "anything") is None


def test_correct_fact_returns_none_for_forgotten():
    store = _make_store()
    fact = Fact(
        id="forgottenfac",
        category=FactCategory.preference,
        content="x",
        confidence=0.0,
    )
    store.append_facts([fact])
    assert store.correct_fact("forgottenfac", "y") is None


def test_correct_fact_append_failure_leaves_original_active(monkeypatch):
    """Failure inside ``append_events`` must leave the original fact active.

    In the event-log model ``correct_fact`` commits its supersedes + created
    events through ``append_events``; if that raises, no events land and the
    original fact remains the sole active record.
    """
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="oldaaaaaaaaa",
                category=FactCategory.preference,
                content="prefers vim",
            )
        ]
    )

    def fail_append(events):
        raise OSError("append failed")

    monkeypatch.setattr(store, "append_events", fail_append)

    with pytest.raises(OSError, match="append failed"):
        store.correct_fact("oldaaaaaaaaa", "prefers neovim")

    active = store.load_active_facts()
    assert len(active) == 1
    assert active[0].id == "oldaaaaaaaaa"
    assert active[0].content == "prefers vim"


def test_correct_fact_log_failure_keeps_single_active_replacement(monkeypatch):
    """If ``log_ingestion`` fails after events are appended, the state still
    reflects a single active replacement (new fact active, old superseded).
    """
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="oldaaaaaaaaa",
                category=FactCategory.preference,
                content="prefers vim",
            )
        ]
    )

    def fail_log(record):
        raise OSError("audit failed")

    monkeypatch.setattr(store, "log_ingestion", fail_log)

    with pytest.raises(OSError, match="audit failed"):
        store.correct_fact("oldaaaaaaaaa", "prefers neovim")

    active = store.load_active_facts()
    assert len(active) == 1
    assert active[0].id != "oldaaaaaaaaa"
    assert active[0].content == "prefers neovim"
    old = next(f for f in store.load_facts() if f.id == "oldaaaaaaaaa")
    # Superseded facts surface as confidence==0 in the materialized view.
    assert old.confidence == 0.0


# ---------------------------------------------------------------------------
# merge_facts
# ---------------------------------------------------------------------------


def test_merge_facts_consolidates_two():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="srcaaaaaaaaa",
                category=FactCategory.preference,
                content="uses vim daily",
            ),
            Fact(
                id="srcbbbbbbbbb",
                category=FactCategory.preference,
                content="uses neovim daily",
            ),
        ]
    )

    result = store.merge_facts(
        ["srcaaaaaaaaa", "srcbbbbbbbbb"],
        "uses neovim daily",
        reason="dedupe vim/neovim",
    )
    assert result is not None
    new_fact, superseded = result
    assert set(superseded) == {"srcaaaaaaaaa", "srcbbbbbbbbb"}

    active_ids = {f.id for f in store.load_active_facts()}
    assert new_fact.id in active_ids
    assert "srcaaaaaaaaa" not in active_ids
    assert "srcbbbbbbbbb" not in active_ids


def test_merge_rejects_single_source():
    store = _make_store()
    store.append_facts(
        [
            Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="a"),
        ]
    )
    assert store.merge_facts(["aaaaaaaaaaaa"], "merged") is None


def test_merge_rejects_duplicates_collapsed_below_two():
    store = _make_store()
    store.append_facts(
        [
            Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="a"),
        ]
    )
    # Same ID twice collapses to one — invalid merge.
    assert store.merge_facts(["aaaaaaaaaaaa", "aaaaaaaaaaaa"], "merged") is None


def test_merge_rejects_mixed_projects():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="aaaaaaaaaaaa",
                category=FactCategory.preference,
                content="a",
                project="proj1",
            ),
            Fact(
                id="bbbbbbbbbbbb",
                category=FactCategory.preference,
                content="b",
                project="proj2",
            ),
        ]
    )
    assert store.merge_facts(["aaaaaaaaaaaa", "bbbbbbbbbbbb"], "merged") is None


def test_merge_facts_append_failure_leaves_sources_active(monkeypatch):
    """Failure inside ``append_events`` must leave both source facts active."""
    store = _make_store()
    store.append_facts(
        [
            Fact(id="srcaaaaaaaaa", category=FactCategory.preference, content="a"),
            Fact(id="srcbbbbbbbbb", category=FactCategory.preference, content="b"),
        ]
    )

    def fail_append(events):
        raise OSError("append failed")

    monkeypatch.setattr(store, "append_events", fail_append)

    with pytest.raises(OSError, match="append failed"):
        store.merge_facts(["srcaaaaaaaaa", "srcbbbbbbbbb"], "merged")

    active_ids = {fact.id for fact in store.load_active_facts()}
    assert active_ids == {"srcaaaaaaaaa", "srcbbbbbbbbb"}


def test_merge_facts_log_failure_keeps_single_active_replacement(monkeypatch):
    """When ``log_ingestion`` fails after events are appended, the merged
    state still resolves to exactly one active record."""
    store = _make_store()
    store.append_facts(
        [
            Fact(id="srcaaaaaaaaa", category=FactCategory.preference, content="a"),
            Fact(id="srcbbbbbbbbb", category=FactCategory.preference, content="b"),
        ]
    )

    def fail_log(record):
        raise OSError("audit failed")

    monkeypatch.setattr(store, "log_ingestion", fail_log)

    with pytest.raises(OSError, match="audit failed"):
        store.merge_facts(["srcaaaaaaaaa", "srcbbbbbbbbb"], "merged")

    active = store.load_active_facts()
    assert len(active) == 1
    assert active[0].content == "merged"
    inactive = {fact.id: fact for fact in store.load_facts() if fact.id != active[0].id}
    # Superseded sources surface as confidence==0 in the materialized view.
    assert inactive["srcaaaaaaaaa"].confidence == 0.0
    assert inactive["srcbbbbbbbbb"].confidence == 0.0


# ---------------------------------------------------------------------------
# mark_stale
# ---------------------------------------------------------------------------


def test_mark_stale_excludes_from_active_recall():
    store = _make_store()
    store.append_facts(
        [Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="a")]
    )
    fact = store.mark_stale("aaaaaaaaaaaa", reason="outdated")
    assert fact is not None
    assert fact.stale is True

    active_ids = [f.id for f in store.load_active_facts()]
    assert "aaaaaaaaaaaa" not in active_ids

    # But the fact is still inspectable with include_stale=True.
    all_ids = [f.id for f in store.load_active_facts(include_stale=True)]
    assert "aaaaaaaaaaaa" in all_ids


def test_mark_stale_returns_none_for_missing():
    store = _make_store()
    assert store.mark_stale("missingmissi") is None


def test_unmark_stale_restores_recall_eligibility():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="aaaaaaaaaaaa",
                category=FactCategory.preference,
                content="a",
                stale=True,
            )
        ]
    )
    store.unmark_stale("aaaaaaaaaaaa")
    active_ids = [f.id for f in store.load_active_facts()]
    assert "aaaaaaaaaaaa" in active_ids


# ---------------------------------------------------------------------------
# MCP surfaces
# ---------------------------------------------------------------------------


def test_correct_memory_mcp_success(monkeypatch):
    store = _setup_server(monkeypatch)
    store.append_facts(
        [
            Fact(
                id="oldaaaaaaaaa",
                category=FactCategory.preference,
                content="vim",
            )
        ]
    )
    result = asyncio.run(
        _call(
            "correct_memory",
            fact_id="oldaaaaaaaaa",
            new_content="neovim",
            reason="updated",
        )
    )
    parsed = _structured(result)
    assert parsed["status"] == "ok"
    assert parsed["data"]["superseded_fact_id"] == "oldaaaaaaaaa"


def test_correct_memory_mcp_not_found(monkeypatch):
    _setup_server(monkeypatch)
    result = asyncio.run(
        _call("correct_memory", fact_id="missingmissi", new_content="x")
    )
    parsed = _structured(result)
    assert parsed["status"] == "error"
    assert parsed["errors"][0]["code"] == "not_found"


def test_merge_memories_mcp_success(monkeypatch):
    store = _setup_server(monkeypatch)
    store.append_facts(
        [
            Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="a"),
            Fact(id="bbbbbbbbbbbb", category=FactCategory.preference, content="b"),
        ]
    )
    result = asyncio.run(
        _call(
            "merge_memories",
            source_ids=["aaaaaaaaaaaa", "bbbbbbbbbbbb"],
            merged_content="merged",
        )
    )
    parsed = _structured(result)
    assert parsed["status"] == "ok"
    assert set(parsed["data"]["superseded_fact_ids"]) == {
        "aaaaaaaaaaaa",
        "bbbbbbbbbbbb",
    }


def test_merge_memories_mcp_validation_for_one_source(monkeypatch):
    _setup_server(monkeypatch)
    result = asyncio.run(
        _call("merge_memories", source_ids=["onlyoneonlyo"], merged_content="x")
    )
    parsed = _structured(result)
    assert parsed["status"] == "error"
    assert parsed["errors"][0]["code"] == "validation_error"


def test_mark_stale_mcp(monkeypatch):
    store = _setup_server(monkeypatch)
    store.append_facts(
        [Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="a")]
    )
    result = asyncio.run(_call("mark_stale", fact_id="aaaaaaaaaaaa", reason="old"))
    parsed = _structured(result)
    assert parsed["status"] == "ok"
    assert parsed["data"]["stale"] is True


def test_inspect_excludes_stale_by_default(monkeypatch):
    store = _setup_server(monkeypatch)
    store.append_facts(
        [
            Fact(
                id="staleaaaaaaa",
                category=FactCategory.preference,
                content="stale",
                stale=True,
            ),
            Fact(
                id="liveaaaaaaaa",
                category=FactCategory.preference,
                content="live",
            ),
        ]
    )
    result = asyncio.run(_call("inspect", format="json"))
    parsed = _structured(result)
    ids = [f["id"] for f in parsed["data"]]
    assert "liveaaaaaaaa" in ids
    assert "staleaaaaaaa" not in ids


def test_inspect_with_include_stale_shows_stale(monkeypatch):
    store = _setup_server(monkeypatch)
    store.append_facts(
        [
            Fact(
                id="staleaaaaaaa",
                category=FactCategory.preference,
                content="stale",
                stale=True,
                stale_reason="outdated",
            ),
        ]
    )
    result = asyncio.run(_call("inspect", format="json", include_stale=True))
    parsed = _structured(result)
    ids = [f["id"] for f in parsed["data"]]
    assert "staleaaaaaaa" in ids
    entry = next(f for f in parsed["data"] if f["id"] == "staleaaaaaaa")
    assert entry["stale"] is True
    assert entry["stale_reason"] == "outdated"
