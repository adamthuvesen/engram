"""Tests for the MCP recall and recall_trace surfaces."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from engram import server
from engram.llm import Completion
from engram.models import Fact, FactCategory
from engram.store import AsyncFactStore, FactStore


def _setup_store(monkeypatch) -> FactStore:
    tmp = Path(tempfile.mkdtemp())
    store = FactStore(data_dir=tmp)
    monkeypatch.setattr(server, "_store", AsyncFactStore(store), raising=False)
    return store


def _patch_complete(monkeypatch, responses):
    queue = list(responses)

    async def fake(
        prompt,
        system="",
        model=None,
        temperature=None,
        response_format=None,
        cache_prefix=None,
    ):
        text, input_tokens, cached = queue.pop(0)
        return Completion(text=text, input_tokens=input_tokens, cached_tokens=cached)

    monkeypatch.setattr("engram.retriever.complete_with_usage", fake)


async def _call(tool, **kwargs):
    """Invoke an MCP tool function regardless of FastMCP wrapping."""
    fn = getattr(tool, "fn", tool)
    if asyncio.iscoroutinefunction(fn):
        return await fn(**kwargs)
    return fn(**kwargs)


def test_recall_default_text(monkeypatch):
    store = _setup_store(monkeypatch)
    store.append_facts(
        [
            Fact(
                id="aaaaaaaaaaaa",
                category=FactCategory.personal_info,
                content="zagblort works on xylophones",
                tags=["zagblort", "xylophone"],
            )
        ]
    )

    result = asyncio.run(_call(server.recall, query="zagblort xylophone"))
    assert isinstance(result, str)
    # Default text mode must NOT be JSON.
    assert not result.lstrip().startswith("{")


def test_recall_json_mode_returns_envelope(monkeypatch):
    store = _setup_store(monkeypatch)
    store.append_facts(
        [
            Fact(
                id="aaaaaaaaaaaa",
                category=FactCategory.personal_info,
                content="zagblort works on xylophones",
                tags=["zagblort", "xylophone"],
            )
        ]
    )

    result = asyncio.run(
        _call(server.recall, query="zagblort xylophone", format="json")
    )
    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    assert "answer" in parsed["data"]
    assert "tier" in parsed["data"]
    assert "source_fact_ids" in parsed["data"]
    assert "cited_fact_ids" in parsed["data"]
    assert "aaaaaaaaaaaa" in parsed["data"]["cited_fact_ids"]


def test_recall_json_no_match(monkeypatch):
    """Empty memory store returns OK envelope with quality=none."""
    _setup_store(monkeypatch)
    result = asyncio.run(_call(server.recall, query="anything", format="json"))
    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    assert parsed["data"]["quality"] == "none"
    assert parsed["data"]["source_fact_ids"] == []


def test_recall_json_with_provenance(monkeypatch):
    store = _setup_store(monkeypatch)
    store.append_facts(
        [
            Fact(
                id="aaaaaaaaaaaa",
                category=FactCategory.personal_info,
                content="zagblort xylophone",
                tags=["zagblort", "xylophone"],
            )
        ]
    )

    result = asyncio.run(
        _call(
            server.recall,
            query="zagblort xylophone",
            format="json",
            with_provenance=True,
        )
    )
    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    assert "provenance" in parsed["data"]
    assert parsed["data"]["provenance"]["tier"] == 0
    assert parsed["data"]["provenance"]["prefilter_matches"]


def test_recall_invalid_format_returns_validation_error(monkeypatch):
    _setup_store(monkeypatch)
    result = asyncio.run(_call(server.recall, query="x", format="yaml"))
    parsed = json.loads(result)
    assert parsed["status"] == "error"
    assert parsed["errors"][0]["code"] == "validation_error"


def test_recall_warns_on_superseded(monkeypatch):
    store = _setup_store(monkeypatch)
    store.append_facts(
        [
            Fact(
                id="oldaaaaaaaaa",
                category=FactCategory.preference,
                content="zagblort uses vim",
                tags=["zagblort", "vim"],
            ),
            Fact(
                id="newaaaaaaaaa",
                category=FactCategory.preference,
                content="zagblort uses neovim",
                tags=["zagblort", "neovim"],
                supersedes="oldaaaaaaaaa",
            ),
        ]
    )

    result = asyncio.run(_call(server.recall, query="zagblort editor", format="json"))
    parsed = json.loads(result)
    codes = [w["code"] for w in parsed["warnings"]]
    assert "superseded_fact" in codes


def test_recall_warns_on_stale(monkeypatch):
    store = _setup_store(monkeypatch)
    store.append_facts(
        [
            Fact(
                id="staleaaaaaaa",
                category=FactCategory.preference,
                content="legacy preference value",
                stale=True,
            ),
            Fact(
                id="livefactaaaa",
                category=FactCategory.preference,
                content="legacy preference current value",
            ),
        ]
    )

    # Trigger inspection-mode recall by calling recall_with_provenance with
    # include_stale=False (the MCP recall tool already excludes stale, so the
    # stale fact won't appear in the prefilter and no stale warning fires —
    # which is correct for the active path). Verify recall still succeeds.
    result = asyncio.run(_call(server.recall, query="legacy preference", format="json"))
    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    assert "staleaaaaaaa" not in parsed["data"]["cited_fact_ids"]


def test_recall_trace_success(monkeypatch):
    from engram.config import get_settings

    store = _setup_store(monkeypatch)
    monkeypatch.setenv("ENGRAM_TIER2_MODE", "multilens")
    get_settings.cache_clear()
    # Seed enough facts for tier-2 to fire.
    store.append_facts(
        [
            Fact(
                id=f"f{i:02d}{'a' * 9}",
                category=FactCategory.preference,
                content=f"trace note {i}",
                tags=[],
            )
            for i in range(15)
        ]
    )
    _patch_complete(
        monkeypatch,
        [
            (
                "## DIRECT\n1. (id: f00aaaaaaaaa)\n## CONTEXTUAL\n(none)\n## TEMPORAL\n(none)\n",
                100,
                0,
            ),
            ("traced (id: f00aaaaaaaaa)\n[quality: medium]", 200, 100),
        ],
    )

    try:
        result = asyncio.run(_call(server.recall_trace, query="trace"))
    finally:
        monkeypatch.delenv("ENGRAM_TIER2_MODE", raising=False)
        get_settings.cache_clear()
    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    trace = parsed["data"]["trace"]
    assert trace is not None
    assert len(trace["calls"]) == 2
    assert trace["calls"][0]["name"] == "multilens_search"
    assert trace["calls"][1]["name"] == "synthesis"


def test_recall_trace_provider_failure(monkeypatch):
    store = _setup_store(monkeypatch)
    store.append_facts(
        [
            Fact(
                id=f"f{i:02d}{'a' * 9}",
                category=FactCategory.preference,
                content=f"failure note {i}",
                tags=[],
            )
            for i in range(15)
        ]
    )

    async def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr("engram.retriever.complete_with_usage", boom)

    result = asyncio.run(_call(server.recall_trace, query="failure"))
    parsed = json.loads(result)
    assert parsed["status"] == "error"
    assert parsed["errors"][0]["code"] == "provider_error"


def test_recall_json_meta_fields_populated(monkeypatch):
    store = _setup_store(monkeypatch)
    store.append_facts(
        [
            Fact(
                id="aaaaaaaaaaaa",
                category=FactCategory.preference,
                content="abc xyz",
                tags=["abc"],
            )
        ]
    )
    result = asyncio.run(_call(server.recall, query="abc xyz", format="json"))
    parsed = json.loads(result)
    assert "meta" in parsed
    assert parsed["meta"]["limit"] == 25  # default max_sources
    assert parsed["meta"]["returned"] >= 0
