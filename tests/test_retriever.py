"""Tests for card recall, answer mode, zero-hit selection, and recall logging."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from engram.core.interfaces import WarningCode
from engram.core.models import Fact, FactCategory
from engram.llm import Completion
from engram.recall.retriever import (
    ZERO_HIT_MAX_CANDIDATES,
    RecallSelection,
    _extract_quality,
    _relevant_hits,
    _scrub_invalid_citations,
    recall,
    recall_with_provenance,
)
from engram.storage.search import SearchHit
from engram.storage.store import AsyncFactStore, FactStore


def _make_store() -> FactStore:
    return FactStore(data_dir=Path(tempfile.mkdtemp()))


def _hit(fact_id: str, score: float, coverage: float) -> SearchHit:
    fact = Fact(id=fact_id, category=FactCategory.preference, content=fact_id)
    return SearchHit(fact=fact, score=score, coverage=coverage)


def _xylophone_store() -> FactStore:
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="aaaaaaaaaaa1",
                category=FactCategory.personal_info,
                content="Zagblort works on the xylophone repair department",
                tags=["zagblort", "xylophone"],
                project="acme",
            ),
            Fact(
                id="aaaaaaaaaaa2",
                category=FactCategory.preference,
                content="Completely unrelated content about banana bread",
            ),
        ]
    )
    return store


def _patch_complete(monkeypatch, responses):
    """Queue-backed stand-in for ``complete_with_usage``; records calls."""
    calls: dict = {"n": 0, "prompts": [], "kwargs": []}
    queue = list(responses)

    async def fake(prompt, system="", **kwargs):
        calls["n"] += 1
        calls["prompts"].append(prompt)
        calls["kwargs"].append(kwargs)
        text, input_tokens, cached = queue.pop(0)
        return Completion(text=text, input_tokens=input_tokens, cached_tokens=cached)

    monkeypatch.setattr("engram.recall.retriever.complete_with_usage", fake)
    return calls


def _patch_select(monkeypatch, result: list[str] | Exception):
    """Stand-in for the structured selection call via ``complete_model``."""
    calls: dict = {"n": 0, "prompts": [], "kwargs": []}

    async def fake(prompt, system, response_model, **kwargs):
        calls["n"] += 1
        calls["prompts"].append(prompt)
        calls["kwargs"].append(kwargs)
        assert response_model is RecallSelection
        if isinstance(result, Exception):
            raise result
        return RecallSelection(relevant_ids=result)

    monkeypatch.setattr("engram.recall.retriever.complete_model", fake)
    return calls


# ---------------------------------------------------------------------------
# Relevance bar
# ---------------------------------------------------------------------------


def test_relevant_hits_require_coverage_and_relative_score():
    hits = [
        _hit("top", 10.0, 0.8),
        _hit("close", 6.0, 0.5),
        _hit("low-coverage", 9.0, 0.1),
        _hit("low-score", 2.0, 0.9),
    ]
    assert [hit.fact.id for hit in _relevant_hits(hits)] == ["top", "close"]


def test_relevant_hits_empty_when_top_hit_lacks_coverage():
    assert _relevant_hits([_hit("weak", 10.0, 0.1)]) == []
    assert _relevant_hits([]) == []


# ---------------------------------------------------------------------------
# Cards mode (default, no LLM)
# ---------------------------------------------------------------------------


def test_cards_mode_returns_dated_cards_without_llm(monkeypatch):
    monkeypatch.setattr("engram.recall.retriever._llm_available", lambda: True)
    calls = _patch_complete(monkeypatch, [])
    store = _xylophone_store()

    text, quality, provenance, _ = asyncio.run(
        recall_with_provenance("zagblort xylophone", project="acme", store=store)
    )

    assert calls["n"] == 0
    lines = text.splitlines()
    assert lines[0] == "Memory (1 of 1 matches, project acme):"
    assert lines[1].startswith("- [personal_info · acme · ")
    assert lines[1].endswith("(id: aaaaaaaaaaa1)")
    assert quality == "high"
    assert provenance.tier == 0
    assert provenance.usage.llm_calls == 0
    assert provenance.cited_fact_ids == ["aaaaaaaaaaa1"]
    assert provenance.source_fact_ids == ["aaaaaaaaaaa1"]
    match = provenance.prefilter_matches[0]
    assert isinstance(match.score, float) and match.coverage > 0.6


def test_cards_capped_at_max_sources():
    store = _make_store()
    store.append_facts(
        [
            Fact(category=FactCategory.preference, content=f"zagblort note {i}")
            for i in range(6)
        ]
    )

    text, _, provenance, _ = asyncio.run(
        recall_with_provenance("zagblort note", store=store, max_sources=3)
    )

    assert text.splitlines()[0] == "Memory (3 of 6 matches):"
    assert len(provenance.cited_fact_ids) == 3
    assert len(provenance.sources) == 3
    assert all(source.cited for source in provenance.sources)


def test_no_hits_returns_no_relevant_memories():
    store = _xylophone_store()
    text, quality, provenance, _ = asyncio.run(
        recall_with_provenance("quantum chromodynamics", store=store)
    )
    assert text == "No relevant memories found for this query."
    assert quality == "none"
    assert provenance.prefilter_count == 0


def test_recall_returns_text_and_logs_record():
    store = _xylophone_store()

    text = asyncio.run(recall("zagblort xylophone", store=store))

    assert "Zagblort" in text
    [record] = store.load_recall_log()
    assert record.tier == 0
    assert record.mode == "cards"
    assert record.delivered_ids == ["aaaaaaaaaaa1"]
    assert record.selector_version == "v4"
    assert record.quality == "high"
    assert record.llm_calls == 0


def test_cards_mode_with_async_store():
    store = _xylophone_store()
    text = asyncio.run(recall("zagblort xylophone", store=AsyncFactStore(store)))
    assert "aaaaaaaaaaa1" in text


# ---------------------------------------------------------------------------
# Zero-hit selection (cards mode, weak hits only)
# ---------------------------------------------------------------------------

# Matches the fixture facts only through the weak word "telemetry" amid many
# unmatched terms, so every hit falls below the coverage floor.
_ZERO_HIT_QUERY = "which analytics platform keeps our event telemetry history?"


def _weak_hit_store(count: int = 1) -> FactStore:
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id=f"{i:012x}",
                category=FactCategory.preference,
                content=f"Snowflake warehouse stores raw telemetry, note {i}",
            )
            for i in range(count)
        ]
    )
    return store


def test_zero_hit_selection_returns_llm_chosen_cards(monkeypatch):
    monkeypatch.setattr("engram.recall.retriever._llm_available", lambda: True)
    calls = _patch_select(monkeypatch, ["000000000000", "ffffffffffff"])
    store = _weak_hit_store()

    text, _quality, provenance, trace = asyncio.run(
        recall_with_provenance(_ZERO_HIT_QUERY, store=store, with_trace=True)
    )

    assert calls["n"] == 1
    assert calls["kwargs"][0]["reasoning_effort"] == "low"
    # The invented ID is dropped; the real candidate becomes a card.
    assert provenance.cited_fact_ids == ["000000000000"]
    assert text.startswith("Memory (1 of 1 matches):")
    assert provenance.tier == 1
    assert provenance.selected_decision.zero_hit_escalation is True
    assert provenance.selected_decision.relevant_count == 0
    assert trace is not None and [c.name for c in trace.calls] == ["select"]
    [record] = store.load_recall_log()
    assert record.llm_calls == 1
    assert record.delivered_ids == ["000000000000"]


def test_zero_hit_selection_bounds_candidates(monkeypatch):
    monkeypatch.setattr("engram.recall.retriever._llm_available", lambda: True)
    calls = _patch_select(monkeypatch, [])
    store = _weak_hit_store(count=ZERO_HIT_MAX_CANDIDATES + 10)

    text, quality, _, _ = asyncio.run(
        recall_with_provenance(_ZERO_HIT_QUERY, store=store)
    )

    assert calls["n"] == 1
    assert calls["prompts"][0].count("(id: ") == ZERO_HIT_MAX_CANDIDATES
    assert text == "No relevant memories found for this query."
    assert quality == "none"


def test_zero_hit_without_key_makes_no_call(monkeypatch):
    calls = _patch_select(monkeypatch, ["000000000000"])
    store = _weak_hit_store()

    text, _, provenance, _ = asyncio.run(
        recall_with_provenance(_ZERO_HIT_QUERY, store=store)
    )

    assert calls["n"] == 0
    assert text == "No relevant memories found for this query."
    assert provenance.selected_decision.zero_hit_escalation is False
    assert provenance.prefilter_count == 1


def test_zero_hit_selection_failure_degrades_with_warning(monkeypatch):
    monkeypatch.setattr("engram.recall.retriever._llm_available", lambda: True)
    _patch_select(monkeypatch, TimeoutError())
    store = _weak_hit_store()

    text, quality, provenance, _ = asyncio.run(
        recall_with_provenance(_ZERO_HIT_QUERY, store=store)
    )

    assert text == "No relevant memories found for this query."
    assert quality == "none"
    assert provenance.tier == 0
    codes = [w.code for w in provenance.warnings]
    assert codes == [WarningCode.provider_unavailable]


def test_strong_hit_never_calls_llm_with_key(monkeypatch):
    monkeypatch.setattr("engram.recall.retriever._llm_available", lambda: True)
    select = _patch_select(monkeypatch, [])
    complete = _patch_complete(monkeypatch, [])

    asyncio.run(recall("zagblort xylophone", store=_xylophone_store()))

    assert select["n"] == 0
    assert complete["n"] == 0


# ---------------------------------------------------------------------------
# Answer mode
# ---------------------------------------------------------------------------


def test_answer_mode_makes_one_call_over_relevant_facts(monkeypatch):
    monkeypatch.setattr("engram.recall.retriever._llm_available", lambda: True)
    calls = _patch_complete(
        monkeypatch,
        [("Zagblort fixes xylophones (id: aaaaaaaaaaa1).\n[quality: high]", 300, 100)],
    )
    store = _xylophone_store()

    text, quality, provenance, trace = asyncio.run(
        recall_with_provenance(
            "zagblort xylophone", store=store, mode="answer", with_trace=True
        )
    )

    assert calls["n"] == 1
    assert calls["kwargs"][0]["reasoning_effort"] == "low"
    # Only relevant facts reach the prompt, with dates and IDs.
    assert "aaaaaaaaaaa1" in calls["prompts"][0]
    assert "banana" not in calls["prompts"][0]
    assert text == "Zagblort fixes xylophones (id: aaaaaaaaaaa1)."
    assert quality == "high"
    assert provenance.tier == 1
    assert provenance.cited_fact_ids == ["aaaaaaaaaaa1"]
    assert provenance.usage.input_tokens == 300
    assert provenance.usage.cache_hit_ratio == pytest.approx(1 / 3)
    assert trace is not None and [c.name for c in trace.calls] == ["answer"]
    [record] = store.load_recall_log()
    assert record.mode == "answer"
    assert record.delivered_ids == ["aaaaaaaaaaa1"]


def test_answer_mode_zero_hit_uses_top_candidates(monkeypatch):
    monkeypatch.setattr("engram.recall.retriever._llm_available", lambda: True)
    calls = _patch_complete(
        monkeypatch, [("Nothing relevant.\n[quality: none]", 50, 0)]
    )
    store = _weak_hit_store(count=3)

    _, quality, provenance, _ = asyncio.run(
        recall_with_provenance(_ZERO_HIT_QUERY, store=store, mode="answer")
    )

    assert calls["n"] == 1
    assert calls["prompts"][0].count("(id: ") == 3
    assert quality == "none"
    assert provenance.selected_decision.zero_hit_escalation is True


def test_answer_mode_without_key_returns_cards_with_warning(monkeypatch):
    calls = _patch_complete(monkeypatch, [])
    store = _xylophone_store()

    text, _, provenance, _ = asyncio.run(
        recall_with_provenance("zagblort xylophone", store=store, mode="answer")
    )

    assert calls["n"] == 0
    assert text.startswith("Memory (1 of 1 matches):")
    assert [w.code for w in provenance.warnings] == [WarningCode.provider_unavailable]


def test_answer_mode_provider_error_returns_cards_with_warning(monkeypatch):
    monkeypatch.setattr("engram.recall.retriever._llm_available", lambda: True)

    async def boom(prompt, system="", **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr("engram.recall.retriever.complete_with_usage", boom)

    text, _, provenance, _ = asyncio.run(
        recall_with_provenance(
            "zagblort xylophone", store=_xylophone_store(), mode="answer"
        )
    )

    assert text.startswith("Memory (1 of 1 matches):")
    assert provenance.cited_fact_ids == ["aaaaaaaaaaa1"]
    assert [w.code for w in provenance.warnings] == [WarningCode.provider_unavailable]


# ---------------------------------------------------------------------------
# Quality extraction
# ---------------------------------------------------------------------------


def test_extract_quality():
    clean, level = _extract_quality("The answer is yes.\n\n[quality: high]")
    assert level == "high"
    assert "[quality:" not in clean


def test_extract_quality_missing():
    text = "Just a plain answer."
    assert _extract_quality(text) == (text, "")


# ---------------------------------------------------------------------------
# Citation scrubbing — invented/mangled IDs never reach the answer
# ---------------------------------------------------------------------------

_VALID_IDS = {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}


def test_scrub_keeps_valid_citations():
    text = "Use stg_* models [aaaaaaaaaaaa] and int__* [bbbbbbbbbbbb]."
    assert _scrub_invalid_citations(text, _VALID_IDS) == text


def test_scrub_removes_wrong_length_id():
    # 15-hex run: a real ID with extra digits merged in by the model.
    text = "Rename carefully ([aaaaaaaaaaaa], [179c6a107298f24])."
    assert (
        _scrub_invalid_citations(text, _VALID_IDS)
        == "Rename carefully ([aaaaaaaaaaaa])."
    )


def test_scrub_removes_unknown_id_from_facts_group():
    text = "Looker shuts down 2026-08-31 [facts: aaaaaaaaaaaa, cccccccccccc]."
    assert (
        _scrub_invalid_citations(text, _VALID_IDS)
        == "Looker shuts down 2026-08-31 [facts: aaaaaaaaaaaa]."
    )


def test_scrub_drops_fully_invalid_citation_group():
    text = "Prefer natural keys [facts: cccccccccccc, dddddddddddd]."
    assert _scrub_invalid_citations(text, _VALID_IDS) == "Prefer natural keys."


def test_scrub_leaves_plain_text_untouched():
    text = "No citations here, just a decade of prose."
    assert _scrub_invalid_citations(text, _VALID_IDS) is text
