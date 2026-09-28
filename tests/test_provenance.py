"""Tests for recall provenance and trace assembly."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from engram.core.interfaces import WarningCode
from engram.core.models import Fact, FactCategory
from engram.core.provenance import RecallTrace
from engram.recall.retriever import (
    _extract_cited_ids,
    recall,
    recall_with_provenance,
)
from engram.storage.store import FactStore


def _make_store() -> FactStore:
    tmp = Path(tempfile.mkdtemp())
    return FactStore(data_dir=tmp)


# ---------------------------------------------------------------------------
# default text recall
# ---------------------------------------------------------------------------


def test_default_recall_returns_string_only():
    """recall() returns a plain string unless callers request structured data."""
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="f1aaaaaaaaaa",
                category=FactCategory.personal_info,
                content="Alex works on Engram",
                tags=["alex"],
            )
        ]
    )

    answer = asyncio.run(recall("Alex Engram", store=store))
    assert isinstance(answer, str)
    assert answer  # non-empty


def test_default_recall_no_provenance_in_text():
    """The text answer must not start emitting JSON or warning structures."""
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="f1aaaaaaaaaa",
                category=FactCategory.personal_info,
                content="Alex works on Engram",
            )
        ]
    )

    answer = asyncio.run(recall("Alex Engram", store=store))
    # Default text response should not embed the structured envelope.
    assert "source_fact_ids" not in answer
    assert "selected_decision" not in answer


# ---------------------------------------------------------------------------
# Provenance content
# ---------------------------------------------------------------------------


def test_cards_provenance_has_no_llm_calls():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="f1aaaaaaaaaa",
                category=FactCategory.personal_info,
                content="Alex works on Engram zagblort xylophone",
                tags=["zagblort", "xylophone"],
            )
        ]
    )
    _, _, provenance, _ = asyncio.run(
        recall_with_provenance("zagblort xylophone Engram", store=store)
    )
    assert provenance.tier == 0
    assert provenance.usage.llm_calls == 0
    assert provenance.cited_fact_ids == ["f1aaaaaaaaaa"]
    assert provenance.selected_decision.rules == "v4"


def test_provenance_includes_scored_matches():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="f1aaaaaaaaaa",
                category=FactCategory.preference,
                content="prefers tabs over spaces",
                tags=["tabs"],
            ),
            Fact(
                id="f2aaaaaaaaaa",
                category=FactCategory.preference,
                content="unrelated content",
            ),
        ]
    )
    _, _, provenance, _ = asyncio.run(
        recall_with_provenance("tabs spaces", store=store)
    )
    assert [m.id for m in provenance.prefilter_matches] == ["f1aaaaaaaaaa"]
    top = provenance.prefilter_matches[0]
    assert top.above_floor is True
    assert top.coverage == 1.0
    assert provenance.selected_decision.top_score == top.score


def test_recall_does_not_reload_facts_once_index_is_warm():
    """Provenance is built from search hits, not a second full fact load."""
    store = _make_store()
    store.append_facts(
        [Fact(category=FactCategory.preference, content="prefers tabs over spaces")]
    )
    asyncio.run(recall_with_provenance("tabs", store=store))

    def fail_load():
        raise AssertionError("recall reloaded every fact")

    store.load_facts = fail_load  # type: ignore[method-assign]
    _, _, provenance, _ = asyncio.run(recall_with_provenance("tabs", store=store))
    assert provenance.cited_fact_ids


def test_superseded_fact_is_not_delivered_and_not_warned():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="oldaaaaaaaaa",
                category=FactCategory.preference,
                content="zagblort prefers vim",
                tags=["zagblort", "vim"],
            ),
            Fact(
                id="newaaaaaaaaa",
                category=FactCategory.preference,
                content="zagblort prefers neovim",
                tags=["zagblort", "neovim"],
                supersedes="oldaaaaaaaaa",
            ),
        ]
    )

    _, _, provenance, _ = asyncio.run(
        recall_with_provenance("zagblort prefers vim or neovim", store=store)
    )
    assert provenance.cited_fact_ids == ["newaaaaaaaaa"]
    assert provenance.warnings == []


def test_stale_facts_are_not_delivered():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="staleaaaaaaa",
                category=FactCategory.preference,
                content="prefers stale option",
                stale=True,
            ),
            Fact(
                id="liveaaaaaaaa",
                category=FactCategory.preference,
                content="prefers live option",
            ),
        ]
    )
    _, _, provenance, _ = asyncio.run(
        recall_with_provenance("prefers option", store=store)
    )
    assert "staleaaaaaaa" not in provenance.cited_fact_ids
    assert "staleaaaaaaa" not in [m.id for m in provenance.prefilter_matches]


def test_conflicting_facts_need_a_shared_memory_key():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="runner1aaaaa",
                category=FactCategory.convention,
                project="atlas",
                memory_key="test-runner",
                content="Atlas runs tests with pytest",
            ),
            Fact(
                id="runner2aaaaa",
                category=FactCategory.convention,
                project="atlas",
                memory_key="test-runner",
                content="Atlas runs tests with unittest",
            ),
            Fact(
                id="lintaaaaaaaa",
                category=FactCategory.convention,
                project="atlas",
                memory_key="linter",
                content="Atlas runs tests after ruff",
            ),
        ]
    )
    _, _, provenance, _ = asyncio.run(
        recall_with_provenance("atlas runs tests", project="atlas", store=store)
    )
    [warning] = provenance.warnings
    assert warning.code == WarningCode.conflicting_facts
    assert warning.ids == ["runner1aaaaa", "runner2aaaaa"]
    assert warning.details == {"project": "atlas", "memory_key": "test-runner"}


def test_suspect_fact_is_delivered_with_warning():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="suspectaaaaa",
                category=FactCategory.convention,
                content="The zagblort parser lives in src/zagblort/parse.py",
                suspect_reason="anchor src/zagblort/parse.py vanished",
            )
        ]
    )
    text, _, provenance, _ = asyncio.run(
        recall_with_provenance("zagblort parser", store=store)
    )
    assert "unverified: anchor src/zagblort/parse.py vanished" in text
    [warning] = provenance.warnings
    assert warning.code == WarningCode.suspect_fact
    assert warning.ids == ["suspectaaaaa"]


def test_trace_is_opt_in():
    store = _make_store()
    store.append_facts(
        [Fact(category=FactCategory.preference, content="prefers tabs over spaces")]
    )
    _, _, provenance, trace = asyncio.run(recall_with_provenance("tabs", store=store))
    assert trace is None
    _, _, provenance, trace = asyncio.run(
        recall_with_provenance("tabs", store=store, with_trace=True)
    )
    assert isinstance(trace, RecallTrace)
    assert trace.provenance == provenance
    assert trace.calls == []


# ---------------------------------------------------------------------------
# Cited-id extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "see (id: aaaaaaaaaaaa) and (id: bbbbbbbbbbbb)",
            ["aaaaaaaaaaaa", "bbbbbbbbbbbb"],
        ),
        ("nothing relevant", []),
        # Hallucinated IDs that aren't in candidate set are dropped.
        ("(id: ffffffffffff)", []),
    ],
)
def test_extract_cited_ids(text, expected):
    candidates = {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert _extract_cited_ids(text, candidates) == expected


def test_extract_cited_ids_dedupes_in_order():
    candidates = {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    text = "(id: aaaaaaaaaaaa) and again (id: aaaaaaaaaaaa) and (id: bbbbbbbbbbbb)"
    assert _extract_cited_ids(text, candidates) == ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]
