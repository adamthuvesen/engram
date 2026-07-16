"""Tests for tiered retrieval and quality logging."""

import asyncio
import tempfile
from pathlib import Path

from engram.llm import Completion
from engram.core.models import Fact, FactCategory
from engram.recall.retriever import (
    _extract_quality,
    _format_direct,
    _scrub_invalid_citations,
    _select_tier,
)
from engram.storage.store import AsyncFactStore, FactStore


def _make_store() -> FactStore:
    tmp = Path(tempfile.mkdtemp())
    return FactStore(data_dir=tmp)


# ---------------------------------------------------------------------------
# Tier selection — gap-based heuristic
# ---------------------------------------------------------------------------


def test_select_tier_0_empty():
    """No scored facts → Tier 0."""
    assert _select_tier([]) == 0


def test_select_tier_0_no_relevant():
    """All scores below relevance floor → Tier 0 (nothing to work with)."""
    scored = [(3, Fact(category=FactCategory.preference, content="noise"))]
    assert _select_tier(scored) == 0


def test_select_tier_0_few_strong_with_gap():
    """Few strong matches with a clear top result → Tier 0."""
    scored = [
        (20, Fact(category=FactCategory.personal_info, content="Alex's team")),
        (8, Fact(category=FactCategory.preference, content="unrelated")),
    ]
    # Only 1 above RELEVANCE_FLOOR=5, gap is inf → Tier 0
    assert _select_tier(scored) == 0


def test_select_tier_0_rejected_when_five_plus():
    """5+ relevant facts → not Tier 0."""
    scored = [
        (15, Fact(category=FactCategory.preference, content=f"fact {i}"))
        for i in range(5)
    ]
    assert _select_tier(scored) != 0


def test_select_tier_1_high_score_strong_gap():
    """High top score (≥15) with strong gap (≥1.5) → Tier 1."""
    scored = [
        (25, Fact(category=FactCategory.preference, content="clear winner")),
        *[
            (8, Fact(category=FactCategory.preference, content=f"noise {i}"))
            for i in range(20)
        ],
    ]
    # top=25, 5th=8, gap=3.1 → Tier 1
    assert _select_tier(scored) == 1


def test_select_tier_2_flat_distribution():
    """All scores equal (gap=1.0) → Tier 2."""
    scored = [
        (12, Fact(category=FactCategory.preference, content=f"fact {i}"))
        for i in range(20)
    ]
    # gap = 12/12 = 1.0 → needs multi-perspective
    assert _select_tier(scored) == 2


def test_select_tier_2_weak_gap():
    """Gap below threshold → Tier 2."""
    scored = [
        (15, Fact(category=FactCategory.preference, content="top")),
        (14, Fact(category=FactCategory.preference, content="close")),
        (13, Fact(category=FactCategory.preference, content="closer")),
        (12, Fact(category=FactCategory.preference, content="near")),
        (11, Fact(category=FactCategory.preference, content="fifth")),
    ]
    # gap = 15/11 = 1.36 → below 1.5 → Tier 2
    assert _select_tier(scored) == 2


def test_select_tier_2_many_results():
    """Large result set with flat scores → definitely Tier 2."""
    scored = [
        (8, Fact(category=FactCategory.preference, content=f"fact {i}"))
        for i in range(50)
    ]
    assert _select_tier(scored) == 2


def test_select_tier_0_few_strong_matches():
    """2 matches with top=10 and gap=2.0 → Tier 0 (trivial lookup)."""
    scored = [
        (10, Fact(category=FactCategory.preference, content="strong match")),
        (5, Fact(category=FactCategory.preference, content="weaker")),
    ]
    # ≤4 relevant and top ≥10 → Tier 0
    assert _select_tier(scored) == 0


# ---------------------------------------------------------------------------
# Direct formatting (Tier 0 output)
# ---------------------------------------------------------------------------


def test_format_direct_returns_facts():
    scored = [
        (10, Fact(id="abc", category=FactCategory.preference, content="Likes Python")),
    ]
    result = _format_direct(scored, "python")
    assert "abc" in result
    assert "Likes Python" in result


def test_format_direct_empty():
    result = _format_direct(
        [(0, Fact(category=FactCategory.preference, content="x"))], "q"
    )
    assert "No relevant memories" in result


def test_format_direct_store_empty():
    result = _format_direct([], "q")
    assert "No memories stored yet" in result


# ---------------------------------------------------------------------------
# Quality extraction
# ---------------------------------------------------------------------------


def test_extract_quality():
    text = "The answer is yes.\n\n[quality: high]"
    clean, level = _extract_quality(text)
    assert level == "high"
    assert "[quality:" not in clean


def test_extract_quality_missing():
    text = "Just a plain answer."
    clean, level = _extract_quality(text)
    assert level == ""
    assert clean == text


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


# ---------------------------------------------------------------------------
# Integration: recall logs to store
# ---------------------------------------------------------------------------


def test_recall_tier0_logs_to_store():
    """Tier 0 recall (no LLM) logs a RecallRecord."""
    store = _make_store()
    # Single fact with unique keywords → only strong match → Tier 0
    store.append_facts(
        [
            Fact(
                id="f1",
                category=FactCategory.personal_info,
                content="Zagblort works on the xylophone repair department",
                tags=["zagblort", "xylophone"],
            ),
        ]
    )

    from engram.recall.retriever import recall

    asyncio.run(recall("zagblort xylophone", store=store))

    records = store.load_recall_log()
    assert len(records) == 1
    assert records[0].tier == 0
    assert records[0].latency_ms > 0
    assert records[0].quality == "high"


def test_select_tier_caps_small_corpus_to_tier_1():
    """The small-corpus cap demotes would-be-tier-2 queries below threshold."""
    # Flat distribution over 8 facts would be tier 2 without the cap.
    scored = [
        (12, Fact(category=FactCategory.preference, content=f"fact {i}"))
        for i in range(8)
    ]
    assert _select_tier(scored, min_prefilter_for_tier2=0) == 2
    assert _select_tier(scored, min_prefilter_for_tier2=11) == 1


def test_select_tier_cap_preserves_tier_0():
    """The cap never promotes tier-0 decisions to tier-1."""
    scored = [
        (15, Fact(category=FactCategory.personal_info, content="only fact")),
    ]
    assert _select_tier(scored, min_prefilter_for_tier2=11) == 0


def test_select_tier_large_corpus_unchanged_by_cap():
    """Above the threshold, the cap does not change tier selection."""
    scored = [
        (12, Fact(category=FactCategory.preference, content=f"fact {i}"))
        for i in range(20)
    ]
    uncapped = _select_tier(scored, min_prefilter_for_tier2=0)
    capped = _select_tier(scored, min_prefilter_for_tier2=11)
    assert uncapped == capped == 2


def test_select_tier_threshold_zero_disables_cap():
    """min_prefilter_for_tier2=0 disables the small-corpus cap."""
    scored = [
        (12, Fact(category=FactCategory.preference, content=f"fact {i}"))
        for i in range(5)
    ]
    assert _select_tier(scored, min_prefilter_for_tier2=0) == 2


def test_recall_stamps_selector_version():
    """Recall logs stamp selector_version='v3' (v2 rules + zero-hit escalation)."""
    from engram.core.config import get_settings

    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="f1",
                category=FactCategory.personal_info,
                content="Zagblort works on the xylophone repair department",
                tags=["zagblort", "xylophone"],
            ),
        ]
    )

    from engram.recall.retriever import recall

    get_settings.cache_clear()
    asyncio.run(recall("zagblort xylophone", store=store))
    records = store.load_recall_log()
    assert records[0].selector_version == "v3"


def _flat_tier2_facts(store: FactStore, count: int = 15) -> None:
    """Seed a store with facts that produce a flat, tier-2 score distribution.

    Facts all share one token with the query ("retrieval") so every one scores
    exactly +5 from unigram overlap — no clear top cluster → tier selector
    returns 2. No category/project/tags/recency boost-stacking.

    Count stays above `ENGRAM_TIER2_MIN_PREFILTER_COUNT` (default 11) so the
    small-corpus cap doesn't demote these fixtures to tier-1.
    """
    store.append_facts(
        [
            Fact(
                id=f"f{i:02d}",
                category=FactCategory.preference,
                content=f"retrieval note number {i}",
                tags=[],
            )
            for i in range(count)
        ]
    )


def _patch_complete(monkeypatch, responses):
    """Replace `complete_with_usage` with a queue-backed stub.

    `responses` is a list of (text, input_tokens, cached_tokens) tuples consumed
    in call order. A counter and the received prompts are recorded so tests can
    assert call count and prompt contents.
    """
    calls = {"n": 0, "prompts": []}
    queue = list(responses)

    async def fake(
        prompt,
        system="",
        model=None,
        temperature=None,
        response_format=None,
        cache_prefix=None,
    ):
        calls["n"] += 1
        calls["prompts"].append(prompt)
        text, input_tokens, cached = queue.pop(0)
        return Completion(text=text, input_tokens=input_tokens, cached_tokens=cached)

    monkeypatch.setattr("engram.recall.retriever.complete_with_usage", fake)
    return calls


def test_recall_tier2_with_async_store_makes_one_llm_call(monkeypatch):
    store = _make_store()
    _flat_tier2_facts(store)
    async_store = AsyncFactStore(store)

    calls = _patch_complete(
        monkeypatch,
        [("done (id:f00)\n[quality: low]", 100, 0)],
    )

    from engram.recall.retriever import recall

    asyncio.run(recall("retrieval", store=async_store))

    assert calls["n"] == 1
    records = store.load_recall_log()
    assert records[0].tier == 2
    assert records[0].llm_calls == 1
    assert records[0].quality == "low"


def test_recall_tier2_default_single_mode_makes_one_llm_call(monkeypatch):
    from engram.recall.retriever import recall_with_provenance

    store = _make_store()
    _flat_tier2_facts(store)

    calls = _patch_complete(
        monkeypatch,
        [("Retrieval notes are available (id:f00).\n[quality: high]", 500, 0)],
    )

    _answer, quality, provenance, trace = asyncio.run(
        recall_with_provenance("retrieval", store=store, with_trace=True)
    )

    assert calls["n"] == 1
    assert quality == "high"
    assert provenance.tier == 2
    assert provenance.usage.llm_calls == 1
    assert trace is not None
    assert [call.name for call in trace.calls] == ["tier2_single"]


# ---------------------------------------------------------------------------
# Zero-hit escalation — paraphrased queries reach the LLM tier
# ---------------------------------------------------------------------------

# Shares zero unigrams/bigrams/tags with the fixture facts below, so the
# prefilter scores everything 0 and only the escalated LLM tier can answer.
_ZERO_HIT_QUERY = "which analytics platform keeps our event data?"


def _zero_overlap_store(count: int = 1) -> FactStore:
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id=f"pad{i:09d}",
                category=FactCategory.preference,
                content=f"Snowflake warehouse stores raw telemetry, note {i}",
                tags=[],
            )
            for i in range(count)
        ]
    )
    return store


def test_zero_hit_query_escalates_to_llm_tier(monkeypatch):
    """No fact above the floor + non-empty corpus + key → tier-1 LLM search."""
    monkeypatch.setattr("engram.recall.retriever._llm_available", lambda: True)
    store = _zero_overlap_store()
    calls = _patch_complete(
        monkeypatch,
        [("Snowflake stores the telemetry (id: pad000000000).\n[quality: low]", 80, 0)],
    )

    from engram.recall.retriever import recall_with_provenance

    _answer, _quality, provenance, _ = asyncio.run(
        recall_with_provenance(_ZERO_HIT_QUERY, store=store)
    )

    assert calls["n"] == 1
    assert provenance.tier == 1
    assert provenance.selected_decision.zero_hit_escalation is True
    assert provenance.selected_decision.relevant_count == 0
    records = store.load_recall_log()
    assert records[0].tier == 1
    assert records[0].llm_calls == 1


def test_zero_hit_query_without_key_keeps_tier0(monkeypatch):
    """Without a configured LLM key, zero-hit recall degrades to today's answer."""
    monkeypatch.setattr("engram.recall.retriever._llm_available", lambda: False)
    store = _zero_overlap_store()
    calls = _patch_complete(monkeypatch, [])

    from engram.recall.retriever import recall_with_provenance

    answer, _quality, provenance, _ = asyncio.run(
        recall_with_provenance(_ZERO_HIT_QUERY, store=store)
    )

    assert calls["n"] == 0
    assert provenance.tier == 0
    assert provenance.selected_decision.zero_hit_escalation is False
    assert "No relevant memories" in answer


def test_zero_hit_empty_corpus_never_escalates(monkeypatch):
    """An empty store stays tier 0 even with an LLM key configured."""
    monkeypatch.setattr("engram.recall.retriever._llm_available", lambda: True)
    store = _make_store()
    calls = _patch_complete(monkeypatch, [])

    from engram.recall.retriever import recall

    answer = asyncio.run(recall(_ZERO_HIT_QUERY, store=store))

    assert calls["n"] == 0
    assert "No memories stored yet" in answer


def test_strong_hit_fast_path_survives_with_key(monkeypatch):
    """Strong deterministic hits still resolve at tier 0 with zero LLM calls."""
    monkeypatch.setattr("engram.recall.retriever._llm_available", lambda: True)
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="f1",
                category=FactCategory.personal_info,
                content="Zagblort works on the xylophone repair department",
                tags=["zagblort", "xylophone"],
            ),
        ]
    )
    calls = _patch_complete(monkeypatch, [])

    from engram.recall.retriever import recall

    answer = asyncio.run(recall("zagblort xylophone", store=store))

    assert calls["n"] == 0
    assert "Zagblort" in answer
    records = store.load_recall_log()
    assert records[0].tier == 0


def test_zero_hit_escalation_bounds_candidates(monkeypatch):
    """The escalated call sends at most ZERO_HIT_MAX_CANDIDATES facts."""
    from engram.recall.retriever import ZERO_HIT_MAX_CANDIDATES

    monkeypatch.setattr("engram.recall.retriever._llm_available", lambda: True)
    store = _zero_overlap_store(count=ZERO_HIT_MAX_CANDIDATES + 30)
    calls = _patch_complete(monkeypatch, [("nothing relevant\n[quality: none]", 40, 0)])

    from engram.recall.retriever import recall

    asyncio.run(recall(_ZERO_HIT_QUERY, store=store))

    assert calls["n"] == 1
    assert calls["prompts"][0].count("(id: ") <= ZERO_HIT_MAX_CANDIDATES


def test_recall_tier0_unrelated_boost_only_fact_logs_no_quality():
    """Tier 0 recall does not treat recency/confidence boosts as relevance."""
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="f_weak",
                category=FactCategory.preference,
                content="Completely unrelated content about xyzzy",
                tags=[],
            ),
        ]
    )

    from engram.recall.retriever import recall

    answer = asyncio.run(recall("zagblort xylophone", store=store))

    records = store.load_recall_log()
    assert "No relevant memories" in answer
    assert len(records) == 1
    assert records[0].tier == 0
    assert records[0].prefilter_count == 0
    assert records[0].quality == "none"
