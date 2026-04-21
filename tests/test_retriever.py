"""Tests for tiered retrieval and quality logging."""

import asyncio
import tempfile
from pathlib import Path

from engram.llm import Completion
from engram.models import Fact, FactCategory
from engram.retriever import (
    _build_prefix,
    _extract_quality,
    _format_direct,
    _parse_multilens_sections,
    _resolve_pipeline,
    _resolve_tier_rules,
    _select_tier,
    _unknown_pipeline_warned,
    _unknown_tier_rules_warned,
)
from engram.store import FactStore


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
    """≤5 matches, all ≥10, clear gap → Tier 0."""
    scored = [
        (20, Fact(category=FactCategory.personal_info, content="Adam's team")),
        (8, Fact(category=FactCategory.preference, content="unrelated")),
    ]
    # Only 1 above RELEVANCE_FLOOR=5, gap is inf → Tier 0
    assert _select_tier(scored) == 0


def test_select_tier_0_rejected_when_four_plus():
    """4+ relevant facts → not Tier 0."""
    scored = [
        (15, Fact(category=FactCategory.preference, content=f"fact {i}"))
        for i in range(4)
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
    # ≤5 relevant, top ≥10, gap=2.0 ≥2.0 → Tier 0
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
    assert "No memories" in result


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

    from engram.retriever import recall

    asyncio.run(recall("zagblort xylophone", store=store))

    records = store.load_recall_log()
    assert len(records) == 1
    assert records[0].tier == 0
    assert records[0].latency_ms > 0
    assert records[0].quality == "high"


def test_resolve_tier_rules_valid():
    assert _resolve_tier_rules("v1") == "v1"
    assert _resolve_tier_rules("v2") == "v2"


def test_resolve_tier_rules_unknown_falls_back_with_single_warning(caplog):
    _unknown_tier_rules_warned.clear()
    with caplog.at_level("WARNING", logger="engram.retriever"):
        assert _resolve_tier_rules("v99") == "v2"
    assert any("v99" in rec.message for rec in caplog.records)
    caplog.clear()
    with caplog.at_level("WARNING", logger="engram.retriever"):
        assert _resolve_tier_rules("v99") == "v2"
    assert not caplog.records


def test_select_tier_v2_caps_small_corpus_to_tier_1():
    """v2 demotes would-be-tier-2 queries with < threshold positive scores."""
    # Flat distribution over 8 facts → would be tier 2 under v1.
    scored = [
        (12, Fact(category=FactCategory.preference, content=f"fact {i}"))
        for i in range(8)
    ]
    assert _select_tier(scored, rules="v1") == 2
    assert _select_tier(scored, rules="v2", min_prefilter_for_tier2=11) == 1


def test_select_tier_v2_preserves_tier_0():
    """The cap never promotes tier-0 decisions to tier-1."""
    scored = [
        (15, Fact(category=FactCategory.personal_info, content="only fact")),
    ]
    assert _select_tier(scored, rules="v2", min_prefilter_for_tier2=11) == 0


def test_select_tier_v2_large_corpus_unchanged():
    """Above the threshold, v2 returns the same tier as v1."""
    scored = [
        (12, Fact(category=FactCategory.preference, content=f"fact {i}"))
        for i in range(20)
    ]
    v1 = _select_tier(scored, rules="v1")
    v2 = _select_tier(scored, rules="v2", min_prefilter_for_tier2=11)
    assert v1 == v2 == 2


def test_select_tier_v1_matches_legacy_behaviour():
    """v1 branch must return the same tier as the pre-v2 selector on a suite
    of representative score distributions."""
    cases = [
        [],
        [(3, Fact(category=FactCategory.preference, content="noise"))],
        [(20, Fact(category=FactCategory.personal_info, content="one")),
         (8, Fact(category=FactCategory.preference, content="two"))],
        [(25, Fact(category=FactCategory.preference, content="winner"))]
        + [(8, Fact(category=FactCategory.preference, content=f"n{i}")) for i in range(20)],
        [(12, Fact(category=FactCategory.preference, content=f"f{i}")) for i in range(20)],
    ]
    # v1 should always match itself regardless of min_prefilter_for_tier2.
    for case in cases:
        assert _select_tier(case, rules="v1", min_prefilter_for_tier2=99) == _select_tier(case)


def test_select_tier_threshold_zero_disables_cap():
    """min_prefilter_for_tier2=0 disables the cap even under v2."""
    scored = [
        (12, Fact(category=FactCategory.preference, content=f"fact {i}"))
        for i in range(5)
    ]
    assert _select_tier(scored, rules="v2", min_prefilter_for_tier2=0) == 2


def test_recall_stamps_selector_version(monkeypatch):
    """Default recall stamps v2 on the log; legacy flag stamps v1."""
    from engram.config import get_settings

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

    from engram.retriever import recall

    get_settings.cache_clear()
    asyncio.run(recall("zagblort xylophone", store=store))
    records = store.load_recall_log()
    assert records[0].selector_version == "v2"

    monkeypatch.setenv("ENGRAM_TIER_RULES", "v1")
    get_settings.cache_clear()
    try:
        asyncio.run(recall("zagblort xylophone", store=store))
    finally:
        monkeypatch.delenv("ENGRAM_TIER_RULES", raising=False)
        get_settings.cache_clear()
    records = store.load_recall_log()
    # Most recent first
    assert records[0].selector_version == "v1"


def _flat_tier2_facts(store: FactStore, count: int = 15) -> None:
    """Seed a store with facts that produce a flat, tier-2 score distribution.

    Facts all share one token with the query ("retrieval") so every one scores
    exactly +5 from unigram overlap — no clear top cluster → tier selector
    returns 2. No category/project/tags/recency boost-stacking.

    Count stays above `ENGRAM_TIER2_MIN_PREFILTER_COUNT` (default 11) so the
    small-corpus cap doesn't demote these fixtures to tier-1 under v2.
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
    in call order. A counter is attached so tests can assert call count.
    """
    calls = {"n": 0}
    queue = list(responses)

    async def fake(prompt, system="", model=None, temperature=None, response_format=None, cache_prefix=None):
        calls["n"] += 1
        text, input_tokens, cached = queue.pop(0)
        return Completion(text=text, input_tokens=input_tokens, cached_tokens=cached)

    monkeypatch.setattr("engram.retriever.complete_with_usage", fake)
    return calls


def test_build_prefix_is_deterministic():
    assert _build_prefix("ABC") == "STORED FACTS:\nABC\n\n"


def test_parse_multilens_all_sections():
    text = """## DIRECT
1. foo [id:a1]

## CONTEXTUAL
1. bar [id:b2]

## TEMPORAL
1. baz [id:c3]
"""
    parsed = _parse_multilens_sections(text)
    assert "foo" in parsed["direct"]
    assert "bar" in parsed["contextual"]
    assert "baz" in parsed["temporal"]


def test_parse_multilens_missing_section():
    text = "## DIRECT\n1. only this\n"
    parsed = _parse_multilens_sections(text)
    assert "only this" in parsed["direct"]
    assert parsed["contextual"] == ""
    assert parsed["temporal"] == ""


def test_parse_multilens_malformed_treated_as_direct():
    text = "no headings at all, just prose"
    parsed = _parse_multilens_sections(text)
    assert parsed["direct"] == text
    assert parsed["contextual"] == ""


def test_resolve_pipeline_valid():
    assert _resolve_pipeline("multilens") == "multilens"
    assert _resolve_pipeline("legacy") == "legacy"


def test_resolve_pipeline_unknown_falls_back_with_warning(caplog):
    _unknown_pipeline_warned.clear()
    with caplog.at_level("WARNING", logger="engram.retriever"):
        assert _resolve_pipeline("bogus") == "multilens"
    assert any("bogus" in rec.message for rec in caplog.records)
    # Second call doesn't re-warn.
    caplog.clear()
    with caplog.at_level("WARNING", logger="engram.retriever"):
        assert _resolve_pipeline("bogus") == "multilens"
    assert not caplog.records


def test_recall_tier2_multilens_makes_two_llm_calls(monkeypatch):
    store = _make_store()
    _flat_tier2_facts(store)

    multilens_response = (
        "## DIRECT\n1. f00 is about retrieval (id:f00)\n\n"
        "## CONTEXTUAL\n(none)\n\n"
        "## TEMPORAL\n(none)\n"
    )
    synthesis_response = "The retrieval notes are logged.\n[quality: high]"
    calls = _patch_complete(
        monkeypatch,
        [
            (multilens_response, 1000, 0),
            (synthesis_response, 1200, 900),
        ],
    )

    from engram.retriever import recall

    asyncio.run(recall("retrieval", store=store))

    assert calls["n"] == 2
    records = store.load_recall_log()
    assert len(records) == 1
    assert records[0].tier == 2
    assert records[0].llm_calls == 2
    assert records[0].input_tokens == 2200
    assert records[0].cached_tokens == 900
    assert records[0].quality == "high"


def test_recall_tier2_legacy_makes_four_llm_calls(monkeypatch):
    from engram.config import get_settings

    monkeypatch.setenv("ENGRAM_RECALL_PIPELINE", "legacy")
    get_settings.cache_clear()

    store = _make_store()
    _flat_tier2_facts(store)

    calls = _patch_complete(
        monkeypatch,
        [
            ("agent A output", 500, 0),
            ("agent B output", 500, 0),
            ("agent C output", 500, 0),
            ("final synthesis\n[quality: medium]", 800, 0),
        ],
    )

    from engram.retriever import recall

    try:
        asyncio.run(recall("retrieval", store=store))
    finally:
        monkeypatch.delenv("ENGRAM_RECALL_PIPELINE", raising=False)
        get_settings.cache_clear()

    assert calls["n"] == 4
    records = store.load_recall_log()
    assert records[0].tier == 2
    assert records[0].llm_calls == 4
    assert records[0].quality == "medium"


def test_recall_tier2_flag_fallback_on_unknown_value(monkeypatch, caplog):
    from engram.config import get_settings

    monkeypatch.setenv("ENGRAM_RECALL_PIPELINE", "bogus")
    get_settings.cache_clear()
    _unknown_pipeline_warned.clear()

    store = _make_store()
    _flat_tier2_facts(store)

    calls = _patch_complete(
        monkeypatch,
        [
            ("## DIRECT\n1. ok (id:f00)\n## CONTEXTUAL\n(none)\n## TEMPORAL\n(none)\n", 100, 0),
            ("done\n[quality: low]", 100, 0),
        ],
    )

    from engram.retriever import recall

    try:
        with caplog.at_level("WARNING", logger="engram.retriever"):
            asyncio.run(recall("retrieval", store=store))
    finally:
        monkeypatch.delenv("ENGRAM_RECALL_PIPELINE", raising=False)
        get_settings.cache_clear()

    assert any("bogus" in rec.message for rec in caplog.records)
    assert calls["n"] == 2  # fell back to multilens


def test_recall_tier2_multilens_uses_stable_prefix(monkeypatch):
    """Both calls in tier-2 multilens must pass the same cache_prefix."""
    store = _make_store()
    _flat_tier2_facts(store)

    captured: list[dict] = []

    async def fake(prompt, system="", model=None, temperature=None, response_format=None, cache_prefix=None):
        captured.append({"prompt": prompt, "cache_prefix": cache_prefix, "system": system})
        if len(captured) == 1:
            return Completion(
                text="## DIRECT\n1. x (id:f00)\n## CONTEXTUAL\n(none)\n## TEMPORAL\n(none)\n",
                input_tokens=100,
                cached_tokens=0,
            )
        return Completion(text="done\n[quality: high]", input_tokens=100, cached_tokens=50)

    monkeypatch.setattr("engram.retriever.complete_with_usage", fake)

    from engram.retriever import recall

    asyncio.run(recall("retrieval", store=store))

    assert len(captured) == 2
    assert captured[0]["cache_prefix"] is not None
    assert captured[0]["cache_prefix"] == captured[1]["cache_prefix"]
    # The cache_prefix must actually be a prefix of each call's prompt.
    assert captured[0]["prompt"].startswith(captured[0]["cache_prefix"])
    assert captured[1]["prompt"].startswith(captured[1]["cache_prefix"])


def test_recall_tier0_weak_match_logs_low_quality():
    """Tier 0 recall with only sub-threshold matches logs quality='low'."""
    store = _make_store()
    # No keyword overlap with the query, but recency (+2) and confidence (+1) boosts
    # produce a score of 1-3 (> 0 but < RELEVANCE_FLOOR=5) → quality="low", not "high" or "none"
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

    from engram.retriever import recall

    asyncio.run(recall("zagblort xylophone", store=store))

    records = store.load_recall_log()
    assert len(records) == 1
    assert records[0].tier == 0
    assert records[0].quality == "low"
