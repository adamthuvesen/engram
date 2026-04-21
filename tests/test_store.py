"""Tests for the JSONL fact store."""

import tempfile
from pathlib import Path

from engram.models import CandidateStatus, Fact, FactCategory, MemoryCandidate
from engram.store import FactStore, format_facts_for_llm


def _make_store() -> FactStore:
    """Create a store with a temp directory."""
    tmp = Path(tempfile.mkdtemp())
    return FactStore(data_dir=tmp)


def test_empty_store():
    store = _make_store()
    assert store.load_facts() == []
    assert store.load_active_facts() == []


def test_append_and_load():
    store = _make_store()
    facts = [
        Fact(
            category=FactCategory.preference, content="User prefers polars over pandas"
        ),
        Fact(category=FactCategory.personal_info, content="User works on AI team"),
    ]
    store.append_facts(facts)

    loaded = store.load_facts()
    assert len(loaded) == 2
    assert loaded[0].content == "User prefers polars over pandas"
    assert loaded[1].content == "User works on AI team"


def test_filter_by_category():
    store = _make_store()
    store.append_facts(
        [
            Fact(category=FactCategory.preference, content="Prefers polars"),
            Fact(category=FactCategory.personal_info, content="Works on AI team"),
            Fact(category=FactCategory.preference, content="Uses ruff for formatting"),
        ]
    )

    prefs = store.load_active_facts(category=FactCategory.preference)
    assert len(prefs) == 2
    assert all(f.category == FactCategory.preference for f in prefs)


def test_filter_by_project():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                category=FactCategory.project,
                content="Uses dbt",
                project="acme-dw",
            ),
            Fact(
                category=FactCategory.project,
                content="Uses FastAPI",
                project="ai-research",
            ),
            Fact(category=FactCategory.preference, content="Global pref"),
        ]
    )

    filtered = store.load_active_facts(project="acme-dw")
    assert len(filtered) == 2  # acme-dw + global (no project filter excludes)


def test_forget():
    store = _make_store()
    fact = Fact(category=FactCategory.preference, content="Old preference")
    store.append_facts([fact])

    result = store.forget(fact.id, reason="outdated")
    assert result is not None
    assert result.confidence == 0.0

    # Should not appear in active facts
    active = store.load_active_facts()
    assert len(active) == 0


def test_update_fact():
    store = _make_store()
    fact = Fact(category=FactCategory.preference, content="Original content")
    store.append_facts([fact])

    updated = store.update_fact(fact.id, content="Updated content")
    assert updated is not None
    assert updated.content == "Updated content"

    loaded = store.load_facts()
    assert loaded[0].content == "Updated content"


def test_stats():
    store = _make_store()
    store.append_facts(
        [
            Fact(category=FactCategory.preference, content="Pref 1"),
            Fact(category=FactCategory.personal_info, content="Info 1"),
            Fact(category=FactCategory.preference, content="Pref 2", confidence=0.0),
        ]
    )

    stats = store.stats()
    assert stats["total_facts"] == 3
    assert stats["active_facts"] == 2
    assert stats["forgotten_facts"] == 1
    assert stats["by_category"]["preference"] == 1
    assert stats["by_category"]["personal_info"] == 1


def test_candidate_approval_promotes_fact():
    store = _make_store()
    candidate = MemoryCandidate(
        id="cand123",
        category=FactCategory.preference,
        content="The user prefers concise summaries",
        why_store="This should shape future responses",
    )
    store.append_candidates([candidate])

    approved = store.approve_candidates(["cand123"])

    assert len(approved) == 1
    assert approved[0].content == "The user prefers concise summaries"

    candidates = store.load_candidates(status=CandidateStatus.approved)
    assert len(candidates) == 1
    assert candidates[0].status == CandidateStatus.approved

    facts = store.load_active_facts()
    assert len(facts) == 1
    assert facts[0].content == "The user prefers concise summaries"


def test_candidate_rejection_updates_status():
    store = _make_store()
    candidate = MemoryCandidate(
        id="cand456",
        category=FactCategory.workflow,
        content="The user might maybe sometimes prefer verbose output",
    )
    store.append_candidates([candidate])

    rejected = store.reject_candidates(["cand456"], reason="Too speculative")

    assert len(rejected) == 1
    assert rejected[0].status == CandidateStatus.rejected
    assert rejected[0].review_note == "Too speculative"
    assert store.load_active_facts() == []


def test_approving_superseding_candidate_softens_old_fact():
    store = _make_store()
    old_fact = Fact(
        id="old123", category=FactCategory.preference, content="User prefers pandas"
    )
    store.append_facts([old_fact])
    candidate = MemoryCandidate(
        id="cand789",
        category=FactCategory.preference,
        content="User prefers polars",
        supersedes="old123",
    )
    store.append_candidates([candidate])

    approved = store.approve_candidates(["cand789"])

    assert len(approved) == 1

    all_facts = store.load_facts()
    original = next(f for f in all_facts if f.id == "old123")
    replacement = next(f for f in all_facts if f.id != "old123")
    assert original.confidence == 0.3
    assert replacement.supersedes == "old123"


def test_format_facts_for_llm():
    facts = [
        Fact(id="abc123", category=FactCategory.preference, content="Prefers polars"),
        Fact(
            id="def456",
            category=FactCategory.project,
            content="Uses dbt",
            project="acme-dw",
        ),
    ]

    formatted = format_facts_for_llm(facts)
    assert "abc123" in formatted
    assert "preference" in formatted
    assert "acme-dw" in formatted


def test_format_empty():
    assert format_facts_for_llm([]) == "(no facts stored)"


def test_prefilter_facts_prioritizes_matching_content():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                category=FactCategory.preference,
                content="The user prefers polars for dataframes",
                tags=["python", "polars"],
            ),
            Fact(category=FactCategory.workflow, content="Use ruff for formatting"),
            Fact(
                category=FactCategory.project,
                content="acme-dw uses dbt models",
                project="acme-dw",
            ),
        ]
    )

    filtered = store.prefilter_facts(
        "What dataframe library does the user prefer? polars", limit=2
    )

    assert len(filtered) == 2
    # prefilter now returns (score, Fact) tuples
    _, top_fact = filtered[0]
    assert "polars" in top_fact.content.lower()


def test_prefilter_bigram_and_normalization():
    """Bigrams and underscore normalization catch near-misses."""
    store = _make_store()
    store.append_facts(
        [
            Fact(
                category=FactCategory.convention,
                content="The team uses coding_style based on PEP8",
            ),
            Fact(
                category=FactCategory.preference, content="Unrelated fact about pizza"
            ),
        ]
    )

    filtered = store.prefilter_facts("coding style conventions")
    scores = [score for score, _ in filtered]
    # The coding_style fact should score higher than the unrelated one
    assert scores[0] > scores[-1]


def test_prefilter_returns_score_tuples():
    store = _make_store()
    store.append_facts(
        [
            Fact(category=FactCategory.preference, content="Likes Python"),
        ]
    )
    results = store.prefilter_facts("python")
    assert len(results) == 1
    score, fact = results[0]
    assert isinstance(score, int)
    assert score > 0
    assert fact.content == "Likes Python"
