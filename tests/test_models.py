"""Tests for model improvements — category migration, FactBase inheritance."""

from engram.models import (
    CandidateStatus,
    Fact,
    FactCategory,
    MemoryCandidate,
    RecallRecord,
    migrate_category,
)


def test_migrate_category_temporal_to_event():
    assert migrate_category("temporal") == "event"


def test_migrate_category_update_to_correction():
    assert migrate_category("update") == "correction"


def test_migrate_category_passthrough():
    assert migrate_category("preference") == "preference"
    assert migrate_category("decision") == "decision"


def test_fact_loads_legacy_temporal_as_event():
    """JSONL with category='temporal' should load as 'event'."""
    import json

    data = {
        "category": "temporal",
        "content": "Currently working on auth rewrite",
        "id": "abc123",
    }
    fact = Fact.model_validate_json(json.dumps(data))
    assert fact.category == FactCategory.event


def test_fact_loads_legacy_update_as_correction():
    import json

    data = {
        "category": "update",
        "content": "Actually uses polars now",
        "id": "def456",
    }
    fact = Fact.model_validate_json(json.dumps(data))
    assert fact.category == FactCategory.correction


def test_new_categories_exist():
    assert FactCategory.decision == "decision"
    assert FactCategory.pitfall == "pitfall"
    assert FactCategory.convention == "convention"


def test_memory_candidate_from_fact_model_dump():
    """MemoryCandidate(**fact.model_dump()) should work cleanly."""
    fact = Fact(
        category=FactCategory.preference,
        content="Likes polars",
        tags=["python"],
        project="engram",
    )
    candidate = MemoryCandidate(**fact.model_dump())
    assert candidate.content == fact.content
    assert candidate.category == fact.category
    assert candidate.tags == fact.tags
    assert candidate.project == fact.project
    assert candidate.status == CandidateStatus.pending


def test_legacy_temporal_string_migrates_via_model_validator():
    """Fact(category='temporal') should normalize to FactCategory.event via model_post_init."""
    fact = Fact(category="temporal", content="x")
    assert fact.category == FactCategory.event


def test_legacy_update_string_migrates_via_model_validator():
    """Fact(category='update') should normalize to FactCategory.correction."""
    fact = Fact(category="update", content="x")
    assert fact.category == FactCategory.correction


def test_legacy_temporal_not_in_enum():
    """FactCategory.temporal should no longer exist."""
    import pytest

    with pytest.raises(AttributeError):
        _ = FactCategory.temporal  # type: ignore[attr-defined]


def test_legacy_update_not_in_enum():
    """FactCategory.update should no longer exist."""
    import pytest

    with pytest.raises(AttributeError):
        _ = FactCategory.update  # type: ignore[attr-defined]


def test_model_copy_with_legacy_category_migrates():
    """model_copy with a legacy category string migrates via model_post_init."""
    fact = Fact(category=FactCategory.preference, content="test")
    fact2 = fact.model_copy(update={"category": "temporal"})
    assert fact2.category == FactCategory.event

    fact3 = fact.model_copy(update={"category": "update"})
    assert fact3.category == FactCategory.correction


def test_recall_record_legacy_lines_parse_with_defaults():
    """Old recall_log lines without token fields must parse with None defaults."""
    import json

    legacy = {
        "id": "r1",
        "query": "who am i",
        "tier": 2,
        "prefilter_count": 10,
        "latency_ms": 1234.5,
        "quality": "high",
        "timestamp": "2026-04-17T12:00:00+00:00",
    }
    record = RecallRecord.model_validate_json(json.dumps(legacy))
    assert record.llm_calls is None
    assert record.input_tokens is None
    assert record.cached_tokens is None
    assert record.selector_version is None
    assert record.tier == 2
    assert record.quality == "high"


def test_candidate_inherits_factbase_fields():
    """MemoryCandidate has all FactBase fields plus status/review_note."""
    candidate = MemoryCandidate(
        category=FactCategory.decision,
        content="Chose JSONL over SQLite",
        status=CandidateStatus.approved,
        review_note="Good call",
    )
    assert candidate.confidence == 1.0
    assert candidate.status == CandidateStatus.approved
    assert candidate.review_note == "Good call"
