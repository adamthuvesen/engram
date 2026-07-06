"""Tests for model contracts."""

import json

import pytest
from pydantic import ValidationError

from engram.core.models import (
    CandidateStatus,
    Fact,
    FactCategory,
    MemoryCandidate,
    RecallRecord,
)


def test_fact_loads_canonical_category_string():
    data = {
        "category": "correction",
        "content": "Uses polars now",
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


def test_removed_category_names_are_rejected():
    with pytest.raises(ValidationError):
        Fact(category="temporal", content="x")
    with pytest.raises(ValidationError):
        Fact(category="update", content="x")


def test_recall_record_usage_fields_default_to_none_when_omitted():
    """Recall log entries can omit provider usage fields."""

    payload = {
        "id": "r1",
        "query": "who am i",
        "tier": 2,
        "prefilter_count": 10,
        "latency_ms": 1234.5,
        "quality": "high",
        "timestamp": "2026-04-17T12:00:00+00:00",
    }
    record = RecallRecord.model_validate_json(json.dumps(payload))
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
