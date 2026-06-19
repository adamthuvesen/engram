"""CI guard for the cross-project recall quality benchmark."""

from __future__ import annotations

import json

import pytest

from tests.run_cross_project_recall_evals import (
    DATASET_PATH,
    MIN_QUALITY_SCORE,
    MIN_RELATIVE_GAIN,
    evaluate,
)


@pytest.fixture(scope="module")
def summary():
    return evaluate()


@pytest.fixture(scope="module")
def dataset():
    return json.loads(DATASET_PATH.read_text())


def test_cross_project_quality_meets_floor(summary):
    assert summary.quality_score >= MIN_QUALITY_SCORE, (
        f"quality {summary.quality_score:.3f} below floor {MIN_QUALITY_SCORE:.3f}"
    )


def test_cross_project_quality_gain_clears_goal(summary):
    assert summary.relative_gain > MIN_RELATIVE_GAIN, (
        f"relative gain {summary.relative_gain:.3f} did not clear {MIN_RELATIVE_GAIN:.3f}"
    )


def test_bad_evidence_is_excluded(summary):
    offenders = [r for r in summary.results if r.bad_evidence]
    assert offenders == []


def test_no_match_returns_no_evidence(summary):
    assert summary.nomatch_ok


def test_answerable_queries_keep_expected_evidence(summary):
    assert summary.hit_rate == 1.0
    assert summary.recall_at_5 == 1.0


def test_dataset_covers_required_memory_cases(dataset):
    corpus = dataset["corpus"]
    queries = dataset["queries"]
    fact_ids = {fact["id"] for fact in corpus}

    assert any(fact.get("stale") is True for fact in corpus)
    assert any(fact.get("supersedes") for fact in corpus)
    assert len({fact.get("project") for fact in corpus if fact.get("project")}) >= 3

    kinds = {query["kind"] for query in queries}
    assert {"project-specific", "stale", "contradiction", "global", "no-match"} <= kinds

    for query in queries:
        for fact_id in query.get("expected", []):
            assert fact_id in fact_ids
        for fact_id in query.get("excluded", []):
            assert fact_id in fact_ids
