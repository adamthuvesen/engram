"""CI guard for the deterministic prefilter recall number.

Runs the labeled dataset through ``tests/run_evals.py`` and asserts the recall
floors, the no-match behavior, and dataset integrity. Fully deterministic — no
LLM calls, no API keys — so this is safe to run in CI on every push. If the
prefilter scorer regresses, the recall floors here trip.
"""

from __future__ import annotations

import json

import pytest

from tests.run_evals import (
    DATASET_PATH,
    MIN_RECALL_AT_1,
    MIN_RECALL_AT_5,
    evaluate,
)


@pytest.fixture(scope="module")
def summary():
    return evaluate()


@pytest.fixture(scope="module")
def dataset():
    return json.loads(DATASET_PATH.read_text())


def test_no_match_returns_nothing(summary):
    # The off-domain query must surface nothing above the relevance floor.
    assert summary.nomatch_ok


def test_recall_at_1_meets_floor(summary):
    assert summary.recall_at_1 >= MIN_RECALL_AT_1, (
        f"recall@1 {summary.recall_at_1:.2f} below floor {MIN_RECALL_AT_1}"
    )


def test_recall_at_5_meets_floor(summary):
    assert summary.recall_at_5 >= MIN_RECALL_AT_5, (
        f"recall@5 {summary.recall_at_5:.2f} below floor {MIN_RECALL_AT_5}"
    )


def test_all_metrics_reported(summary):
    # recall@1 (not just @5), recall@5, hit-rate, and MRR must all be present
    # and in range — the README cites these exact numbers.
    assert summary.n_answerable >= 40
    for value in (summary.hit_rate, summary.recall_at_1, summary.recall_at_5):
        assert 0.0 <= value <= 1.0
    assert 0.0 < summary.mrr <= 1.0
    assert summary.recall_at_1 <= summary.recall_at_5 <= summary.hit_rate


def test_dataset_is_well_formed(dataset):
    corpus_ids = [f["id"] for f in dataset["corpus"]]
    assert len(corpus_ids) == len(set(corpus_ids)), "duplicate corpus ids"

    queries = dataset["queries"]
    answerable = [q for q in queries if q.get("expected")]
    nomatch = [q for q in queries if not q.get("expected")]
    assert len(answerable) >= 40
    assert len(nomatch) == 1, "exactly one no-match query expected"

    known = set(corpus_ids)
    for q in answerable:
        for fid in q["expected"]:
            assert fid in known, f"query labels unknown fact id: {fid}"
