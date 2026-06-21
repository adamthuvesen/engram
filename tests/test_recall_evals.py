"""CI guard for the deterministic prefilter recall numbers.

Runs the labeled dataset through ``tests/run_evals.py`` and asserts the recall
floors, the tier-0 (zero-LLM) cost floor, the no-match behavior, the per-kind
division of labor, and dataset integrity. Fully deterministic — no LLM calls,
no API keys — so this is safe to run in CI on every push. If the prefilter
scorer regresses, the floors here trip.
"""

from __future__ import annotations

import json

import pytest

from tests.run_evals import (
    DATASET_PATH,
    MIN_HIT_RATE,
    MIN_MRR,
    MIN_RECALL_AT_1,
    MIN_RECALL_AT_5,
    MIN_TIER0_FRACTION,
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


def test_tier0_cost_floor(summary):
    # A representative query mix must resolve a real share at tier-0 (no LLM).
    assert summary.tier0_fraction >= MIN_TIER0_FRACTION, (
        f"tier-0 share {summary.tier0_fraction:.2f} below floor {MIN_TIER0_FRACTION}"
    )


def test_recall_at_1_meets_floor(summary):
    assert summary.recall_at_1 >= MIN_RECALL_AT_1, (
        f"recall@1 {summary.recall_at_1:.2f} below floor {MIN_RECALL_AT_1}"
    )


def test_recall_at_5_meets_floor(summary):
    assert summary.recall_at_5 >= MIN_RECALL_AT_5, (
        f"recall@5 {summary.recall_at_5:.2f} below floor {MIN_RECALL_AT_5}"
    )


def test_candidate_recall_meets_floor(summary):
    # Hit-rate is the prefilter's candidate-recall job: keep the answer in pool.
    assert summary.hit_rate >= MIN_HIT_RATE, (
        f"candidate recall {summary.hit_rate:.2f} below floor {MIN_HIT_RATE}"
    )


def test_mrr_meets_floor(summary):
    assert summary.mrr >= MIN_MRR, f"MRR {summary.mrr:.2f} below floor {MIN_MRR}"


def test_literal_queries_are_a_prefilter_win(summary):
    # The whole point: the deterministic keyword pass nails literal/exact-term
    # queries on its own. If this drops, the cost story no longer holds.
    hits, total = summary.recall1_by_kind["literal"]
    assert total >= 15
    assert hits / total >= 0.9, f"literal recall@1 {hits}/{total} too low"


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
    assert len(nomatch) >= 1, "at least one no-match query expected"

    # A representative set, not an all-hard one: a real share of literal queries
    # and a real share of synonym/semantic ones.
    kinds = [q.get("kind", "") for q in answerable]
    assert kinds.count("literal") >= 15
    assert sum(k in ("synonym", "semantic") for k in kinds) >= 5

    known = set(corpus_ids)
    for q in answerable:
        for fid in q["expected"]:
            assert fid in known, f"query labels unknown fact id: {fid}"
