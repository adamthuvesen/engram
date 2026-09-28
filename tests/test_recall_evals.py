"""CI guard for the deterministic lexical recall numbers.

Runs the labeled dataset through ``tests/run_evals.py`` and asserts the recall
floors, the zero-LLM cost floor, the no-match behavior, the per-kind division
of labor, and dataset integrity. Fully deterministic — no LLM calls, no API
keys — so this is safe to run in CI on every push. If search ranking or the
relevance bar regresses, the floors here trip.
"""

from __future__ import annotations

import json

import pytest

from engram.recall.evals import EvalFixture, run_fixture_sync
from tests.run_evals import (
    DATASET_PATH,
    MIN_HIT_RATE,
    MIN_MRR,
    MIN_RECALL_AT_1,
    MIN_RECALL_AT_5,
    MIN_LEXICAL_FRACTION,
    evaluate,
)


@pytest.fixture(scope="module")
def summary():
    return evaluate()


@pytest.fixture(scope="module")
def dataset():
    return json.loads(DATASET_PATH.read_text())


def test_no_match_returns_nothing(summary):
    # Off-domain queries must return no cards.
    assert summary.nomatch_ok


def test_lexical_cost_floor(summary):
    # Most of a representative query mix must resolve with no LLM call, even
    # when a key is configured.
    assert summary.lexical_fraction >= MIN_LEXICAL_FRACTION, (
        f"lexical share {summary.lexical_fraction:.2f} below floor "
        f"{MIN_LEXICAL_FRACTION}"
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
    # Hit-rate: the answer is reachable by the path recall takes (cards, or the
    # zero-hit LLM selection pool).
    assert summary.hit_rate >= MIN_HIT_RATE, (
        f"candidate recall {summary.hit_rate:.2f} below floor {MIN_HIT_RATE}"
    )


def test_mrr_meets_floor(summary):
    assert summary.mrr >= MIN_MRR, f"MRR {summary.mrr:.2f} below floor {MIN_MRR}"


@pytest.mark.parametrize("kind", ["literal", "paraphrase"])
def test_worded_queries_are_a_lexical_win(summary, kind):
    # The whole point: lexical cards nail queries that use the fact's own terms
    # (literal) or a natural rewording of them. If this drops, the zero-LLM
    # default no longer holds.
    hits, total = summary.recall1_by_kind[kind]
    assert total >= 15
    assert hits / total >= 0.9, f"{kind} recall@1 {hits}/{total} too low"


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


KNOWLEDGE_UPDATE_PATH = DATASET_PATH.parent / "knowledge_update_eval_fixtures.json"


def _knowledge_update_fixtures() -> list[EvalFixture]:
    raw = json.loads(KNOWLEDGE_UPDATE_PATH.read_text())
    return [EvalFixture.model_validate(item) for item in raw["fixtures"]]


def test_knowledge_update_set_is_large_enough():
    assert len(_knowledge_update_fixtures()) >= 6


@pytest.mark.parametrize(
    "fixture", _knowledge_update_fixtures(), ids=lambda fixture: fixture.name
)
def test_knowledge_update_fixture(fixture):
    result = run_fixture_sync(fixture)
    failed = [check for check in result.checks if not check.passed]
    assert result.passed, failed
