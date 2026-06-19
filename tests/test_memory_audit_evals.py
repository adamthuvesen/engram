"""Regression tests for the memory audit eval harness."""

from tests.run_memory_audit_evals import (
    MAX_BURDEN_MULTIPLIER,
    MIN_NEW_PRECISION,
    MIN_NEW_RECALL,
    MIN_RECALL_GAIN,
    evaluate,
)


def test_memory_audit_eval_meets_significant_improvement_floor():
    summary = evaluate()

    assert summary.recall_gain >= MIN_RECALL_GAIN
    assert summary.new.recall >= MIN_NEW_RECALL
    assert summary.new.precision >= MIN_NEW_PRECISION
    assert summary.new.burden_multiplier <= MAX_BURDEN_MULTIPLIER
