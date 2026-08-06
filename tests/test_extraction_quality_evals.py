"""Regression tests for the extraction-quality eval harness."""

from __future__ import annotations

from tests.run_extraction_quality_evals import (
    DATASET_PATH,
    MAX_FRAGMENTATION,
    MIN_DURABLE_CLAIM_COVERAGE,
    MIN_EMITTED_CARD_PRECISION,
    POLICY_REQUIRED_CLAUSES,
    evaluate,
    load_dataset,
    main,
)


def _fixture_outputs():
    dataset = load_dataset()
    return {
        case.id: [card.model_dump() for card in case.fixture_extracted_cards]
        for case in dataset.cases
    }


def test_fixture_meets_extraction_quality_gates():
    summary = evaluate()

    assert summary.durable_claim_coverage >= MIN_DURABLE_CLAIM_COVERAGE
    assert summary.emitted_card_precision >= MIN_EMITTED_CARD_PRECISION
    assert summary.fragmentation <= MAX_FRAGMENTATION
    assert summary.policy_regression_ok
    assert summary.forbidden_transient_claims == 0
    assert summary.coherent_unit_separation_ok
    assert summary.card_metadata_ok
    assert summary.passes_gates


def test_dataset_covers_required_memory_shapes():
    dataset = load_dataset(DATASET_PATH)
    cases_by_kind = {case.kind: case for case in dataset.cases}

    assert {
        "coupled-policy",
        "independent-units",
        "transient-exclusion",
        "correction-update",
        "project-scope",
        "related-distinct-units",
    } <= cases_by_kind.keys()

    policy_case = cases_by_kind["coupled-policy"]
    assert len(policy_case.canonical_memory_units) == 1
    assert (
        len(policy_case.canonical_memory_units[0].required_durable_claims)
        == POLICY_REQUIRED_CLAUSES
    )
    assert len(policy_case.fixture_extracted_cards) == 1

    correction_unit = cases_by_kind["correction-update"].canonical_memory_units[0]
    assert correction_unit.expected_supersedes is not None
    assert {
        unit.project for unit in cases_by_kind["project-scope"].canonical_memory_units
    } == {"atlas", "beacon"}

    for case in dataset.cases:
        assert case.raw_input
        assert case.forbidden_transient_claims
        for unit in case.canonical_memory_units:
            assert unit.label
            assert unit.required_durable_claims
            assert unit.future_recall_queries


def test_provider_outputs_can_be_scored_without_fixture_cards():
    outputs = _fixture_outputs()
    policy_card = outputs["dual-memory-policy"][0]
    policy_card["content"] = policy_card["content"].replace(
        "Engram is the canonical cross-tool copy. ", ""
    )

    summary = evaluate(extracted_cards_by_case=outputs)
    policy_result = next(
        result for result in summary.results if result.case_id == "dual-memory-policy"
    )

    assert summary.durable_claim_coverage < MIN_DURABLE_CLAIM_COVERAGE
    assert policy_result.policy_regression_ok is False
    assert not summary.passes_gates


def test_missing_memory_key_or_retrieval_hints_fails_metadata_gate():
    outputs = _fixture_outputs()
    outputs["durable-rule-with-progress"][0]["memory_key"] = ""
    outputs["independent-decisions"][0]["retrieval_hints"] = []

    summary = evaluate(extracted_cards_by_case=outputs)

    assert not summary.card_metadata_ok
    assert not summary.passes_gates


def test_fragmented_policy_fails_one_card_and_fragmentation_gates():
    dataset = load_dataset()
    outputs = _fixture_outputs()
    policy_unit = next(
        case for case in dataset.cases if case.id == "dual-memory-policy"
    ).canonical_memory_units[0]
    outputs["dual-memory-policy"] = [
        {"content": claim, "project": None, "supersedes": None}
        for claim in policy_unit.required_durable_claims
    ]

    summary = evaluate(extracted_cards_by_case=outputs)

    assert summary.fragmentation > MAX_FRAGMENTATION
    assert not summary.policy_regression_ok
    assert not summary.passes_gates


def test_transient_claim_is_counted_and_makes_card_imprecise():
    outputs = _fixture_outputs()
    outputs["durable-rule-with-progress"].append(
        {
            "content": "The restore rehearsal is on step four of nine.",
            "project": "orion",
            "supersedes": None,
        }
    )

    summary = evaluate(extracted_cards_by_case=outputs)

    assert summary.forbidden_transient_claims == 1
    assert summary.emitted_card_precision == MIN_EMITTED_CARD_PRECISION
    assert not summary.passes_gates


def test_merged_related_memories_fail_coherent_unit_separation():
    outputs = _fixture_outputs()
    outputs["related-resilience-decisions"] = [
        {
            "content": (
                "Retry failed requests up to three times with exponential backoff. "
                "Open the circuit breaker after five consecutive failures."
            ),
            "project": "helios",
            "supersedes": None,
        }
    ]

    summary = evaluate(extracted_cards_by_case=outputs)
    result = next(
        result
        for result in summary.results
        if result.case_id == "related-resilience-decisions"
    )

    assert result.merged_card_indexes == [1]
    assert summary.emitted_card_precision < MIN_EMITTED_CARD_PRECISION
    assert not summary.coherent_unit_separation_ok
    assert not summary.passes_gates


def test_scope_and_supersedes_are_part_of_claim_coverage():
    outputs = _fixture_outputs()
    outputs["preference-correction"][0]["supersedes"] = None
    outputs["project-scoping"][0]["project"] = "beacon"

    summary = evaluate(extracted_cards_by_case=outputs)
    results = {result.case_id: result for result in summary.results}

    assert results["preference-correction"].covered_claims == 0
    assert results["project-scoping"].covered_claims == 1
    assert not summary.passes_gates


def test_cli_reports_no_provider_call(capsys):
    assert main() == 0
    output = capsys.readouterr().out

    assert "no provider call" in output
    assert "durable claim coverage" in output


def test_live_cli_reports_provider_failure_without_traceback(monkeypatch, capsys):
    async def fail_live_run():
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "tests.run_extraction_quality_evals.extract_live_provider_outputs",
        fail_live_run,
    )

    assert main(live=True) == 2
    assert capsys.readouterr().err == (
        "Live extraction eval failed: provider unavailable\n"
    )
