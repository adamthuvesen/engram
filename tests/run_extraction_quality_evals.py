#!/usr/bin/env python
"""Score coherent memory-card extraction on a no-secret fixture.

Run directly:

    uv run python tests/run_extraction_quality_evals.py

The default run scores committed gold cards and never calls an LLM provider.
Pass ``--live`` to run the configured provider through extraction and dedup, or
pass provider-produced cards to ``evaluate`` to reuse the labels and gates.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from engram.core.models import Fact, FactCategory
from engram.extraction.observer import extract_facts
from engram.storage.store import FactStore

DATASET_PATH = Path(__file__).parent / "extraction_quality_eval_dataset.json"

MIN_DURABLE_CLAIM_COVERAGE = 0.95
MIN_EMITTED_CARD_PRECISION = 0.90
MAX_FRAGMENTATION = 1.25
POLICY_REQUIRED_CLAUSES = 6


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CanonicalMemoryUnit(_StrictModel):
    id: str
    label: str
    project: str | None = None
    expected_supersedes: str | None = None
    required_durable_claims: list[str] = Field(min_length=1)
    future_recall_queries: list[str] = Field(min_length=1)


class ExtractedCard(_StrictModel):
    memory_key: str = ""
    content: str = Field(min_length=1)
    retrieval_hints: list[str] = Field(default_factory=list)
    project: str | None = None
    supersedes: str | None = None


class ExtractionEvalCase(_StrictModel):
    id: str
    kind: str
    policy_regression: bool = False
    raw_input: str = Field(min_length=1)
    canonical_memory_units: list[CanonicalMemoryUnit] = Field(min_length=1)
    forbidden_transient_claims: list[str] = Field(default_factory=list)
    fixture_extracted_cards: list[ExtractedCard] = Field(default_factory=list)


class ExtractionEvalDataset(_StrictModel):
    version: Literal[1]
    description: str
    cases: list[ExtractionEvalCase] = Field(min_length=1)


class CaseScore(_StrictModel):
    case_id: str
    kind: str
    canonical_units: int
    emitted_cards: int
    required_claims: int
    covered_claims: int
    precise_cards: int
    card_unit_assignments: int
    forbidden_transient_hits: list[str]
    merged_card_indexes: list[int]
    uncovered_claims: list[str]
    policy_regression: bool
    policy_regression_ok: bool | None


class EvalSummary(_StrictModel):
    cases: int
    canonical_units: int
    emitted_cards: int
    required_claims: int
    covered_claims: int
    precise_cards: int
    durable_claim_coverage: float
    emitted_card_precision: float
    fragmentation: float
    forbidden_transient_claims: int
    policy_regression_ok: bool
    coherent_unit_separation_ok: bool
    card_metadata_ok: bool
    passes_gates: bool
    results: list[CaseScore]


ProviderOutputs = Mapping[str, Sequence[ExtractedCard | Mapping[str, Any]]]


def load_dataset(path: Path = DATASET_PATH) -> ExtractionEvalDataset:
    dataset = ExtractionEvalDataset.model_validate_json(path.read_text())
    case_ids = [case.id for case in dataset.cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Extraction eval case IDs must be unique")

    policy_cases = [case for case in dataset.cases if case.policy_regression]
    if len(policy_cases) != 1:
        raise ValueError("Dataset must contain exactly one policy regression case")
    policy_case = policy_cases[0]
    if len(policy_case.canonical_memory_units) != 1:
        raise ValueError("Policy regression must label exactly one canonical unit")
    policy_claims = policy_case.canonical_memory_units[0].required_durable_claims
    if len(policy_claims) != POLICY_REQUIRED_CLAUSES:
        raise ValueError(
            f"Policy regression must label {POLICY_REQUIRED_CLAUSES} durable clauses"
        )
    return dataset


def evaluate(
    dataset_path: Path = DATASET_PATH,
    extracted_cards_by_case: ProviderOutputs | None = None,
) -> EvalSummary:
    """Score fixture cards or caller-supplied provider cards against the labels."""
    dataset = load_dataset(dataset_path)
    outputs = _resolve_outputs(dataset, extracted_cards_by_case)
    results = [_score_case(case, outputs[case.id]) for case in dataset.cases]

    canonical_units = sum(result.canonical_units for result in results)
    emitted_cards = sum(result.emitted_cards for result in results)
    required_claims = sum(result.required_claims for result in results)
    covered_claims = sum(result.covered_claims for result in results)
    precise_cards = sum(result.precise_cards for result in results)
    card_unit_assignments = sum(result.card_unit_assignments for result in results)
    forbidden_transient_claims = sum(
        len(result.forbidden_transient_hits) for result in results
    )

    durable_claim_coverage = covered_claims / required_claims
    emitted_card_precision = precise_cards / emitted_cards if emitted_cards else 1.0
    fragmentation = card_unit_assignments / canonical_units
    policy_regression_ok = all(
        result.policy_regression_ok is True
        for result in results
        if result.policy_regression
    )
    coherent_unit_separation_ok = all(
        not result.merged_card_indexes for result in results
    )
    card_metadata_ok = all(
        card.memory_key and 1 <= len(card.retrieval_hints) <= 5
        for cards in outputs.values()
        for card in cards
    )
    passes_gates = (
        durable_claim_coverage >= MIN_DURABLE_CLAIM_COVERAGE
        and emitted_card_precision >= MIN_EMITTED_CARD_PRECISION
        and fragmentation <= MAX_FRAGMENTATION
        and policy_regression_ok
        and forbidden_transient_claims == 0
        and coherent_unit_separation_ok
        and card_metadata_ok
    )

    return EvalSummary(
        cases=len(results),
        canonical_units=canonical_units,
        emitted_cards=emitted_cards,
        required_claims=required_claims,
        covered_claims=covered_claims,
        precise_cards=precise_cards,
        durable_claim_coverage=durable_claim_coverage,
        emitted_card_precision=emitted_card_precision,
        fragmentation=fragmentation,
        forbidden_transient_claims=forbidden_transient_claims,
        policy_regression_ok=policy_regression_ok,
        coherent_unit_separation_ok=coherent_unit_separation_ok,
        card_metadata_ok=card_metadata_ok,
        passes_gates=passes_gates,
        results=results,
    )


async def extract_live_provider_outputs(
    dataset_path: Path = DATASET_PATH,
) -> dict[str, list[ExtractedCard]]:
    """Run the real extraction and dedup path with the configured provider."""
    dataset = load_dataset(dataset_path)
    outputs: dict[str, list[ExtractedCard]] = {}

    with tempfile.TemporaryDirectory(prefix="engram-extraction-eval-") as data_dir:
        for case in dataset.cases:
            store = FactStore(data_dir=Path(data_dir) / case.id)
            for unit in case.canonical_memory_units:
                if unit.expected_supersedes:
                    store.append_facts(
                        [
                            Fact(
                                id=unit.expected_supersedes,
                                category=FactCategory.preference,
                                memory_key="rowan-editor-preference",
                                content="Rowan uses Vim as the default editor.",
                                project=unit.project,
                            )
                        ]
                    )

            facts = await extract_facts(case.raw_input, store=store)
            outputs[case.id] = [
                ExtractedCard(
                    memory_key=fact.memory_key,
                    content=fact.content,
                    retrieval_hints=fact.retrieval_hints,
                    project=fact.project,
                    supersedes=fact.supersedes,
                )
                for fact in facts
            ]

    return outputs


def _resolve_outputs(
    dataset: ExtractionEvalDataset,
    extracted_cards_by_case: ProviderOutputs | None,
) -> dict[str, list[ExtractedCard]]:
    if extracted_cards_by_case is None:
        return {case.id: list(case.fixture_extracted_cards) for case in dataset.cases}

    expected_case_ids = {case.id for case in dataset.cases}
    supplied_case_ids = set(extracted_cards_by_case)
    if supplied_case_ids != expected_case_ids:
        missing = sorted(expected_case_ids - supplied_case_ids)
        unknown = sorted(supplied_case_ids - expected_case_ids)
        raise ValueError(
            f"Provider outputs must cover every case. Missing: {missing}. Unknown: {unknown}."
        )
    return {
        case_id: [ExtractedCard.model_validate(card) for card in cards]
        for case_id, cards in extracted_cards_by_case.items()
    }


def _score_case(case: ExtractionEvalCase, cards: list[ExtractedCard]) -> CaseScore:
    normalized_cards = [_normalize(card.content) for card in cards]
    unit_card_indexes: dict[str, set[int]] = {
        unit.id: set() for unit in case.canonical_memory_units
    }
    card_unit_ids: dict[int, set[str]] = {index: set() for index in range(len(cards))}
    uncovered_claims: list[str] = []
    covered_claims = 0

    for unit in case.canonical_memory_units:
        for claim in unit.required_durable_claims:
            matching_indexes = {
                index
                for index, card in enumerate(cards)
                if _card_aligns_with_unit(card, unit)
                and _contains(normalized_cards[index], claim)
            }
            if matching_indexes:
                covered_claims += 1
                unit_card_indexes[unit.id].update(matching_indexes)
                for index in matching_indexes:
                    card_unit_ids[index].add(unit.id)
            else:
                uncovered_claims.append(f"{unit.id}: {claim}")

    forbidden_hits_by_card: dict[int, list[str]] = {}
    forbidden_transient_hits: list[str] = []
    for index, normalized_card in enumerate(normalized_cards):
        hits = [
            claim
            for claim in case.forbidden_transient_claims
            if _contains(normalized_card, claim)
        ]
        if hits:
            forbidden_hits_by_card[index] = hits
            forbidden_transient_hits.extend(
                f"card {index + 1}: {claim}" for claim in hits
            )

    merged_card_indexes = [
        index + 1 for index, unit_ids in card_unit_ids.items() if len(unit_ids) > 1
    ]
    precise_cards = sum(
        len(card_unit_ids[index]) == 1 and index not in forbidden_hits_by_card
        for index in range(len(cards))
    )
    card_unit_assignments = sum(len(indexes) for indexes in unit_card_indexes.values())
    required_claims = sum(
        len(unit.required_durable_claims) for unit in case.canonical_memory_units
    )

    policy_regression_ok: bool | None = None
    if case.policy_regression:
        policy_unit = case.canonical_memory_units[0]
        policy_regression_ok = (
            len(cards) == 1
            and _card_aligns_with_unit(cards[0], policy_unit)
            and all(
                _contains(normalized_cards[0], claim)
                for claim in policy_unit.required_durable_claims
            )
        )

    return CaseScore(
        case_id=case.id,
        kind=case.kind,
        canonical_units=len(case.canonical_memory_units),
        emitted_cards=len(cards),
        required_claims=required_claims,
        covered_claims=covered_claims,
        precise_cards=precise_cards,
        card_unit_assignments=card_unit_assignments,
        forbidden_transient_hits=forbidden_transient_hits,
        merged_card_indexes=merged_card_indexes,
        uncovered_claims=uncovered_claims,
        policy_regression=case.policy_regression,
        policy_regression_ok=policy_regression_ok,
    )


def _card_aligns_with_unit(
    card: ExtractedCard,
    unit: CanonicalMemoryUnit,
) -> bool:
    return card.project == unit.project and card.supersedes == unit.expected_supersedes


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


def _contains(normalized_card: str, claim: str) -> bool:
    return _normalize(claim) in normalized_card


def _pct(value: float) -> str:
    return f"{value:.0%}"


def main(*, live: bool = False) -> int:
    if live:
        try:
            outputs = asyncio.run(extract_live_provider_outputs())
        except Exception as exc:
            print(f"Live extraction eval failed: {exc}", file=sys.stderr)
            return 2
        summary = evaluate(extracted_cards_by_case=outputs)
        run_label = "Live configured provider through extraction and dedup"
    else:
        summary = evaluate()
        run_label = "Deterministic contract fixture, no provider call, no API key"

    print("Extraction quality - coherent memory cards")
    print(
        f"{summary.cases} cases, {summary.canonical_units} canonical units, "
        f"{summary.required_claims} durable claims"
    )
    print(run_label)
    print("")
    print(f"{'metric':<32}{'value':>10}{'gate':>12}")
    print("-" * 54)
    print(
        f"{'durable claim coverage':<32}"
        f"{_pct(summary.durable_claim_coverage):>10}"
        f"{'>= 95%':>12}"
    )
    print(
        f"{'emitted-card precision':<32}"
        f"{_pct(summary.emitted_card_precision):>10}"
        f"{'>= 90%':>12}"
    )
    print(
        f"{'cards per canonical unit':<32}{summary.fragmentation:>10.2f}{'<= 1.25':>12}"
    )
    print(
        f"{'policy regression':<32}"
        f"{str(summary.policy_regression_ok):>10}"
        f"{'one card':>12}"
    )
    print(
        f"{'forbidden transient claims':<32}"
        f"{summary.forbidden_transient_claims:>10}"
        f"{'zero':>12}"
    )
    print(
        f"{'canonical units stay separate':<32}"
        f"{str(summary.coherent_unit_separation_ok):>10}"
        f"{'required':>12}"
    )
    print(
        f"{'keys and retrieval hints':<32}"
        f"{str(summary.card_metadata_ok):>10}"
        f"{'required':>12}"
    )

    if summary.passes_gates:
        return 0

    print("\nGATE FAILED")
    for result in summary.results:
        if result.uncovered_claims:
            print(f"- {result.case_id} uncovered: {result.uncovered_claims}")
        if result.forbidden_transient_hits:
            print(f"- {result.case_id} transient: {result.forbidden_transient_hits}")
        if result.merged_card_indexes:
            print(f"- {result.case_id} merged cards: {result.merged_card_indexes}")
        if result.policy_regression_ok is False:
            print(f"- {result.case_id} policy regression did not stay in one card")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run extraction and dedup with the configured LLM provider",
    )
    args = parser.parse_args()
    sys.exit(main(live=args.live))
