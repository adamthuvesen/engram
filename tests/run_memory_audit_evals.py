#!/usr/bin/env python
"""Evaluate read-only memory audit suggestions on a labeled no-secret dump.

Run directly:

    uv run python tests/run_memory_audit_evals.py

The baseline mirrors the current no-key audit floor: exact duplicate signatures
like ``doctor`` checks today. The new loop must improve labeled issue recall by
at least 50 percentage points while keeping precision high and reviewer burden
bounded.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from engram.memory_audit import audit_facts
from engram.models import Fact, FactCategory, MIN_ACTIVE_CONFIDENCE

DATASET_PATH = Path(__file__).parent / "memory_audit_eval_dataset.json"

MIN_RECALL_GAIN = 0.50
MIN_NEW_RECALL = 0.80
MIN_NEW_PRECISION = 0.80
MAX_BURDEN_MULTIPLIER = 1.50


class ExpectedIssue(BaseModel):
    kind: str
    fact_ids: list[str] = Field(min_length=1)


class MetricBlock(BaseModel):
    predicted: int
    true_positive: int
    recall: float
    precision: float
    burden_multiplier: float


class EvalSummary(BaseModel):
    facts: int
    expected: int
    baseline: MetricBlock
    new: MetricBlock
    recall_gain: float
    baseline_keys: list[str]
    new_keys: list[str]
    expected_keys: list[str]


@dataclass(frozen=True)
class _LoadedDataset:
    facts: list[Fact]
    now: datetime
    expected: set[str]


def evaluate(path: Path = DATASET_PATH) -> EvalSummary:
    dataset = _load_dataset(path)
    baseline_keys = _doctor_like_baseline(dataset.facts)
    result = audit_facts(dataset.facts, now=dataset.now)
    new_keys = {_suggestion_key(s.kind, s.fact_ids) for s in result.suggestions}

    baseline = _metrics(baseline_keys, dataset.expected)
    new = _metrics(new_keys, dataset.expected)
    return EvalSummary(
        facts=len(dataset.facts),
        expected=len(dataset.expected),
        baseline=baseline,
        new=new,
        recall_gain=new.recall - baseline.recall,
        baseline_keys=sorted(baseline_keys),
        new_keys=sorted(new_keys),
        expected_keys=sorted(dataset.expected),
    )


def _load_dataset(path: Path) -> _LoadedDataset:
    raw = json.loads(path.read_text())
    now = _parse_dt(raw["now"])
    facts = [_fact_from_json(item) for item in raw["corpus"]]
    expected = {
        _suggestion_key(issue.kind, issue.fact_ids)
        for issue in (ExpectedIssue.model_validate(item) for item in raw["expected"])
    }
    return _LoadedDataset(facts=facts, now=now, expected=expected)


def _fact_from_json(item: dict[str, Any]) -> Fact:
    observed_at = _parse_dt(item["observed_at"])
    expires_at = _parse_dt(item["expires_at"]) if item.get("expires_at") else None
    return Fact(
        id=item["id"],
        category=FactCategory(item["category"]),
        content=item["content"],
        project=item.get("project"),
        tags=item.get("tags", []),
        observed_at=observed_at,
        updated_at=observed_at,
        expires_at=expires_at,
    )


def _doctor_like_baseline(facts: list[Fact]) -> set[str]:
    """Current deterministic audit floor: exact active duplicate signatures."""
    by_signature: dict[tuple[str | None, str, str], list[str]] = defaultdict(list)
    for fact in facts:
        if fact.confidence < MIN_ACTIVE_CONFIDENCE or fact.stale:
            continue
        signature = (
            fact.project,
            fact.category.value,
            " ".join(fact.content.lower().split()),
        )
        by_signature[signature].append(fact.id)
    return {
        _suggestion_key("duplicate", fact_ids)
        for fact_ids in by_signature.values()
        if len(fact_ids) > 1
    }


def _metrics(predicted: set[str], expected: set[str]) -> MetricBlock:
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted) if predicted else 1.0
    return MetricBlock(
        predicted=len(predicted),
        true_positive=true_positive,
        recall=true_positive / len(expected),
        precision=precision,
        burden_multiplier=(len(predicted) / len(expected)) if expected else 0.0,
    )


def _suggestion_key(kind: str, fact_ids: list[str]) -> str:
    return f"{kind}:{'|'.join(sorted(fact_ids))}"


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _pct(value: float) -> str:
    return f"{value:.0%}"


def main() -> int:
    summary = evaluate()
    print("Memory audit suggestions - realistic no-secret fixture")
    print(f"{summary.facts} facts, {summary.expected} labeled issue groups")
    print("")
    print(
        "baseline (doctor-like exact checks): "
        f"recall {_pct(summary.baseline.recall)} "
        f"({summary.baseline.true_positive}/{summary.expected}), "
        f"precision {_pct(summary.baseline.precision)}, "
        f"suggestions {summary.baseline.predicted}"
    )
    print(
        "new audit loop: "
        f"recall {_pct(summary.new.recall)} "
        f"({summary.new.true_positive}/{summary.expected}), "
        f"precision {_pct(summary.new.precision)}, "
        f"suggestions {summary.new.predicted}, "
        f"burden {summary.new.burden_multiplier:.2f}x"
    )
    print(f"recall gain: +{_pct(summary.recall_gain)}")

    failures = []
    if summary.recall_gain < MIN_RECALL_GAIN:
        failures.append(f"recall gain below floor {_pct(MIN_RECALL_GAIN)}")
    if summary.new.recall < MIN_NEW_RECALL:
        failures.append(f"new recall below floor {_pct(MIN_NEW_RECALL)}")
    if summary.new.precision < MIN_NEW_PRECISION:
        failures.append(f"new precision below floor {_pct(MIN_NEW_PRECISION)}")
    if summary.new.burden_multiplier > MAX_BURDEN_MULTIPLIER:
        failures.append(
            f"reviewer burden above {MAX_BURDEN_MULTIPLIER:.2f}x expected issues"
        )

    if failures:
        print("")
        print("FAILED:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
