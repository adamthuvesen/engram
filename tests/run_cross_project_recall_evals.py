#!/usr/bin/env python
"""Deterministic cross-project recall quality benchmark.

Run directly:

    uv run python tests/run_cross_project_recall_evals.py

The fixture is no-secret fictional memory data. It measures whether deterministic
recall keeps useful evidence while excluding stale, superseded, contradictory,
or wrong-project evidence. No LLM provider or API key is used.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pydantic import BaseModel, Field

DATASET_PATH = Path(__file__).parent / "cross_project_recall_eval_dataset.json"

BASELINE_QUALITY_SCORE = 0.4166666666666667
MIN_QUALITY_SCORE = 0.88
MIN_RELATIVE_GAIN = 0.20


class LabeledQuery(BaseModel):
    query: str
    expected: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    kind: str
    note: str = ""


class QueryResult(BaseModel):
    query: str
    kind: str
    expected: list[str]
    excluded: list[str]
    tier: int
    ranked: list[str]
    hit: bool
    rank: int | None
    recall_at_1: bool
    recall_at_5: bool
    bad_evidence: list[str]
    score: float


class Summary(BaseModel):
    n_corpus: int
    n_queries: int
    n_answerable: int
    quality_score: float
    absolute_gain: float
    relative_gain: float
    recall_at_1: float
    recall_at_5: float
    hit_rate: float
    bad_evidence_rate: float
    nomatch_ok: bool
    score_by_kind: dict[str, list[float]]
    results: list[QueryResult]


def _no_op_completion():
    from engram.llm import Completion

    async def fake(
        prompt,
        system="",
        model=None,
        temperature=None,
        response_format=None,
        cache_prefix=None,
    ):
        return Completion(
            text="(no deterministic answer) [quality: low]",
            input_tokens=0,
            cached_tokens=0,
        )

    return fake


def _score_query(
    *,
    expected: set[str],
    excluded: set[str],
    ranked: list[str],
) -> tuple[float, bool, int | None, bool, bool, list[str]]:
    bad_evidence = [fact_id for fact_id in ranked if fact_id in excluded]
    if not expected:
        return (
            1.0 if not ranked else 0.0,
            not ranked,
            None,
            False,
            False,
            bad_evidence,
        )

    rank: int | None = None
    for i, fact_id in enumerate(ranked, start=1):
        if fact_id in expected:
            rank = i
            break

    hit = rank is not None
    recall_at_1 = rank == 1
    recall_at_5 = rank is not None and rank <= 5
    rank_credit = 1.0 if recall_at_1 else (0.5 if recall_at_5 else 0.0)
    bad_penalty = min(1.0, len(bad_evidence) / max(1, len(excluded)))
    score = max(0.0, (0.70 if hit else 0.0) + (0.30 * rank_credit) - bad_penalty)
    return score, hit, rank, recall_at_1, recall_at_5, bad_evidence


async def _run_query(store, lq: LabeledQuery) -> QueryResult:
    import engram.retriever as retriever_mod
    from engram.retriever import recall_with_provenance

    saved = retriever_mod.complete_with_usage
    retriever_mod.complete_with_usage = _no_op_completion()
    try:
        _, _, provenance, _ = await recall_with_provenance(lq.query, store=store)
    finally:
        retriever_mod.complete_with_usage = saved

    ranked = [m.id for m in provenance.prefilter_matches if m.above_floor]
    score, hit, rank, recall_at_1, recall_at_5, bad_evidence = _score_query(
        expected=set(lq.expected),
        excluded=set(lq.excluded),
        ranked=ranked,
    )
    return QueryResult(
        query=lq.query,
        kind=lq.kind,
        expected=lq.expected,
        excluded=lq.excluded,
        tier=provenance.tier,
        ranked=ranked,
        hit=hit,
        rank=rank,
        recall_at_1=recall_at_1,
        recall_at_5=recall_at_5,
        bad_evidence=bad_evidence,
        score=score,
    )


async def _evaluate_async(dataset: dict[str, Any]) -> Summary:
    from engram.evals import EvalFactSpec, _materialize_facts
    from engram.store import FactStore

    corpus = [EvalFactSpec.model_validate(f) for f in dataset["corpus"]]
    queries = [LabeledQuery.model_validate(q) for q in dataset["queries"]]

    with TemporaryDirectory() as tmp_dir:
        store = FactStore(data_dir=Path(tmp_dir))
        store.append_facts(_materialize_facts(corpus))
        results = [await _run_query(store, query) for query in queries]

    answerable = [r for r in results if r.expected]
    nomatch = [r for r in results if not r.expected]
    by_kind: dict[str, list[float]] = {}
    for result in results:
        by_kind.setdefault(result.kind, []).append(result.score)

    quality_score = sum(r.score for r in results) / len(results)
    absolute_gain = quality_score - BASELINE_QUALITY_SCORE
    relative_gain = absolute_gain / BASELINE_QUALITY_SCORE

    return Summary(
        n_corpus=len(corpus),
        n_queries=len(results),
        n_answerable=len(answerable),
        quality_score=quality_score,
        absolute_gain=absolute_gain,
        relative_gain=relative_gain,
        recall_at_1=sum(r.recall_at_1 for r in answerable) / len(answerable),
        recall_at_5=sum(r.recall_at_5 for r in answerable) / len(answerable),
        hit_rate=sum(r.hit for r in answerable) / len(answerable),
        bad_evidence_rate=sum(bool(r.bad_evidence) for r in results) / len(results),
        nomatch_ok=all(not r.ranked for r in nomatch),
        score_by_kind=by_kind,
        results=results,
    )


def evaluate(dataset_path: Path = DATASET_PATH) -> Summary:
    dataset = json.loads(dataset_path.read_text())
    return asyncio.run(_evaluate_async(dataset))


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def main() -> int:
    summary = evaluate()
    bad = [r for r in summary.results if r.bad_evidence]
    misses = [r for r in summary.results if r.expected and not r.hit]

    print("Cross-project recall quality — stale/contradiction/project benchmark")
    print(
        f"{summary.n_queries} queries over a {summary.n_corpus}-fact corpus  "
        "·  deterministic, no LLM, no embeddings\n"
    )
    print(f"{'metric':<28}{'value':>8}")
    print("-" * 36)
    print(f"{'quality score':<28}{summary.quality_score:>8.3f}")
    print(f"{'baseline score':<28}{BASELINE_QUALITY_SCORE:>8.3f}")
    print(f"{'absolute gain':<28}{summary.absolute_gain:>8.3f}")
    print(f"{'relative gain':<28}{_pct(summary.relative_gain):>8}")
    print(f"{'recall@1':<28}{_pct(summary.recall_at_1):>8}")
    print(f"{'recall@5':<28}{_pct(summary.recall_at_5):>8}")
    print(f"{'candidate recall':<28}{_pct(summary.hit_rate):>8}")
    print(f"{'bad-evidence query rate':<28}{_pct(summary.bad_evidence_rate):>8}")
    print(f"{'no-match clean':<28}{str(summary.nomatch_ok):>8}")
    print()
    print("mean score by kind:")
    for kind, scores in sorted(summary.score_by_kind.items()):
        print(f"  {kind:<18}{sum(scores) / len(scores):.3f} ({len(scores)} queries)")

    if bad:
        print(f"\nbad evidence surfaced ({len(bad)} queries):")
        for result in bad:
            print(
                f'  [{result.kind}] "{result.query}" -> '
                f"bad {result.bad_evidence}, ranked {result.ranked}"
            )

    if misses:
        print(f"\nmisses ({len(misses)} queries):")
        for result in misses:
            print(
                f'  [{result.kind}] "{result.query}" -> '
                f"expected {result.expected}, ranked {result.ranked}"
            )

    ok = (
        summary.quality_score >= MIN_QUALITY_SCORE
        and summary.relative_gain >= MIN_RELATIVE_GAIN
        and summary.bad_evidence_rate == 0
        and summary.nomatch_ok
    )
    if not ok:
        print(
            "\nGATE FAILED: "
            f"quality={summary.quality_score:.3f} "
            f"(floor {MIN_QUALITY_SCORE:.3f}), "
            f"relative_gain={summary.relative_gain:.3f} "
            f"(floor {MIN_RELATIVE_GAIN:.3f}), "
            f"bad_evidence_rate={summary.bad_evidence_rate:.3f}, "
            f"nomatch_ok={summary.nomatch_ok}"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
