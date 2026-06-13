#!/usr/bin/env python
"""Deterministic recall@k over the labeled prefilter dataset.

This is a *script*, not a CLI subcommand and not part of the shipped surface.
Run it directly:

    uv run python tests/run_evals.py

It measures ONE thing: how well the deterministic keyword prefilter — the free,
zero-LLM first pass of engram's tiered recall — surfaces the labeled fact for a
fresh, paraphrased query. It does NOT measure the LLM search/synthesis tiers;
those are non-deterministic and cost spend. Queries the prefilter misses at
tier-0 escalate to the LLM in production, which this script does not exercise.

Determinism: every query runs through ``recall_with_provenance`` against a temp
store, but ``complete_with_usage`` is replaced with a no-op stub, so there are
zero LLM calls and no API keys are needed. The metrics read only the
deterministic prefilter artifacts:

- ``seen``  = ``cited_fact_ids | {m.id for m in prefilter_matches if above_floor}``
              (the harness's existing relevant-set definition in evals.py)
- ``ranked`` = above-floor prefilter matches in score-descending order

Metrics, computed over the answerable queries (those with a non-empty label):

- hit-rate : a labeled fact appears anywhere in ``seen`` (any rank)
- recall@1 : a labeled fact is the top-ranked above-floor match
- recall@5 : a labeled fact is within the top 5 ranked above-floor matches
- MRR      : mean reciprocal rank of the first labeled fact

The single no-match query (empty label) is checked separately: its above-floor
set must be empty.

The script exits non-zero if the no-match case regresses or if recall drops
below the floors below, so ``uv run python tests/run_evals.py`` is a real gate.
``tests/test_recall_evals.py`` enforces the same floors inside pytest/CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pydantic import BaseModel, Field

DATASET_PATH = Path(__file__).parent / "recall_eval_dataset.json"

# Regression floors. Set conservatively below the measured numbers (recall@1
# ≈0.51, recall@5 ≈0.72) so an honest scorer tweak won't flap, but a real
# prefilter regression trips the gate. The metric is fully deterministic.
MIN_RECALL_AT_1 = 0.45
MIN_RECALL_AT_5 = 0.65


class LabeledQuery(BaseModel):
    query: str
    expected: list[str] = Field(default_factory=list)
    kind: str = ""
    note: str = ""


class QueryResult(BaseModel):
    query: str
    kind: str
    expected: list[str]
    tier: int
    llm_calls: int
    seen: list[str]
    ranked: list[str]
    hit: bool
    rank: int | None  # 1-based rank of the first labeled fact in ``ranked``
    recall_at_1: bool
    recall_at_5: bool
    reciprocal_rank: float


class Summary(BaseModel):
    n_corpus: int
    n_answerable: int
    hit_rate: float
    recall_at_1: float
    recall_at_5: float
    mrr: float
    tier0_fraction: float  # share of ALL queries resolved at tier-0 (zero LLM)
    llm_free_fraction: float  # share of ALL queries that made zero LLM calls
    tier_counts: dict[int, int]
    nomatch_ok: bool
    results: list[QueryResult]


def _no_op_completion():
    """A deterministic stand-in for ``complete_with_usage``.

    Returns innocuous text with no fact IDs and no project tokens, so any query
    that escalates past tier-0 produces empty ``cited_fact_ids`` and the metric
    reduces to the pure deterministic prefilter. Records every call so we can
    report how many queries would have spent on the LLM.
    """
    from engram.llm import Completion

    calls = {"n": 0}

    async def fake(
        prompt,
        system="",
        model=None,
        temperature=None,
        response_format=None,
        cache_prefix=None,
    ):
        calls["n"] += 1
        return Completion(
            text="(no answer) [quality: low]", input_tokens=0, cached_tokens=0
        )

    return fake, calls


async def _run_query(store, lq: LabeledQuery) -> QueryResult:
    import engram.retriever as retriever_mod
    from engram.retriever import recall_with_provenance

    fake, calls = _no_op_completion()
    saved = retriever_mod.complete_with_usage
    retriever_mod.complete_with_usage = fake
    try:
        _, _, provenance, _ = await recall_with_provenance(lq.query, store=store)
    finally:
        retriever_mod.complete_with_usage = saved

    above_floor = [m.id for m in provenance.prefilter_matches if m.above_floor]
    seen = set(provenance.cited_fact_ids) | set(above_floor)
    expected = set(lq.expected)

    rank: int | None = None
    for i, fid in enumerate(above_floor, start=1):
        if fid in expected:
            rank = i
            break

    return QueryResult(
        query=lq.query,
        kind=lq.kind,
        expected=lq.expected,
        tier=provenance.tier,
        llm_calls=calls["n"],
        seen=sorted(seen),
        ranked=above_floor,
        hit=bool(expected & seen),
        rank=rank,
        recall_at_1=rank is not None and rank <= 1,
        recall_at_5=rank is not None and rank <= 5,
        reciprocal_rank=(1.0 / rank) if rank else 0.0,
    )


async def _evaluate_async(dataset: dict[str, Any]) -> Summary:
    from engram.evals import EvalFactSpec, _materialize_facts
    from engram.store import FactStore

    corpus = [EvalFactSpec.model_validate(f) for f in dataset["corpus"]]
    queries = [LabeledQuery.model_validate(q) for q in dataset["queries"]]

    with TemporaryDirectory() as tmp_dir:
        store = FactStore(data_dir=Path(tmp_dir))
        store.append_facts(_materialize_facts(corpus))

        results = [await _run_query(store, lq) for lq in queries]

    answerable = [r for r in results if r.expected]
    nomatch = [r for r in results if not r.expected]
    n = len(answerable)

    tier_counts = {0: 0, 1: 0, 2: 0}
    for r in results:
        tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1

    nomatch_ok = all(not r.seen for r in nomatch)

    return Summary(
        n_corpus=len(corpus),
        n_answerable=n,
        hit_rate=sum(r.hit for r in answerable) / n,
        recall_at_1=sum(r.recall_at_1 for r in answerable) / n,
        recall_at_5=sum(r.recall_at_5 for r in answerable) / n,
        mrr=sum(r.reciprocal_rank for r in answerable) / n,
        tier0_fraction=tier_counts.get(0, 0) / len(results),
        llm_free_fraction=sum(r.llm_calls == 0 for r in results) / len(results),
        tier_counts=tier_counts,
        nomatch_ok=nomatch_ok,
        results=results,
    )


def evaluate(dataset_path: Path = DATASET_PATH) -> Summary:
    """Load the dataset and compute the deterministic recall summary."""
    import asyncio

    dataset = json.loads(dataset_path.read_text())
    return asyncio.run(_evaluate_async(dataset))


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def main() -> int:
    summary = evaluate()

    misses = [r for r in summary.results if r.expected and not r.recall_at_5]
    print(
        f"Deterministic prefilter recall  ·  {summary.n_answerable} labeled queries "
        f"over a {summary.n_corpus}-fact corpus  ·  no LLM, no embeddings\n"
    )
    print(f"{'metric':<14}{'value':>8}")
    print("-" * 22)
    print(f"{'hit-rate':<14}{_pct(summary.hit_rate):>8}")
    print(f"{'recall@1':<14}{_pct(summary.recall_at_1):>8}")
    print(f"{'recall@5':<14}{_pct(summary.recall_at_5):>8}")
    print(f"{'MRR':<14}{summary.mrr:>8.2f}")
    print()
    print(
        f"tier-0 (zero LLM): {_pct(summary.tier0_fraction)} of all queries  "
        f"·  llm-free: {_pct(summary.llm_free_fraction)}  "
        f"·  tiers {dict(sorted(summary.tier_counts.items()))}"
    )
    print(
        f"no-match returns nothing above floor: {'ok' if summary.nomatch_ok else 'FAIL'}"
    )

    if misses:
        print(f"\nrecall@5 misses ({len(misses)}):")
        for r in misses:
            where = (
                f"rank {r.rank}"
                if r.rank
                else ("not above floor" if not r.hit else "below top-5")
            )
            print(f'  [{r.kind:<13}] "{r.query}"  → expected {r.expected}, {where}')

    ok = (
        summary.nomatch_ok
        and summary.recall_at_1 >= MIN_RECALL_AT_1
        and summary.recall_at_5 >= MIN_RECALL_AT_5
    )
    if not ok:
        print(
            f"\nGATE FAILED: recall@1={_pct(summary.recall_at_1)} "
            f"(floor {_pct(MIN_RECALL_AT_1)}), recall@5={_pct(summary.recall_at_5)} "
            f"(floor {_pct(MIN_RECALL_AT_5)}), nomatch_ok={summary.nomatch_ok}"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
