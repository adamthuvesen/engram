#!/usr/bin/env python
"""Deterministic recall@k over the labeled recall dataset.

This is a *script*, not a CLI subcommand and not part of the shipped surface.
Run it directly:

    uv run python tests/run_evals.py

It measures the deterministic lexical pass (BM25 search plus the relevance
bar) that produces engram's default memory cards with zero LLM calls, over a
*representative* query mix: terse literal lookups (where lexical search wins)
plus paraphrased and pure-synonym questions (where it can fall short and, with
an LLM key, a zero-hit query spends one selection call instead).

Determinism: every query runs through ``recall_with_provenance`` against a temp
store with ``_llm_available`` pinned False, so there are zero LLM calls and no
API keys are needed. The metrics read only deterministic artifacts:

- ``ranked`` = hits that cleared the relevance bar, in score order (the cards)
- ``pool``   = the facts recall acts on: the cards, or when no hit cleared the
               bar, the top ``ZERO_HIT_MAX_CANDIDATES`` hits that one LLM
               selection call chooses from when a key is configured

Metrics, computed over the answerable queries (those with a non-empty label):

- hit-rate : a labeled fact is in ``pool`` (reachable by the path recall takes)
- recall@5 : a labeled fact is within the top 5 cards
- recall@1 : a labeled fact is the top card
- MRR      : mean reciprocal rank of the first labeled fact

``lexical_fraction`` is the share of ALL queries that resolve without an LLM
call even when a key is configured (some card cleared the bar, or the search
found nothing at all). The no-match queries (empty label) are checked
separately: their relevant set must be empty.

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

# Regression floors match the recall-performance mission gate. Deterministic.
# Synonym/semantic queries are deliberately lexical misses (LLM territory), so
# the card-level floors sit well below 1.0; literal and paraphrase queries are
# held to 0.9 recall@1 in tests/test_recall_evals.py.
MIN_RECALL_AT_1 = 0.64
MIN_RECALL_AT_5 = 0.69
MIN_HIT_RATE = 0.87
MIN_MRR = 0.66
MIN_LEXICAL_FRACTION = 0.75


class LabeledQuery(BaseModel):
    query: str
    expected: list[str] = Field(default_factory=list)
    kind: str = ""
    note: str = ""


class QueryResult(BaseModel):
    query: str
    kind: str
    expected: list[str]
    lexical: bool  # resolved without an LLM call even with a key
    pool: list[str]
    ranked: list[str]
    hit: bool
    rank: int | None  # 1-based rank of the first labeled fact in ``ranked``
    recall_at_1: bool
    recall_at_5: bool
    reciprocal_rank: float


class Summary(BaseModel):
    n_corpus: int
    n_queries: int
    n_answerable: int
    n_nomatch: int
    hit_rate: float
    recall_at_1: float
    recall_at_5: float
    mrr: float
    lexical_fraction: float  # share of ALL queries needing no LLM call with a key
    recall1_by_kind: dict[str, list[int]]  # kind -> [recall@1 hits, total]
    nomatch_ok: bool
    results: list[QueryResult]


async def _run_query(store, lq: LabeledQuery) -> QueryResult:
    import engram.recall.retriever as retriever_mod
    from engram.recall.retriever import ZERO_HIT_MAX_CANDIDATES, recall_with_provenance

    saved_available = retriever_mod._llm_available
    # Pin the LLM off: this script measures the deterministic lexical pass and
    # must report the same numbers with or without a local key.
    retriever_mod._llm_available = lambda: False
    try:
        _, _, provenance, _ = await recall_with_provenance(lq.query, store=store)
    finally:
        retriever_mod._llm_available = saved_available

    ranked = [m.id for m in provenance.prefilter_matches if m.above_floor]
    pool = ranked or [
        m.id for m in provenance.prefilter_matches[:ZERO_HIT_MAX_CANDIDATES]
    ]
    expected = set(lq.expected)

    rank: int | None = None
    for i, fid in enumerate(ranked, start=1):
        if fid in expected:
            rank = i
            break

    decision = provenance.selected_decision
    return QueryResult(
        query=lq.query,
        kind=lq.kind,
        expected=lq.expected,
        lexical=decision.relevant_count > 0 or provenance.prefilter_count == 0,
        pool=pool,
        ranked=ranked,
        hit=bool(expected & set(pool)),
        rank=rank,
        recall_at_1=rank is not None and rank <= 1,
        recall_at_5=rank is not None and rank <= 5,
        reciprocal_rank=(1.0 / rank) if rank else 0.0,
    )


async def _evaluate_async(dataset: dict[str, Any]) -> Summary:
    from engram.recall.evals import EvalFactSpec, _materialize_facts
    from engram.storage.store import FactStore

    corpus = [EvalFactSpec.model_validate(f) for f in dataset["corpus"]]
    queries = [LabeledQuery.model_validate(q) for q in dataset["queries"]]

    with TemporaryDirectory() as tmp_dir:
        store = FactStore(data_dir=Path(tmp_dir))
        store.append_facts(_materialize_facts(corpus))

        results = [await _run_query(store, lq) for lq in queries]

    answerable = [r for r in results if r.expected]
    nomatch = [r for r in results if not r.expected]
    n = len(answerable)

    recall1_by_kind: dict[str, list[int]] = {}
    for r in answerable:
        bucket = recall1_by_kind.setdefault(r.kind, [0, 0])
        bucket[1] += 1
        if r.recall_at_1:
            bucket[0] += 1

    nomatch_ok = all(not r.ranked for r in nomatch)

    return Summary(
        n_corpus=len(corpus),
        n_queries=len(results),
        n_answerable=n,
        n_nomatch=len(nomatch),
        hit_rate=sum(r.hit for r in answerable) / n,
        recall_at_1=sum(r.recall_at_1 for r in answerable) / n,
        recall_at_5=sum(r.recall_at_5 for r in answerable) / n,
        mrr=sum(r.reciprocal_rank for r in answerable) / n,
        lexical_fraction=sum(r.lexical for r in results) / len(results),
        recall1_by_kind=recall1_by_kind,
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
        f"Deterministic lexical recall — representative query mix\n"
        f"{summary.n_answerable} answerable labeled queries + "
        f"{summary.n_nomatch} no-match queries over a {summary.n_corpus}-fact "
        f"corpus  ·  no LLM, no embeddings\n"
    )
    print(
        f"{_pct(summary.lexical_fraction)} of queries resolve with zero LLM calls "
        f"even when a key is configured\n"
    )
    print(f"{'metric':<26}{'value':>8}")
    print("-" * 34)
    print(f"{'recall@1':<26}{_pct(summary.recall_at_1):>8}")
    print(f"{'recall@5':<26}{_pct(summary.recall_at_5):>8}")
    print(f"{'candidate recall (hit-rate)':<26}{_pct(summary.hit_rate):>8}")
    print(f"{'MRR':<26}{summary.mrr:>8.2f}")
    print()
    print(
        "recall@1 by query kind (where lexical search wins vs. where the LLM earns it):"
    )
    for kind, (hits, total) in sorted(summary.recall1_by_kind.items()):
        print(f"  {kind:<14}{hits:>3}/{total:<3}  {_pct(hits / total):>4}")
    print(f"\nno-match returns no cards: {'ok' if summary.nomatch_ok else 'FAIL'}")

    if misses:
        print(f"\nrecall@5 misses ({len(misses)}) — the LLM's territory:")
        for r in misses:
            where = (
                f"rank {r.rank}"
                if r.rank
                else ("no card" if not r.hit else "below top-5")
            )
            print(f'  [{r.kind:<13}] "{r.query}"  → expected {r.expected}, {where}')

    ok = (
        summary.nomatch_ok
        and summary.recall_at_1 >= MIN_RECALL_AT_1
        and summary.recall_at_5 >= MIN_RECALL_AT_5
        and summary.hit_rate >= MIN_HIT_RATE
        and summary.mrr >= MIN_MRR
        and summary.lexical_fraction >= MIN_LEXICAL_FRACTION
    )
    if not ok:
        print(
            f"\nGATE FAILED: recall@1={_pct(summary.recall_at_1)} "
            f"(floor {_pct(MIN_RECALL_AT_1)}), recall@5={_pct(summary.recall_at_5)} "
            f"(floor {_pct(MIN_RECALL_AT_5)}), hit-rate={_pct(summary.hit_rate)} "
            f"(floor {_pct(MIN_HIT_RATE)}), MRR={summary.mrr:.2f} "
            f"(floor {MIN_MRR:.2f}), lexical={_pct(summary.lexical_fraction)} "
            f"(floor {_pct(MIN_LEXICAL_FRACTION)}), nomatch_ok={summary.nomatch_ok}"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
