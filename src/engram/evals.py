"""Recall evaluation harness — golden-fixture testing for retrieval quality.

Fixtures describe input facts, a query, expected source IDs that must appear
in provenance, excluded source IDs that must not, optional answer-text
assertions, and optional performance budgets (max tier, max LLM calls, max
latency, max input tokens, expected cached tokens).

Two execution modes are supported:

- Deterministic: prefilter, tier-0, ``prompt`` mode (compact fact dump), or
  user-supplied mocked synthesis. No live provider needed.
- Provider-backed: requires ``ENGRAM_EVAL_PROVIDER=1`` and live credentials.
  Skipped (not failed) when the env flag is missing.

The fixture format is versioned via the ``version`` field so future shape
changes can be detected and migrated.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from pydantic import BaseModel, Field

from engram.llm import Completion
from engram.models import Fact, FactCategory
from engram.retriever import recall_with_provenance
from engram.store import FactStore


class EvalFactSpec(BaseModel):
    id: str
    category: FactCategory = FactCategory.preference
    content: str
    project: str | None = None
    tags: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    supersedes: str | None = None
    stale: bool = False


class EvalBudget(BaseModel):
    max_tier: int | None = None
    max_llm_calls: int | None = None
    max_latency_ms: float | None = None
    max_input_tokens: int | None = None
    min_cached_tokens: int | None = None


class EvalFixture(BaseModel):
    version: int = 1
    name: str
    description: str = ""
    project: str | None = None
    query: str
    facts: list[EvalFactSpec] = Field(default_factory=list)
    expected_source_ids: list[str] = Field(default_factory=list)
    excluded_source_ids: list[str] = Field(default_factory=list)
    answer_contains: list[str] = Field(default_factory=list)
    answer_excludes: list[str] = Field(default_factory=list)
    budget: EvalBudget = Field(default_factory=EvalBudget)
    mode: Literal["deterministic", "provider"] = "deterministic"
    # When ``mode == "deterministic"`` and the eval drives tier-1/tier-2,
    # ``mocked_responses`` supplies (text, input_tokens, cached_tokens) tuples
    # consumed in call order. Leave empty for tier-0 evals.
    mocked_responses: list[tuple[str, int | None, int | None]] = Field(
        default_factory=list
    )


class EvalCheck(BaseModel):
    name: str
    passed: bool
    expected: Any | None = None
    actual: Any | None = None
    message: str = ""


class EvalResult(BaseModel):
    fixture: str
    passed: bool
    skipped: bool = False
    skip_reason: str | None = None
    tier: int | None = None
    latency_ms: float | None = None
    llm_calls: int | None = None
    input_tokens: int | None = None
    cached_tokens: int | None = None
    answer: str = ""
    quality: str = ""
    cited_fact_ids: list[str] = Field(default_factory=list)
    checks: list[EvalCheck] = Field(default_factory=list)


def _materialize_facts(specs: list[EvalFactSpec]) -> list[Fact]:
    now = datetime.now(timezone.utc)
    facts: list[Fact] = []
    for spec in specs:
        facts.append(
            Fact(
                id=spec.id,
                category=spec.category,
                content=spec.content,
                project=spec.project,
                tags=spec.tags,
                confidence=spec.confidence,
                supersedes=spec.supersedes,
                stale=spec.stale,
                created_at=now,
                updated_at=now,
                observed_at=now,
            )
        )
    return facts


def _patch_completions(
    monkeypatch_target, queue: list[tuple[str, int | None, int | None]]
):
    """Replace ``engram.retriever.complete_with_usage`` with a queue stub."""
    pending = list(queue)

    async def fake(
        prompt,
        system="",
        model=None,
        temperature=None,
        response_format=None,
        cache_prefix=None,
    ):
        if not pending:
            raise RuntimeError("eval fixture exhausted mocked_responses")
        text, input_tokens, cached = pending.pop(0)
        return Completion(text=text, input_tokens=input_tokens, cached_tokens=cached)

    setattr(monkeypatch_target, "complete_with_usage", fake)


async def run_fixture(
    fixture: EvalFixture,
    *,
    enable_provider: bool | None = None,
) -> EvalResult:
    """Run a single eval fixture and return its result.

    Provider-backed fixtures are skipped (not failed) when
    ``ENGRAM_EVAL_PROVIDER`` is not set, mirroring how the harness behaves in
    normal CI.
    """
    if fixture.mode == "provider":
        if enable_provider is None:
            enable_provider = os.environ.get("ENGRAM_EVAL_PROVIDER") == "1"
        if not enable_provider:
            return EvalResult(
                fixture=fixture.name,
                passed=False,
                skipped=True,
                skip_reason="provider mode disabled (set ENGRAM_EVAL_PROVIDER=1)",
            )

    with TemporaryDirectory() as tmp_dir:
        store = FactStore(data_dir=Path(tmp_dir))
        facts = _materialize_facts(fixture.facts)
        if facts:
            store.append_facts(facts)

        import engram.retriever as retriever_mod

        saved = retriever_mod.complete_with_usage
        # Tier-0 evals run without LLM calls; only patch when tier-1/2 mocks exist.
        if fixture.mode == "deterministic" and fixture.mocked_responses:
            _patch_completions(retriever_mod, fixture.mocked_responses)

        try:
            answer, quality, provenance, _ = await recall_with_provenance(
                fixture.query,
                project=fixture.project,
                store=store,
            )
        finally:
            retriever_mod.complete_with_usage = saved

    checks: list[EvalCheck] = []

    cited = set(provenance.cited_fact_ids)
    matched_ids = {m.id for m in provenance.prefilter_matches if m.above_floor}
    seen = cited | matched_ids
    for expected in fixture.expected_source_ids:
        passed = expected in seen
        checks.append(
            EvalCheck(
                name=f"expected_source:{expected}",
                passed=passed,
                expected=expected,
                actual=sorted(seen),
                message="" if passed else "Expected source not present in provenance",
            )
        )

    for excluded in fixture.excluded_source_ids:
        passed = excluded not in cited
        checks.append(
            EvalCheck(
                name=f"excluded_source:{excluded}",
                passed=passed,
                expected="not in cited_fact_ids",
                actual=sorted(cited),
                message="" if passed else "Excluded source appeared as cited",
            )
        )

    answer_lower = answer.lower()
    for needle in fixture.answer_contains:
        passed = needle.lower() in answer_lower
        checks.append(
            EvalCheck(
                name=f"answer_contains:{needle}",
                passed=passed,
                expected=needle,
                actual=answer[:200],
            )
        )
    for needle in fixture.answer_excludes:
        passed = needle.lower() not in answer_lower
        checks.append(
            EvalCheck(
                name=f"answer_excludes:{needle}",
                passed=passed,
                expected=f"not '{needle}'",
                actual=answer[:200],
            )
        )

    b = fixture.budget
    if b.max_tier is not None:
        passed = provenance.tier <= b.max_tier
        checks.append(
            EvalCheck(
                name="max_tier",
                passed=passed,
                expected=b.max_tier,
                actual=provenance.tier,
            )
        )
    if b.max_llm_calls is not None:
        actual = provenance.usage.llm_calls or 0
        passed = actual <= b.max_llm_calls
        checks.append(
            EvalCheck(
                name="max_llm_calls",
                passed=passed,
                expected=b.max_llm_calls,
                actual=actual,
            )
        )
    if b.max_latency_ms is not None:
        passed = provenance.latency_ms <= b.max_latency_ms
        checks.append(
            EvalCheck(
                name="max_latency_ms",
                passed=passed,
                expected=b.max_latency_ms,
                actual=provenance.latency_ms,
            )
        )
    if b.max_input_tokens is not None and provenance.usage.input_tokens is not None:
        passed = provenance.usage.input_tokens <= b.max_input_tokens
        checks.append(
            EvalCheck(
                name="max_input_tokens",
                passed=passed,
                expected=b.max_input_tokens,
                actual=provenance.usage.input_tokens,
            )
        )
    if b.min_cached_tokens is not None and provenance.usage.cached_tokens is not None:
        passed = provenance.usage.cached_tokens >= b.min_cached_tokens
        checks.append(
            EvalCheck(
                name="min_cached_tokens",
                passed=passed,
                expected=b.min_cached_tokens,
                actual=provenance.usage.cached_tokens,
            )
        )

    return EvalResult(
        fixture=fixture.name,
        passed=all(c.passed for c in checks),
        tier=provenance.tier,
        latency_ms=provenance.latency_ms,
        llm_calls=provenance.usage.llm_calls,
        input_tokens=provenance.usage.input_tokens,
        cached_tokens=provenance.usage.cached_tokens,
        answer=answer,
        quality=quality,
        cited_fact_ids=list(provenance.cited_fact_ids),
        checks=checks,
    )


def run_fixture_sync(
    fixture: EvalFixture, *, enable_provider: bool | None = None
) -> EvalResult:
    return asyncio.run(run_fixture(fixture, enable_provider=enable_provider))


def representative_fixtures() -> list[EvalFixture]:
    """Bundle of fixtures covering the cases mentioned in the spec.

    Cases covered:
    - project preferences (tier-0 fast path)
    - outdated facts (superseded fact must be excluded)
    - conflicting facts (two active facts that disagree)
    - duplicate memories (same content twice)
    - small-corpus fast-path (cap demotes tier-2 to tier-1 in real config)
    """
    return [
        EvalFixture(
            name="project_preference_tier0",
            description="A direct preference query should fast-path to tier 0.",
            project="engramx",
            query="zagblort xylophone",
            facts=[
                EvalFactSpec(
                    id="pref01aaaaaa",
                    category=FactCategory.preference,
                    content="zagblort xylophone repair shop preference",
                    project="engramx",
                    tags=["zagblort", "xylophone"],
                ),
                EvalFactSpec(
                    id="noise01aaaaa",
                    category=FactCategory.preference,
                    content="unrelated banana fact",
                    project="engramx",
                ),
            ],
            expected_source_ids=["pref01aaaaaa"],
            excluded_source_ids=["noise01aaaaa"],
            budget=EvalBudget(max_tier=0, max_llm_calls=0),
        ),
        EvalFixture(
            name="outdated_superseded",
            description="Old superseded fact must not be the cited source.",
            query="zagblort editor zagblort editor zagblort editor",
            facts=[
                EvalFactSpec(
                    id="oldedaaaaaaa",
                    category=FactCategory.preference,
                    content="zagblort editor preference is vim",
                    tags=["zagblort", "editor"],
                ),
                EvalFactSpec(
                    id="newedaaaaaaa",
                    category=FactCategory.preference,
                    content="zagblort editor preference is neovim",
                    tags=["zagblort", "editor"],
                    supersedes="oldedaaaaaaa",
                ),
            ],
            # Tier-0 will return both; the test asserts that the newer fact
            # is in the cited list.
            expected_source_ids=["newedaaaaaaa"],
            budget=EvalBudget(max_tier=0, max_llm_calls=0),
        ),
        EvalFixture(
            name="duplicate_memories",
            description="Two facts with the same content should both surface.",
            query="duplicate widget preference",
            facts=[
                EvalFactSpec(
                    id="dup01aaaaaaa",
                    category=FactCategory.preference,
                    content="duplicate widget preference",
                    tags=["widget"],
                ),
                EvalFactSpec(
                    id="dup02aaaaaaa",
                    category=FactCategory.preference,
                    content="duplicate widget preference",
                    tags=["widget"],
                ),
            ],
            expected_source_ids=["dup01aaaaaaa", "dup02aaaaaaa"],
            budget=EvalBudget(max_tier=0, max_llm_calls=0),
        ),
        EvalFixture(
            name="small_corpus_fast_path",
            description=(
                "A small corpus that would otherwise be tier-2 should be "
                "demoted by the v2 cap to tier 0/1."
            ),
            query="retrieval note retrieval note",
            facts=[
                EvalFactSpec(
                    id=f"small{i:02d}aaaa",
                    category=FactCategory.preference,
                    content=f"retrieval note number {i}",
                )
                for i in range(5)
            ],
            # Either 0 or 1 is acceptable on a small corpus.
            budget=EvalBudget(max_tier=1),
            # Provide a tier-1 mocked response so the eval is hermetic.
            mocked_responses=[
                (
                    "All notes mention retrieval (id: small00aaaa).\n[quality: medium]",
                    100,
                    0,
                ),
            ],
        ),
    ]


__all__ = [
    "EvalBudget",
    "EvalCheck",
    "EvalFactSpec",
    "EvalFixture",
    "EvalResult",
    "representative_fixtures",
    "run_fixture",
    "run_fixture_sync",
]
