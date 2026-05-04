"""Tests for the recall evaluation harness."""

from __future__ import annotations

from engram.config import get_settings
from engram.evals import (
    EvalBudget,
    EvalFactSpec,
    EvalFixture,
    representative_fixtures,
    run_fixture_sync,
)
from engram.models import FactCategory


# ---------------------------------------------------------------------------
# 7.6: passing evals
# ---------------------------------------------------------------------------


def test_eval_passes_when_expected_source_present():
    fixture = EvalFixture(
        name="basic",
        query="prefers tabs over spaces",
        facts=[
            EvalFactSpec(
                id="aaaaaaaaaaaa",
                category=FactCategory.preference,
                content="prefers tabs over spaces",
                tags=["tabs", "spaces"],
            )
        ],
        expected_source_ids=["aaaaaaaaaaaa"],
        budget=EvalBudget(max_tier=0, max_llm_calls=0),
    )
    result = run_fixture_sync(fixture)
    assert result.passed is True
    assert all(c.passed for c in result.checks)
    assert result.tier == 0


def test_eval_fails_when_expected_source_missing():
    fixture = EvalFixture(
        name="missing",
        query="banana",
        facts=[
            EvalFactSpec(
                id="aaaaaaaaaaaa",
                category=FactCategory.preference,
                content="something completely unrelated",
            )
        ],
        expected_source_ids=["aaaaaaaaaaaa"],
        budget=EvalBudget(max_tier=0, max_llm_calls=0),
    )
    result = run_fixture_sync(fixture)
    assert result.passed is False
    failed_names = [c.name for c in result.checks if not c.passed]
    assert any(n.startswith("expected_source:") for n in failed_names)


def test_eval_fails_on_excluded_source_present():
    fixture = EvalFixture(
        name="excluded",
        query="zagblort xylophone",
        facts=[
            EvalFactSpec(
                id="aaaaaaaaaaaa",
                category=FactCategory.preference,
                content="zagblort xylophone preference",
                tags=["zagblort", "xylophone"],
            )
        ],
        excluded_source_ids=["aaaaaaaaaaaa"],
        budget=EvalBudget(max_tier=0),
    )
    result = run_fixture_sync(fixture)
    assert result.passed is False
    failed_names = [c.name for c in result.checks if not c.passed]
    assert any(n.startswith("excluded_source:") for n in failed_names)


# ---------------------------------------------------------------------------
# Budget failures
# ---------------------------------------------------------------------------


def test_eval_fails_on_tier_budget_exceeded(monkeypatch):
    """Force tier-2 by seeding a flat distribution and capping max_tier=0."""
    fixture = EvalFixture(
        name="tier_budget",
        query="retrieval",
        facts=[
            EvalFactSpec(
                id=f"f{i:02d}aaaaaaaa",
                category=FactCategory.preference,
                content=f"retrieval note {i}",
            )
            for i in range(15)
        ],
        budget=EvalBudget(max_tier=0),
        mocked_responses=[
            (
                "## DIRECT\n(none)\n## CONTEXTUAL\n(none)\n## TEMPORAL\n(none)\n",
                100,
                0,
            ),
            ("ok\n[quality: low]", 100, 0),
        ],
    )
    result = run_fixture_sync(fixture)
    assert result.passed is False
    tier_check = next(c for c in result.checks if c.name == "max_tier")
    assert tier_check.passed is False


def test_eval_fails_on_llm_call_budget(monkeypatch):
    monkeypatch.setenv("ENGRAM_TIER2_MODE", "multilens")
    get_settings.cache_clear()
    fixture = EvalFixture(
        name="llm_budget",
        query="retrieval",
        facts=[
            EvalFactSpec(
                id=f"f{i:02d}aaaaaaaa",
                category=FactCategory.preference,
                content=f"retrieval note {i}",
            )
            for i in range(15)
        ],
        budget=EvalBudget(max_llm_calls=1),
        mocked_responses=[
            (
                "## DIRECT\n(none)\n## CONTEXTUAL\n(none)\n## TEMPORAL\n(none)\n",
                100,
                0,
            ),
            ("ok\n[quality: low]", 100, 0),
        ],
    )
    try:
        result = run_fixture_sync(fixture)
    finally:
        monkeypatch.delenv("ENGRAM_TIER2_MODE", raising=False)
        get_settings.cache_clear()
    assert result.passed is False
    call_check = next(c for c in result.checks if c.name == "max_llm_calls")
    assert call_check.passed is False
    assert call_check.actual == 2


# ---------------------------------------------------------------------------
# Provider-backed skip
# ---------------------------------------------------------------------------


def test_provider_eval_skipped_when_disabled(monkeypatch):
    monkeypatch.delenv("ENGRAM_EVAL_PROVIDER", raising=False)
    fixture = EvalFixture(
        name="provider_only",
        query="x",
        mode="provider",
    )
    result = run_fixture_sync(fixture)
    assert result.skipped is True
    assert result.passed is False  # not "passed", but explicitly skipped
    assert "ENGRAM_EVAL_PROVIDER" in (result.skip_reason or "")


def test_provider_eval_runs_when_enabled(monkeypatch):
    """When the env flag is set, provider-mode fixtures still execute deterministically
    if they happen to land in tier-0 (no live calls). This verifies the harness
    plumbing rather than hitting an actual provider."""
    monkeypatch.setenv("ENGRAM_EVAL_PROVIDER", "1")
    fixture = EvalFixture(
        name="provider_tier0",
        query="zagblort xylophone",
        mode="provider",
        facts=[
            EvalFactSpec(
                id="aaaaaaaaaaaa",
                category=FactCategory.preference,
                content="zagblort xylophone preference",
                tags=["zagblort", "xylophone"],
            )
        ],
        expected_source_ids=["aaaaaaaaaaaa"],
        budget=EvalBudget(max_tier=0, max_llm_calls=0),
    )
    result = run_fixture_sync(fixture, enable_provider=True)
    assert result.skipped is False
    assert result.passed is True


# ---------------------------------------------------------------------------
# Representative bundle smoke
# ---------------------------------------------------------------------------


def test_representative_fixtures_at_least_one_passes():
    fixtures = representative_fixtures()
    assert fixtures
    results = [run_fixture_sync(f) for f in fixtures]
    # The project_preference_tier0 fixture is the easiest and must pass.
    pref = next(r for r in results if r.fixture == "project_preference_tier0")
    assert pref.passed is True
