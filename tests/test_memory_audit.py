"""Tests for read-only memory audit suggestions."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from engram.maintenance.memory_audit import audit_facts, format_audit_result
from engram.core.models import Fact, FactCategory
from engram.operations import audit_memories
from engram.storage.store import FactStore


NOW = datetime(2026, 6, 19, tzinfo=timezone.utc)


def _fact(
    fact_id: str,
    content: str,
    *,
    category: FactCategory = FactCategory.project,
    project: str | None = "engram",
    observed_days_ago: int = 0,
    expires_days_ago: int | None = None,
) -> Fact:
    observed_at = NOW - timedelta(days=observed_days_ago)
    expires_at = None
    if expires_days_ago is not None:
        expires_at = NOW - timedelta(days=expires_days_ago)
    return Fact(
        id=fact_id,
        category=category,
        content=content,
        project=project,
        observed_at=observed_at,
        updated_at=observed_at,
        expires_at=expires_at,
        tags=["audit"],
    )


def test_audit_finds_duplicate_stale_and_contradiction_suggestions():
    facts = [
        _fact(
            "dup-jsonl-a",
            "Engram persists memories as an append-only JSONL event log on disk",
        ),
        _fact(
            "dup-jsonl-b",
            "Engram stores memories in an append-only JSONL event log on disk",
        ),
        _fact(
            "stale-auth",
            "Currently using the temporary auth migration branch",
            category=FactCategory.workflow,
            observed_days_ago=45,
        ),
        _fact(
            "pref-polars",
            "The user prefers polars over pandas for dataframes",
            category=FactCategory.preference,
            project=None,
        ),
        _fact(
            "pref-pandas",
            "The user prefers pandas over polars for dataframes",
            category=FactCategory.preference,
            project=None,
        ),
        _fact("clean-mcp", "Engram exposes its tools through FastMCP"),
    ]

    result = audit_facts(facts, now=NOW)
    by_kind = {suggestion.kind: suggestion for suggestion in result.suggestions}

    assert result.total_analyzed == len(facts)
    assert set(by_kind) == {"duplicate", "stale", "contradiction"}
    assert by_kind["duplicate"].action == "merge_memories"
    assert set(by_kind["duplicate"].fact_ids) == {"dup-jsonl-a", "dup-jsonl-b"}
    assert by_kind["stale"].action == "mark_stale"
    assert by_kind["stale"].fact_ids == ["stale-auth"]
    assert by_kind["contradiction"].action == "review_contradiction"
    assert set(by_kind["contradiction"].fact_ids) == {"pref-polars", "pref-pandas"}


def test_audit_does_not_classify_reversed_preferences_as_duplicates():
    facts = [
        _fact(
            "pref-polars",
            "The user prefers polars over pandas for dataframes",
            category=FactCategory.preference,
        ),
        _fact(
            "pref-pandas",
            "The user prefers pandas over polars for dataframes",
            category=FactCategory.preference,
        ),
    ]

    result = audit_facts(facts, now=NOW)

    assert [suggestion.kind for suggestion in result.suggestions] == ["contradiction"]


def test_audit_respects_project_boundaries_for_duplicates():
    facts = [
        _fact("alpha-fact", "Use Snowflake for the warehouse", project="alpha"),
        _fact("beta-fact", "Use Snowflake for the warehouse", project="beta"),
    ]

    result = audit_facts(facts, now=NOW)

    assert result.suggestions == []


def test_audit_flags_expired_facts_without_mutating_store(tmp_path):
    store = FactStore(data_dir=tmp_path)
    expired = _fact(
        "expired-freeze",
        "Mobile release freeze runs until 2026-05-01",
        category=FactCategory.event,
        project=None,
        expires_days_ago=10,
    )
    store.append_facts([expired])

    result = asyncio.run(audit_memories(store=store))

    assert result.envelope.data["stale_facts"] == 1
    assert store.load_facts()[0].stale is False
    assert store.load_facts()[0].confidence == 1.0


def test_expired_facts_do_not_participate_in_duplicate_groups():
    facts = [
        _fact(
            "expired-jsonl",
            "Engram stores memories in an append-only JSONL event log on disk",
            expires_days_ago=1,
        ),
        _fact(
            "active-jsonl",
            "Engram stores memories in an append-only JSONL event log on disk",
        ),
    ]

    result = audit_facts(facts, now=NOW)

    assert [suggestion.kind for suggestion in result.suggestions] == ["stale"]


def test_format_audit_result_states_no_changes_applied():
    result = audit_facts(
        [
            _fact("first-dup", "Engram stores facts as JSONL event logs"),
            _fact("second-dup", "Engram stores facts as JSONL event logs"),
        ],
        now=NOW,
    )

    output = format_audit_result(result)

    assert "No changes were applied." in output
    assert "review command: engram merge-memories" in output
