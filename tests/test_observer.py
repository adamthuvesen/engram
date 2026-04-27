"""Tests for memory suggestion and extraction flows."""

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engram.models import CandidateStatus, Fact, FactCategory
from engram.observer import _dedup, _find_near_matches, extract_facts, suggest_memories
from engram.store import AsyncFactStore, FactStore


def _make_store() -> FactStore:
    tmp = Path(tempfile.mkdtemp())
    return FactStore(data_dir=tmp)


def test_suggest_memories_queues_pending_candidates(monkeypatch):
    store = _make_store()

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        if "Classify each new fact" in prompt:
            return {"new": [0], "updates": [], "duplicates": []}
        return {
            "facts": [
                {
                    "content": "The user prefers concise summaries",
                    "category": "preference",
                    "tags": ["style"],
                    "why_store": "Useful for future responses",
                    "expires_at": None,
                }
            ]
        }

    monkeypatch.setattr("engram.observer.complete_json", fake_complete_json)

    candidates = asyncio.run(
        suggest_memories(
            "Remember that the user prefers concise summaries.",
            store=store,
        )
    )

    assert len(candidates) == 1
    assert candidates[0].status == CandidateStatus.pending
    assert candidates[0].why_store == "Useful for future responses"

    loaded = store.load_candidates(status=CandidateStatus.pending)
    assert len(loaded) == 1
    assert loaded[0].content == "The user prefers concise summaries"


def test_suggest_memories_accepts_async_store(monkeypatch):
    store = _make_store()

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        return {
            "facts": [
                {
                    "content": "The user prefers async storage",
                    "category": "preference",
                    "tags": ["storage"],
                    "why_store": "Guides storage changes",
                    "expires_at": None,
                }
            ]
        }

    monkeypatch.setattr("engram.observer.complete_json", fake_complete_json)

    candidates = asyncio.run(
        suggest_memories("User prefers async storage.", store=AsyncFactStore(store))
    )

    assert len(candidates) == 1
    assert store.load_candidates(status=CandidateStatus.pending)[0].content == (
        "The user prefers async storage"
    )


def test_extract_facts_marks_updates_and_softens_superseded_fact(monkeypatch):
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="oldfact",
                category=FactCategory.preference,
                content="The user prefers pandas",
            )
        ]
    )

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        if "Classify each new fact" in prompt:
            return {
                "new": [],
                "updates": [{"new_idx": 0, "existing_id": "oldfact"}],
                "duplicates": [],
            }
        return {
            "facts": [
                {
                    "content": "The user prefers polars",
                    "category": "preference",
                    "tags": ["python"],
                    "why_store": "Reflects the current dataframe preference",
                    "expires_at": None,
                }
            ]
        }

    monkeypatch.setattr("engram.observer.complete_json", fake_complete_json)

    facts = asyncio.run(extract_facts("The user prefers polars now.", store=store))

    assert len(facts) == 1
    assert facts[0].supersedes == "oldfact"

    all_facts = store.load_facts()
    original = next(f for f in all_facts if f.id == "oldfact")
    updated = next(f for f in all_facts if f.id != "oldfact")
    assert original.confidence == 0.3
    assert updated.supersedes == "oldfact"


def test_extract_facts_accepts_async_store_for_dedup_and_persist(monkeypatch):
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="oldfact",
                category=FactCategory.preference,
                content="The user prefers sync storage",
            )
        ]
    )

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        if "Classify each new fact" in prompt:
            return {
                "new": [],
                "updates": [{"new_idx": 0, "existing_id": "oldfact"}],
                "duplicates": [],
            }
        return {
            "facts": [
                {
                    "content": "The user prefers async storage",
                    "category": "preference",
                    "tags": ["storage"],
                    "why_store": "Reflects current architecture",
                    "expires_at": None,
                }
            ]
        }

    monkeypatch.setattr("engram.observer.complete_json", fake_complete_json)

    facts = asyncio.run(
        extract_facts("User prefers async storage.", store=AsyncFactStore(store))
    )

    assert len(facts) == 1
    assert facts[0].supersedes == "oldfact"
    old = next(f for f in store.load_facts() if f.id == "oldfact")
    assert old.confidence == 0.3


def test_dedup_ignores_malformed_update_entries(monkeypatch):
    existing = [
        Fact(
            id="oldfact",
            category=FactCategory.preference,
            content="The user prefers pandas",
        )
    ]
    candidates = [
        Fact(
            category=FactCategory.preference,
            content="The user prefers polars",
        )
    ]

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        return {
            "new": [],
            "updates": [
                {"new_idx": 99, "existing_id": "oldfact"},
                {"new_idx": "0", "existing_id": "oldfact"},
                {"new_idx": 0},
            ],
            "duplicates": [],
        }

    monkeypatch.setattr("engram.observer.complete_json", fake_complete_json)

    kept = asyncio.run(_dedup(candidates, existing, store=None))

    assert kept == candidates


def test_extract_facts_dedup_respects_project_scope(monkeypatch):
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="beta-fact",
                category=FactCategory.preference,
                content="The user prefers polars",
                project="beta",
            )
        ]
    )

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        if "Classify each new fact" in prompt:
            raise AssertionError("Different project scopes should not be deduped")
        return {
            "facts": [
                {
                    "content": "The user prefers polars",
                    "category": "preference",
                    "tags": ["python"],
                    "why_store": "Library preference",
                    "expires_at": None,
                }
            ]
        }

    monkeypatch.setattr("engram.observer.complete_json", fake_complete_json)

    facts = asyncio.run(
        extract_facts("User prefers polars.", project="alpha", store=store)
    )

    assert len(facts) == 1
    assert facts[0].project == "alpha"
    assert len(store.load_active_facts()) == 2


def test_near_match_is_computed_per_candidate():
    existing = [
        Fact(
            id="oldfact",
            category=FactCategory.preference,
            content="The user prefers pandas dataframes",
        )
    ]
    candidates = [
        Fact(
            category=FactCategory.preference,
            content="The user prefers polars dataframes",
        ),
        Fact(
            category=FactCategory.workflow,
            content="Deployment uses kubernetes nginx docker compose staging production",
        ),
        Fact(
            category=FactCategory.project,
            content="Billing service sends invoices through Stripe webhooks",
        ),
    ]

    near = _find_near_matches(candidates, existing)

    assert near == existing


def test_extract_facts_dedups_against_older_than_200_exact_match(monkeypatch):
    store = _make_store()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    old_duplicate = Fact(
        id="old-duplicate",
        category=FactCategory.preference,
        content="The user prefers polars",
        updated_at=base,
    )
    newer_facts = [
        Fact(
            id=f"newer-{i}",
            category=FactCategory.preference,
            content=f"Unique newer fact {i}",
            updated_at=base + timedelta(days=i + 1),
        )
        for i in range(200)
    ]
    store.append_facts([old_duplicate, *newer_facts])

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        if "Classify each new fact" in prompt:
            raise AssertionError("Exact duplicate should not need LLM dedup")
        return {
            "facts": [
                {
                    "content": "The user prefers polars",
                    "category": "preference",
                    "tags": ["python"],
                    "why_store": "Library preference",
                    "expires_at": None,
                }
            ]
        }

    monkeypatch.setattr("engram.observer.complete_json", fake_complete_json)

    facts = asyncio.run(extract_facts("User prefers polars.", store=store))

    assert facts == []
    assert len(store.load_active_facts()) == 201


def test_extract_facts_skips_fact_missing_content(monkeypatch):
    """A fact without a content field is skipped; others in the batch are extracted."""
    store = _make_store()

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        return {
            "facts": [
                {
                    "category": "preference",
                    "tags": [],
                    "why_store": "no content here",
                    # 'content' is intentionally missing
                },
                {
                    "content": "The user prefers dark mode",
                    "category": "preference",
                    "tags": ["ui"],
                    "why_store": "UX preference",
                    "expires_at": None,
                },
            ]
        }

    monkeypatch.setattr("engram.observer.complete_json", fake_complete_json)

    facts = asyncio.run(extract_facts("User likes dark mode.", store=store))

    assert len(facts) == 1
    assert facts[0].content == "The user prefers dark mode"


def test_dedup_against_candidates_ignores_rejected(monkeypatch):
    """Rejected candidates do not block deduplication of new facts."""
    store = _make_store()

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        # No dedup needed from existing active facts
        if "Classify each new fact" in prompt:
            return {"new": [0], "updates": [], "duplicates": []}
        return {
            "facts": [
                {
                    "content": "The user prefers polars",
                    "category": "preference",
                    "tags": ["python"],
                    "why_store": "Library preference",
                    "expires_at": None,
                }
            ]
        }

    monkeypatch.setattr("engram.observer.complete_json", fake_complete_json)

    from engram.models import MemoryCandidate

    # Seed 3 rejected candidates with the same content
    rejected = [
        MemoryCandidate(
            category=FactCategory.preference,
            content="The user prefers polars",
            status=CandidateStatus.rejected,
        )
        for _ in range(3)
    ]
    store.append_candidates(rejected)

    # Suggest the same fact — should NOT be blocked by the rejected candidates
    candidates = asyncio.run(suggest_memories("User prefers polars.", store=store))

    assert len(candidates) == 1
    assert candidates[0].content == "The user prefers polars"
    assert candidates[0].status == CandidateStatus.pending
