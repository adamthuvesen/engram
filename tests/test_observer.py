"""Tests for memory suggestion and extraction flows."""

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engram.core.models import CandidateStatus, Fact, FactCategory
from engram.extraction.observer import (
    _consolidate_batch,
    _dedup,
    _find_near_matches,
    _normalize_memory_key,
    extract_facts,
    suggest_memories,
)
from engram.storage.store import AsyncFactStore, FactStore


def _make_store() -> FactStore:
    tmp = Path(tempfile.mkdtemp())
    return FactStore(data_dir=tmp)


def _structured(response_model, data):
    if "facts" in data:
        data = {**data, "excluded_claims": data.get("excluded_claims", [])}
        facts = []
        for index, fact in enumerate(data["facts"]):
            content = fact.get("content", "missing content")
            facts.append(
                {
                    "memory_key": f"test-memory-{index}",
                    "project": None,
                    "retrieval_hints": [content],
                    "covered_claims": [content],
                    "why_store": "Useful future context",
                    **fact,
                }
            )
        data["facts"] = facts
    return response_model.model_validate(data)


def test_consolidate_batch_combines_fragments_with_same_memory_identity():
    cards = [
        Fact(
            category=FactCategory.assistant_info,
            memory_key="agent-memory-policy",
            content="Store durable memories in Engram.",
            tags=["memory"],
            retrieval_hints=["where memories live"],
            project="dotfiles",
        ),
        Fact(
            category=FactCategory.assistant_info,
            memory_key="agent-memory-policy",
            content="Also store them in native memory when available.",
            tags=["policy"],
            retrieval_hints=["dual memory"],
            project="dotfiles",
        ),
    ]

    consolidated = _consolidate_batch(cards)

    assert len(consolidated) == 1
    assert "Store durable memories in Engram." in consolidated[0].content
    assert "Also store them in native memory when available." in consolidated[0].content
    assert consolidated[0].tags == ["memory", "policy"]
    assert consolidated[0].retrieval_hints == [
        "where memories live",
        "dual memory",
    ]


def test_consolidate_batch_keeps_independent_memory_keys_separate():
    cards = [
        Fact(
            category=FactCategory.decision,
            memory_key="storage-format",
            content="Use JSONL for storage.",
        ),
        Fact(
            category=FactCategory.decision,
            memory_key="retrieval-strategy",
            content="Use tiered retrieval.",
        ),
    ]

    assert len(_consolidate_batch(cards)) == 2


def test_normalize_memory_key_produces_stable_slug():
    assert _normalize_memory_key(" Agent Memory Policy! ") == "agent-memory-policy"


def test_normalize_memory_key_rejects_empty_slug():
    with pytest.raises(ValueError, match="letter or number"):
        _normalize_memory_key("!!!")


def test_extract_facts_preserves_per_card_project_scope(monkeypatch):
    store = _make_store()

    async def fake_complete_model(
        prompt: str, system: str, response_model, model: str | None = None
    ):
        return _structured(
            response_model,
            {
                "facts": [
                    {
                        "memory_key": "atlas-package-manager",
                        "content": "Atlas uses pnpm.",
                        "category": "project",
                        "project": "atlas",
                    },
                    {
                        "memory_key": "beacon-package-manager",
                        "content": "Beacon uses npm.",
                        "category": "project",
                        "project": "beacon",
                    },
                ]
            },
        )

    monkeypatch.setattr(
        "engram.extraction.observer.complete_model", fake_complete_model
    )

    facts = asyncio.run(extract_facts("Atlas uses pnpm. Beacon uses npm.", store=store))

    assert {fact.project for fact in facts} == {"atlas", "beacon"}


def test_explicit_project_overrides_model_scope(monkeypatch):
    store = _make_store()

    async def fake_complete_model(
        prompt: str, system: str, response_model, model: str | None = None
    ):
        assert "fixed the scope to project 'atlas'" in prompt
        return _structured(
            response_model,
            {
                "facts": [
                    {
                        "content": "Atlas uses PostgreSQL.",
                        "category": "project",
                        "project": "wrong-project",
                    }
                ]
            },
        )

    monkeypatch.setattr(
        "engram.extraction.observer.complete_model", fake_complete_model
    )

    facts = asyncio.run(
        extract_facts("Atlas uses PostgreSQL.", project="atlas", store=store)
    )

    assert facts[0].project == "atlas"


def test_suggest_memories_queues_pending_candidates(monkeypatch):
    store = _make_store()

    async def fake_complete_model(
        prompt: str, system: str, response_model, model: str | None = None
    ):
        if "Classify each new fact" in prompt:
            return _structured(
                response_model, {"new": [0], "updates": [], "duplicates": []}
            )
        return _structured(
            response_model,
            {
                "facts": [
                    {
                        "content": "The user prefers concise summaries",
                        "category": "preference",
                        "tags": ["style"],
                        "why_store": "Useful for future responses",
                        "expires_at": None,
                    }
                ]
            },
        )

    monkeypatch.setattr(
        "engram.extraction.observer.complete_model", fake_complete_model
    )

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

    async def fake_complete_model(
        prompt: str, system: str, response_model, model: str | None = None
    ):
        return _structured(
            response_model,
            {
                "facts": [
                    {
                        "content": "The user prefers async storage",
                        "category": "preference",
                        "tags": ["storage"],
                        "why_store": "Guides storage changes",
                        "expires_at": None,
                    }
                ]
            },
        )

    monkeypatch.setattr(
        "engram.extraction.observer.complete_model", fake_complete_model
    )

    candidates = asyncio.run(
        suggest_memories("User prefers async storage.", store=AsyncFactStore(store))
    )

    assert len(candidates) == 1
    assert store.load_candidates(status=CandidateStatus.pending)[0].content == (
        "The user prefers async storage"
    )


def test_extract_facts_marks_updates_and_removes_superseded_fact_from_active_recall(
    monkeypatch,
):
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

    async def fake_complete_model(
        prompt: str, system: str, response_model, model: str | None = None
    ):
        if "Classify each new fact" in prompt:
            return _structured(
                response_model,
                {
                    "new": [],
                    "updates": [{"new_idx": 0, "existing_id": "oldfact"}],
                    "duplicates": [],
                },
            )
        return _structured(
            response_model,
            {
                "facts": [
                    {
                        "content": "The user prefers polars",
                        "category": "preference",
                        "tags": ["python"],
                        "why_store": "Reflects the current dataframe preference",
                        "expires_at": None,
                    }
                ]
            },
        )

    monkeypatch.setattr(
        "engram.extraction.observer.complete_model", fake_complete_model
    )

    facts = asyncio.run(extract_facts("The user prefers polars now.", store=store))

    assert len(facts) == 1
    assert facts[0].supersedes == "oldfact"

    all_facts = store.load_facts()
    original = next(f for f in all_facts if f.id == "oldfact")
    updated = next(f for f in all_facts if f.id != "oldfact")
    assert original.confidence == 0.0
    assert updated.supersedes == "oldfact"
    assert [f.id for f in store.load_active_facts()] == [updated.id]


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

    async def fake_complete_model(
        prompt: str, system: str, response_model, model: str | None = None
    ):
        if "Classify each new fact" in prompt:
            return _structured(
                response_model,
                {
                    "new": [],
                    "updates": [{"new_idx": 0, "existing_id": "oldfact"}],
                    "duplicates": [],
                },
            )
        return _structured(
            response_model,
            {
                "facts": [
                    {
                        "content": "The user prefers async storage",
                        "category": "preference",
                        "tags": ["storage"],
                        "why_store": "Reflects current architecture",
                        "expires_at": None,
                    }
                ]
            },
        )

    monkeypatch.setattr(
        "engram.extraction.observer.complete_model", fake_complete_model
    )

    facts = asyncio.run(
        extract_facts("User prefers async storage.", store=AsyncFactStore(store))
    )

    assert len(facts) == 1
    assert facts[0].supersedes == "oldfact"
    old = next(f for f in store.load_facts() if f.id == "oldfact")
    assert old.confidence == 0.0


def test_dedup_retries_then_rejects_unclassified_candidates(monkeypatch):
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

    calls = 0

    async def fake_complete_model(
        prompt: str, system: str, response_model, model: str | None = None
    ):
        nonlocal calls
        calls += 1
        return _structured(
            response_model,
            {
                "new": [],
                "updates": [
                    {"new_idx": 99, "existing_id": "oldfact"},
                    {"new_idx": "0", "existing_id": "oldfact"},
                    {"new_idx": 0},
                ],
                "duplicates": [],
            },
        )

    monkeypatch.setattr(
        "engram.extraction.observer.complete_model", fake_complete_model
    )

    with pytest.raises(ValueError, match="remained invalid"):
        asyncio.run(_dedup(candidates, existing, store=None))

    assert calls == 2


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

    async def fake_complete_model(
        prompt: str, system: str, response_model, model: str | None = None
    ):
        if "Classify each new fact" in prompt:
            raise AssertionError("Different project scopes should not be deduped")
        return _structured(
            response_model,
            {
                "facts": [
                    {
                        "content": "The user prefers polars",
                        "category": "preference",
                        "tags": ["python"],
                        "why_store": "Library preference",
                        "expires_at": None,
                    }
                ]
            },
        )

    monkeypatch.setattr(
        "engram.extraction.observer.complete_model", fake_complete_model
    )

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

    async def fake_complete_model(
        prompt: str, system: str, response_model, model: str | None = None
    ):
        if "Classify each new fact" in prompt:
            raise AssertionError("Exact duplicate should not need LLM dedup")
        return _structured(
            response_model,
            {
                "facts": [
                    {
                        "content": "The user prefers polars",
                        "category": "preference",
                        "tags": ["python"],
                        "why_store": "Library preference",
                        "expires_at": None,
                    }
                ]
            },
        )

    monkeypatch.setattr(
        "engram.extraction.observer.complete_model", fake_complete_model
    )

    facts = asyncio.run(extract_facts("User prefers polars.", store=store))

    assert facts == []
    assert len(store.load_active_facts()) == 201


def test_extract_facts_validation_error_returns_no_facts(monkeypatch):
    """Invalid structured extraction responses are dropped as a batch."""
    store = _make_store()

    async def fake_complete_model(
        prompt: str, system: str, response_model, model: str | None = None
    ):
        return _structured(
            response_model,
            {
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
            },
        )

    monkeypatch.setattr(
        "engram.extraction.observer.complete_model", fake_complete_model
    )

    facts = asyncio.run(extract_facts("User likes dark mode.", store=store))

    assert facts == []


def test_dedup_against_candidates_ignores_rejected(monkeypatch):
    """Rejected candidates do not block deduplication of new facts."""
    store = _make_store()

    async def fake_complete_model(
        prompt: str, system: str, response_model, model: str | None = None
    ):
        # No dedup needed from existing active facts
        if "Classify each new fact" in prompt:
            return _structured(
                response_model, {"new": [0], "updates": [], "duplicates": []}
            )
        return _structured(
            response_model,
            {
                "facts": [
                    {
                        "content": "The user prefers polars",
                        "category": "preference",
                        "tags": ["python"],
                        "why_store": "Library preference",
                        "expires_at": None,
                    }
                ]
            },
        )

    monkeypatch.setattr(
        "engram.extraction.observer.complete_model", fake_complete_model
    )

    from engram.core.models import MemoryCandidate

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
