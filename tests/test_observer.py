"""Tests for ingest: one LLM call that extracts and reconciles memory cards."""

import asyncio
import tempfile
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engram.core.models import (
    CandidateStatus,
    Durability,
    EventType,
    Fact,
    FactCategory,
    MemoryCandidate,
)
from engram.extraction.observer import (
    _consolidate_batch,
    _normalize_memory_key,
    ingest,
)
from engram.storage.store import AsyncFactStore, FactStore


def _make_store() -> FactStore:
    return FactStore(data_dir=Path(tempfile.mkdtemp()))


def _card(**fields) -> dict:
    content = fields.get("content", "missing content")
    return {
        "memory_key": "test-memory",
        "category": "preference",
        "project": None,
        "tags": [],
        "retrieval_hints": [content],
        "covered_claims": [content],
        "why_store": "Useful future context",
        "durability": "durable",
        "anchors": [],
        "expires_at": None,
        "replaces": [],
        "duplicate_of": None,
        **fields,
    }


class FakeModel:
    """Stands in for ``complete_model`` and records every call.

    ``during_call`` runs inside each call, simulating a concurrent writer that
    changes the store while the LLM is thinking.
    """

    def __init__(
        self,
        facts: list[dict],
        retire: list[dict] | None = None,
        during_call: Callable[[int], None] | None = None,
    ):
        self.response = {
            "facts": [_card(**fact) for fact in facts],
            "retire": retire or [],
            "excluded_claims": [],
        }
        self.prompts: list[str] = []
        self.during_call = during_call

    async def __call__(self, prompt, system, response_model, **kwargs):
        self.prompts.append(prompt)
        if self.during_call is not None:
            self.during_call(len(self.prompts))
        return response_model.model_validate(self.response)


def _patch(monkeypatch, fake: FakeModel) -> FakeModel:
    monkeypatch.setattr("engram.extraction.observer.complete_model", fake)
    return fake


def _event_types(store: FactStore, fact_id: str) -> list[EventType]:
    return [e.event_type for e in store._load_all_events() if e.fact_id == fact_id]


def test_consolidate_batch_combines_fragments_with_same_memory_identity():
    cards = [
        Fact(
            category=FactCategory.assistant_info,
            memory_key="agent-memory-policy",
            content="Store durable memories in Engram.",
            tags=["memory"],
            retrieval_hints=["where memories live"],
            project="dotfiles",
            consolidates=["old1"],
        ),
        Fact(
            category=FactCategory.assistant_info,
            memory_key="agent-memory-policy",
            content="Also store them in native memory when available.",
            tags=["policy"],
            retrieval_hints=["dual memory"],
            project="dotfiles",
            consolidates=["old2"],
        ),
    ]

    consolidated = _consolidate_batch(cards)

    assert len(consolidated) == 1
    assert consolidated[0].content == (
        "Store durable memories in Engram. "
        "Also store them in native memory when available."
    )
    assert consolidated[0].tags == ["memory", "policy"]
    assert consolidated[0].retrieval_hints == ["where memories live", "dual memory"]
    assert consolidated[0].consolidates == ["old1", "old2"]


def test_consolidate_batch_keeps_independent_memory_keys_separate():
    cards = [
        Fact(category=FactCategory.decision, memory_key="storage", content="JSONL."),
        Fact(category=FactCategory.decision, memory_key="retrieval", content="Tiers."),
    ]

    assert len(_consolidate_batch(cards)) == 2


def test_normalize_memory_key_produces_stable_slug():
    assert _normalize_memory_key(" Agent Memory Policy! ") == "agent-memory-policy"


def test_normalize_memory_key_rejects_empty_slug():
    with pytest.raises(ValueError, match="letter or number"):
        _normalize_memory_key("!!!")


def test_ingest_makes_one_llm_call_and_stores_cards(monkeypatch):
    store = _make_store()
    store.append_facts(
        [Fact(id="neighbor", category=FactCategory.preference, content="Uses pnpm")]
    )
    fake = _patch(
        monkeypatch,
        FakeModel([{"content": "Atlas uses pnpm 10.", "project": "Atlas"}]),
    )

    result = asyncio.run(ingest("Atlas uses pnpm 10.", store=AsyncFactStore(store)))

    assert len(fake.prompts) == 1
    assert "EXISTING CARDS" in fake.prompts[0]
    assert [fact.project for fact in result.created] == ["atlas"]
    assert {f.content for f in store.load_active_facts(project="atlas")} == {
        "Uses pnpm",
        "Atlas uses pnpm 10.",
    }


def test_explicit_project_overrides_model_scope(monkeypatch):
    fake = _patch(
        monkeypatch,
        FakeModel([{"content": "Atlas uses PostgreSQL.", "project": "wrong"}]),
    )

    result = asyncio.run(
        ingest("Atlas uses PostgreSQL.", project="atlas", store=_make_store())
    )

    assert "fixed the scope to project 'atlas'" in fake.prompts[0]
    assert result.created[0].project == "atlas"


def test_unscoped_ingest_shows_project_scoped_neighbors(monkeypatch):
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="atlas-pm",
                category=FactCategory.project,
                project="atlas",
                memory_key="atlas-package-manager",
                content="Atlas uses npm for JavaScript packages.",
            )
        ]
    )
    fake = _patch(
        monkeypatch,
        FakeModel(
            [
                {
                    "content": "Atlas uses pnpm for JavaScript packages.",
                    "category": "project",
                    "project": "atlas",
                    "replaces": ["atlas-pm"],
                }
            ]
        ),
    )

    result = asyncio.run(
        ingest("Atlas switched to pnpm for JavaScript packages.", store=store)
    )

    assert "[id:atlas-pm] [project:atlas]" in fake.prompts[0]
    new_id = result.created[0].id
    assert result.superseded == {"atlas-pm": new_id}


def test_replaces_supersedes_old_fact_atomically(monkeypatch):
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="oldfact",
                category=FactCategory.preference,
                content="The user prefers pandas for dataframes.",
                consolidates=["ancient"],
                observed_at=datetime(2026, 3, 22, tzinfo=timezone.utc),
            )
        ]
    )
    _patch(
        monkeypatch,
        FakeModel(
            [
                {
                    "content": "The user prefers polars for dataframes.",
                    "replaces": ["oldfact"],
                }
            ]
        ),
    )
    appends: list[int] = []
    original_append = store.append_events

    def counting_append(events):
        appends.append(len(events))
        original_append(events)

    monkeypatch.setattr(store, "append_events", counting_append)

    result = asyncio.run(ingest("The user prefers polars dataframes now.", store=store))

    new = result.created[0]
    assert new.supersedes == "oldfact"
    assert new.consolidates == ["oldfact", "ancient"]
    # Details carried forward from the replaced card keep its age.
    assert new.first_observed_at == datetime(2026, 3, 22, tzinfo=timezone.utc)
    assert result.superseded == {"oldfact": new.id}
    assert appends == [2]
    assert _event_types(store, "oldfact")[-1] == EventType.superseded
    assert [f.id for f in store.load_active_facts()] == [new.id]


def test_retire_marks_existing_fact_stale_with_reason(monkeypatch):
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="freeze",
                category=FactCategory.event,
                project="atlas",
                content="Atlas has a deploy freeze.",
            )
        ]
    )
    _patch(
        monkeypatch,
        FakeModel([], retire=[{"id": "freeze", "reason": "freeze lifted"}]),
    )

    result = asyncio.run(
        ingest("The Atlas deploy freeze is lifted.", project="atlas", store=store)
    )

    assert result.created == []
    assert result.retired == {"freeze": "freeze lifted"}
    stale = next(f for f in store.load_facts() if f.id == "freeze")
    assert stale.stale and stale.stale_reason == "freeze lifted"


def test_duplicate_of_creates_nothing(monkeypatch):
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="existing",
                category=FactCategory.preference,
                content="The user prefers concise summaries.",
            )
        ]
    )
    _patch(
        monkeypatch,
        FakeModel(
            [{"content": "User likes concise summaries.", "duplicate_of": "existing"}]
        ),
    )

    result = asyncio.run(ingest("The user wants concise summaries.", store=store))

    assert result.created == []
    assert result.duplicates == ["existing"]
    assert len(store.load_facts()) == 1


def test_exact_content_match_is_a_duplicate_without_model_flag(monkeypatch):
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="existing",
                category=FactCategory.preference,
                content="The user prefers polars.",
            )
        ]
    )
    _patch(monkeypatch, FakeModel([{"content": "The user   prefers Polars."}]))

    result = asyncio.run(ingest("The user prefers polars.", store=store))

    assert result.created == []
    assert result.duplicates == ["existing"]


def test_unknown_ids_are_dropped_without_raising(monkeypatch):
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="unrelated",
                category=FactCategory.workflow,
                content="Deploys run through Argo CD on Fridays.",
            )
        ]
    )
    _patch(
        monkeypatch,
        FakeModel(
            [
                {
                    "content": "The user prefers polars.",
                    "replaces": ["invented", "unrelated-not-shown"],
                    "duplicate_of": "also-invented",
                }
            ],
            retire=[{"id": "invented", "reason": "hallucinated"}],
        ),
    )

    result = asyncio.run(ingest("The user prefers polars.", store=store))

    assert len(result.created) == 1
    assert result.created[0].supersedes is None
    assert result.created[0].consolidates == []
    assert result.superseded == {}
    assert result.retired == {}
    assert result.duplicates == []


def test_cross_project_replace_is_rejected(monkeypatch):
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="global-pm",
                category=FactCategory.preference,
                content="The user prefers pnpm for JavaScript packages.",
            )
        ]
    )
    _patch(
        monkeypatch,
        FakeModel(
            [
                {
                    "content": "Atlas uses npm for JavaScript packages.",
                    "replaces": ["global-pm"],
                }
            ]
        ),
    )

    result = asyncio.run(
        ingest("Atlas uses npm for JavaScript packages.", project="atlas", store=store)
    )

    assert result.superseded == {}
    assert {f.id for f in store.load_active_facts()} == {
        "global-pm",
        result.created[0].id,
    }


def test_existing_id_is_claimed_by_first_card_only(monkeypatch):
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="policy",
                category=FactCategory.convention,
                content="Helios retries requests three times.",
            )
        ]
    )
    _patch(
        monkeypatch,
        FakeModel(
            [
                {
                    "memory_key": "helios-retries",
                    "content": "Helios retries requests five times.",
                    "replaces": ["policy"],
                },
                {
                    "memory_key": "helios-breaker",
                    "content": "Helios opens the circuit breaker after five failures.",
                    "replaces": ["policy"],
                },
            ]
        ),
    )

    result = asyncio.run(ingest("Helios retries requests five times.", store=store))

    first, second = result.created
    assert result.superseded == {"policy": first.id}
    assert second.supersedes is None


def test_ephemeral_card_gets_default_expiry(monkeypatch):
    explicit = datetime(2030, 1, 1, tzinfo=timezone.utc)
    _patch(
        monkeypatch,
        FakeModel(
            [
                {
                    "memory_key": "atlas-incident",
                    "content": "Atlas staging is down; use the backup cluster.",
                    "durability": "ephemeral",
                },
                {
                    "memory_key": "atlas-freeze",
                    "content": "Atlas deploys are frozen until 2030.",
                    "durability": "ephemeral",
                    "expires_at": explicit.isoformat(),
                },
                {"memory_key": "atlas-db", "content": "Atlas uses PostgreSQL."},
            ]
        ),
    )

    result = asyncio.run(ingest("Atlas notes.", project="atlas", store=_make_store()))

    incident, freeze, db = result.created
    assert incident.durability == Durability.ephemeral
    assert incident.expires_at is not None
    expected = datetime.now(timezone.utc) + timedelta(days=45)
    assert abs(incident.expires_at - expected) < timedelta(minutes=1)
    assert freeze.expires_at == explicit
    assert db.expires_at is None


def test_invalid_response_stores_nothing(monkeypatch):
    fake = FakeModel([{"content": "The user prefers dark mode."}])
    fake.response["facts"].append({"category": "preference"})
    _patch(monkeypatch, fake)
    store = _make_store()

    result = asyncio.run(ingest("User likes dark mode.", store=store))

    assert result.created == []
    assert store.load_facts() == []


def test_queue_for_review_then_approval_supersedes(monkeypatch):
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="oldfact",
                category=FactCategory.preference,
                content="The user prefers pandas for dataframes.",
            ),
            Fact(
                id="gone",
                category=FactCategory.workflow,
                content="The user runs pandas profiling for dataframes weekly.",
            ),
        ]
    )
    _patch(
        monkeypatch,
        FakeModel(
            [
                {
                    "content": "The user prefers polars for dataframes.",
                    "replaces": ["oldfact"],
                }
            ],
            retire=[{"id": "gone", "reason": "no longer uses pandas"}],
        ),
    )

    result = asyncio.run(
        ingest(
            "The user prefers polars for dataframes now.",
            store=store,
            queue_for_review=True,
        )
    )

    assert result.created == []
    assert result.retired == {}
    (candidate,) = result.candidates
    assert candidate.replaces == ["oldfact"]
    assert "gone (no longer uses pandas)" in candidate.review_note
    assert len(store.load_active_facts()) == 2

    (approved,) = store.approve_candidates([candidate.id])

    assert _event_types(store, "oldfact")[-1] == EventType.superseded
    assert {f.id for f in store.load_active_facts()} == {"gone", approved.id}


def test_queue_for_review_skips_pending_duplicates_but_not_rejected(monkeypatch):
    store = _make_store()
    store.append_candidates(
        [
            MemoryCandidate(
                category=FactCategory.preference,
                content="The user prefers polars",
                status=CandidateStatus.rejected,
            ),
            MemoryCandidate(
                category=FactCategory.preference,
                content="The user prefers dark mode",
            ),
        ]
    )
    _patch(
        monkeypatch,
        FakeModel(
            [
                {"memory_key": "polars", "content": "The user prefers polars"},
                {"memory_key": "theme", "content": "The user prefers dark mode"},
            ]
        ),
    )

    result = asyncio.run(
        ingest("Polars and dark mode.", store=store, queue_for_review=True)
    )

    assert [c.content for c in result.candidates] == ["The user prefers polars"]
    assert len(store.load_candidates(status=CandidateStatus.pending)) == 2


def _pandas_store() -> FactStore:
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="oldfact",
                category=FactCategory.preference,
                content="The user prefers pandas for dataframes.",
            )
        ]
    )
    return store


_POLARS_CARD = {
    "content": "The user prefers polars for dataframes.",
    "replaces": ["oldfact"],
}


def test_target_changed_during_llm_call_retries_with_fresh_neighbors(monkeypatch):
    store = _pandas_store()

    def edit_on_first_call(call: int) -> None:
        if call == 1:
            store.update_fact("oldfact", tags=["python"])

    fake = _patch(
        monkeypatch, FakeModel([_POLARS_CARD], during_call=edit_on_first_call)
    )

    result = asyncio.run(ingest("The user prefers polars now.", store=store))

    assert len(fake.prompts) == 2
    (new,) = result.created
    assert result.superseded == {"oldfact": new.id}
    assert new.supersedes == "oldfact"
    assert [f.id for f in store.load_active_facts()] == [new.id]


def test_repeated_conflict_stores_card_without_the_changed_target(monkeypatch):
    store = _pandas_store()
    fake = _patch(
        monkeypatch,
        FakeModel(
            [_POLARS_CARD],
            during_call=lambda call: store.update_fact("oldfact", tags=[str(call)]),
        ),
    )

    result = asyncio.run(ingest("The user prefers polars now.", store=store))

    assert len(fake.prompts) == 2
    (new,) = result.created
    assert new.supersedes is None
    assert new.consolidates == []
    assert result.superseded == {}
    assert {f.id for f in store.load_active_facts()} == {"oldfact", new.id}
    stored = next(f for f in store.load_facts() if f.id == new.id)
    assert stored.supersedes is None


def test_retire_changed_during_llm_call_is_not_applied(monkeypatch):
    store = _pandas_store()
    _patch(
        monkeypatch,
        FakeModel(
            [{"memory_key": "theme", "content": "The user prefers dark mode."}],
            retire=[{"id": "oldfact", "reason": "no longer true"}],
            during_call=lambda call: store.update_fact("oldfact", tags=[str(call)]),
        ),
    )

    result = asyncio.run(ingest("Dark mode; pandas is gone.", store=store))

    assert len(result.created) == 1
    assert result.retired == {}
    assert not next(f for f in store.load_facts() if f.id == "oldfact").stale


def test_project_scoped_ingest_cannot_retire_global_or_foreign_cards(monkeypatch):
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="global-pm",
                category=FactCategory.preference,
                content="The user prefers pnpm for JavaScript packages.",
            ),
            Fact(
                id="atlas-pm",
                category=FactCategory.project,
                project="atlas",
                content="Atlas uses pnpm for JavaScript packages.",
            ),
        ]
    )
    _patch(
        monkeypatch,
        FakeModel(
            [],
            retire=[
                {"id": "global-pm", "reason": "atlas switched"},
                {"id": "atlas-pm", "reason": "atlas switched"},
            ],
        ),
    )

    result = asyncio.run(
        ingest("Atlas no longer uses pnpm packages.", project="atlas", store=store)
    )

    assert result.retired == {"atlas-pm": "atlas switched"}
    assert not next(f for f in store.load_facts() if f.id == "global-pm").stale


def test_unscoped_ingest_may_retire_global_cards(monkeypatch):
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="global-pm",
                category=FactCategory.preference,
                content="The user prefers pnpm for JavaScript packages.",
            )
        ]
    )
    _patch(
        monkeypatch,
        FakeModel([], retire=[{"id": "global-pm", "reason": "moved to bun"}]),
    )

    result = asyncio.run(
        ingest("The user no longer prefers pnpm packages.", store=store)
    )

    assert result.retired == {"global-pm": "moved to bun"}


def test_past_expiry_on_ephemeral_card_falls_back_to_ttl(monkeypatch):
    past = datetime(2020, 1, 1, tzinfo=timezone.utc)
    _patch(
        monkeypatch,
        FakeModel(
            [
                {
                    "content": "Atlas staging is down.",
                    "durability": "ephemeral",
                    "expires_at": past.isoformat(),
                }
            ]
        ),
    )

    result = asyncio.run(ingest("Atlas staging is down.", store=_make_store()))

    expires_at = result.created[0].expires_at
    assert expires_at is not None
    assert expires_at > datetime.now(timezone.utc) + timedelta(days=44)
