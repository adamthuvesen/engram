"""Tests for the JSONL fact store."""

import asyncio
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engram.models import (
    CandidateStatus,
    Fact,
    FactCategory,
    MemoryCandidate,
    RecallRecord,
)
from engram.store import AsyncFactStore, FactStore, format_facts_for_llm


def _make_store() -> FactStore:
    """Create a store with a temp directory."""
    tmp = Path(tempfile.mkdtemp())
    return FactStore(data_dir=tmp)


def test_empty_store():
    store = _make_store()
    assert store.load_facts() == []
    assert store.load_active_facts() == []


def test_append_and_load():
    store = _make_store()
    facts = [
        Fact(
            category=FactCategory.preference, content="User prefers polars over pandas"
        ),
        Fact(category=FactCategory.personal_info, content="User works on AI team"),
    ]
    store.append_facts(facts)

    loaded = store.load_facts()
    assert len(loaded) == 2
    assert loaded[0].content == "User prefers polars over pandas"
    assert loaded[1].content == "User works on AI team"


def test_filter_by_category():
    store = _make_store()
    store.append_facts(
        [
            Fact(category=FactCategory.preference, content="Prefers polars"),
            Fact(category=FactCategory.personal_info, content="Works on AI team"),
            Fact(category=FactCategory.preference, content="Uses ruff for formatting"),
        ]
    )

    prefs = store.load_active_facts(category=FactCategory.preference)
    assert len(prefs) == 2
    assert all(f.category == FactCategory.preference for f in prefs)


def test_filter_by_project():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                category=FactCategory.project,
                content="Uses dbt",
                project="acme-dw",
            ),
            Fact(
                category=FactCategory.project,
                content="Uses FastAPI",
                project="ai-research",
            ),
            Fact(category=FactCategory.preference, content="Global pref"),
        ]
    )

    filtered = store.load_active_facts(project="acme-dw")
    assert len(filtered) == 2  # acme-dw + global (no project filter excludes)

    exact = store.load_active_facts(project="acme-dw", include_global=False)
    assert len(exact) == 1
    assert exact[0].project == "acme-dw"


def test_forget():
    store = _make_store()
    fact = Fact(category=FactCategory.preference, content="Old preference")
    store.append_facts([fact])

    result = store.forget(fact.id, reason="outdated")
    assert result is not None
    assert result.confidence == 0.0

    # Should not appear in active facts
    active = store.load_active_facts()
    assert len(active) == 0


def test_update_fact():
    store = _make_store()
    fact = Fact(category=FactCategory.preference, content="Original content")
    store.append_facts([fact])

    updated = store.update_fact(fact.id, content="Updated content")
    assert updated is not None
    assert updated.content == "Updated content"

    loaded = store.load_facts()
    assert loaded[0].content == "Updated content"


def test_stats():
    store = _make_store()
    past = datetime.now(timezone.utc) - timedelta(days=1)
    store.append_facts(
        [
            Fact(category=FactCategory.preference, content="Pref 1"),
            Fact(category=FactCategory.personal_info, content="Info 1"),
            Fact(category=FactCategory.preference, content="Pref 2", confidence=0.0),
            Fact(category=FactCategory.preference, content="Stale", stale=True),
            Fact(category=FactCategory.workflow, content="Expired", expires_at=past),
        ]
    )

    stats = store.stats()
    assert stats["total_facts"] == 5
    assert stats["active_facts"] == 2
    assert stats["forgotten_facts"] == 1
    assert stats["expired_facts"] == 1
    assert stats["by_category"]["preference"] == 1
    assert stats["by_category"]["personal_info"] == 1
    assert "workflow" not in stats["by_category"]


def test_update_fact_does_not_overwrite_concurrent_append():
    store = _make_store()
    original = Fact(
        id="aaaaaaaaaaaa",
        category=FactCategory.preference,
        content="Original",
    )
    appended = Fact(
        id="bbbbbbbbbbbb",
        category=FactCategory.workflow,
        content="Concurrent append",
    )
    store.append_facts([original])

    original_rewrite = store._rewrite
    rewrite_started = threading.Event()
    allow_rewrite = threading.Event()
    errors: list[BaseException] = []

    def slow_rewrite(records, path=None):
        rewrite_started.set()
        assert allow_rewrite.wait(timeout=2)
        return original_rewrite(records, path=path)

    store._rewrite = slow_rewrite

    def update_fact():
        try:
            store.update_fact("aaaaaaaaaaaa", content="Updated")
        except BaseException as exc:
            errors.append(exc)

    update_thread = threading.Thread(target=update_fact)
    update_thread.start()
    assert rewrite_started.wait(timeout=2)

    append_thread = threading.Thread(target=lambda: store.append_facts([appended]))
    append_thread.start()
    time.sleep(0.05)
    allow_rewrite.set()

    update_thread.join(timeout=2)
    append_thread.join(timeout=2)

    assert not update_thread.is_alive()
    assert not append_thread.is_alive()
    assert errors == []

    facts = {fact.id: fact for fact in store.load_facts()}
    assert facts["aaaaaaaaaaaa"].content == "Updated"
    assert facts["bbbbbbbbbbbb"].content == "Concurrent append"


def test_candidate_approval_promotes_fact():
    store = _make_store()
    candidate = MemoryCandidate(
        id="cand123",
        category=FactCategory.preference,
        content="The user prefers concise summaries",
        why_store="This should shape future responses",
    )
    store.append_candidates([candidate])

    approved = store.approve_candidates(["cand123"])

    assert len(approved) == 1
    assert approved[0].content == "The user prefers concise summaries"

    candidates = store.load_candidates(status=CandidateStatus.approved)
    assert len(candidates) == 1
    assert candidates[0].status == CandidateStatus.approved

    facts = store.load_active_facts()
    assert len(facts) == 1
    assert facts[0].content == "The user prefers concise summaries"


def test_candidate_approval_is_idempotent():
    store = _make_store()
    candidate = MemoryCandidate(
        id="cand123",
        category=FactCategory.preference,
        content="The user prefers concise summaries",
    )
    store.append_candidates([candidate])

    first = store.approve_candidates(["cand123"])
    second = store.approve_candidates(["cand123"])

    assert len(first) == 1
    assert second == []
    facts = store.load_active_facts()
    assert len(facts) == 1
    assert facts[0].content == "The user prefers concise summaries"


def test_rejected_candidate_cannot_be_approved():
    store = _make_store()
    candidate = MemoryCandidate(
        id="cand456",
        category=FactCategory.workflow,
        content="Rejected thought",
    )
    store.append_candidates([candidate])
    store.reject_candidates(["cand456"], reason="Nope")

    approved = store.approve_candidates(["cand456"])

    assert approved == []
    assert store.load_active_facts() == []
    rejected = store.load_candidates(status=CandidateStatus.rejected)
    assert len(rejected) == 1


def test_candidate_rejection_updates_status():
    store = _make_store()
    candidate = MemoryCandidate(
        id="cand456",
        category=FactCategory.workflow,
        content="The user might maybe sometimes prefer verbose output",
    )
    store.append_candidates([candidate])

    rejected = store.reject_candidates(["cand456"], reason="Too speculative")

    assert len(rejected) == 1
    assert rejected[0].status == CandidateStatus.rejected
    assert rejected[0].review_note == "Too speculative"
    assert store.load_active_facts() == []


def test_candidate_rejection_is_idempotent_and_does_not_reject_approved():
    store = _make_store()
    pending = MemoryCandidate(
        id="pending",
        category=FactCategory.workflow,
        content="Pending thought",
    )
    approved = MemoryCandidate(
        id="approved",
        category=FactCategory.workflow,
        content="Approved thought",
    )
    store.append_candidates([pending, approved])
    store.approve_candidates(["approved"])

    first = store.reject_candidates(["pending"], reason="Nope")
    second = store.reject_candidates(["pending"], reason="Still nope")
    rejected_approved = store.reject_candidates(["approved"], reason="Too late")

    assert len(first) == 1
    assert second == []
    assert rejected_approved == []
    assert len(store.load_candidates(status=CandidateStatus.rejected)) == 1
    assert len(store.load_candidates(status=CandidateStatus.approved)) == 1


def test_approving_superseding_candidate_softens_old_fact():
    store = _make_store()
    old_fact = Fact(
        id="old123", category=FactCategory.preference, content="User prefers pandas"
    )
    store.append_facts([old_fact])
    candidate = MemoryCandidate(
        id="cand789",
        category=FactCategory.preference,
        content="User prefers polars",
        supersedes="old123",
    )
    store.append_candidates([candidate])

    approved = store.approve_candidates(["cand789"])

    assert len(approved) == 1

    all_facts = store.load_facts()
    original = next(f for f in all_facts if f.id == "old123")
    replacement = next(f for f in all_facts if f.id != "old123")
    assert original.confidence == 0.3
    assert replacement.supersedes == "old123"


def test_format_facts_for_llm():
    facts = [
        Fact(id="abc123", category=FactCategory.preference, content="Prefers polars"),
        Fact(
            id="def456",
            category=FactCategory.project,
            content="Uses dbt",
            project="acme-dw",
        ),
    ]

    formatted = format_facts_for_llm(facts)
    assert "abc123" in formatted
    assert "preference" in formatted
    assert "acme-dw" in formatted


def test_format_empty():
    assert format_facts_for_llm([]) == "(no facts stored)"


def test_prefilter_facts_prioritizes_matching_content():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                category=FactCategory.preference,
                content="The user prefers polars for dataframes",
                tags=["python", "polars"],
            ),
            Fact(category=FactCategory.workflow, content="Use ruff for formatting"),
            Fact(
                category=FactCategory.project,
                content="acme-dw uses dbt models",
                project="acme-dw",
            ),
        ]
    )

    filtered = store.prefilter_facts(
        "What dataframe library does the user prefer? polars", limit=2
    )

    assert len(filtered) == 2
    # prefilter now returns (score, Fact) tuples
    _, top_fact = filtered[0]
    assert "polars" in top_fact.content.lower()


def test_prefilter_bigram_and_normalization():
    """Bigrams and underscore normalization catch near-misses."""
    store = _make_store()
    store.append_facts(
        [
            Fact(
                category=FactCategory.convention,
                content="The team uses coding_style based on PEP8",
            ),
            Fact(
                category=FactCategory.preference, content="Unrelated fact about pizza"
            ),
        ]
    )

    filtered = store.prefilter_facts("coding style conventions")
    scores = [score for score, _ in filtered]
    assert len(filtered) == 1
    assert scores[0] > 0
    assert "coding_style" in filtered[0][1].content


def test_prefilter_returns_score_tuples():
    store = _make_store()
    store.append_facts(
        [
            Fact(category=FactCategory.preference, content="Likes Python"),
        ]
    )
    results = store.prefilter_facts("python")
    assert len(results) == 1
    score, fact = results[0]
    assert isinstance(score, int)
    assert score > 0
    assert fact.content == "Likes Python"


@pytest.mark.asyncio
async def test_async_store_facade_covers_server_operations():
    store = _make_store()
    async_store = AsyncFactStore(store)
    fact = Fact(category=FactCategory.preference, content="Likes concise output")
    candidate = MemoryCandidate(
        id="cand123",
        category=FactCategory.workflow,
        content="Use async storage in MCP tools",
    )

    await async_store.append_facts([fact])
    await async_store.append_candidates([candidate])

    assert [loaded.id for loaded in await async_store.load_facts()] == [fact.id]
    assert (await async_store.load_active_facts())[0].content == fact.content
    assert (await async_store.prefilter_facts("concise"))[0][1].id == fact.id

    updated = await async_store.update_fact(fact.id, content="Likes direct summaries")
    assert updated is not None
    assert updated.content == "Likes direct summaries"

    await async_store.update_candidate("cand123", content="Use async facade")
    approved = await async_store.approve_candidates(["cand123"])
    assert len(approved) == 1
    assert (await async_store.reject_candidates(["cand123"])) == []

    await async_store.log_recall(
        RecallRecord(query="storage", tier=1, prefilter_count=1, latency_ms=1.0)
    )
    assert (await async_store.load_recall_log())[0].query == "storage"
    assert (await async_store.stats())["total_facts"] == 2

    renamed = await async_store.rename_project("missing", "new")
    assert renamed == 0
    forgotten = await async_store.forget(fact.id)
    assert forgotten is not None
    assert forgotten.confidence == 0.0
    assert (await async_store.purge())["purged"] == 1


@pytest.mark.asyncio
async def test_async_store_read_does_not_block_unrelated_coroutine():
    store = _make_store()
    async_store = AsyncFactStore(store)
    original_load_facts = store.load_facts

    def slow_load_facts():
        time.sleep(0.12)
        return original_load_facts()

    store.load_facts = slow_load_facts  # type: ignore[method-assign]
    start = time.perf_counter()

    load_task = asyncio.create_task(async_store.load_facts())
    await asyncio.sleep(0.02)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.08
    assert not load_task.done()
    assert await load_task == []


@pytest.mark.asyncio
async def test_async_store_write_does_not_block_unrelated_coroutine():
    store = _make_store()
    async_store = AsyncFactStore(store)
    original_append_facts = store.append_facts

    def slow_append_facts(facts: list[Fact]) -> None:
        time.sleep(0.12)
        original_append_facts(facts)

    store.append_facts = slow_append_facts  # type: ignore[method-assign]
    start = time.perf_counter()

    append_task = asyncio.create_task(
        async_store.append_facts(
            [Fact(category=FactCategory.workflow, content="Async append")]
        )
    )
    await asyncio.sleep(0.02)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.08
    assert not append_task.done()
    await append_task
    assert len(store.load_facts()) == 1


@pytest.mark.asyncio
async def test_async_concurrent_appends_keep_complete_parseable_lines():
    store = _make_store()
    async_store = AsyncFactStore(store)

    await asyncio.gather(
        *[
            async_store.append_facts(
                [Fact(category=FactCategory.workflow, content=f"Fact {i}")]
            )
            for i in range(30)
        ]
    )

    lines = store.facts_path.read_text().splitlines()
    loaded = [Fact.model_validate_json(line) for line in lines]

    assert len(lines) == 30
    assert {fact.content for fact in loaded} == {f"Fact {i}" for i in range(30)}


@pytest.mark.asyncio
async def test_async_concurrent_rewrites_leave_parseable_file_and_no_tmp_files():
    store = _make_store()
    async_store = AsyncFactStore(store)
    facts = [
        Fact(id=f"fact{i}", category=FactCategory.workflow, content=f"Fact {i}")
        for i in range(10)
    ]
    store.append_facts(facts)

    await asyncio.gather(
        async_store.batch_update_facts(
            {f"fact{i}": {"content": f"First update {i}"} for i in range(5)}
        ),
        async_store.batch_update_facts(
            {f"fact{i}": {"content": f"Second update {i}"} for i in range(5, 10)}
        ),
    )

    lines = store.facts_path.read_text().splitlines()
    assert [Fact.model_validate_json(line) for line in lines]
    assert not list(store.data_dir.glob("facts*.tmp"))
