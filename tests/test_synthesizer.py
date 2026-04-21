"""Tests for the synthesizer — LLM-powered fact consolidation."""

import asyncio
import tempfile
from pathlib import Path

from engram.models import Fact, FactCategory
from engram.store import FactStore
from engram.synthesizer import SynthesisResult, format_synthesis_result, synthesize


def _make_store() -> FactStore:
    tmp = Path(tempfile.mkdtemp())
    return FactStore(data_dir=tmp)


def _seed_facts(store: FactStore, facts: list[Fact]) -> None:
    store.append_facts(facts)


def test_empty_store_returns_zero_counts():
    store = _make_store()
    result = asyncio.run(synthesize(store=store))
    assert result.total_analyzed == 0
    assert result.kept == 0


def test_dry_run_does_not_modify_store(monkeypatch):
    store = _make_store()
    _seed_facts(
        store,
        [
            Fact(
                id="fact1",
                category=FactCategory.preference,
                content="User likes Python",
            ),
            Fact(
                id="fact2",
                category=FactCategory.preference,
                content="User likes Python a lot",
            ),
        ],
    )

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        return {
            "actions": [
                {
                    "fact_id": "fact1",
                    "action": "merge",
                    "merge_with": ["fact2"],
                    "merged_content": "User likes Python",
                    "merged_tags": ["python"],
                    "reason": "duplicates",
                },
                {"fact_id": "fact2", "action": "merge_source", "merge_target": "fact1"},
            ]
        }

    monkeypatch.setattr("engram.synthesizer.complete_json", fake_complete_json)

    result = asyncio.run(synthesize(dry_run=True, store=store))
    assert result.merged_groups == 1
    assert result.merged_sources == 1

    # Store should be untouched
    facts = store.load_active_facts()
    assert len(facts) == 2
    assert all(f.confidence == 1.0 for f in facts)


def test_removes_stale_facts(monkeypatch):
    store = _make_store()
    _seed_facts(
        store,
        [
            Fact(
                id="stale1",
                category=FactCategory.event,
                content="Currently debugging auth",
            ),
            Fact(id="good1", category=FactCategory.event, content="Team uses FastAPI"),
        ],
    )

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        return {
            "actions": [
                {
                    "fact_id": "stale1",
                    "action": "remove",
                    "reason": "no longer relevant",
                },
                {"fact_id": "good1", "action": "keep"},
            ]
        }

    monkeypatch.setattr("engram.synthesizer.complete_json", fake_complete_json)

    result = asyncio.run(synthesize(dry_run=False, store=store))
    assert result.removed == 1
    assert result.kept == 1

    active = store.load_active_facts()
    assert len(active) == 1
    assert active[0].id == "good1"


def test_rewrites_facts(monkeypatch):
    store = _make_store()
    _seed_facts(
        store,
        [
            Fact(id="vague1", category=FactCategory.preference, content="likes ts"),
        ],
    )

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        return {
            "actions": [
                {
                    "fact_id": "vague1",
                    "action": "rewrite",
                    "new_content": "The user prefers TypeScript over JavaScript for new projects",
                    "new_tags": ["typescript", "preference"],
                    "reason": "improved clarity",
                },
            ]
        }

    monkeypatch.setattr("engram.synthesizer.complete_json", fake_complete_json)

    result = asyncio.run(synthesize(dry_run=False, store=store))
    assert result.rewritten == 1

    facts = store.load_active_facts()
    assert len(facts) == 1
    assert (
        facts[0].content
        == "The user prefers TypeScript over JavaScript for new projects"
    )
    assert facts[0].tags == ["typescript", "preference"]


def test_merges_duplicate_facts(monkeypatch):
    store = _make_store()
    _seed_facts(
        store,
        [
            Fact(id="dup1", category=FactCategory.project, content="Engram uses JSONL"),
            Fact(
                id="dup2",
                category=FactCategory.project,
                content="Engram stores facts as JSONL files",
            ),
        ],
    )

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        return {
            "actions": [
                {
                    "fact_id": "dup1",
                    "action": "merge",
                    "merge_with": ["dup2"],
                    "merged_content": "Engram persists facts as JSONL files in ~/.engram/data/",
                    "merged_tags": ["engram", "storage"],
                    "reason": "duplicate information",
                },
                {"fact_id": "dup2", "action": "merge_source", "merge_target": "dup1"},
            ]
        }

    monkeypatch.setattr("engram.synthesizer.complete_json", fake_complete_json)

    result = asyncio.run(synthesize(dry_run=False, store=store))
    assert result.merged_groups == 1
    assert result.merged_sources == 1

    active = store.load_active_facts()
    assert len(active) == 1
    assert active[0].id == "dup1"
    assert (
        active[0].content == "Engram persists facts as JSONL files in ~/.engram/data/"
    )
    assert active[0].supersedes == "dup2"

    # dup2 should be forgotten
    all_facts = store.load_facts()
    dup2 = next(f for f in all_facts if f.id == "dup2")
    assert dup2.confidence == 0.0


def test_project_filter_only_processes_matching(monkeypatch):
    store = _make_store()
    _seed_facts(
        store,
        [
            Fact(
                id="proj_a",
                category=FactCategory.project,
                content="Fact A",
                project="alpha",
            ),
            Fact(
                id="proj_b",
                category=FactCategory.project,
                content="Fact B",
                project="beta",
            ),
        ],
    )

    calls = []

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        calls.append(prompt)
        return {
            "actions": [
                {"fact_id": "proj_a", "action": "keep"},
            ]
        }

    monkeypatch.setattr("engram.synthesizer.complete_json", fake_complete_json)

    result = asyncio.run(synthesize(project="alpha", store=store))
    assert result.total_analyzed == 1
    assert result.kept == 1

    # Should only have called LLM once, with project alpha's fact
    assert len(calls) == 1
    assert "proj_a" in calls[0]
    assert "proj_b" not in calls[0]


def test_invalid_llm_response_defaults_to_keep(monkeypatch):
    store = _make_store()
    _seed_facts(
        store,
        [
            Fact(
                id="ok1", category=FactCategory.preference, content="User likes tests"
            ),
            Fact(id="ok2", category=FactCategory.preference, content="User likes CI"),
        ],
    )

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        # LLM only returns action for one fact, misses the other
        return {
            "actions": [
                {"fact_id": "ok1", "action": "keep"},
            ]
        }

    monkeypatch.setattr("engram.synthesizer.complete_json", fake_complete_json)

    result = asyncio.run(synthesize(dry_run=False, store=store))
    assert result.kept == 2  # both kept (one explicit, one defaulted)

    active = store.load_active_facts()
    assert len(active) == 2


def test_llm_error_defaults_all_to_keep(monkeypatch):
    store = _make_store()
    _seed_facts(
        store,
        [
            Fact(
                id="safe1", category=FactCategory.preference, content="Important fact"
            ),
        ],
    )

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        raise RuntimeError("API error")

    monkeypatch.setattr("engram.synthesizer.complete_json", fake_complete_json)

    result = asyncio.run(synthesize(dry_run=False, store=store))
    assert result.kept == 1

    active = store.load_active_facts()
    assert len(active) == 1


def test_three_way_merge_preserves_all_absorbed_ids(monkeypatch):
    """Multi-way merge encodes all absorbed IDs in the survivor's tags."""
    store = _make_store()
    _seed_facts(
        store,
        [
            Fact(
                id="survivor", category=FactCategory.preference, content="Primary fact"
            ),
            Fact(id="src1", category=FactCategory.preference, content="Duplicate 1"),
            Fact(id="src2", category=FactCategory.preference, content="Duplicate 2"),
        ],
    )

    async def fake_complete_json(
        prompt: str, system: str = "", model: str | None = None
    ):
        return {
            "actions": [
                {
                    "fact_id": "survivor",
                    "action": "merge",
                    "merge_with": ["src1", "src2"],
                    "merged_content": "Consolidated fact",
                    "merged_tags": ["merged"],
                    "reason": "three duplicates",
                },
                {
                    "fact_id": "src1",
                    "action": "merge_source",
                    "merge_target": "survivor",
                },
                {
                    "fact_id": "src2",
                    "action": "merge_source",
                    "merge_target": "survivor",
                },
            ]
        }

    monkeypatch.setattr("engram.synthesizer.complete_json", fake_complete_json)

    asyncio.run(synthesize(dry_run=False, store=store))

    all_facts = store.load_facts()
    merged = next(f for f in all_facts if f.id == "survivor")

    # Primary supersedes link
    assert merged.supersedes == "src1"
    # Second absorbed ID encoded in tags
    assert "merged:src2" in merged.tags


def test_format_synthesis_result_dry_run():
    result = SynthesisResult(
        total_analyzed=10,
        kept=7,
        removed=1,
        rewritten=1,
        merged_groups=1,
        merged_sources=1,
        details=[
            {"action": "remove", "fact_id": "abc", "reason": "stale"},
            {
                "action": "rewrite",
                "fact_id": "def",
                "new_content": "Better wording",
                "reason": "clarity",
            },
            {
                "action": "merge",
                "fact_id": "ghi",
                "merge_with": ["jkl"],
                "merged_content": "Combined fact",
                "reason": "duplicates",
            },
        ],
    )
    output = format_synthesis_result(result, dry_run=True)
    assert "Preview (dry run)" in output
    assert "Analyzed:** 10" in output
    assert "dry_run=False" in output
    assert "`abc`" in output
    assert "Better wording" in output
    assert "Combined fact" in output


def test_format_synthesis_result_applied():
    result = SynthesisResult(total_analyzed=5, kept=5)
    output = format_synthesis_result(result, dry_run=False)
    assert "Applied" in output
    assert "dry_run=False" not in output
