"""Tests for shared MCP/CLI operation functions."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from engram import operations
from engram.models import CandidateStatus, Fact, FactCategory, MemoryCandidate
from engram.store import AsyncFactStore, FactStore


def _store() -> FactStore:
    return FactStore(data_dir=Path(tempfile.mkdtemp()))


def test_inspect_operation_returns_text_and_envelope():
    store = _store()
    store.append_facts(
        [Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="uses vim")]
    )

    result = asyncio.run(operations.inspect(store=AsyncFactStore(store)))

    assert result.exit_code == operations.EXIT_OK
    assert result.envelope.status == "ok"
    assert result.envelope.data[0]["id"] == "aaaaaaaaaaaa"
    assert "uses vim" in result.text


def test_list_candidates_operation_validates_status():
    result = asyncio.run(operations.list_candidates(status="bogus", store=_store()))

    assert result.exit_code == operations.EXIT_VALIDATION
    assert result.envelope.errors[0].code == "validation_error"
    assert "Unsupported status" in result.text


def test_candidate_review_operations_use_shared_payloads():
    store = _store()
    store.append_candidates(
        [
            MemoryCandidate(
                id="candidateaaa",
                category=FactCategory.workflow,
                content="Use the shared operation layer",
                status=CandidateStatus.pending,
            )
        ]
    )

    approved = asyncio.run(
        operations.approve_candidates(["candidateaaa"], store=AsyncFactStore(store))
    )

    assert approved.exit_code == operations.EXIT_OK
    assert approved.envelope.data["facts"][0]["content"] == (
        "Use the shared operation layer"
    )
    assert "Approved 1 candidate" in approved.text


def test_approve_candidates_rejects_edits_for_missing_ids():
    store = _store()
    store.append_candidates(
        [
            MemoryCandidate(
                id="candidateaaa",
                category=FactCategory.workflow,
                content="original content",
                status=CandidateStatus.pending,
            )
        ]
    )

    result = asyncio.run(
        operations.approve_candidates(
            ["candidateaaa"],
            edits={"candidateaaa": "edited", "candidateZZZ": "ghost"},
            store=AsyncFactStore(store),
        )
    )

    assert result.exit_code == operations.EXIT_VALIDATION
    assert "candidateZZZ" in result.text
    # No fact promoted because the validation aborted before approval.
    assert store.load_active_facts() == []


def test_approve_candidates_applies_valid_edits():
    store = _store()
    store.append_candidates(
        [
            MemoryCandidate(
                id="candidateaaa",
                category=FactCategory.workflow,
                content="original content",
                status=CandidateStatus.pending,
            )
        ]
    )

    result = asyncio.run(
        operations.approve_candidates(
            ["candidateaaa"],
            edits={"candidateaaa": "edited content"},
            store=AsyncFactStore(store),
        )
    )

    assert result.exit_code == operations.EXIT_OK
    assert result.envelope.data["facts"][0]["content"] == "edited content"
