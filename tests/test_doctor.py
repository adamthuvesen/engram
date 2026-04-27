"""Tests for the doctor diagnostics module."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from engram import server
from engram.doctor import DoctorIssue, repair_store, run_doctor
from engram.models import (
    Fact,
    FactCategory,
    MemoryCandidate,
    StoreTransaction,
    TransactionStatus,
)
from engram.store import AsyncFactStore, FactStore


def _make_store() -> FactStore:
    return FactStore(data_dir=Path(tempfile.mkdtemp()))


# ---------------------------------------------------------------------------
# 5.1: read-only checks
# ---------------------------------------------------------------------------


def test_doctor_healthy_store_reports_ok():
    store = _make_store()
    store.append_facts(
        [
            Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="x"),
        ]
    )
    report = run_doctor(store)
    assert report.status == "ok"
    assert all(issue.severity != "error" for issue in report.issues)


def test_doctor_detects_malformed_facts_jsonl():
    store = _make_store()
    # Write a corrupt line directly.
    store.facts_path.write_text("not json\n")
    report = run_doctor(store)
    codes = [issue.code for issue in report.issues]
    assert "facts_jsonl_corrupt" in codes


def test_doctor_detects_pending_transactions():
    store = _make_store()
    txn = StoreTransaction(
        type="approve_candidates",
        status=TransactionStatus.prepared,
    )
    store.transaction_log_path.write_text(txn.model_dump_json() + "\n")
    report = run_doctor(store)
    codes = [issue.code for issue in report.issues]
    assert "prepared_transactions_pending" in codes
    issue = next(i for i in report.issues if i.code == "prepared_transactions_pending")
    assert issue.repairable is True


def test_doctor_detects_orphaned_supersession():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="orphanaaaaaa",
                category=FactCategory.preference,
                content="orphan",
                supersedes="missingmissi",
            )
        ]
    )
    report = run_doctor(store)
    codes = [issue.code for issue in report.issues]
    assert "orphaned_supersession" in codes


def test_doctor_detects_circular_supersession():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="aaaaaaaaaaaa",
                category=FactCategory.preference,
                content="a",
                supersedes="bbbbbbbbbbbb",
            ),
            Fact(
                id="bbbbbbbbbbbb",
                category=FactCategory.preference,
                content="b",
                supersedes="aaaaaaaaaaaa",
            ),
        ]
    )
    report = run_doctor(store)
    codes = [issue.code for issue in report.issues]
    assert "circular_supersession" in codes
    cycle_issue = next(i for i in report.issues if i.code == "circular_supersession")
    assert cycle_issue.severity == "error"


def test_doctor_detects_duplicate_facts():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="aaaaaaaaaaaa",
                category=FactCategory.preference,
                content="prefers tabs",
            ),
            Fact(
                id="bbbbbbbbbbbb",
                category=FactCategory.preference,
                content="prefers tabs",
            ),
        ]
    )
    report = run_doctor(store)
    codes = [issue.code for issue in report.issues]
    assert "duplicate_facts" in codes


def test_doctor_provider_failure_reported_separately():
    store = _make_store()
    failure = DoctorIssue(
        code="provider_call_failed",
        severity="error",
        category="provider",
        message="boom",
    )
    report = run_doctor(store, check_provider_flag=True, provider_issue=failure)
    assert "provider" in report.checks_run
    codes = [issue.code for issue in report.issues]
    assert "provider_call_failed" in codes
    # Storage status is still tracked separately on the issue category.
    storage_errors = [i for i in report.issues if i.category == "storage"]
    assert all(i.severity != "error" for i in storage_errors)


def test_doctor_skips_provider_by_default():
    store = _make_store()
    report = run_doctor(store)
    assert "provider" not in report.checks_run
    assert "provider" in report.checks_skipped


def test_doctor_counts_active_stale_forgotten():
    store = _make_store()
    store.append_facts(
        [
            Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="a"),
            Fact(
                id="bbbbbbbbbbbb",
                category=FactCategory.preference,
                content="b",
                stale=True,
            ),
            Fact(
                id="cccccccccccc",
                category=FactCategory.preference,
                content="c",
                confidence=0.0,
            ),
        ]
    )
    report = run_doctor(store)
    assert report.counts["active_facts"] == 1
    assert report.counts["stale_facts"] == 1
    assert report.counts["forgotten_facts"] == 1


def test_doctor_flags_stale_pending_candidates():
    store = _make_store()
    from datetime import datetime, timedelta, timezone

    old = datetime.now(timezone.utc) - timedelta(days=45)
    candidate = MemoryCandidate(
        id="oldpendingcan",
        category=FactCategory.preference,
        content="old",
        updated_at=old,
        created_at=old,
    )
    store.append_candidates([candidate])
    report = run_doctor(store)
    codes = [i.code for i in report.issues]
    assert "stale_pending_candidates" in codes


# ---------------------------------------------------------------------------
# 5.4: repairs
# ---------------------------------------------------------------------------


def test_repair_jsonl_drops_corrupt_lines():
    store = _make_store()
    fact = Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="ok")
    store.append_facts([fact])
    # Corrupt a line by appending garbage.
    with store.facts_path.open("a") as f:
        f.write("garbage line\n")

    summary = repair_store(store, repair_jsonl=True)
    assert summary["jsonl_repair"]["facts_corrupt"] == 1


# ---------------------------------------------------------------------------
# MCP doctor surface
# ---------------------------------------------------------------------------


def test_doctor_mcp_returns_envelope(monkeypatch):
    store = _make_store()
    monkeypatch.setattr(server, "_store", AsyncFactStore(store), raising=False)

    async def call():
        fn = getattr(server.doctor, "fn", server.doctor)
        return await fn()

    result = asyncio.run(call())
    parsed = json.loads(result)
    assert parsed["status"] == "ok"
    assert "report" in parsed["data"]
    assert parsed["data"]["report"]["status"] in ("ok", "warning", "error")
