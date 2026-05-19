"""Tests for the doctor diagnostics module."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
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
    issue = next(i for i in report.issues if i.code == "orphaned_supersession")
    assert issue.repairable is True


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
    past = datetime.now(timezone.utc) - timedelta(days=1)
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
            Fact(
                id="dddddddddddd",
                category=FactCategory.preference,
                content="d",
                expires_at=past,
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


def test_repair_orphaned_supersessions_clears_only_broken_links():
    store = _make_store()
    store.append_facts(
        [
            Fact(
                id="parentaaaaaa",
                category=FactCategory.preference,
                content="parent",
            ),
            Fact(
                id="validchildaa",
                category=FactCategory.preference,
                content="valid child",
                supersedes="parentaaaaaa",
            ),
            Fact(
                id="orphanchild",
                category=FactCategory.preference,
                content="orphan child",
                supersedes="missingmissi",
            ),
        ]
    )

    summary = repair_store(store, repair_orphaned_supersessions=True)
    facts = {fact.id: fact for fact in store.load_facts()}

    assert summary["orphaned_supersessions"] == {
        "cleared": 1,
        "ids": ["orphanchild"],
    }
    assert facts["orphanchild"].supersedes is None
    assert facts["validchildaa"].supersedes == "parentaaaaaa"


# ---------------------------------------------------------------------------
# MCP doctor surface
# ---------------------------------------------------------------------------


def test_doctor_mcp_returns_envelope(monkeypatch):
    store = _make_store()
    app = server.create_mcp(AsyncFactStore(store))

    result = asyncio.run(app._call_tool_mcp("doctor", {}))
    parsed = result[1]
    assert parsed["status"] == "ok"
    assert "report" in parsed["data"]
    assert parsed["data"]["report"]["status"] in ("ok", "warning", "error")


# ---------------------------------------------------------------------------
# Sync diagnostics
# ---------------------------------------------------------------------------


def test_doctor_sync_group_when_not_a_git_repo():
    """A data directory with no .git folder reports remote_configured=False
    and does not degrade overall health solely because sync is not set up."""
    store = _make_store()
    report = run_doctor(store)
    assert "sync" in report.counts
    sync = report.counts["sync"]
    assert sync["remote_configured"] is False
    assert sync["last_sync_at"] is None
    assert sync["unpushed_commits"] == 0
    assert sync["conflicting_facts"] == []
    # "Sync not configured" must NOT bump overall status to warning/error.
    assert report.status == "ok"


def test_doctor_sync_group_is_local_only(monkeypatch):
    """The sync check must not invoke ``git fetch`` or any network command."""
    store = _make_store()

    import subprocess as _subprocess

    real_run = _subprocess.run
    network_calls: list[list[str]] = []

    def watching_run(cmd, *args, **kwargs):
        if cmd and isinstance(cmd, list) and len(cmd) >= 2 and cmd[0] == "git":
            # Allow read-only local subcommands; flag anything that would touch
            # the network.
            forbidden = {"fetch", "pull", "push", "ls-remote"}
            if cmd[1] in forbidden:
                network_calls.append(cmd)
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr("engram.doctor.subprocess.run", watching_run)
    run_doctor(store)
    assert network_calls == []
