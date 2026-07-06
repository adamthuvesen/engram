"""Read-only health diagnostics for the Engram memory store.

Doctor reports problems but does not modify state. The `repair` flag on
recoverable issues is opt-in and limited to well-tested operations:

- Recovering prepared (uncommitted) transactions by rolling them forward.
- Repairing JSONL files with corrupt lines via ``FactStore.repair`` (drops
  unparseable lines after copying valid records).
- Clearing supersession links that point to missing ancestor facts.

Provider readiness is reported separately so a missing API key never masks
storage health and vice versa.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from engram.core.config import get_settings
from engram.core.models import (
    CandidateStatus,
    EVENT_LOG_META_VERSION,
    EventLogMeta,
    Fact,
    FactEvent,
    MIN_ACTIVE_CONFIDENCE,
    MemoryCandidate,
    RecallRecord,
    StoreTransaction,
    TransactionStatus,
)
from engram.storage.store import AsyncFactStore, FactStore, _is_active_fact

logger = logging.getLogger(__name__)


class DoctorIssue(BaseModel):
    code: str
    severity: str  # "info" | "warning" | "error"
    category: (
        str  # "storage" | "config" | "provider" | "recall_log" | "facts" | "candidates"
    )
    message: str
    ids: list[str] = Field(default_factory=list)
    repair: str | None = None
    repairable: bool = False


class DoctorReport(BaseModel):
    """``status`` rolls up the worst severity: ``ok``, ``warning``, or ``error``."""

    status: str
    issues: list[DoctorIssue] = Field(default_factory=list)
    counts: dict[str, Any] = Field(default_factory=dict)
    checks_run: list[str] = Field(default_factory=list)
    checks_skipped: list[str] = Field(default_factory=list)


def _resolve_store(store: FactStore | AsyncFactStore | None) -> FactStore:
    if store is None:
        return FactStore()
    if isinstance(store, AsyncFactStore):
        return store.sync_store
    return store


def _check_data_dir(store: FactStore, issues: list[DoctorIssue]) -> None:
    if not store.data_dir.exists():
        issues.append(
            DoctorIssue(
                code="data_dir_missing",
                severity="error",
                category="storage",
                message=f"Data directory missing: {store.data_dir}",
                repair="Create the directory or set ENGRAM_DATA_DIR.",
            )
        )
        return
    if not os.access(store.data_dir, os.W_OK):
        issues.append(
            DoctorIssue(
                code="data_dir_not_writable",
                severity="error",
                category="storage",
                message=f"Data directory is not writable: {store.data_dir}",
                repair="Adjust filesystem permissions.",
            )
        )


def _check_jsonl_integrity(
    path: Path,
    label: str,
    model_cls: type,
    issues: list[DoctorIssue],
) -> int:
    """Return the count of valid records; emit issues for corrupt lines.

    For ``facts.jsonl``, validates each non-empty line against the current
    event-log shape: the first line must be the ``EventLogMeta`` sentinel and
    the remaining lines must be ``FactEvent`` records.
    """
    if not path.exists():
        return 0

    if label == "facts":
        first_lineno = 0
        with path.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                first_lineno = lineno
                try:
                    payload = EventLogMeta.model_validate_json(line)
                except (ValueError, ValidationError):
                    _append_jsonl_issue(path, label, [lineno], issues)
                    return 0
                if payload.meta != EVENT_LOG_META_VERSION:
                    _append_jsonl_issue(path, label, [lineno], issues)
                    return 0
                break
        if first_lineno == 0:
            return 0

        valid = 0
        corrupt: list[int] = []
        first_data_line = True
        with path.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                if first_data_line:
                    first_data_line = False
                    continue
                try:
                    FactEvent.model_validate_json(line)
                    valid += 1
                except (ValueError, ValidationError):
                    corrupt.append(lineno)
        if corrupt:
            _append_jsonl_issue(path, label, corrupt, issues)
        return valid

    valid = 0
    corrupt: list[int] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                model_cls.model_validate_json(line)
                valid += 1
            except (ValueError, ValidationError):
                corrupt.append(lineno)
    if corrupt:
        _append_jsonl_issue(path, label, corrupt, issues)
    return valid


def _append_jsonl_issue(
    path: Path,
    label: str,
    corrupt: list[int],
    issues: list[DoctorIssue],
) -> None:
    issues.append(
        DoctorIssue(
            code=f"{label}_jsonl_corrupt",
            severity="error",
            category="storage",
            message=(
                f"{len(corrupt)} corrupt line(s) in {path.name}: "
                f"{corrupt[:5]}{'…' if len(corrupt) > 5 else ''}"
            ),
            repair="Run `engram doctor --repair-jsonl` to drop corrupt lines.",
            repairable=True,
        )
    )


def _check_transactions(store: FactStore, issues: list[DoctorIssue]) -> None:
    path = store.transaction_log_path
    if not path.exists():
        return
    transactions: list[StoreTransaction] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                transactions.append(StoreTransaction.model_validate_json(line))
            except (ValueError, ValidationError):
                issues.append(
                    DoctorIssue(
                        code="transaction_log_corrupt",
                        severity="warning",
                        category="storage",
                        message=f"Corrupt transaction log line {lineno}",
                        repair="Inspect transactions.jsonl manually.",
                    )
                )
    prepared: dict[str, StoreTransaction] = {}
    committed: set[str] = set()
    for txn in transactions:
        if txn.status == TransactionStatus.prepared:
            prepared[txn.id] = txn
        elif txn.status == TransactionStatus.committed:
            committed.add(txn.id)
    pending = [tid for tid in prepared if tid not in committed]
    if pending:
        issues.append(
            DoctorIssue(
                code="prepared_transactions_pending",
                severity="warning",
                category="storage",
                message=(
                    f"{len(pending)} prepared transaction(s) need to be rolled forward."
                ),
                ids=pending,
                repair="Run `engram doctor --recover-transactions` (or any store write) to roll forward.",
                repairable=True,
            )
        )


def _check_recall_log(store: FactStore, issues: list[DoctorIssue]) -> None:
    path = store.recall_log_path
    if not path.exists():
        return
    bad: list[int] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                RecallRecord.model_validate_json(line)
            except (ValueError, ValidationError):
                bad.append(lineno)
    if bad:
        issues.append(
            DoctorIssue(
                code="recall_log_unparseable",
                severity="warning",
                category="recall_log",
                message=f"{len(bad)} unparseable recall_log lines: {bad[:5]}",
                repair="Truncate recall_log.jsonl or drop the bad lines manually.",
            )
        )


def _check_sync(
    store: FactStore,
    issues: list[DoctorIssue],
    *,
    counts: dict[str, object],
) -> None:
    """Populate sync diagnostics in ``counts['sync']``.

    Local-only by default — no network calls. Reports whether the data dir
    is a git repo, the configured remote (if any), the last sync timestamp
    read from ``.engram-sync-state``, and the number of local commits ahead
    of the upstream tracking branch.
    """
    from engram.storage.sync import read_sync_state

    sync_info: dict[str, object] = {
        "remote_configured": False,
        "last_sync_at": None,
        "unpushed_commits": 0,
        "conflicting_facts": [],
    }

    data_dir = store.data_dir
    if not (data_dir / ".git").exists():
        counts["sync"] = sync_info
        return

    # Remote configured?
    remotes_proc = subprocess.run(
        ["git", "remote"], cwd=data_dir, capture_output=True, text=True, check=False
    )
    sync_info["remote_configured"] = bool(remotes_proc.stdout.strip())

    # Last successful sync timestamp (from local state file).
    state = read_sync_state(data_dir)
    if state:
        sync_info["last_sync_at"] = state.get("completed_at")

    # Local commits ahead of upstream — read-only, no network.
    if sync_info["remote_configured"]:
        ahead_proc = subprocess.run(
            ["git", "rev-list", "--count", "@{upstream}..HEAD"],
            cwd=data_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if ahead_proc.returncode == 0:
            try:
                sync_info["unpushed_commits"] = int(ahead_proc.stdout.strip() or "0")
            except ValueError:
                sync_info["unpushed_commits"] = 0

    counts["sync"] = sync_info


def _check_config(issues: list[DoctorIssue]) -> None:
    settings = get_settings()
    if not settings.llm_model:
        issues.append(
            DoctorIssue(
                code="missing_llm_model",
                severity="error",
                category="config",
                message="ENGRAM_LLM_MODEL is empty.",
                repair="Set ENGRAM_LLM_MODEL to a litellm-supported model.",
            )
        )


def _check_facts_relationships(
    facts: list[Fact],
    issues: list[DoctorIssue],
) -> None:
    """Detect duplicates, broken supersession, and stale/forgotten conflicts."""
    if not facts:
        return

    issues.extend(issue for issue in _fact_relationship_issues(facts) if issue)


def _fact_relationship_issues(facts: list[Fact]) -> list[DoctorIssue | None]:
    fact_ids = {fact.id for fact in facts}
    edges: dict[str, str] = {
        fact.id: fact.supersedes for fact in facts if fact.supersedes
    }
    return [
        _duplicate_fact_issue(facts),
        _orphaned_supersession_issue(facts, fact_ids),
        _circular_supersession_issue(edges),
        _stale_forgotten_issue(facts),
    ]


def _duplicate_fact_issue(facts: list[Fact]) -> DoctorIssue | None:
    by_signature: dict[tuple[str | None, str, str], list[str]] = defaultdict(list)
    for fact in facts:
        if fact.confidence < MIN_ACTIVE_CONFIDENCE or fact.stale:
            continue
        sig = (
            fact.project,
            fact.category.value,
            " ".join(fact.content.lower().split()),
        )
        by_signature[sig].append(fact.id)
    duplicates: list[list[str]] = [ids for ids in by_signature.values() if len(ids) > 1]
    if not duplicates:
        return None

    flat: list[str] = []
    for group in duplicates:
        flat.extend(group)
    return DoctorIssue(
        code="duplicate_facts",
        severity="warning",
        category="facts",
        message=f"Detected {len(duplicates)} duplicate fact group(s).",
        ids=flat,
        repair="Use `merge_memories` to consolidate, or `forget` the duplicates.",
    )


def _orphaned_supersession_issue(
    facts: list[Fact],
    fact_ids: set[str],
) -> DoctorIssue | None:
    orphans = [
        fact.id for fact in facts if fact.supersedes and fact.supersedes not in fact_ids
    ]
    if not orphans:
        return None
    return DoctorIssue(
        code="orphaned_supersession",
        severity="warning",
        category="facts",
        message=f"{len(orphans)} fact(s) reference a missing supersession ancestor.",
        ids=orphans,
        repair=(
            "Run `engram doctor --repair --repair-orphaned-supersessions` "
            "to clear missing supersedes links."
        ),
        repairable=True,
    )


def _circular_supersession_issue(edges: dict[str, str]) -> DoctorIssue | None:
    cycles: list[str] = []
    for start in edges:
        seen: set[str] = set()
        node = start
        while node and node not in seen:
            seen.add(node)
            node = edges.get(node)
            if node == start:
                cycles.append(start)
                break
    if not cycles:
        return None
    return DoctorIssue(
        code="circular_supersession",
        severity="error",
        category="facts",
        message=f"{len(cycles)} circular supersession chain(s).",
        ids=cycles,
        repair="Break the cycle by editing one fact to remove its supersedes link.",
    )


def _stale_forgotten_issue(facts: list[Fact]) -> DoctorIssue | None:
    contradictory = [fact.id for fact in facts if fact.stale and fact.confidence == 0.0]
    if not contradictory:
        return None
    return DoctorIssue(
        code="stale_and_forgotten",
        severity="info",
        category="facts",
        message=(f"{len(contradictory)} fact(s) are marked both stale and forgotten."),
        ids=contradictory,
        repair="Pick one — usually `forget` is sufficient.",
    )


def _check_candidates(
    candidates: list[MemoryCandidate],
    issues: list[DoctorIssue],
) -> None:
    if not candidates:
        return
    now = datetime.now(timezone.utc)
    stale_pending = [
        c.id
        for c in candidates
        if c.status == CandidateStatus.pending and (now - c.updated_at).days > 30
    ]
    if stale_pending:
        issues.append(
            DoctorIssue(
                code="stale_pending_candidates",
                severity="info",
                category="candidates",
                message=(
                    f"{len(stale_pending)} candidate(s) have been pending for over 30 days."
                ),
                ids=stale_pending,
                repair="Approve, reject, or delete via the maintenance tools.",
            )
        )


async def check_provider() -> DoctorIssue | None:
    """Optional provider readiness check.

    Reports configuration problems and minimal-call failures separately from
    storage health. Returns ``None`` if the provider responds.
    """
    settings = get_settings()
    if not settings.llm_model:
        return DoctorIssue(
            code="provider_not_configured",
            severity="error",
            category="provider",
            message="No LLM model configured.",
            repair="Set ENGRAM_LLM_MODEL.",
        )

    try:
        from engram.llm import complete

        await complete(prompt="ping", system="reply 'pong'")
    except Exception as exc:
        return DoctorIssue(
            code="provider_call_failed",
            severity="error",
            category="provider",
            message=f"Minimal provider call failed: {exc}",
            repair="Check API keys, network, and rate limits.",
        )
    return None


def _record_provider_check(
    *,
    check_provider_flag: bool,
    provider_issue: DoctorIssue | None,
    issues: list[DoctorIssue],
    checks_run: list[str],
    checks_skipped: list[str],
) -> None:
    if check_provider_flag:
        if provider_issue is not None:
            issues.append(provider_issue)
        checks_run.append("provider")
    else:
        checks_skipped.append("provider")


def _doctor_status(issues: list[DoctorIssue]) -> str:
    severities = Counter(issue.severity for issue in issues)
    if severities.get("error"):
        return "error"
    if severities.get("warning"):
        return "warning"
    return "ok"


def _doctor_counts(
    *,
    facts: list[Fact],
    facts_valid: int,
    candidates_valid: int,
    issues: list[DoctorIssue],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "facts_valid": facts_valid,
        "candidates_valid": candidates_valid,
        "active_facts": sum(1 for fact in facts if _is_active_fact(fact, now)),
        "stale_facts": sum(1 for fact in facts if fact.stale),
        "forgotten_facts": sum(1 for fact in facts if fact.confidence == 0.0),
        "issues": len(issues),
    }


def run_doctor(
    store: FactStore | AsyncFactStore | None = None,
    *,
    check_provider_flag: bool = False,
    provider_issue: DoctorIssue | None = None,
) -> DoctorReport:
    """Run the synchronous doctor checks.

    ``check_provider_flag`` toggles whether the provider check should be
    represented in the report; the actual call is async and lives in
    :func:`check_provider` so callers can run it under their own event loop.
    """
    sync_store = _resolve_store(store)
    issues: list[DoctorIssue] = []
    checks_run: list[str] = []
    checks_skipped: list[str] = []

    _check_data_dir(sync_store, issues)
    checks_run.append("data_dir")

    facts_valid = _check_jsonl_integrity(sync_store.facts_path, "facts", Fact, issues)
    candidates_valid = _check_jsonl_integrity(
        sync_store.candidates_path, "candidates", MemoryCandidate, issues
    )
    checks_run.append("jsonl_integrity")

    _check_transactions(sync_store, issues)
    checks_run.append("transactions")

    _check_recall_log(sync_store, issues)
    checks_run.append("recall_log")

    _check_config(issues)
    checks_run.append("config")

    facts = sync_store.load_facts()
    candidates = sync_store.load_candidates()
    _check_facts_relationships(facts, issues)
    _check_candidates(candidates, issues)
    checks_run.extend(["facts_relationships", "candidates_lifecycle"])

    _record_provider_check(
        check_provider_flag=check_provider_flag,
        provider_issue=provider_issue,
        issues=issues,
        checks_run=checks_run,
        checks_skipped=checks_skipped,
    )
    counts = _doctor_counts(
        facts=facts,
        facts_valid=facts_valid,
        candidates_valid=candidates_valid,
        issues=issues,
    )

    _check_sync(sync_store, issues, counts=counts)
    checks_run.append("sync")

    return DoctorReport(
        status=_doctor_status(issues),
        issues=issues,
        counts=counts,
        checks_run=checks_run,
        checks_skipped=checks_skipped,
    )


def repair_store(
    store: FactStore | AsyncFactStore | None = None,
    *,
    repair_jsonl: bool = False,
    recover_transactions: bool = False,
    repair_orphaned_supersessions: bool = False,
) -> dict[str, Any]:
    """Run safe, opt-in repairs and return a summary."""
    sync_store = _resolve_store(store)
    summary: dict[str, Any] = {}

    if recover_transactions:
        summary["recovered_transactions"] = sync_store.recover_transactions()

    if repair_jsonl:
        summary["jsonl_repair"] = sync_store.repair()

    if repair_orphaned_supersessions:
        facts = sync_store.load_facts()
        fact_ids = {fact.id for fact in facts}
        orphan_ids = [
            fact.id
            for fact in facts
            if fact.supersedes and fact.supersedes not in fact_ids
        ]
        updated = sync_store.batch_update_facts(
            {fact_id: {"supersedes": None} for fact_id in orphan_ids}
        )
        summary["orphaned_supersessions"] = {
            "cleared": len(updated),
            "ids": [fact.id for fact in updated],
        }

    return summary


__all__ = [
    "DoctorIssue",
    "DoctorReport",
    "check_provider",
    "repair_store",
    "run_doctor",
]
