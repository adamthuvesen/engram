"""Append-only candidate snapshots and deterministic offline review projection."""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from engram.core import clock
from engram.core.fact_state import is_fact_eligible
from engram.core.models import (
    CandidateStatus,
    EventType,
    Fact,
    FactCategory,
    FactEvent,
    MemoryCandidate,
    StoreTransaction,
    TransactionStatus,
)
from engram.storage.jsonl import load_jsonl_records
from engram.storage.transactions import (
    ensure_trailing_newline,
)
from engram.storage.transactions import (
    locked_store as _locked_store,
)
from engram.storage.transactions import (
    locked_write as _locked_write,
)

logger = logging.getLogger(__name__)


def _candidate_key(candidate: MemoryCandidate) -> tuple[str | None, FactCategory, str]:
    return candidate.project, candidate.category, candidate.content.lower()


def _candidate_revision(candidate: MemoryCandidate) -> str:
    data = candidate.model_dump(mode="json", exclude={"status", "review_note", "updated_at"})
    if candidate.replaces_expected is None:
        data.pop("replaces_expected")
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def candidate_promotion_id(candidate: MemoryCandidate) -> str:
    return hashlib.sha256(
        f"candidate:{candidate.id}:{_candidate_revision(candidate)}".encode()
    ).hexdigest()[:24]


def _candidate_order(candidate: MemoryCandidate) -> tuple[datetime, int, str]:
    rank = {
        CandidateStatus.pending: 0,
        CandidateStatus.rejected: 1,
        CandidateStatus.approved: 2,
        CandidateStatus.conflicted: 3,
    }
    return candidate.updated_at, rank[candidate.status], candidate.model_dump_json()


class CandidateSnapshots:
    """Materialize reviews before filtering and serialize pending deduplication."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.path = data_dir / "candidates.jsonl"

    def load(
        self,
        status: CandidateStatus | None = None,
        project: str | None = None,
        limit: int | None = None,
        *,
        strict: bool = True,
    ) -> list[MemoryCandidate]:
        """Load memory candidates filtered by status and project."""
        loaded, _ = load_jsonl_records(self.path, MemoryCandidate, "candidate", strict=strict)
        by_id: dict[str, list[MemoryCandidate]] = {}
        for candidate in loaded:
            by_id.setdefault(candidate.id, []).append(candidate)
        candidates = []
        for rows in by_id.values():
            candidate = max(rows, key=_candidate_order).model_copy(deep=True)
            revision = _candidate_revision(candidate)
            decisions = {
                row.status
                for row in rows
                if _candidate_revision(row) == revision
                and row.status in {CandidateStatus.approved, CandidateStatus.rejected}
            }
            if len(decisions) > 1:
                candidate.status = CandidateStatus.conflicted
                candidate.review_note = "Conflicting approval and rejection decisions"
            if status and candidate.status != status:
                continue
            if project and candidate.project and candidate.project != project:
                continue
            candidates.append(candidate)

        candidates.sort(key=lambda c: c.updated_at, reverse=True)
        if limit:
            candidates = candidates[:limit]
        return candidates

    def append(self, candidates: list[MemoryCandidate]) -> None:
        with _locked_store(self.data_dir), _locked_write(self.path):
            ensure_trailing_newline(self.path)
            with self.path.open("a") as f:
                for candidate in candidates:
                    f.write(candidate.model_dump_json() + "\n")
                f.flush()
                os.fsync(f.fileno())
        logger.info("Appended %d memory candidate(s) to queue", len(candidates))

    def append_pending(self, candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
        """Append new pending content once across all materialized candidate states."""
        with _locked_store(self.data_dir):
            keys = {_candidate_key(row) for row in self.load(status=CandidateStatus.pending)}
            added = []
            for candidate in candidates:
                if candidate.status is not CandidateStatus.pending:
                    raise ValueError("Only pending candidates can be proposed")
                key = _candidate_key(candidate)
                if key not in keys:
                    keys.add(key)
                    added.append(candidate)
            self.append(added)
            return added


def prepare_approval(
    candidate_ids: list[str],
    pending_candidates: dict[str, MemoryCandidate],
    facts_by_id: dict[str, Fact],
    *,
    edits: dict[str, str] | None,
    unresolved: dict[str, str],
) -> StoreTransaction | None:
    """Build the approval transaction for pending candidates, without its event snapshot."""
    replacements: dict[str, list[str]] = {}
    for candidate_id in dict.fromkeys(candidate_ids):
        candidate = pending_candidates.get(candidate_id)
        if candidate is None:
            continue
        for source_id in set(
            candidate.replaces or ([candidate.supersedes] if candidate.supersedes else [])
        ):
            replacements.setdefault(source_id, []).append(candidate_id)
    for source_id, competing_ids in sorted(replacements.items()):
        if len(competing_ids) < 2:
            continue
        conflict = (
            f"Source {source_id} has competing replacement candidates: {', '.join(competing_ids)}"
        )
        for candidate_id in competing_ids:
            previous = unresolved.get(candidate_id)
            unresolved[candidate_id] = f"{previous}. {conflict}" if previous else conflict

    fact_updates: dict[str, dict[str, Any]] = {}
    supersessions: dict[str, str] = {}
    candidate_updates: dict[str, dict[str, Any]] = {}
    new_facts: list[Fact] = []
    now = clock.now()
    seen_ids: set[str] = set()
    selected_ids: list[str] = []

    for candidate_id in candidate_ids:
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)

        candidate = pending_candidates.get(candidate_id)
        if not candidate or candidate_id in unresolved:
            continue
        replaced = candidate.replaces or ([candidate.supersedes] if candidate.supersedes else [])
        if candidate.replaces_expected is not None and (
            set(candidate.replaces_expected) != set(replaced)
            or any(
                old_id not in facts_by_id
                or not is_fact_eligible(facts_by_id[old_id], now)
                or facts_by_id[old_id].updated_at != revision
                for old_id, revision in candidate.replaces_expected.items()
            )
        ):
            continue
        if edits and candidate_id in edits:
            candidate = MemoryCandidate.model_validate(
                {**candidate.model_dump(), "content": edits[candidate_id]}
            )
        selected_ids.append(candidate_id)

        candidate_updates[candidate_id] = {"status": CandidateStatus.approved}
        if edits and candidate_id in edits:
            candidate_updates[candidate_id]["content"] = candidate.content

        data = candidate.model_dump(
            exclude={"status", "review_note", "replaces", "replaces_expected"}
        )
        data.update(
            id=candidate_promotion_id(candidate),
            created_at=candidate.created_at,
            updated_at=candidate.created_at,
        )
        new_fact = Fact(**data)
        new_facts.append(new_fact)
        replaced = candidate.replaces or ([candidate.supersedes] if candidate.supersedes else [])
        for old_id in replaced:
            supersessions[old_id] = new_fact.id

    if not new_facts:
        return None

    return StoreTransaction(
        type="approve_candidates",
        status=TransactionStatus.prepared,
        candidate_ids=selected_ids,
        fact_updates=fact_updates,
        supersessions=supersessions,
        candidate_updates=candidate_updates,
        new_facts=new_facts,
        created_at=now,
    )


def approval_snapshot(
    transaction: StoreTransaction, rows: dict[str, MemoryCandidate]
) -> StoreTransaction:
    """Attach the fact events and candidate rows an approval transaction appends."""
    events = [
        FactEvent(
            event_type=EventType.created,
            fact_id=fact.id,
            timestamp=fact.created_at,
            payload=fact.model_dump(),
        )
        for fact in transaction.new_facts
    ]
    events.extend(
        FactEvent(
            event_type=EventType.superseded,
            fact_id=old,
            timestamp=transaction.created_at,
            payload={
                "superseded_by": new,
                "reason": "approved candidate replaces this memory",
            },
        )
        for old, new in transaction.supersessions.items()
    )
    for fact_id, updates in transaction.fact_updates.items():
        event_type = EventType.forgotten if updates == {"confidence": 0.0} else EventType.edited
        events.append(
            FactEvent(
                event_type=event_type,
                fact_id=fact_id,
                timestamp=transaction.created_at,
                payload={} if event_type is EventType.forgotten else updates,
            )
        )
    candidates = [
        MemoryCandidate.model_validate(
            {
                **rows[candidate_id].model_dump(),
                **updates,
                "updated_at": transaction.created_at,
            }
        )
        for candidate_id, updates in transaction.candidate_updates.items()
        if candidate_id in rows
    ]
    return transaction.model_copy(update={"events": events, "candidates": candidates})
