"""Append-only candidate snapshots and deterministic offline review projection."""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from engram.core.models import CandidateStatus, FactCategory, MemoryCandidate
from engram.storage.jsonl import _load_jsonl_records
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
        loaded, _ = _load_jsonl_records(self.path, MemoryCandidate, "candidate", strict=strict)
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
        """Append memory candidates to the queue."""
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
