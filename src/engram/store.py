"""JSONL-based knowledge store for facts."""

import asyncio
import fcntl
import logging
import os
import re
import sys
import tempfile
import threading
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import ParamSpec, TypeVar

from uuid import uuid4

from pydantic import ValidationError

from engram.config import get_settings
from engram.models import (
    CandidateStatus,
    Fact,
    FactCategory,
    IngestionRecord,
    MemoryCandidate,
    RecallRecord,
    StoreTransaction,
    TransactionStatus,
)

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_THREAD_LOCKS: dict[Path, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
P = ParamSpec("P")
R = TypeVar("R")

# Lightweight suffix-stripping stemmer (no NLTK dependency)
_STEM_SUFFIXES = (
    "tion",
    "sion",
    "ment",
    "ness",
    "ible",
    "able",
    "ing",
    "ies",
    "ous",
    "ive",
    "ers",
    "ed",
    "ly",
    "es",
    "er",
    "al",
    "s",
)


def _stem(word: str) -> str:
    """Cheap suffix strip — good enough for prefilter scoring."""
    if len(word) <= 4:
        return word
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _content_hash(text: str) -> str:
    """Normalize and hash fact content for exact-match dedup."""
    return " ".join(text.lower().split())


def _thread_lock_for(path: Path) -> threading.RLock:
    """Return a process-local lock for the given filesystem lock path."""
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(path)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[path] = lock
        return lock


def format_facts_for_llm(facts: list[Fact]) -> str:
    """Format facts as a numbered list for LLM context."""
    if not facts:
        return "(no facts stored)"
    lines = []
    for i, fact in enumerate(facts, 1):
        meta = f"[{fact.category.value}]"
        if fact.project:
            meta += f" [{fact.project}]"
        if fact.confidence < 1.0:
            meta += f" [confidence: {fact.confidence:.1f}]"
        if fact.source_ref:
            meta += f" [source: {fact.source_ref}]"
        if fact.supersedes:
            meta += f" [supersedes: {fact.supersedes}]"
        lines.append(f"{i}. {meta} {fact.content} (id: {fact.id})")
    return "\n".join(lines)


@contextmanager
def _locked_write(path: Path):
    """Acquire an exclusive flock on <path>.lock before writing."""
    if sys.platform == "win32":
        logger.warning(
            "File locking not supported on Windows; concurrent writes are not serialized"
        )
        yield
        return
    lock_path = path.with_suffix(".lock")
    lock_fd = lock_path.open("a")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


@contextmanager
def _locked_store(data_dir: Path):
    """Acquire the coarse store lock for multi-file operations."""
    lock_path = data_dir / "store.lock"
    thread_lock = _thread_lock_for(lock_path)
    with thread_lock:
        if sys.platform == "win32":
            logger.warning(
                "File locking not supported on Windows; store transactions are not serialized"
            )
            yield
            return

        lock_fd = lock_path.open("a")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()


class FactStore:
    """Read/write/filter operations on the JSONL fact store."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or get_settings().data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Cache: fact.id -> (updated_at_iso, unigrams, bigrams)
        self._tok_cache: dict[str, tuple[str, set[str], set[str]]] = {}
        self.recover_transactions()

    @property
    def facts_path(self) -> Path:
        return self.data_dir / "facts.jsonl"

    @property
    def ingestion_log_path(self) -> Path:
        return self.data_dir / "ingestion_log.jsonl"

    @property
    def candidates_path(self) -> Path:
        return self.data_dir / "candidates.jsonl"

    @property
    def transaction_log_path(self) -> Path:
        return self.data_dir / "transactions.jsonl"

    def load_facts(self) -> list[Fact]:
        """Load all facts from JSONL, skipping corrupt lines."""
        if not self.facts_path.exists():
            return []
        facts = []
        for lineno, line in enumerate(self.facts_path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                facts.append(Fact.model_validate_json(line))
            except (ValueError, ValidationError) as exc:
                logger.warning("Skipping corrupt fact at line %d: %s", lineno, exc)
        return facts

    def load_candidates(
        self,
        status: CandidateStatus | None = None,
        project: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryCandidate]:
        """Load memory candidates filtered by status and project."""
        if not self.candidates_path.exists():
            return []

        candidates = []
        for lineno, line in enumerate(self.candidates_path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = MemoryCandidate.model_validate_json(line)
            except (ValueError, ValidationError) as exc:
                logger.warning("Skipping corrupt candidate at line %d: %s", lineno, exc)
                continue
            if status and candidate.status != status:
                continue
            if project and candidate.project and candidate.project != project:
                continue
            candidates.append(candidate)

        candidates.sort(key=lambda c: c.updated_at, reverse=True)
        if limit:
            candidates = candidates[:limit]
        return candidates

    def load_active_facts(
        self,
        category: FactCategory | None = None,
        project: str | None = None,
        include_global: bool = True,
        min_confidence: float = 0.1,
        limit: int | None = None,
        include_stale: bool = False,
    ) -> list[Fact]:
        """Load facts filtered by confidence, category, project, and stale status.

        Stale facts are excluded by default. Pass ``include_stale=True`` for
        inspection workflows that need to see the full set.
        """
        now = datetime.now(timezone.utc)
        facts = []
        for fact in self.load_facts():
            if fact.confidence < min_confidence:
                continue
            if fact.expires_at and fact.expires_at < now:
                continue
            if not include_stale and getattr(fact, "stale", False):
                continue
            if category and fact.category != category:
                continue
            if project:
                if fact.project != project and not (
                    include_global and fact.project is None
                ):
                    continue
            facts.append(fact)

        # Sort by recency
        facts.sort(key=lambda f: f.updated_at, reverse=True)
        if limit:
            facts = facts[:limit]
        return facts

    def prefilter_facts(
        self,
        query: str,
        project: str | None = None,
        limit: int | None = None,
    ) -> list[tuple[int, Fact]]:
        """Prefilter facts deterministically before agentic retrieval.

        Returns list of (score, Fact) tuples sorted by score descending.
        Score is included so the retriever can use it for tier selection.
        """
        facts = self.load_active_facts(project=project)
        if not facts:
            return []

        query_unigrams, query_bigrams = self._tokenize_extended(query)
        if not query_unigrams:
            return [(0, f) for f in (facts[:limit] if limit else facts)]

        now = datetime.now(timezone.utc)
        recency_cutoff = now - timedelta(days=7)

        scored: list[tuple[int, Fact]] = []
        for fact in facts:
            evidence_score = 0
            content_unigrams, content_bigrams = self._get_cached_tokens(fact)

            # Unigram overlap (stemmed)
            overlap = len(query_unigrams & content_unigrams)
            evidence_score += overlap * 5

            # Bigram overlap — catches "coding style" vs "coding_style"
            bigram_overlap = len(query_bigrams & content_bigrams)
            evidence_score += bigram_overlap * 3

            # Tag overlap (stem query tokens, compare to tags)
            tag_set = {_stem(t) for t in fact.tags}
            tag_overlap = len(query_unigrams & tag_set)
            evidence_score += tag_overlap * 4

            if fact.project and fact.project in query.lower():
                evidence_score += 6
            if any(token in fact.category.value for token in query_unigrams):
                evidence_score += 3

            score = evidence_score
            if fact.supersedes:
                score += 1
            if fact.expires_at:
                score += 1

            # Recency boost
            if fact.updated_at >= recency_cutoff:
                score += 2

            # Confidence boost
            if fact.confidence > 0.8:
                score += 1

            if evidence_score > 0:
                scored.append((score, fact))

        if not scored:
            return [(0, f) for f in (facts[:limit] if limit else facts)]

        scored.sort(
            key=lambda item: (item[0], item[1].updated_at.timestamp()),
            reverse=True,
        )
        if limit and len(scored) < limit:
            seen_ids = {fact.id for _, fact in scored}
            scored.extend((0, fact) for fact in facts if fact.id not in seen_ids)
        return scored[:limit] if limit else scored

    def append_facts(self, facts: list[Fact]) -> None:
        """Append facts to the JSONL store."""
        with _locked_write(self.facts_path):
            self._ensure_trailing_newline(self.facts_path)
            with self.facts_path.open("a") as f:
                for fact in facts:
                    f.write(fact.model_dump_json() + "\n")
        logger.info("Appended %d facts to store", len(facts))

    def append_candidates(self, candidates: list[MemoryCandidate]) -> None:
        """Append memory candidates to the queue."""
        with _locked_write(self.candidates_path):
            self._ensure_trailing_newline(self.candidates_path)
            with self.candidates_path.open("a") as f:
                for candidate in candidates:
                    f.write(candidate.model_dump_json() + "\n")
        logger.info("Appended %d memory candidate(s) to queue", len(candidates))

    def update_fact(self, fact_id: str, **updates) -> Fact | None:
        """Update a fact in-place by rewriting the JSONL file.

        Returns None if the fact is not found or is forgotten (confidence == 0.0),
        unless the update explicitly sets confidence (e.g. forget/restore operations).
        """
        facts = self.load_facts()
        updated = None
        for i, fact in enumerate(facts):
            if fact.id == fact_id:
                # Refuse to edit forgotten facts unless the caller is changing confidence
                if fact.confidence == 0.0 and "confidence" not in updates:
                    return None
                for key, value in updates.items():
                    setattr(fact, key, value)
                fact.updated_at = datetime.now(timezone.utc)
                facts[i] = fact
                updated = fact
                break

        if updated:
            self._rewrite(facts)
        return updated

    def update_candidate(self, candidate_id: str, **updates) -> MemoryCandidate | None:
        """Update a candidate in-place by rewriting the candidate JSONL file."""
        candidates = self.load_candidates()
        updated = None
        for i, candidate in enumerate(candidates):
            if candidate.id == candidate_id:
                for key, value in updates.items():
                    setattr(candidate, key, value)
                candidate.updated_at = datetime.now(timezone.utc)
                candidates[i] = candidate
                updated = candidate
                break

        if updated:
            self._rewrite(candidates, path=self.candidates_path)
        return updated

    def rename_project(self, old_project: str, new_project: str) -> int:
        """Bulk-rename the project field on all facts and candidates."""
        # Rename in facts
        facts = self.load_facts()
        fact_count = 0
        now = datetime.now(timezone.utc)
        for i, fact in enumerate(facts):
            if fact.project == old_project:
                fact.project = new_project
                fact.updated_at = now
                facts[i] = fact
                fact_count += 1
        if fact_count > 0:
            self._rewrite(facts)
            logger.info(
                "Renamed project %s → %s for %d fact(s)",
                old_project,
                new_project,
                fact_count,
            )

        # Rename in candidates
        candidates = self.load_candidates()
        cand_count = 0
        for i, candidate in enumerate(candidates):
            if candidate.project == old_project:
                candidate.project = new_project
                candidate.updated_at = now
                candidates[i] = candidate
                cand_count += 1
        if cand_count > 0:
            self._rewrite(candidates, path=self.candidates_path)
            logger.info(
                "Renamed project %s → %s for %d candidate(s)",
                old_project,
                new_project,
                cand_count,
            )

        return fact_count + cand_count

    def batch_update_facts(self, updates: dict[str, dict]) -> list[Fact]:
        """Apply multiple fact updates in a single JSONL rewrite.

        Args:
            updates: Mapping of fact_id -> field updates to apply.

        Returns:
            List of updated Fact objects.
        """
        if not updates:
            return []

        facts = self.load_facts()
        now = datetime.now(timezone.utc)
        updated: list[Fact] = []

        for i, fact in enumerate(facts):
            if fact.id in updates:
                for key, value in updates[fact.id].items():
                    setattr(fact, key, value)
                fact.updated_at = now
                facts[i] = fact
                updated.append(fact)

        if updated:
            self._rewrite(facts)
            logger.info("Batch-updated %d facts in single rewrite", len(updated))

        return updated

    def batch_update_candidates(
        self, updates: dict[str, dict]
    ) -> list[MemoryCandidate]:
        """Apply multiple candidate updates in a single JSONL rewrite.

        Args:
            updates: Mapping of candidate_id -> field updates to apply.

        Returns:
            List of updated MemoryCandidate objects.
        """
        if not updates:
            return []

        candidates = self.load_candidates()
        now = datetime.now(timezone.utc)
        updated: list[MemoryCandidate] = []

        for i, candidate in enumerate(candidates):
            if candidate.id in updates:
                for key, value in updates[candidate.id].items():
                    if key == "status":
                        value = CandidateStatus(value)
                    setattr(candidate, key, value)
                candidate.updated_at = now
                candidates[i] = candidate
                updated.append(candidate)

        if updated:
            self._rewrite(candidates, path=self.candidates_path)
            logger.info("Batch-updated %d candidates in single rewrite", len(updated))

        return updated

    def forget(self, fact_id: str, reason: str = "") -> Fact | None:
        """Soft-delete a fact by setting confidence to 0."""
        fact = self.update_fact(fact_id, confidence=0.0)
        if fact and reason:
            logger.info("Forgot fact %s: %s", fact_id, reason)
        return fact

    def correct_fact(
        self,
        fact_id: str,
        new_content: str,
        *,
        category: FactCategory | None = None,
        tags: list[str] | None = None,
        project: str | None = None,
        reason: str = "",
    ) -> Fact | None:
        """Replace ``fact_id`` with a new active fact that supersedes it.

        The original fact is preserved (its confidence is reduced so it falls
        out of active recall) and a new fact with ``supersedes=fact_id`` is
        created. The new fact inherits the original's category/project/tags
        unless overrides are supplied.

        Returns the new fact, or ``None`` if the original fact was not found
        or was already forgotten.
        """
        existing = next((f for f in self.load_facts() if f.id == fact_id), None)
        if existing is None or existing.confidence == 0.0:
            return None

        now = datetime.now(timezone.utc)
        new_fact = Fact(
            id=uuid4().hex[:12],
            category=category or existing.category,
            content=new_content,
            source=existing.source,
            confidence=existing.confidence,
            created_at=now,
            updated_at=now,
            observed_at=now,
            project=project if project is not None else existing.project,
            tags=tags if tags is not None else list(existing.tags),
            supersedes=fact_id,
            evidence_kind=existing.evidence_kind,
            why_store=reason or existing.why_store,
        )
        self.append_facts([new_fact])
        # Mark the old fact superseded by reducing its confidence so default
        # recall paths (min_confidence=0.1) skip it. Audit trail is preserved
        # via the supersedes link on the new fact.
        self.update_fact(fact_id, confidence=0.05)
        self.log_ingestion(
            IngestionRecord(
                source="correct_fact",
                facts_created=[new_fact.id],
                facts_updated=[fact_id],
                agent_model="manual_correction",
            )
        )
        logger.info("Corrected fact %s -> %s: %s", fact_id, new_fact.id, reason)
        return new_fact

    def merge_facts(
        self,
        source_ids: list[str],
        merged_content: str,
        *,
        category: FactCategory | None = None,
        tags: list[str] | None = None,
        project: str | None = None,
        reason: str = "",
    ) -> tuple[Fact, list[str]] | None:
        """Consolidate multiple active facts into one new fact.

        Creates a single new active fact that supersedes the first source and
        marks every other source fact superseded as well (by reducing their
        confidence). Returns ``(new_fact, superseded_ids)`` on success.

        Returns ``None`` if fewer than two valid active source facts are
        provided or the source set is invalid.
        """
        if len(source_ids) < 2:
            return None

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique_ids = [sid for sid in source_ids if not (sid in seen or seen.add(sid))]
        if len(unique_ids) < 2:
            return None

        all_facts = {f.id: f for f in self.load_facts()}
        sources: list[Fact] = []
        for sid in unique_ids:
            fact = all_facts.get(sid)
            if fact is None or fact.confidence == 0.0:
                return None
            sources.append(fact)

        # Same project scope across sources or merge is ambiguous.
        scopes = {f.project for f in sources}
        if len(scopes) > 1:
            return None

        primary = sources[0]
        now = datetime.now(timezone.utc)
        new_fact = Fact(
            id=uuid4().hex[:12],
            category=category or primary.category,
            content=merged_content,
            source=primary.source,
            confidence=max(f.confidence for f in sources),
            created_at=now,
            updated_at=now,
            observed_at=now,
            project=project if project is not None else primary.project,
            tags=tags if tags is not None else list(primary.tags),
            supersedes=primary.id,
            evidence_kind=primary.evidence_kind,
            why_store=reason or "merged",
        )
        self.append_facts([new_fact])

        # Reduce confidence on every source so they fall out of active recall.
        updates = {sid: {"confidence": 0.05} for sid in unique_ids}
        self.batch_update_facts(updates)

        self.log_ingestion(
            IngestionRecord(
                source="merge_facts",
                facts_created=[new_fact.id],
                facts_updated=list(unique_ids),
                agent_model="manual_merge",
            )
        )
        logger.info("Merged %d facts -> %s: %s", len(unique_ids), new_fact.id, reason)
        return new_fact, unique_ids

    def mark_stale(self, fact_id: str, reason: str = "") -> Fact | None:
        """Mark a fact stale so it is excluded from active recall.

        Stale facts remain inspectable and keep their supersession history.
        """
        existing = next((f for f in self.load_facts() if f.id == fact_id), None)
        if existing is None:
            return None
        fact = self.update_fact(fact_id, stale=True, stale_reason=reason)
        if fact:
            self.log_ingestion(
                IngestionRecord(
                    source="mark_stale",
                    facts_updated=[fact_id],
                    agent_model="manual_stale",
                )
            )
            logger.info("Marked fact %s stale: %s", fact_id, reason)
        return fact

    def unmark_stale(self, fact_id: str) -> Fact | None:
        """Reverse a stale marking on a fact."""
        return self.update_fact(fact_id, stale=False, stale_reason="")

    def approve_candidates(self, candidate_ids: list[str]) -> list[Fact]:
        """Approve candidates and promote them into active facts with recovery."""
        with _locked_store(self.data_dir):
            self._recover_transactions_locked()
            transaction = self._prepare_approval_transaction(candidate_ids)
            if not transaction:
                return []

            self._append_transaction(transaction)
            self._apply_approval_transaction(transaction)
            self._commit_transaction(transaction)
            return transaction.new_facts

    def reject_candidates(
        self, candidate_ids: list[str], reason: str = ""
    ) -> list[MemoryCandidate]:
        """Reject candidates without promoting them into active facts."""
        pending_candidates = {
            c.id: c for c in self.load_candidates(status=CandidateStatus.pending)
        }
        updates: dict[str, dict] = {}
        for candidate_id in candidate_ids:
            if candidate_id in pending_candidates:
                updates[candidate_id] = {
                    "status": CandidateStatus.rejected,
                    "review_note": reason,
                }
        return self.batch_update_candidates(updates)

    def recover_transactions(self) -> int:
        """Roll forward any prepared store transactions that lack a commit marker."""
        with _locked_store(self.data_dir):
            return self._recover_transactions_locked()

    def _recover_transactions_locked(self) -> int:
        recovered = 0
        for transaction in self._pending_transactions():
            logger.warning(
                "Recovering prepared %s transaction %s",
                transaction.type,
                transaction.id,
            )
            self._apply_transaction(transaction)
            self._commit_transaction(transaction)
            recovered += 1
        return recovered

    def _prepare_approval_transaction(
        self, candidate_ids: list[str]
    ) -> StoreTransaction | None:
        pending_candidates = {
            c.id: c for c in self.load_candidates(status=CandidateStatus.pending)
        }

        fact_updates: dict[str, dict] = {}
        candidate_updates: dict[str, dict] = {}
        new_facts: list[Fact] = []
        now = datetime.now(timezone.utc)
        seen_ids: set[str] = set()
        selected_ids: list[str] = []

        for candidate_id in candidate_ids:
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)

            candidate = pending_candidates.get(candidate_id)
            if not candidate:
                continue
            selected_ids.append(candidate_id)

            candidate_updates[candidate_id] = {"status": CandidateStatus.approved}

            if candidate.supersedes:
                fact_updates[candidate.supersedes] = {"confidence": 0.3}

            data = candidate.model_dump(exclude={"status", "review_note"})
            data.update(id=uuid4().hex[:12], created_at=now, updated_at=now)
            new_facts.append(Fact(**data))

        if not new_facts:
            return None

        return StoreTransaction(
            type="approve_candidates",
            status=TransactionStatus.prepared,
            candidate_ids=selected_ids,
            fact_updates=fact_updates,
            candidate_updates=candidate_updates,
            new_facts=new_facts,
            created_at=now,
        )

    def _apply_transaction(self, transaction: StoreTransaction) -> None:
        if transaction.type != "approve_candidates":
            logger.warning(
                "Skipping unsupported transaction %s of type %s",
                transaction.id,
                transaction.type,
            )
            return
        self._apply_approval_transaction(transaction)

    def _apply_approval_transaction(self, transaction: StoreTransaction) -> None:
        missing_facts = self._missing_facts(transaction.new_facts)
        if missing_facts:
            self.append_facts(missing_facts)
        if transaction.fact_updates:
            self.batch_update_facts(transaction.fact_updates)
        if transaction.candidate_updates:
            self.batch_update_candidates(transaction.candidate_updates)

    def _missing_facts(self, facts: list[Fact]) -> list[Fact]:
        existing_ids = {fact.id for fact in self.load_facts()}
        return [fact for fact in facts if fact.id not in existing_ids]

    def _commit_transaction(self, transaction: StoreTransaction) -> None:
        committed = transaction.model_copy(
            update={
                "status": TransactionStatus.committed,
                "committed_at": datetime.now(timezone.utc),
            }
        )
        self._append_transaction(committed)

    def _append_transaction(self, transaction: StoreTransaction) -> None:
        with _locked_write(self.transaction_log_path):
            self._ensure_trailing_newline(self.transaction_log_path)
            with self.transaction_log_path.open("a") as f:
                f.write(transaction.model_dump_json() + "\n")
                f.flush()
                os.fsync(f.fileno())

    def _load_transactions(self) -> list[StoreTransaction]:
        if not self.transaction_log_path.exists():
            return []

        transactions: list[StoreTransaction] = []
        for lineno, line in enumerate(
            self.transaction_log_path.read_text().splitlines(), 1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                transactions.append(StoreTransaction.model_validate_json(line))
            except (ValueError, ValidationError) as exc:
                logger.warning(
                    "Skipping corrupt transaction at line %d: %s",
                    lineno,
                    exc,
                )
        return transactions

    def _pending_transactions(self) -> list[StoreTransaction]:
        prepared: dict[str, StoreTransaction] = {}
        committed_ids: set[str] = set()
        for transaction in self._load_transactions():
            if transaction.status == TransactionStatus.prepared:
                prepared[transaction.id] = transaction
            elif transaction.status == TransactionStatus.committed:
                committed_ids.add(transaction.id)
        return [
            transaction
            for transaction_id, transaction in prepared.items()
            if transaction_id not in committed_ids
        ]

    def log_ingestion(self, record: IngestionRecord) -> None:
        """Append an ingestion record to the audit log."""
        with _locked_write(self.ingestion_log_path):
            with self.ingestion_log_path.open("a") as f:
                f.write(record.model_dump_json() + "\n")

    def purge(self) -> dict:
        """Remove forgotten and expired facts from the JSONL file.

        Returns counts of purged and retained facts.
        """
        facts = self.load_facts()
        now = datetime.now(timezone.utc)
        kept = []
        purged = 0
        for fact in facts:
            is_forgotten = fact.confidence < 0.1
            is_expired = fact.expires_at is not None and fact.expires_at < now
            if is_forgotten or is_expired:
                purged += 1
            else:
                kept.append(fact)

        if purged > 0:
            self._rewrite(kept)
            logger.info("Purged %d facts (%d retained)", purged, len(kept))

        return {"purged": purged, "retained": len(kept)}

    def stats(self) -> dict:
        """Get store statistics."""
        facts = self.load_facts()
        now = datetime.now(timezone.utc)
        active = [
            f
            for f in facts
            if f.confidence >= 0.1 and not (f.expires_at and f.expires_at < now)
        ]
        forgotten = [f for f in facts if f.confidence < 0.1]
        expired = [
            f
            for f in facts
            if f.confidence >= 0.1 and f.expires_at and f.expires_at < now
        ]
        by_category = Counter(f.category.value for f in active)
        by_project = Counter(f.project or "(none)" for f in active)

        return {
            "total_facts": len(facts),
            "active_facts": len(active),
            "forgotten_facts": len(forgotten),
            "expired_facts": len(expired),
            "by_category": dict(by_category.most_common()),
            "by_project": dict(by_project.most_common(10)),
            "storage_bytes": self.facts_path.stat().st_size
            if self.facts_path.exists()
            else 0,
            "pending_candidates": len(
                self.load_candidates(status=CandidateStatus.pending)
            ),
        }

    def format_candidates_for_review(self, candidates: list[MemoryCandidate]) -> str:
        """Format candidates as a numbered list for review."""
        if not candidates:
            return "(no candidates queued)"

        lines = []
        for i, candidate in enumerate(candidates, 1):
            meta = f"[{candidate.status.value}] [{candidate.category.value}]"
            if candidate.project:
                meta += f" [{candidate.project}]"
            if candidate.supersedes:
                meta += f" [supersedes: {candidate.supersedes}]"
            lines.append(
                f"{i}. {meta} {candidate.content} "
                f"(id: {candidate.id}, why: {candidate.why_store or 'useful future context'})"
            )
        return "\n".join(lines)

    @property
    def recall_log_path(self) -> Path:
        return self.data_dir / "recall_log.jsonl"

    def log_recall(self, record: RecallRecord) -> None:
        """Append a recall record to the recall log."""
        with _locked_write(self.recall_log_path):
            with self.recall_log_path.open("a") as f:
                f.write(record.model_dump_json() + "\n")

    def load_recall_log(self, limit: int = 500) -> list[RecallRecord]:
        """Load recent recall log entries."""
        if not self.recall_log_path.exists():
            return []
        records = []
        for line in self.recall_log_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(RecallRecord.model_validate_json(line))
                except Exception:
                    continue
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:limit]

    def repair(self) -> dict:
        """Recover from truncated/corrupt JSONL lines.

        Rewrites facts.jsonl and candidates.jsonl keeping only parseable lines.
        Returns counts of valid and corrupt lines found.
        """
        result: dict[str, int] = {}
        for label, path, model_cls in [
            ("facts", self.facts_path, Fact),
            ("candidates", self.candidates_path, MemoryCandidate),
        ]:
            if not path.exists():
                result[f"{label}_valid"] = 0
                result[f"{label}_corrupt"] = 0
                continue

            valid = []
            corrupt = 0
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    valid.append(model_cls.model_validate_json(line))
                except Exception:
                    corrupt += 1
                    logger.warning("Corrupt %s line dropped: %s", label, line[:80])

            if corrupt > 0:
                self._rewrite(valid, path=path)
            result[f"{label}_valid"] = len(valid)
            result[f"{label}_corrupt"] = corrupt
        return result

    def _ensure_trailing_newline(self, path: Path) -> None:
        """Ensure the file ends with a newline to prevent JSON concatenation."""
        if not path.exists() or path.stat().st_size == 0:
            return
        with path.open("rb") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                with path.open("ab") as af:
                    af.write(b"\n")

    def _rewrite(
        self, records: list[Fact | MemoryCandidate], path: Path | None = None
    ) -> None:
        """Rewrite the entire JSONL file atomically with fsync and unique tmp path."""
        target_path = path or self.facts_path
        parent = target_path.parent
        name = target_path.stem
        tmp_path: Path | None = None
        with _locked_write(target_path):
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    dir=parent,
                    prefix=name,
                    suffix=".tmp",
                    delete=False,
                ) as f:
                    tmp_path = Path(f.name)
                    for record in records:
                        f.write(record.model_dump_json() + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, target_path)
                tmp_path = None  # Successfully replaced; don't clean up
            finally:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()

        # Evict cache entries for any facts removed by this rewrite
        if path is None or path == self.facts_path:
            kept_ids = {r.id for r in records if isinstance(r, Fact)}
            for stale_id in [k for k in self._tok_cache if k not in kept_ids]:
                del self._tok_cache[stale_id]

    def _get_cached_tokens(self, fact: Fact) -> tuple[set[str], set[str]]:
        """Return cached stemmed unigram/bigram sets, recomputing if stale."""
        updated_iso = fact.updated_at.isoformat()
        cached = self._tok_cache.get(fact.id)
        if cached and cached[0] == updated_iso:
            return cached[1], cached[2]
        unigrams, bigrams = self._tokenize_extended(fact.content)
        self._tok_cache[fact.id] = (updated_iso, unigrams, bigrams)
        return unigrams, bigrams

    def _tokenize_extended(self, text: str) -> tuple[set[str], set[str]]:
        """Tokenize text into stemmed unigrams and bigrams.

        Normalizes underscore/hyphen so "fact_store" and "fact-store" both
        produce the same tokens as "fact store".
        """
        normalized = text.lower().replace("_", " ").replace("-", " ")
        raw = _TOKEN_RE.findall(normalized)
        unigrams = {_stem(t) for t in raw}
        bigrams = {f"{raw[i]}_{raw[i + 1]}" for i in range(len(raw) - 1)}
        return unigrams, bigrams


class AsyncFactStore:
    """Async facade for running blocking JSONL store operations off the event loop."""

    def __init__(self, store: FactStore | None = None):
        self.sync_store = store or FactStore()

    @property
    def data_dir(self) -> Path:
        return self.sync_store.data_dir

    @property
    def facts_path(self) -> Path:
        return self.sync_store.facts_path

    @property
    def ingestion_log_path(self) -> Path:
        return self.sync_store.ingestion_log_path

    @property
    def candidates_path(self) -> Path:
        return self.sync_store.candidates_path

    @property
    def transaction_log_path(self) -> Path:
        return self.sync_store.transaction_log_path

    @property
    def recall_log_path(self) -> Path:
        return self.sync_store.recall_log_path

    async def _run(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        return await asyncio.to_thread(func, *args, **kwargs)

    async def load_facts(self) -> list[Fact]:
        return await self._run(self.sync_store.load_facts)

    async def load_candidates(
        self,
        status: CandidateStatus | None = None,
        project: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryCandidate]:
        return await self._run(self.sync_store.load_candidates, status, project, limit)

    async def load_active_facts(
        self,
        category: FactCategory | None = None,
        project: str | None = None,
        include_global: bool = True,
        min_confidence: float = 0.1,
        limit: int | None = None,
        include_stale: bool = False,
    ) -> list[Fact]:
        return await self._run(
            self.sync_store.load_active_facts,
            category,
            project,
            include_global,
            min_confidence,
            limit,
            include_stale,
        )

    async def prefilter_facts(
        self,
        query: str,
        project: str | None = None,
        limit: int | None = None,
    ) -> list[tuple[int, Fact]]:
        return await self._run(self.sync_store.prefilter_facts, query, project, limit)

    async def append_facts(self, facts: list[Fact]) -> None:
        await self._run(self.sync_store.append_facts, facts)

    async def append_candidates(self, candidates: list[MemoryCandidate]) -> None:
        await self._run(self.sync_store.append_candidates, candidates)

    async def update_fact(self, fact_id: str, **updates) -> Fact | None:
        return await self._run(self.sync_store.update_fact, fact_id, **updates)

    async def update_candidate(
        self, candidate_id: str, **updates
    ) -> MemoryCandidate | None:
        return await self._run(
            self.sync_store.update_candidate, candidate_id, **updates
        )

    async def rename_project(self, old_project: str, new_project: str) -> int:
        return await self._run(self.sync_store.rename_project, old_project, new_project)

    async def batch_update_facts(self, updates: dict[str, dict]) -> list[Fact]:
        return await self._run(self.sync_store.batch_update_facts, updates)

    async def batch_update_candidates(
        self, updates: dict[str, dict]
    ) -> list[MemoryCandidate]:
        return await self._run(self.sync_store.batch_update_candidates, updates)

    async def forget(self, fact_id: str, reason: str = "") -> Fact | None:
        return await self._run(self.sync_store.forget, fact_id, reason)

    async def correct_fact(
        self,
        fact_id: str,
        new_content: str,
        **kwargs,
    ) -> Fact | None:
        return await self._run(
            self.sync_store.correct_fact, fact_id, new_content, **kwargs
        )

    async def merge_facts(
        self,
        source_ids: list[str],
        merged_content: str,
        **kwargs,
    ) -> tuple[Fact, list[str]] | None:
        return await self._run(
            self.sync_store.merge_facts, source_ids, merged_content, **kwargs
        )

    async def mark_stale(self, fact_id: str, reason: str = "") -> Fact | None:
        return await self._run(self.sync_store.mark_stale, fact_id, reason)

    async def unmark_stale(self, fact_id: str) -> Fact | None:
        return await self._run(self.sync_store.unmark_stale, fact_id)

    async def approve_candidates(self, candidate_ids: list[str]) -> list[Fact]:
        return await self._run(self.sync_store.approve_candidates, candidate_ids)

    async def reject_candidates(
        self, candidate_ids: list[str], reason: str = ""
    ) -> list[MemoryCandidate]:
        return await self._run(self.sync_store.reject_candidates, candidate_ids, reason)

    async def recover_transactions(self) -> int:
        return await self._run(self.sync_store.recover_transactions)

    async def log_ingestion(self, record: IngestionRecord) -> None:
        await self._run(self.sync_store.log_ingestion, record)

    async def purge(self) -> dict:
        return await self._run(self.sync_store.purge)

    async def stats(self) -> dict:
        return await self._run(self.sync_store.stats)

    def format_candidates_for_review(self, candidates: list[MemoryCandidate]) -> str:
        return self.sync_store.format_candidates_for_review(candidates)

    async def log_recall(self, record: RecallRecord) -> None:
        await self._run(self.sync_store.log_recall, record)

    async def load_recall_log(self, limit: int = 500) -> list[RecallRecord]:
        return await self._run(self.sync_store.load_recall_log, limit)

    async def repair(self) -> dict:
        return await self._run(self.sync_store.repair)
