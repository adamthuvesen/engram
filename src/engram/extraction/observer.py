"""Ingest: turn raw text into memory cards reconciled against existing memory.

One LLM call extracts cards from the input and, in the same pass, decides how
each relates to the nearest existing cards (new, duplicate, or replacement)
and which existing cards the input shows are no longer true. The result is
applied as one atomic change set, or queued as candidates for review.
"""

import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from engram.core.config import get_settings
from engram.core.models import (
    CandidateStatus,
    Durability,
    EvidenceKind,
    Fact,
    FactCategory,
    MemoryCandidate,
)
from engram.core.projects import canonical_project
from engram.core.structured_outputs import (
    ExcludedClaim,
    ExtractedFact,
    ExtractionResponse,
)
from engram.llm import complete_model
from engram.storage.store import AsyncFactStore, ChangeSet, FactStore, _content_hash

logger = logging.getLogger(__name__)

NEIGHBOR_LIMIT = 25
NEIGHBOR_QUERY_CHARS = 4000

INGEST_SYSTEM = """You maintain a durable memory store for coding agents. From the
INPUT, extract the minimum set of memory cards worth keeping, and reconcile each
with the EXISTING CARDS shown to you.

WHAT TO STORE
Store a card only when it is durable, non-obvious knowledge that will change how a
future agent behaves: a policy, a decision with its rationale, a workflow, a
pitfall with its remedy, a stable preference, a correction, or a project
constraint. When in doubt, leave it out.

Never store (list these in excluded_claims instead):
- Test, lint, build, or CI outcomes ("ran make check, 158 tests green")
- Branch, PR, commit, merge, or release bookkeeping ("built on branch adam/feat-x")
- Progress reports and task status ("migration is 60% done", "next I will...")
- Implementation details readable from a named file or doc, unless the card
  records a pitfall or the rationale behind a decision
- Guesses, one-off failures, and secrets or credentials

HOW TO WRITE A CARD
- One card per future-use context. A card may hold several coupled clauses; split
  only when claims can be corrected, contradicted, expired, or acted on
  independently. Never split one policy or workflow into sentence fragments.
- Self-contained: name the subject explicitly (the project, tool, service, or
  component). Never write "this workflow", "the script", or "the repo".
- Third person ("The user prefers...", not "I prefer...").
- memory_key: a stable lowercase semantic identity such as "snowflake-read-only".
  When the card is about the same thing as an existing card, REUSE that card's
  memory_key exactly.
- retrieval_hints: 1-5 terms or questions a future agent might search for.
- covered_claims: the source claims this card preserves; map every stored claim
  to exactly one card.
- why_store: one short reason this changes future agent behavior.
- durability:
  - evergreen: identity and lasting personal preferences
  - durable: conventions, decisions, pitfalls, workflows (true until contradicted)
  - ephemeral: in-flight state, temporary workarounds, current incidents, anything
    true only for weeks
- anchors: repo-relative file paths (with extension) or code identifiers inside
  the project's git repository, explicitly named in the input, that an agent
  needs to act on this memory. Never database tables or warehouse objects,
  URLs, external IDs, directories, or files outside the repo. Empty when none
  are named. Never invent paths.
- expires_at: ISO datetime only when the input states an explicit time bound,
  else null.

RECONCILING WITH EXISTING CARDS
- replaces: IDs of existing cards this card updates, extends, merges, or
  contradicts. The new card replaces them entirely, so it must be complete on its
  own: carry forward every still-true detail from the cards it replaces. Only
  replace cards in the same project scope as the new card.
- duplicate_of: the ID of an existing card when the input adds nothing to it. The
  card is then skipped, so still fill its other fields briefly. Otherwise null.
- retire (top level): existing cards the input shows are no longer true and that
  no new card replaces, each with a short reason.
- Only use IDs listed in EXISTING CARDS. Each existing ID may appear in at most
  one card's replaces. Related but independent cards stay separate: do not merge
  complementary policies merely because they share terms.

categories:
- personal_info: user identity, role, team, responsibilities
- preference: how the user likes to work, tool and style choices
- event: incidents, investigations, time-bound states
- decision: architectural decisions and tradeoff resolutions
- pitfall: gotchas, failure modes, known bugs
- convention: naming, code style, process norms
- correction: corrections to prior knowledge
- assistant_info: how AI agents should behave for this user
- project: project-specific context, architecture, ownership
- workflow: reusable procedures, tool locations, process knowledge

Return JSON with "facts" (the cards), "retire", and "excluded_claims" (a list of
{"claim", "reason"} for source claims intentionally not stored).

Example: EXISTING CARDS holds [id:a1b2c3] preference "The user prefers pandas for
dataframes." and the input says "Switched to Polars for large datasets because of
lazy execution; pandas is still fine for small notebooks. Tests pass."
{"facts": [
  {"memory_key": "dataframe-library-preference", "content": "The user prefers Polars for large datasets because of its lazy execution, and still uses pandas for small notebooks.", "category": "preference", "project": null, "tags": ["python", "data"], "retrieval_hints": ["preferred dataframe library", "polars or pandas"], "covered_claims": ["Prefers Polars for large datasets because of lazy execution", "pandas is fine for small notebooks"], "why_store": "Guides future library choices", "durability": "durable", "anchors": [], "expires_at": null, "replaces": ["a1b2c3"], "duplicate_of": null}
], "retire": [], "excluded_claims": [
  {"claim": "Tests pass", "reason": "Test outcome"}
]}"""


class IngestResult(BaseModel):
    """What one ingest call stored, queued, reconciled, or left out."""

    created: list[Fact] = Field(default_factory=list)
    candidates: list[MemoryCandidate] = Field(default_factory=list)
    # Existing fact ID -> ID of the new fact replacing it.
    superseded: dict[str, str] = Field(default_factory=dict)
    # Existing fact ID -> why it was retired (marked stale).
    retired: dict[str, str] = Field(default_factory=dict)
    # Existing fact IDs the input merely restated.
    duplicates: list[str] = Field(default_factory=list)
    excluded: list[ExcludedClaim] = Field(default_factory=list)


async def ingest(
    content: str,
    *,
    source: str = "conversation",
    project: str | None = None,
    store: FactStore | AsyncFactStore,
    queue_for_review: bool = False,
) -> IngestResult:
    """Extract memory cards from ``content`` and reconcile them with the store.

    ``project`` must already be canonical. With ``queue_for_review`` the cards
    become pending candidates instead of active facts.

    Replacements and retirements only apply to targets unchanged since they
    were shown to the model. If one changed during the LLM call, ingest re-runs
    once against fresh neighbors; if that also conflicts, the new cards are
    stored without the conflicting targets rather than dropped.
    """
    if isinstance(store, FactStore):
        store = AsyncFactStore(store)

    plan = await _plan(content, source, project, store)
    if plan is None:
        return IngestResult()
    if queue_for_review:
        return IngestResult(
            candidates=await _queue_candidates(store, plan),
            duplicates=plan.duplicates,
            excluded=plan.excluded,
        )

    applied = await store.apply_changes(plan.change_set(source), all_or_nothing=True)
    if applied.skipped:
        logger.warning(
            "Ingest targets changed during reconciliation (%s); retrying once",
            ", ".join(sorted(applied.skipped)),
        )
        plan = await _plan(content, source, project, store)
        if plan is None:
            return IngestResult()
        applied = await store.apply_changes(
            plan.change_set(source), all_or_nothing=True
        )
    # Each pass removes the conflicting targets, so this ends at worst with the
    # new cards alone, which cannot conflict.
    while applied.skipped and plan.drop_targets(set(applied.skipped)):
        logger.warning(
            "Ingest targets changed again (%s); storing new cards without them",
            ", ".join(sorted(applied.skipped)),
        )
        applied = await store.apply_changes(
            plan.change_set(source), all_or_nothing=True
        )

    return IngestResult(
        created=applied.created,
        superseded=applied.superseded,
        retired=applied.staled,
        duplicates=plan.duplicates,
        excluded=plan.excluded,
    )


@dataclass
class _Plan:
    """Validated cards and reconciliation decisions from one LLM call."""

    neighbors: dict[str, Fact]
    # New cards, not yet linked to what they replace.
    drafts: list[Fact]
    # New card ID -> existing IDs it replaces.
    replaces: dict[str, list[str]]
    retire: dict[str, str]
    duplicates: list[str]
    excluded: list[ExcludedClaim]

    def linked_facts(self) -> list[Fact]:
        facts: list[Fact] = []
        for draft in self.drafts:
            replaced = self.replaces.get(draft.id, [])
            consolidates = _ordered_unique(
                [
                    *replaced,
                    *(i for old in replaced for i in self.neighbors[old].consolidates),
                ]
            )
            facts.append(
                draft.model_copy(
                    update={
                        "supersedes": replaced[0] if replaced else None,
                        "consolidates": consolidates,
                    }
                )
            )
        return facts

    def change_set(self, source: str) -> ChangeSet:
        supersede = {old: new for new, olds in self.replaces.items() for old in olds}
        targets = [*supersede, *self.retire]
        return ChangeSet(
            new_facts=self.linked_facts(),
            supersede=supersede,
            stale=dict(self.retire),
            expected={i: self.neighbors[i].updated_at for i in targets},
            reason=f"ingest from {source}",
            actor="ingest",
        )

    def drop_targets(self, ids: set[str]) -> bool:
        """Forget replacements and retirements of ``ids``; True if any were set."""
        before = sum(map(len, self.replaces.values())) + len(self.retire)
        self.replaces = {
            new: [old for old in olds if old not in ids]
            for new, olds in self.replaces.items()
        }
        self.retire = {i: r for i, r in self.retire.items() if i not in ids}
        return sum(map(len, self.replaces.values())) + len(self.retire) < before


async def _plan(
    content: str, source: str, project: str | None, store: AsyncFactStore
) -> _Plan | None:
    """Fetch neighbors, run the one LLM call, and validate its decisions."""
    hits = await store.search_facts(
        content[:NEIGHBOR_QUERY_CHARS], project=project, limit=NEIGHBOR_LIMIT
    )
    neighbors = {hit.fact.id: hit.fact for hit in hits}

    try:
        response = await complete_model(
            prompt=_ingest_prompt(content, project, list(neighbors.values())),
            system=INGEST_SYSTEM,
            response_model=ExtractionResponse,
        )
    except ValidationError as e:
        logger.warning("Skipping invalid ingest response: %s", e)
        return None

    duplicates: list[str] = []
    drafts = _consolidate_batch(
        _draft_facts(response.facts, source, project, neighbors, duplicates)
    )
    replaces = _resolve_replacements(drafts, neighbors)
    claimed = {old for olds in replaces.values() for old in olds}
    retire = _resolve_retirements(response, neighbors, claimed, project)
    logger.info(
        "Ingest: %d card(s), %d replacement(s), %d retirement(s), %d duplicate(s), "
        "%d excluded claim(s)",
        len(drafts),
        len(claimed),
        len(retire),
        len(duplicates),
        len(response.excluded_claims),
    )
    return _Plan(
        neighbors=neighbors,
        drafts=drafts,
        replaces=replaces,
        retire=retire,
        duplicates=duplicates,
        excluded=response.excluded_claims,
    )


def _ingest_prompt(content: str, project: str | None, neighbors: list[Fact]) -> str:
    if project is None:
        scope_instruction = (
            "Infer each card's project from the input. Use null only for truly "
            "cross-project or personal memories."
        )
    else:
        scope_instruction = (
            f"The caller fixed the scope to project {project!r}. Return that exact "
            "project value for every card."
        )
    existing = "\n".join(_format_neighbor(fact) for fact in neighbors) or "(none)"
    return f"{scope_instruction}\n\nEXISTING CARDS:\n{existing}\n\nINPUT:\n{content}"


def _format_neighbor(fact: Fact) -> str:
    project = fact.project if fact.project is not None else "global"
    key = fact.memory_key or "(none)"
    return (
        f"[id:{fact.id}] [project:{project}] [{fact.category.value}] "
        f"[memory_key:{key}] [observed:{fact.observed_at.date().isoformat()}] "
        f"{fact.content}"
    )


def _draft_facts(
    raw_facts: list[ExtractedFact],
    source: str,
    project: str | None,
    neighbors: dict[str, Fact],
    duplicates: list[str],
) -> list[Fact]:
    """Build facts from model cards, dropping restatements of existing cards.

    Until replacements are validated, ``consolidates`` holds the model's raw
    ``replaces`` IDs so sibling consolidation can merge them.
    """
    neighbor_by_hash = {
        _content_hash(fact.content): fact.id for fact in neighbors.values()
    }
    now = datetime.now(timezone.utc)
    ttl = timedelta(days=get_settings().ephemeral_ttl_days)
    source_group_id = uuid4().hex[:12]
    drafts: list[Fact] = []
    for raw in raw_facts:
        exact_id = neighbor_by_hash.get(_content_hash(raw.content))
        if exact_id is not None:
            duplicates.append(exact_id)
            continue
        if raw.duplicate_of is not None:
            if raw.duplicate_of in neighbors:
                duplicates.append(raw.duplicate_of)
                continue
            logger.warning(
                "Ignoring duplicate_of %r: not an existing card shown to the model",
                raw.duplicate_of,
            )
        expires_at = raw.expires_at
        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at is not None and expires_at <= now:
            logger.warning("Ignoring past expires_at %s", expires_at.isoformat())
            expires_at = None
        if expires_at is None and raw.durability == Durability.ephemeral:
            expires_at = now + ttl
        drafts.append(
            Fact(
                category=raw.category,
                memory_key=_normalize_memory_key(raw.memory_key),
                content=raw.content,
                source=source,
                project=project
                if project is not None
                else canonical_project(raw.project),
                tags=raw.tags,
                retrieval_hints=raw.retrieval_hints,
                created_at=now,
                updated_at=now,
                observed_at=now,
                expires_at=expires_at,
                durability=raw.durability,
                anchors=_ordered_unique(anchor.strip() for anchor in raw.anchors),
                consolidates=list(raw.replaces),
                evidence_kind=_infer_evidence_kind(source),
                source_ref=source,
                source_group_id=source_group_id,
                why_store=raw.why_store,
            )
        )
    return drafts


def _resolve_replacements(
    drafts: list[Fact], neighbors: dict[str, Fact]
) -> dict[str, list[str]]:
    """Map each draft ID to the existing IDs it validly replaces.

    A replaced ID must be an existing card shown to the model, in the same
    project as the new card, and not already claimed by an earlier card. The
    drafts carry the model's raw ``replaces`` in ``consolidates``.
    """
    claimed: set[str] = set()
    replaces: dict[str, list[str]] = {}
    for draft in drafts:
        replaced: list[str] = []
        for old_id in draft.consolidates:
            old = neighbors.get(old_id)
            if old is None:
                logger.warning("Ignoring replaces %r: not an existing card", old_id)
            elif old.project != draft.project:
                logger.warning(
                    "Ignoring replaces %r: project %r differs from new card's %r",
                    old_id,
                    old.project,
                    draft.project,
                )
            elif old_id in claimed:
                logger.warning("Ignoring replaces %r: already claimed", old_id)
            else:
                claimed.add(old_id)
                replaced.append(old_id)
        replaces[draft.id] = replaced
    return replaces


def _resolve_retirements(
    response: ExtractionResponse,
    neighbors: dict[str, Fact],
    claimed: set[str],
    project: str | None,
) -> dict[str, str]:
    """Keep retirements of shown cards within the caller's project scope.

    A project-scoped ingest may only retire that project's cards; global cards
    are retired only by an unscoped ingest.
    """
    retire: dict[str, str] = {}
    for item in response.retire:
        target = neighbors.get(item.id)
        if target is None:
            logger.warning("Ignoring retire %r: not an existing card", item.id)
        elif project is not None and target.project != project:
            logger.warning(
                "Ignoring retire %r: project %r is outside ingest scope %r",
                item.id,
                target.project,
                project,
            )
        elif item.id not in claimed:
            retire.setdefault(item.id, item.reason)
    return retire


async def _queue_candidates(
    store: AsyncFactStore, plan: _Plan
) -> list[MemoryCandidate]:
    pending = await store.load_candidates(status=CandidateStatus.pending, limit=200)
    pending_keys = {
        (candidate.project, candidate.category, candidate.content.lower())
        for candidate in pending
    }
    review_note = ""
    if plan.retire:
        review_note = "Also suggested retiring: " + "; ".join(
            f"{fact_id} ({reason})" for fact_id, reason in plan.retire.items()
        )
    candidates = [
        MemoryCandidate(
            **fact.model_dump(),
            replaces=plan.replaces.get(fact.id, []),
            review_note=review_note,
        )
        for fact in plan.linked_facts()
        if (fact.project, fact.category, fact.content.lower()) not in pending_keys
    ]
    if candidates:
        await store.append_candidates(candidates)
    return candidates


def _normalize_memory_key(value: str) -> str:
    """Normalize a model-supplied semantic identity into a stable key."""
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not normalized:
        raise ValueError("memory_key must contain at least one letter or number")
    return normalized[:120]


def _consolidate_batch(facts: list[Fact]) -> list[Fact]:
    """Combine sibling cards with the same identity and lifecycle.

    The prompt should normally emit one card per memory key. This
    deterministic pass preserves every clause when the model returns multiple
    fragments for the same card.
    """
    by_identity: dict[
        tuple[str | None, FactCategory, str, datetime | None], list[Fact]
    ] = defaultdict(list)
    seen_content: set[str] = set()

    for fact in facts:
        content_key = _content_hash(fact.content)
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        identity = (fact.project, fact.category, fact.memory_key, fact.expires_at)
        by_identity[identity].append(fact)

    return [_merge_sibling_cards(cards) for cards in by_identity.values()]


def _merge_sibling_cards(cards: list[Fact]) -> Fact:
    primary = cards[0]
    if len(cards) == 1:
        return primary

    return primary.model_copy(
        update={
            "content": " ".join(card.content.strip() for card in cards),
            "tags": _ordered_unique(tag for card in cards for tag in card.tags)[:5],
            "retrieval_hints": _ordered_unique(
                hint for card in cards for hint in card.retrieval_hints
            )[:5],
            "why_store": max((card.why_store for card in cards), key=len),
            "anchors": _ordered_unique(a for card in cards for a in card.anchors),
            "consolidates": _ordered_unique(
                i for card in cards for i in card.consolidates
            ),
        }
    )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _infer_evidence_kind(source: str) -> EvidenceKind:
    """Map source strings into a stable evidence kind."""
    if source.startswith("claude_code:"):
        return EvidenceKind.imported_memory
    if source == "conversation":
        return EvidenceKind.conversation
    if source.startswith("file:"):
        return EvidenceKind.file
    if source.startswith("tool:"):
        return EvidenceKind.tool_output
    return EvidenceKind.unknown
