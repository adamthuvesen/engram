"""Observer agent — extracts structured facts from raw text."""

import logging
import re
from collections.abc import Iterable
from collections import defaultdict
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import ValidationError

from engram.llm import complete_model
from engram.core.models import (
    CandidateStatus,
    EvidenceKind,
    Fact,
    FactCategory,
    MemoryCandidate,
)
from engram.storage.store import (
    AsyncFactStore,
    FactStore,
    _content_hash,
    _stem,
    _TOKEN_RE,
)
from engram.core.structured_outputs import DedupResponse, ExtractionResponse

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """You are a durable-memory extraction agent. Extract the minimum set
of independently useful memory cards from the input.

A memory card captures one future-use context: a policy, decision with rationale,
workflow, pitfall with remedy, stable preference, correction, or project constraint.
It may contain several coupled clauses. Split cards only when the claims can be
corrected, contradicted, expired, or acted on independently. Never split a policy
or workflow into sentence-level fragments merely because it contains several rules.

Store a card only when it is durable, non-obvious, actionable, and likely to change
future agent behavior. Exclude progress updates, routine implementation details,
guesses, one-off failures, secrets, and facts already obvious from the referenced
code or docs. Preserve every durable claim by mapping it to exactly one card.

Each card must be:
- Self-contained and understandable without the original context
- Categorized into exactly one category
- Written in third person ("The user prefers..." not "I prefer...")
- Specific and actionable
- Assigned a stable semantic memory_key such as "agent-memory-policy" or
  "snowflake-read-only". Use the same key when later wording updates the same memory
- Given 1-5 retrieval_hints phrased as terms or questions a future agent may use
- Linked to the durable source claims it covers

Categories:
- personal_info: User identity, role, team, responsibilities
- preference: How the user likes to work, tool preferences, style choices
- event: Things that happened, time-bound states, incidents, investigations
- decision: Architectural decisions, tradeoff resolutions, choices made
- pitfall: Gotchas, things that don't work, known bugs, failure modes
- convention: Naming patterns, code style rules, process norms
- correction: Corrections or changes to prior knowledge
- assistant_info: Meta-knowledge about how AI agents should behave for this user
- project: Project-specific context, architecture, team ownership
- workflow: Reusable patterns, tool locations, process knowledge

Return a JSON object with a "facts" array. Each fact has:
- "memory_key": stable lowercase semantic identity for this memory card
- "content": the fact as a clear sentence
- "category": one of the categories above
- "project": repository or project scope when the claim is project-specific, null for global memories
- "tags": 1-3 relevant tags (lowercase)
- "retrieval_hints": 1-5 likely search phrases or questions
- "covered_claims": the durable source claims preserved by this card
- "why_store": one short reason this would be useful for future agent behavior
- "effective_at": ISO datetime if the fact became true at a known time, null otherwise
- "expires_at": ISO datetime if this is temporal/time-bound, null otherwise

Also return "excluded_claims", a list of {"claim": ..., "reason": ...} entries for
source claims intentionally omitted. This coverage ledger is reviewed but not stored.

Example output:
{"facts": [
  {"memory_key": "dataframe-library-preference", "content": "The user prefers Polars over pandas for large datasets because its execution model fits their workloads.", "category": "preference", "project": null, "tags": ["python", "data"], "retrieval_hints": ["preferred dataframe library", "polars or pandas", "large dataset tooling"], "covered_claims": ["Prefer Polars over pandas for large datasets", "The preference is based on workload fit"], "why_store": "Guides future library choices", "effective_at": null, "expires_at": null}
], "excluded_claims": [
  {"claim": "The current migration is 60% complete", "reason": "Transient progress state"}
]}"""

DEDUP_SYSTEM = """You are a memory-card deduplication agent. Given EXISTING cards and
NEW cards, identify which new cards are:
1. Duplicates of existing cards (same memory_key and same meaning)
2. Updates to existing cards (same memory_key, newer or materially changed meaning)
3. Genuinely new cards

Cards with different memory_key values are related but distinct unless one key is
clearly malformed. Do not merge complementary policies merely because they share
terms or tags.

Return a JSON object with:
- "new": list of indices (0-based) of genuinely new facts to add
- "updates": list of {"new_idx": int, "existing_id": str} for facts that supersede existing ones
- "duplicates": list of indices to skip"""


async def extract_facts(
    content: str,
    source: str = "conversation",
    project: str | None = None,
    store: FactStore | AsyncFactStore | None = None,
) -> list[Fact]:
    """Extract structured facts from raw text input.

    1. LLM extracts candidate facts from the input
    2. Dedup check against existing facts
    3. Returns new/updated facts ready to store
    """
    store = store or FactStore()
    candidates = await _extract_candidate_facts(content, source=source, project=project)
    if not candidates:
        return []

    existing = await _load_dedup_facts_for_store(store, project)
    if existing:
        candidates = await _dedup(candidates, existing, store)

    if candidates:
        await _append_facts(store, candidates)

    logger.info("Extracted %d facts from input", len(candidates))
    return candidates


async def suggest_memories(
    content: str,
    source: str = "conversation",
    project: str | None = None,
    store: FactStore | AsyncFactStore | None = None,
) -> list[MemoryCandidate]:
    """Extract and queue proposed memories for review."""
    store = store or FactStore()
    facts = await _extract_candidate_facts(content, source=source, project=project)
    if not facts:
        return []

    existing = await _load_dedup_facts_for_store(store, project)
    if existing:
        facts = await _dedup(facts, existing, store=None)

    pending = await _load_pending_candidates(store, limit=200)
    if pending:
        facts = _dedup_against_candidates(facts, pending)

    candidates = [MemoryCandidate(**fact.model_dump()) for fact in facts]

    if candidates:
        await _append_candidates(store, candidates)
    return candidates


async def _extract_candidate_facts(
    content: str,
    source: str,
    project: str | None,
) -> list[Fact]:
    """Run extraction without persisting the results."""
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
    prompt = (
        f"{scope_instruction}\n\n"
        f"Extract structured facts from the following input:\n\n{content}"
    )
    try:
        result = await complete_model(
            prompt=prompt,
            system=EXTRACTION_SYSTEM,
            response_model=ExtractionResponse,
        )
    except ValidationError as e:
        logger.warning("Skipping invalid extraction response: %s", e)
        return []

    raw_facts = result.facts
    if not raw_facts:
        logger.info("No facts extracted from input")
        return []

    now = datetime.now(timezone.utc)
    source_group_id = uuid4().hex[:12]
    extracted: list[Fact] = []
    for raw in raw_facts:
        fact = Fact(
            category=raw.category,
            memory_key=_normalize_memory_key(raw.memory_key),
            content=raw.content,
            source=source,
            project=project if project is not None else raw.project,
            tags=raw.tags,
            retrieval_hints=raw.retrieval_hints,
            created_at=now,
            updated_at=now,
            observed_at=now,
            effective_at=raw.effective_at,
            expires_at=raw.expires_at,
            evidence_kind=_infer_evidence_kind(source),
            source_ref=source,
            source_group_id=source_group_id,
            why_store=raw.why_store,
        )
        extracted.append(fact)

    consolidated = _consolidate_batch(extracted)
    logger.info(
        "Extraction coverage: %d durable claim(s), %d excluded claim(s), "
        "%d raw card(s), %d consolidated card(s)",
        sum(len(raw.covered_claims) for raw in raw_facts),
        len(result.excluded_claims),
        len(extracted),
        len(consolidated),
    )
    return consolidated


def _normalize_memory_key(value: str) -> str:
    """Normalize a model-supplied semantic identity into a stable key."""
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not normalized:
        raise ValueError("memory_key must contain at least one letter or number")
    return normalized[:120]


def _consolidate_batch(facts: list[Fact]) -> list[Fact]:
    """Combine sibling cards with the same identity and lifecycle.

    The extraction prompt should normally emit one card per memory key. This
    deterministic pass preserves every clause when the model returns multiple
    fragments for the same card.
    """
    by_identity: dict[
        tuple[str | None, FactCategory, str, datetime | None, datetime | None],
        list[Fact],
    ] = defaultdict(list)
    order: list[
        tuple[str | None, FactCategory, str, datetime | None, datetime | None]
    ] = []
    seen_content: set[str] = set()

    for fact in facts:
        content_key = _content_hash(fact.content)
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        identity = (
            fact.project,
            fact.category,
            fact.memory_key,
            fact.effective_at,
            fact.expires_at,
        )
        if identity not in by_identity:
            order.append(identity)
        by_identity[identity].append(fact)

    return [_merge_sibling_cards(by_identity[identity]) for identity in order]


def _merge_sibling_cards(cards: list[Fact]) -> Fact:
    primary = cards[0]
    if len(cards) == 1:
        return primary

    contents: list[str] = []
    seen: set[str] = set()
    for card in cards:
        normalized = _content_hash(card.content)
        if normalized in seen:
            continue
        seen.add(normalized)
        contents.append(card.content.strip())

    return primary.model_copy(
        update={
            "content": " ".join(contents),
            "tags": _ordered_unique(tag for card in cards for tag in card.tags)[:5],
            "retrieval_hints": _ordered_unique(
                hint for card in cards for hint in card.retrieval_hints
            )[:5],
            "why_store": max(
                (card.why_store for card in cards),
                key=len,
                default=primary.why_store,
            ),
        }
    )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


async def _dedup(
    candidates: list[Fact],
    existing: list[Fact],
    store: FactStore | AsyncFactStore | None,
) -> list[Fact]:
    """Two-phase dedup: exact content hash, then scoped LLM dedup for near-matches."""
    after_exact = _drop_exact_duplicates(candidates, existing)
    if not after_exact:
        return []

    near_matches = _find_near_matches(after_exact, existing)
    if not near_matches:
        return after_exact

    prompt = _dedup_prompt(after_exact, near_matches)
    for attempt in range(2):
        result = await complete_model(
            prompt=prompt,
            system=DEDUP_SYSTEM,
            response_model=DedupResponse,
        )
        new_indices = _valid_dedup_indices(result.new, after_exact)
        duplicate_indices = _valid_dedup_indices(result.duplicates, after_exact)
        raw_update_map = _dedup_update_map(result, after_exact, near_matches)
        issue = _dedup_classification_issue(
            len(after_exact),
            new_indices,
            duplicate_indices,
            set(raw_update_map),
        )
        if issue is None:
            break
        if attempt == 0:
            logger.warning("Retrying incomplete dedup response: %s", issue)
            prompt += (
                "\n\nYour prior response was incomplete. Classify every candidate "
                "exactly once as new, duplicate, or update."
            )
    else:
        raise ValueError(f"Dedup response remained invalid after retry: {issue}")

    resolved_update_map = _resolve_update_collisions(raw_update_map, after_exact)

    return await _apply_dedup_decisions(
        after_exact,
        new_indices,
        duplicate_indices,
        raw_update_map,
        resolved_update_map,
        store,
    )


def _dedup_classification_issue(
    candidate_count: int,
    new_indices: set[int],
    duplicate_indices: set[int],
    update_indices: set[int],
) -> str | None:
    groups = [new_indices, duplicate_indices, update_indices]
    overlaps = set().union(
        new_indices & duplicate_indices,
        new_indices & update_indices,
        duplicate_indices & update_indices,
    )
    classified = set().union(*groups)
    missing = set(range(candidate_count)) - classified
    if overlaps or missing:
        return f"overlapping={sorted(overlaps)}, missing={sorted(missing)}"
    return None


def _drop_exact_duplicates(candidates: list[Fact], existing: list[Fact]) -> list[Fact]:
    existing_hashes = {_content_hash(fact.content) for fact in existing}
    after_exact: list[Fact] = []
    for fact in candidates:
        if _content_hash(fact.content) in existing_hashes:
            logger.info("Exact-match dedup dropped: %s", fact.content[:60])
            continue
        after_exact.append(fact)
    return after_exact


def _dedup_prompt(candidates: list[Fact], near_matches: list[Fact]) -> str:
    existing_summary = "\n".join(_format_fact_for_dedup(fact) for fact in near_matches)
    candidate_summary = "\n".join(
        f"[{i}] {_format_fact_for_dedup(fact)}" for i, fact in enumerate(candidates)
    )
    return f"""EXISTING FACTS:
{existing_summary}

NEW CANDIDATE FACTS:
{candidate_summary}

Classify each new fact as genuinely new, a duplicate, or an update to an existing fact."""


def _valid_dedup_indices(indices: list[int], facts: list[Fact]) -> set[int]:
    return {idx for idx in indices if isinstance(idx, int) and 0 <= idx < len(facts)}


def _dedup_update_map(
    result: DedupResponse,
    candidates: list[Fact],
    near_matches: list[Fact],
) -> dict[int, str]:
    near_ids = {fact.id for fact in near_matches}
    raw_update_map: dict[int, str] = {}
    for update in result.updates:
        new_idx = update.new_idx
        existing_id = update.existing_id
        if not isinstance(new_idx, int) or not 0 <= new_idx < len(candidates):
            logger.warning("Skipping dedup update with invalid new_idx: %s", update)
            continue
        if not isinstance(existing_id, str) or existing_id not in near_ids:
            logger.warning("Skipping dedup update with invalid existing_id: %s", update)
            continue
        raw_update_map[new_idx] = existing_id
    return raw_update_map


def _resolve_update_collisions(
    raw_update_map: dict[int, str],
    candidates: list[Fact],
) -> dict[int, str]:
    ancestor_to_candidates: dict[str, list[tuple[int, Fact]]] = defaultdict(list)
    for new_idx, old_id in raw_update_map.items():
        ancestor_to_candidates[old_id].append((new_idx, candidates[new_idx]))

    resolved_update_map: dict[int, str] = {}
    for old_id, cands in ancestor_to_candidates.items():
        if len(cands) == 1:
            idx, _ = cands[0]
            resolved_update_map[idx] = old_id
        else:
            best_idx, _ = max(cands, key=lambda x: x[1].confidence)
            resolved_update_map[best_idx] = old_id
            dropped_count = len(cands) - 1
            logger.info(
                "Dedup collision: dropped %d candidate(s) targeting ancestor %s",
                dropped_count,
                old_id,
            )
    return resolved_update_map


async def _apply_dedup_decisions(
    candidates: list[Fact],
    new_indices: set[int],
    duplicate_indices: set[int],
    raw_update_map: dict[int, str],
    resolved_update_map: dict[int, str],
    store: FactStore | AsyncFactStore | None,
) -> list[Fact]:
    dropped_update_indices = set(raw_update_map) - set(resolved_update_map)
    kept = []
    for i, fact in enumerate(candidates):
        if i in new_indices:
            kept.append(fact)
        elif i in resolved_update_map:
            old_id = resolved_update_map[i]
            fact.supersedes = old_id
            if store is not None:
                await _update_fact(store, old_id, confidence=0.0)
            kept.append(fact)
        elif i in duplicate_indices:
            continue
        elif i in dropped_update_indices:
            continue
        else:
            raise AssertionError(f"Validated dedup response omitted candidate {i}")

    return kept


async def _load_dedup_facts_for_store(
    store: FactStore | AsyncFactStore,
    project: str | None,
) -> list[Fact]:
    """Load facts allowed to deduplicate a new fact in this scope."""
    if project:
        if isinstance(store, AsyncFactStore):
            return await store.load_active_facts(project=project)
        return store.load_active_facts(project=project)
    if isinstance(store, AsyncFactStore):
        facts = await store.load_active_facts()
    else:
        facts = store.load_active_facts()
    return [fact for fact in facts if fact.project is None]


async def _load_pending_candidates(
    store: FactStore | AsyncFactStore,
    limit: int,
) -> list[MemoryCandidate]:
    if isinstance(store, AsyncFactStore):
        return await store.load_candidates(status=CandidateStatus.pending, limit=limit)
    return store.load_candidates(status=CandidateStatus.pending, limit=limit)


async def _append_facts(store: FactStore | AsyncFactStore, facts: list[Fact]) -> None:
    if isinstance(store, AsyncFactStore):
        await store.append_facts(facts)
    else:
        store.append_facts(facts)


async def _append_candidates(
    store: FactStore | AsyncFactStore,
    candidates: list[MemoryCandidate],
) -> None:
    if isinstance(store, AsyncFactStore):
        await store.append_candidates(candidates)
    else:
        store.append_candidates(candidates)


async def _update_fact(
    store: FactStore | AsyncFactStore,
    fact_id: str,
    **updates,
) -> Fact | None:
    if isinstance(store, AsyncFactStore):
        return await store.update_fact(fact_id, **updates)
    return store.update_fact(fact_id, **updates)


def _format_fact_for_dedup(fact: Fact) -> str:
    project = fact.project if fact.project is not None else "(global)"
    key = fact.memory_key or "(legacy)"
    return (
        f"[id:{fact.id}] [{fact.category.value}] [project:{project}] "
        f"[memory_key:{key}] {fact.content}"
    )


def _find_near_matches(candidates: list[Fact], existing: list[Fact]) -> list[Fact]:
    """Find existing facts with meaningful token overlap to any candidate.

    Returns the subset of existing facts worth sending to LLM dedup.
    No hard cap — scales with actual overlap, not arbitrary limits.
    """
    candidate_token_sets = [
        {
            _stem(t)
            for t in _TOKEN_RE.findall(
                c.content.lower().replace("_", " ").replace("-", " ")
            )
        }
        for c in candidates
    ]

    near: list[Fact] = []
    for fact in existing:
        if any(
            candidate.memory_key
            and candidate.memory_key == fact.memory_key
            and candidate.category == fact.category
            and candidate.project == fact.project
            for candidate in candidates
        ):
            near.append(fact)
            continue
        normalized = fact.content.lower().replace("_", " ").replace("-", " ")
        fact_tokens = {_stem(t) for t in _TOKEN_RE.findall(normalized)}
        if not fact_tokens:
            continue
        for candidate_tokens in candidate_token_sets:
            shared = candidate_tokens & fact_tokens
            union = candidate_tokens | fact_tokens
            jaccard = len(shared) / len(union) if union else 0.0
            if jaccard >= 0.3:
                near.append(fact)
                break
    return near


def _dedup_against_candidates(
    facts: list[Fact],
    candidates: list[MemoryCandidate],
) -> list[Fact]:
    """Drop facts that match any of the given candidates on project/category/content.

    Callers are responsible for pre-filtering candidates to the desired status
    (e.g. pending only) before passing them in.
    """
    existing_keys = {
        (candidate.project, candidate.category, candidate.content.lower())
        for candidate in candidates
    }
    return [
        fact
        for fact in facts
        if (fact.project, fact.category, fact.content.lower()) not in existing_keys
    ]


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
