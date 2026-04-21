"""Observer agent — extracts structured facts from raw text."""

import logging
from collections import defaultdict
from datetime import datetime, timezone

from engram.config import get_settings
from engram.llm import complete_json
from engram.models import (
    CandidateStatus,
    EvidenceKind,
    Fact,
    FactCategory,
    IngestionRecord,
    MemoryCandidate,
)
from engram.store import FactStore, _content_hash, _stem, _TOKEN_RE

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """You are a knowledge extraction agent. Your job is to extract discrete,
structured facts from user input. Each fact should be:
- Self-contained (understandable without the original context)
- Categorized into exactly one category
- Written in third person ("The user prefers..." not "I prefer...")
- Specific and actionable, not vague

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
- "content": the fact as a clear sentence
- "category": one of the categories above
- "tags": 1-3 relevant tags (lowercase)
- "why_store": one short reason this would be useful for future agent behavior
- "expires_at": ISO datetime if this is temporal/time-bound, null otherwise

Example output:
{"facts": [
  {"content": "The user works on a data platform team", "category": "personal_info", "tags": ["team", "role"], "why_store": "Useful background for project context", "expires_at": null},
  {"content": "The user prefers polars over pandas for large datasets", "category": "preference", "tags": ["python", "data"], "why_store": "Guides future tool and library choices", "expires_at": null}
]}"""

DEDUP_SYSTEM = """You are a deduplication agent. Given a list of EXISTING facts and a list of
NEW facts, identify which new facts are:
1. Duplicates of existing facts (same information, possibly different wording)
2. Updates to existing facts (newer version of the same knowledge)
3. Genuinely new facts

Return a JSON object with:
- "new": list of indices (0-based) of genuinely new facts to add
- "updates": list of {"new_idx": int, "existing_id": str} for facts that supersede existing ones
- "duplicates": list of indices to skip"""


async def extract_facts(
    content: str,
    source: str = "conversation",
    project: str | None = None,
    store: FactStore | None = None,
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

    # Step 2: Dedup against existing facts
    existing = store.load_active_facts(limit=200)
    if existing:
        candidates = await _dedup(candidates, existing, store)

    # Step 3: Persist
    if candidates:
        store.append_facts(candidates)
        store.log_ingestion(
            IngestionRecord(
                source=source,
                facts_created=[f.id for f in candidates if not f.supersedes],
                facts_updated=[f.id for f in candidates if f.supersedes],
                agent_model=get_settings().llm_model,
            )
        )

    logger.info("Extracted %d facts from input", len(candidates))
    return candidates


async def suggest_memories(
    content: str,
    source: str = "conversation",
    project: str | None = None,
    store: FactStore | None = None,
) -> list[MemoryCandidate]:
    """Extract and queue proposed memories for review."""
    store = store or FactStore()
    facts = await _extract_candidate_facts(content, source=source, project=project)
    if not facts:
        return []

    existing = store.load_active_facts(limit=200)
    if existing:
        facts = await _dedup(facts, existing, store=None)

    pending = store.load_candidates(status=CandidateStatus.pending, limit=200)
    if pending:
        facts = _dedup_against_candidates(facts, pending)

    candidates = [MemoryCandidate(**fact.model_dump()) for fact in facts]

    if candidates:
        store.append_candidates(candidates)
    return candidates


async def _extract_candidate_facts(
    content: str,
    source: str,
    project: str | None,
) -> list[Fact]:
    """Run extraction without persisting the results."""
    prompt = f"Extract structured facts from the following input:\n\n{content}"
    result = await complete_json(prompt=prompt, system=EXTRACTION_SYSTEM)

    raw_facts = result.get("facts", [])
    if not raw_facts:
        logger.info("No facts extracted from input")
        return []

    now = datetime.now(timezone.utc)
    extracted: list[Fact] = []
    for raw in raw_facts:
        fact_content = raw.get("content")
        if not fact_content:
            logger.warning("Skipping fact with missing content: %s", raw)
            continue

        try:
            category = FactCategory(raw["category"])
        except (ValueError, KeyError):
            logger.warning(
                "Skipping fact with invalid category: %s", raw.get("category")
            )
            continue

        fact = Fact(
            category=category,
            content=fact_content,
            source=source,
            project=project,
            tags=raw.get("tags", []),
            created_at=now,
            updated_at=now,
            observed_at=now,
            effective_at=_parse_datetime(raw.get("effective_at")),
            expires_at=_parse_datetime(raw.get("expires_at")),
            evidence_kind=_infer_evidence_kind(source),
            source_ref=source,
            why_store=raw.get("why_store", ""),
        )
        extracted.append(fact)

    return extracted


async def _dedup(
    candidates: list[Fact],
    existing: list[Fact],
    store: FactStore | None,
) -> list[Fact]:
    """Two-phase dedup: exact content hash, then scoped LLM dedup for near-matches."""
    # Phase 1: Exact-match dedup via content hash — no LLM call needed
    existing_hashes = {_content_hash(f.content) for f in existing}
    after_exact: list[Fact] = []
    for fact in candidates:
        if _content_hash(fact.content) in existing_hashes:
            logger.info("Exact-match dedup dropped: %s", fact.content[:60])
            continue
        after_exact.append(fact)

    if not after_exact:
        return []

    # Phase 2: Scoped LLM dedup — only compare candidates to existing facts
    # with meaningful token overlap (no hard truncation)
    near_matches = _find_near_matches(after_exact, existing)
    if not near_matches:
        return after_exact

    existing_summary = "\n".join(
        f"[id:{f.id}] [{f.category.value}] {f.content}" for f in near_matches
    )
    candidate_summary = "\n".join(
        f"[{i}] [{f.category.value}] {f.content}" for i, f in enumerate(after_exact)
    )

    prompt = f"""EXISTING FACTS:
{existing_summary}

NEW CANDIDATE FACTS:
{candidate_summary}

Classify each new fact as genuinely new, a duplicate, or an update to an existing fact."""

    result = await complete_json(prompt=prompt, system=DEDUP_SYSTEM)

    new_indices = set(result.get("new", []))
    updates = result.get("updates", [])
    raw_update_map = {u["new_idx"]: u["existing_id"] for u in updates}

    # Detect collisions: multiple candidates targeting the same ancestor.
    # Keep only the highest-confidence candidate per ancestor.
    ancestor_to_candidates: dict[str, list[tuple[int, Fact]]] = defaultdict(list)
    for new_idx, old_id in raw_update_map.items():
        ancestor_to_candidates[old_id].append((new_idx, after_exact[new_idx]))

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

    kept = []
    for i, fact in enumerate(after_exact):
        if i in new_indices:
            kept.append(fact)
        elif i in resolved_update_map:
            old_id = resolved_update_map[i]
            fact.supersedes = old_id
            if store is not None:
                store.update_fact(old_id, confidence=0.3)
            kept.append(fact)
        # else: duplicate, skip

    return kept


def _find_near_matches(candidates: list[Fact], existing: list[Fact]) -> list[Fact]:
    """Find existing facts with meaningful token overlap to any candidate.

    Returns the subset of existing facts worth sending to LLM dedup.
    No hard cap — scales with actual overlap, not arbitrary limits.
    """
    candidate_tokens: set[str] = set()
    for c in candidates:
        normalized = c.content.lower().replace("_", " ").replace("-", " ")
        candidate_tokens.update(_stem(t) for t in _TOKEN_RE.findall(normalized))

    near: list[Fact] = []
    for fact in existing:
        normalized = fact.content.lower().replace("_", " ").replace("-", " ")
        fact_tokens = {_stem(t) for t in _TOKEN_RE.findall(normalized)}
        if not fact_tokens:
            continue
        shared = candidate_tokens & fact_tokens
        union = candidate_tokens | fact_tokens
        jaccard = len(shared) / len(union) if union else 0.0
        if jaccard >= 0.3:
            near.append(fact)
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


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse an ISO datetime string, returning None on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


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
