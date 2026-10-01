"""LLM consolidation: rewrite related cards into coherent ones.

Facts in one project scope are partitioned into disjoint clusters of lexical
neighbours. One LLM call per cluster returns the minimal set of cards that
preserves every durable, still-true claim, plus the cards to retire as junk.
Cards about one policy are merged, and a card holding independent claims is
split, so merging can be undone and no card grows without bound.
Each cluster's result is applied as one all-or-nothing change set, so a
cluster touched concurrently (by recall-time writes, another upkeep) is left
alone rather than half-merged.

Incremental runs only consolidate projects that changed since their last
consolidation, and only clusters seeded by a changed fact (or by a seed whose
cluster failed last time). Bookkeeping lives in ``maintenance_state.jsonl``;
see :mod:`engram.maintenance.upkeep_state`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, ConfigDict

from engram.core.config import get_settings
from engram.core.models import BRIEF_MEMORY_KEY, Durability, Fact, FactCategory
from engram.llm import complete_model
from engram.maintenance.upkeep_report import StepReport
from engram.maintenance.upkeep_state import (
    MAX_SEED_ATTEMPTS,
    StateLine,
    append_state,
    load_state,
)
from engram.storage.search import SearchIndex
from engram.storage.store import AsyncFactStore, ChangeSet

logger = logging.getLogger(__name__)

CONSOLIDATE_SOURCE = "engram:consolidate"
BRIEF_KEY = BRIEF_MEMORY_KEY
MAX_CLUSTER_SIZE = 25
# A merge or rewrite may not produce a longer card. Well above the size of a
# card holding one policy with its coupled clauses, so hitting it means the
# card bundles claims that should be corrected and expired independently.
MAX_CARD_CHARS = 1200
_MAX_HINTS = 5
_MAX_TAGS = 3

CONSOLIDATE_SYSTEM = f"""You curate a memory store for coding agents. You receive a
CLUSTER of related memory cards from one project scope. Rewrite the cluster as the
minimal set of coherent, self-contained cards that preserves every durable,
still-true claim.

HOW TO REWRITE
- Merge cards that state the same policy, workflow, decision, or pitfall into one
  card and carry forward every still-true detail. Sharing a subject, tool, or
  terms is not enough: claims that can be corrected, contradicted, expired, or
  acted on independently belong in separate cards.
- Split a card that holds several such independent claims into one card per
  claim, and list the original card in the source_ids of each. A card longer
  than {MAX_CARD_CHARS} characters almost always needs splitting.
- No card you write may be longer than {MAX_CARD_CHARS} characters. The only
  exception is a single source card copied verbatim that cannot be divided.
- When cards conflict, the newer claim wins; drop the outdated one. `date` is
  when a card last took in new information. `oldest_claim`, when present, is
  how old some of its claims may be.
- A card that is already coherent and needs no merge: output it with a single
  source id and its content copied VERBATIM. You may still improve its metadata.
- Self-contained: name the subject explicitly (project, tool, service,
  component). Never write "this workflow", "the script", or "the repo". Third
  person.
- Never invent facts, paths, or symbols.

RETIRE (list in retire with a short reason, not in any card) cards that are:
- test, lint, build, or CI outcomes
- branch, PR, commit, merge, or release bookkeeping
- progress reports and task status
- implementation details trivially readable from the code, unless the card
  records a pitfall or the rationale behind a decision
- context-free fragments a future agent could not act on
- contradicted by newer cards in the cluster
- flagged "suspect" (files it names are gone from the repository) when the
  claim depends on those files

COVERAGE: every input id is either kept or retired. A kept id appears in the
source_ids of one card, or of several cards when that card is being split. A
retired id appears once in retire. Never both, never an id that is not in the
input.

FIELDS
- memory_key: stable lowercase semantic identity; reuse a source card's key when
  it fits.
- category: personal_info, preference, event, decision, pitfall, convention,
  assistant_info, project, workflow, or correction.
- durability: evergreen (identity, lasting personal preferences), durable
  (conventions, decisions, pitfalls, workflows; true until contradicted), or
  ephemeral (in-flight state, temporary workarounds, incidents; true for weeks).
- anchors: repo-relative file paths (with extension) or code identifiers inside
  this project's git repository that the card depends on, taken only from the
  source cards. Never database tables or warehouse objects, URLs, external
  IDs, directories, or files outside the repo. Empty when none.
- retrieval_hints: 1-5 terms or questions a future agent might search for.
- tags: 1-3 short topic tags."""


class ConsolidatedCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ids: list[str]
    memory_key: str
    content: str
    category: FactCategory
    durability: Durability
    anchors: list[str]
    retrieval_hints: list[str]
    tags: list[str]


class RetiredInputCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    reason: str


class ConsolidationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cards: list[ConsolidatedCard]
    retire: list[RetiredInputCard]


# --- Clustering --------------------------------------------------------------


def partition_clusters(
    facts: list[Fact],
    index: SearchIndex,
    *,
    is_new: Callable[[Fact], bool],
    max_size: int = MAX_CLUSTER_SIZE,
) -> list[list[Fact]]:
    """Split one scope's facts into disjoint clusters seeded by new facts.

    Seeds are taken oldest-first; each cluster is the seed plus its closest
    unassigned lexical neighbours. A scope small enough for one call is one
    cluster.
    """
    if not any(is_new(fact) for fact in facts):
        return []
    ordered = sorted(facts, key=lambda fact: (fact.created_at, fact.id))
    if len(ordered) <= max_size:
        return [ordered]
    pool = {fact.id for fact in ordered}
    assigned: set[str] = set()
    clusters: list[list[Fact]] = []
    for seed in ordered:
        if seed.id in assigned or not is_new(seed):
            continue
        assigned.add(seed.id)
        hits = index.neighbors(
            seed,
            accept=lambda fact: fact.id in pool and fact.id not in assigned,
            limit=max_size - 1,
        )
        members = [seed, *(hit.fact for hit in hits)]
        assigned.update(fact.id for fact in members)
        clusters.append(members)
    return clusters


# --- One cluster -------------------------------------------------------------


def _handles(cluster: list[Fact]) -> dict[str, str]:
    """Short prompt handles (``c1``…) mapped to fact IDs, oldest first.

    Models copy ``c7`` reliably; they garble 12-hex IDs often enough to fail
    a cluster's coverage check.
    """
    ordered = sorted(cluster, key=lambda fact: (fact.observed_at, fact.id))
    return {f"c{index}": fact.id for index, fact in enumerate(ordered, 1)}


def _with_fact_ids(
    response: ConsolidationResponse, handles: dict[str, str]
) -> ConsolidationResponse:
    """Translate handles back to fact IDs; unknown handles pass through as-is."""
    return response.model_copy(
        update={
            "cards": [
                card.model_copy(
                    update={
                        "source_ids": [
                            handles.get(handle, handle) for handle in card.source_ids
                        ]
                    }
                )
                for card in response.cards
            ],
            "retire": [
                retired.model_copy(update={"id": handles.get(retired.id, retired.id)})
                for retired in response.retire
            ],
        }
    )


def _cluster_prompt(cluster: list[Fact], scope: str) -> str:
    by_id = {fact.id: fact for fact in cluster}
    cards = [
        {
            "id": handle,
            "date": fact.observed_at.date().isoformat(),
            **(
                {"oldest_claim": fact.oldest_claim_at.date().isoformat()}
                if fact.oldest_claim_at.date() < fact.observed_at.date()
                else {}
            ),
            "category": fact.category.value,
            "durability": fact.durability.value,
            "memory_key": fact.memory_key,
            "content": fact.content,
            "anchors": fact.anchors,
            "suspect": fact.suspect_reason,
        }
        for handle, fact in (
            (handle, by_id[fact_id]) for handle, fact_id in _handles(cluster).items()
        )
    ]
    return f"PROJECT SCOPE: {scope}\n\nCLUSTER (oldest first):\n" + json.dumps(
        cards, indent=1, ensure_ascii=False
    )


def _source_uses(response: ConsolidationResponse) -> Counter[str]:
    """How many output cards each input id feeds; more than one is a split."""
    return Counter(
        source_id for card in response.cards for source_id in set(card.source_ids)
    )


def _is_verbatim(
    card: ConsolidatedCard, by_id: dict[str, Fact], uses: Counter[str]
) -> bool:
    """True when ``card`` is one unsplit source card with its content unchanged."""
    if len(card.source_ids) != 1 or uses[card.source_ids[0]] != 1:
        return False
    source = by_id.get(card.source_ids[0])
    return source is not None and _same_text(card.content, source.content)


def response_problem(cluster: list[Fact], response: ConsolidationResponse) -> str:
    """Why ``response`` cannot be applied: bad coverage or an oversized card."""
    by_id = {fact.id: fact for fact in cluster}
    uses = _source_uses(response)
    retired = [item.id for item in response.retire]
    seen = set(uses) | set(retired)
    problems: list[str] = []
    unknown = sorted(seen - set(by_id))
    missing = sorted(set(by_id) - seen)
    both = sorted(set(uses) & set(retired))
    twice = sorted(
        {fact_id for fact_id in retired if retired.count(fact_id) > 1}
        | {
            source_id
            for card in response.cards
            for source_id in card.source_ids
            if card.source_ids.count(source_id) > 1
        }
    )
    if unknown:
        problems.append(f"ids not in the input: {', '.join(unknown)}")
    if missing:
        problems.append(f"input ids not accounted for: {', '.join(missing)}")
    if both:
        problems.append(f"ids both kept and retired: {', '.join(both)}")
    if twice:
        problems.append(f"ids listed twice in one place: {', '.join(twice)}")
    if any(not card.source_ids for card in response.cards):
        problems.append("a card has no source_ids")
    # A split must divide a card's claims, never repeat them.
    texts = [" ".join(card.content.split()) for card in response.cards]
    if len(set(texts)) < len(texts):
        problems.append("two cards have the same content")
    copied = sorted(
        {
            source_id
            for card in response.cards
            for source_id in card.source_ids
            if uses[source_id] > 1
            and source_id in by_id
            and _same_text(card.content, by_id[source_id].content)
        }
    )
    if copied:
        problems.append(f"split cards repeated whole in one piece: {', '.join(copied)}")
    too_long = [
        f"{card.memory_key} ({len(card.content.strip())})"
        for card in response.cards
        if len(card.content.strip()) > MAX_CARD_CHARS
        and not _is_verbatim(card, by_id, uses)
    ]
    if too_long:
        problems.append(
            f"cards longer than {MAX_CARD_CHARS} characters, split them into "
            f"independent cards: {', '.join(too_long)}"
        )
    return "; ".join(problems)


async def consolidate_cluster(
    cluster: list[Fact], scope: str, report: StepReport
) -> ConsolidationResponse | None:
    """Ask the LLM to rewrite ``cluster``; one retry on an invalid answer."""
    settings = get_settings()
    prompt = _cluster_prompt(cluster, scope)
    handles = _handles(cluster)
    problem = ""
    for attempt in range(2):
        attempt_prompt = prompt
        if problem:
            attempt_prompt += (
                f"\n\nCORRECTION: your previous answer was invalid ({problem}). "
                "Fix exactly that and keep or retire every input id."
            )
        report.llm_calls += 1
        try:
            response = await complete_model(
                attempt_prompt,
                system=CONSOLIDATE_SYSTEM,
                response_model=ConsolidationResponse,
                reasoning_effort=settings.llm_reasoning_effort,
            )
        except Exception as exc:  # noqa: BLE001 - one bad cluster must not stop upkeep
            problem = f"LLM call failed: {exc}"
            logger.warning("consolidate %s attempt %d: %s", scope, attempt + 1, exc)
            continue
        response = _with_fact_ids(response, handles)
        problem = response_problem(cluster, response)
        if not problem:
            return response
    report.errors.append(f"{scope}: skipped cluster of {len(cluster)} ({problem})")
    return None


def _unique(values: list[str], limit: int) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))[
        :limit
    ]


def _same_text(left: str, right: str) -> bool:
    return " ".join(left.split()) == " ".join(right.split())


@dataclass(frozen=True)
class ClusterPlan:
    changes: ChangeSet
    merged: int
    rewritten: int
    # Input cards divided across more than one output card.
    split: int


def plan_cluster_changes(
    cluster: list[Fact],
    response: ConsolidationResponse,
    *,
    project: str | None,
    now: datetime,
    ephemeral_ttl: timedelta,
) -> ClusterPlan:
    """Translate a validated response into one atomic change set."""
    by_id = {fact.id: fact for fact in cluster}
    changes = ChangeSet(reason="upkeep: consolidation", actor=CONSOLIDATE_SOURCE)
    merged = rewritten = 0
    uses = _source_uses(response)
    for card in response.cards:
        sources = [by_id[source_id] for source_id in card.source_ids]
        hints = _unique(card.retrieval_hints, _MAX_HINTS)
        tags = _unique(card.tags, _MAX_TAGS)
        anchors = _unique(card.anchors, 50)
        if _is_verbatim(card, by_id, uses):
            source = sources[0]
            fields: dict[str, object] = {
                "memory_key": card.memory_key,
                "category": card.category,
                "durability": card.durability,
                "anchors": anchors,
                "retrieval_hints": hints or source.retrieval_hints,
                "tags": tags,
            }
            if card.durability is Durability.ephemeral and source.expires_at is None:
                fields["expires_at"] = now + ephemeral_ttl
            edits = {
                name: value
                for name, value in fields.items()
                if getattr(source, name) != value
            }
            if edits:
                changes.edits[source.id] = edits
            continue

        primary = sources[0]
        observed_at = max(source.observed_at for source in sources)
        # The newest source dates the card for conflict resolution, but the
        # claims it carries forward are only as fresh as the oldest source.
        oldest_claim_at = min(source.oldest_claim_at for source in sources)
        expires_at = None
        if card.durability is Durability.ephemeral:
            # Keep a stated future expiry; otherwise the TTL starts now, so a
            # card upkeep just classified never arrives already expired.
            explicit = [
                source.expires_at
                for source in sources
                if source.expires_at and source.expires_at > now
            ]
            expires_at = max(explicit) if explicit else now + ephemeral_ttl
        fact = Fact(
            category=card.category,
            memory_key=card.memory_key,
            content=card.content.strip(),
            source=CONSOLIDATE_SOURCE,
            confidence=max(source.confidence for source in sources),
            created_at=now,
            updated_at=now,
            observed_at=observed_at,
            first_observed_at=(
                oldest_claim_at if oldest_claim_at < observed_at else None
            ),
            expires_at=expires_at,
            tags=tags,
            retrieval_hints=hints,
            project=project,
            supersedes=primary.id,
            consolidates=list(
                dict.fromkeys(
                    fact_id
                    for source in sources
                    for fact_id in [source.id, *source.consolidates]
                )
            ),
            evidence_kind=primary.evidence_kind,
            source_ref=primary.source_ref,
            why_store=primary.why_store,
            durability=card.durability,
            anchors=anchors,
        )
        changes.new_facts.append(fact)
        for source in sources:
            # A split source is superseded by the first card it feeds; the
            # others reach it through ``consolidates``.
            changes.supersede.setdefault(source.id, fact.id)
        if len(sources) > 1:
            merged += 1
        else:
            rewritten += 1
    for retired in response.retire:
        changes.stale[retired.id] = f"consolidation: {retired.reason}"
    changes.expected = {
        fact_id: by_id[fact_id].updated_at
        for fact_id in [*changes.supersede, *changes.stale, *changes.edits]
    }
    split = sum(1 for count in uses.values() if count > 1)
    return ClusterPlan(changes=changes, merged=merged, rewritten=rewritten, split=split)


# --- Step --------------------------------------------------------------------


async def run_consolidation(
    store: AsyncFactStore,
    *,
    project: str | None,
    report: StepReport,
    dry_run: bool,
    now: datetime,
    full: bool,
) -> None:
    """Consolidate every scope (or just ``project``) that changed."""
    settings = get_settings()
    # Taken before reading: anything written after this is seen next run.
    snapshot = datetime.now(timezone.utc)
    index = await store.search_index()
    state = await asyncio.to_thread(load_state, store.data_dir)
    by_scope: dict[str | None, list[Fact]] = {}
    for fact in index.facts:
        if fact.memory_key == BRIEF_KEY:
            continue
        if project is not None and fact.project != project:
            continue
        by_scope.setdefault(fact.project, []).append(fact)

    work: list[tuple[str | None, list[Fact]]] = []
    retry_counts: dict[str | None, dict[str, int]] = {}
    for scope, facts in sorted(by_scope.items(), key=lambda item: item[0] or ""):
        prior = state.projects.get(scope)
        active_ids = {fact.id for fact in facts}
        retry = {
            seed_id: attempts
            for seed_id, attempts in (prior.retry_seeds if prior else {}).items()
            if seed_id in active_ids
        }
        since = None if full or prior is None else prior.consolidated_at
        written = prior.written if prior else {}

        def is_new(
            fact: Fact,
            since: datetime | None = since,
            written: dict[str, datetime] = written,
            retry: dict[str, int] = retry,
        ) -> bool:
            if since is None or fact.id in retry:
                return True
            # Upkeep's own last write is not a change worth reconsidering.
            if written.get(fact.id) == fact.updated_at:
                return False
            # New cards only: metadata edits (anchors, suspect flags from
            # verify) must not re-trigger consolidation of settled cards.
            return fact.created_at > since

        clusters = partition_clusters(facts, index, is_new=is_new)
        if not clusters:
            report.bump("projects_unchanged")
            continue
        report.bump("projects")
        retry_counts[scope] = retry
        work.extend((scope, cluster) for cluster in clusters)
    report.bump("clusters", len(work))

    semaphore = asyncio.Semaphore(max(1, settings.maintenance_concurrency))
    ttl = timedelta(days=settings.ephemeral_ttl_days)
    failed_seeds: dict[str | None, list[str]] = {}
    written_ids: dict[str | None, list[str]] = {}

    async def handle(scope: str | None, cluster: list[Fact]) -> None:
        label = scope or "(global)"
        async with semaphore:
            response = await consolidate_cluster(cluster, label, report)
        if response is None:
            failed_seeds.setdefault(scope, []).append(cluster[0].id)
            return
        plan = plan_cluster_changes(
            cluster,
            response,
            project=scope,
            now=now,
            ephemeral_ttl=ttl,
        )
        changes = plan.changes
        if dry_run:
            applied_created = [fact.id for fact in changes.new_facts]
            superseded = list(changes.supersede)
            staled = list(changes.stale)
            edited = list(changes.edits)
        else:
            result = await store.apply_changes(changes, all_or_nothing=True)
            if result.skipped:
                failed_seeds.setdefault(scope, []).append(cluster[0].id)
                report.errors.append(
                    f"{label}: cluster changed concurrently, left alone "
                    f"({len(result.skipped)} targets changed or inactive)"
                )
                return
            applied_created = [fact.id for fact in result.created]
            superseded = list(result.superseded)
            staled = list(result.staled)
            edited = result.edited
            written_ids.setdefault(scope, []).extend([*applied_created, *edited])
        report.bump("cards_in", len(cluster))
        report.bump("cards_out", len(response.cards))
        report.bump("merged_cards", plan.merged)
        report.bump("rewritten_cards", plan.rewritten)
        report.bump("split_cards", plan.split)
        report.add("created", *applied_created)
        report.add("superseded", *superseded)
        report.add("staled", *staled)
        report.add("edited", *edited)
        for retired in response.retire:
            report.notes.append(f"retired {retired.id}: {retired.reason}")

    await asyncio.gather(*(handle(scope, cluster) for scope, cluster in work))
    if dry_run:
        report.notes.insert(0, "dry run: LLM was called, nothing applied")
        return

    current = {fact.id: fact for fact in await store.load_facts()}
    lines: list[StateLine] = []
    for scope, retry in retry_counts.items():
        next_retry: dict[str, int] = {}
        for seed_id in failed_seeds.get(scope, []):
            attempts = retry.get(seed_id, 0) + 1
            if attempts >= MAX_SEED_ATTEMPTS:
                report.notes.append(
                    f"gave up on cluster seeded by {seed_id} after {attempts} attempts"
                )
            else:
                next_retry[seed_id] = attempts
        lines.append(
            StateLine(
                kind="consolidated",
                at=snapshot,
                project=scope,
                written={
                    fact_id: current[fact_id].updated_at
                    for fact_id in written_ids.get(scope, [])
                    if fact_id in current
                },
                retry_seeds=next_retry,
            )
        )
    await asyncio.to_thread(append_state, store.data_dir, lines)
