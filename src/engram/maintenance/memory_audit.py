"""Read-only memory audit suggestions for compaction review."""

from __future__ import annotations

import hashlib
import re
import shlex
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from engram.core.models import Fact, MIN_ACTIVE_CONFIDENCE
from engram.storage.store import AsyncFactStore, FactStore, _STOPWORDS, _TOKEN_RE, _stem

SuggestionKind = Literal["duplicate", "stale", "contradiction"]
SuggestionAction = Literal["merge_memories", "mark_stale", "review_contradiction"]

DEFAULT_TEMPORAL_STALE_DAYS = 30

_TEMPORAL_STALE_RE = re.compile(
    r"\b("
    r"currently|temporary|temporarily|right now|today|tomorrow|yesterday|"
    r"this week|next week|this sprint|for now|during the migration|"
    r"until further notice"
    r")\b",
    re.IGNORECASE,
)
_DATED_WINDOW_RE = re.compile(
    r"\b(until|before|by|through|ends? on)\s+(?P<date>20\d{2}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
_PREFERENCE_PATTERNS = (
    re.compile(
        r"\bprefers?\s+(?P<winner>.+?)\s+over\s+(?P<loser>.+?)"
        r"(?:\s+for\s+(?P<context>.+?))?(?:[.;,]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:decided to\s+)?use(?:s|d)?\s+(?P<winner>.+?)\s+instead of\s+"
        r"(?P<loser>.+?)(?:\s+for\s+(?P<context>.+?))?(?:[.;,]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:must|should)\s+use\s+(?P<winner>.+?),?\s+not\s+"
        r"(?P<loser>.+?)(?:\s+for\s+(?P<context>.+?))?(?:[.;,]|$)",
        re.IGNORECASE,
    ),
)


class MemoryAuditSuggestion(BaseModel):
    """One reviewable maintenance suggestion."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: SuggestionKind
    action: SuggestionAction
    fact_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    suggested_content: str | None = None
    suggested_command: str | None = None
    evidence: list[str] = Field(default_factory=list)


class MemoryAuditResult(BaseModel):
    """Summary of a read-only memory audit run."""

    model_config = ConfigDict(extra="forbid")

    total_analyzed: int
    suggestions: list[MemoryAuditSuggestion] = Field(default_factory=list)

    @property
    def duplicate_groups(self) -> int:
        return sum(1 for s in self.suggestions if s.kind == "duplicate")

    @property
    def stale_facts(self) -> int:
        return sum(1 for s in self.suggestions if s.kind == "stale")

    @property
    def contradiction_groups(self) -> int:
        return sum(1 for s in self.suggestions if s.kind == "contradiction")


@dataclass(frozen=True)
class _ChoiceClaim:
    winner: str
    loser: str
    context: frozenset[str]


@dataclass(frozen=True)
class _DuplicateSignal:
    score: float
    reason: str


class _UnionFind:
    def __init__(self, ids: Sequence[str]):
        self.parents = {fact_id: fact_id for fact_id in ids}

    def find(self, fact_id: str) -> str:
        parent = self.parents[fact_id]
        if parent != fact_id:
            self.parents[fact_id] = self.find(parent)
        return self.parents[fact_id]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parents[right_root] = left_root


async def audit_memory_store(
    *,
    project: str | None = None,
    store: FactStore | AsyncFactStore | None = None,
    now: datetime | None = None,
) -> MemoryAuditResult:
    """Return read-only suggestions for duplicates, stale facts, and contradictions."""
    store = store or FactStore()
    if isinstance(store, AsyncFactStore):
        facts = await store.load_facts()
    else:
        facts = store.load_facts()
    if project is not None:
        facts = [fact for fact in facts if fact.project == project]
    return audit_facts(facts, now=now)


def audit_facts(
    facts: Sequence[Fact],
    *,
    now: datetime | None = None,
    temporal_stale_days: int = DEFAULT_TEMPORAL_STALE_DAYS,
) -> MemoryAuditResult:
    """Analyze a fact list and return safe maintenance suggestions.

    The audit is deliberately conservative: it proposes commands for human
    review, and leaves execution to existing explicit maintenance tools.
    """
    checked_at = _aware(now or datetime.now(timezone.utc))
    reviewable = [
        fact
        for fact in facts
        if fact.confidence >= MIN_ACTIVE_CONFIDENCE and not fact.stale
    ]
    active = [fact for fact in reviewable if not _is_expired(fact, checked_at)]

    suggestions: list[MemoryAuditSuggestion] = []
    suggestions.extend(_duplicate_suggestions(active))
    suggestions.extend(
        _stale_suggestions(
            reviewable,
            now=checked_at,
            stale_after_days=temporal_stale_days,
        )
    )
    suggestions.extend(_contradiction_suggestions(active))
    suggestions.sort(key=lambda item: (_kind_order(item.kind), item.fact_ids, item.id))
    return MemoryAuditResult(total_analyzed=len(reviewable), suggestions=suggestions)


def format_audit_result(result: MemoryAuditResult) -> str:
    """Format audit suggestions for CLI and MCP text output."""
    lines = ["## Memory Audit Suggestions", ""]
    lines.append(f"Analyzed: {result.total_analyzed} active fact(s)")
    lines.append(
        "Suggestions: "
        f"{len(result.suggestions)} total | "
        f"{result.duplicate_groups} duplicate | "
        f"{result.stale_facts} stale | "
        f"{result.contradiction_groups} contradiction"
    )
    lines.append("No changes were applied.")

    if not result.suggestions:
        lines.append("")
        lines.append("No review suggestions found.")
        return "\n".join(lines)

    for suggestion in result.suggestions:
        ids = ", ".join(f"`{fact_id}`" for fact_id in suggestion.fact_ids)
        lines.append("")
        lines.append(
            f"- [{suggestion.kind}] `{suggestion.id}` -> {suggestion.action} "
            f"({ids}, confidence {suggestion.confidence:.2f})"
        )
        lines.append(f"  reason: {suggestion.reason}")
        if suggestion.suggested_content:
            lines.append(f"  suggested content: {suggestion.suggested_content}")
        if suggestion.suggested_command:
            lines.append(f"  review command: {suggestion.suggested_command}")
    return "\n".join(lines)


def _duplicate_suggestions(facts: Sequence[Fact]) -> list[MemoryAuditSuggestion]:
    suggestions: list[MemoryAuditSuggestion] = []
    groups: dict[tuple[str | None, str], list[Fact]] = defaultdict(list)
    for fact in facts:
        groups[(fact.project, fact.category.value)].append(fact)

    for group_facts in groups.values():
        if len(group_facts) < 2:
            continue
        union = _UnionFind([fact.id for fact in group_facts])
        reasons: dict[tuple[str, str], _DuplicateSignal] = {}
        by_id = {fact.id: fact for fact in group_facts}

        for left, right in combinations(group_facts, 2):
            signal = _duplicate_signal(left, right)
            if signal is None:
                continue
            union.union(left.id, right.id)
            pair = (left.id, right.id) if left.id <= right.id else (right.id, left.id)
            reasons[pair] = signal

        duplicate_sets: dict[str, list[str]] = defaultdict(list)
        for fact in group_facts:
            duplicate_sets[union.find(fact.id)].append(fact.id)

        for fact_ids in duplicate_sets.values():
            if len(fact_ids) < 2:
                continue
            ordered = _order_for_review([by_id[fid] for fid in fact_ids])
            ordered_ids = [fact.id for fact in ordered]
            content = _suggested_merge_content(ordered)
            reason = _duplicate_reason(ordered_ids, reasons)
            suggestions.append(
                MemoryAuditSuggestion(
                    id=_suggestion_id("duplicate", ordered_ids),
                    kind="duplicate",
                    action="merge_memories",
                    fact_ids=ordered_ids,
                    confidence=min(
                        0.98, max(0.80, _duplicate_confidence(ordered_ids, reasons))
                    ),
                    reason=reason,
                    suggested_content=content,
                    suggested_command=_merge_command(ordered_ids, content),
                    evidence=[fact.content for fact in ordered],
                )
            )
    return suggestions


def _stale_suggestions(
    facts: Sequence[Fact],
    *,
    now: datetime,
    stale_after_days: int,
) -> list[MemoryAuditSuggestion]:
    suggestions: list[MemoryAuditSuggestion] = []
    for fact in facts:
        reason = _stale_reason(fact, now=now, stale_after_days=stale_after_days)
        if reason is None:
            continue
        suggestions.append(
            MemoryAuditSuggestion(
                id=_suggestion_id("stale", [fact.id]),
                kind="stale",
                action="mark_stale",
                fact_ids=[fact.id],
                confidence=0.90,
                reason=reason,
                suggested_command=_stale_command(fact.id, reason),
                evidence=[fact.content],
            )
        )
    return suggestions


def _contradiction_suggestions(facts: Sequence[Fact]) -> list[MemoryAuditSuggestion]:
    suggestions: list[MemoryAuditSuggestion] = []
    groups: dict[tuple[str | None, str], list[Fact]] = defaultdict(list)
    for fact in facts:
        groups[(fact.project, fact.category.value)].append(fact)

    for group_facts in groups.values():
        for left, right in combinations(group_facts, 2):
            if not _reversed_choice(left, right):
                continue
            ordered = _order_for_review([left, right])
            ordered_ids = [fact.id for fact in ordered]
            suggestions.append(
                MemoryAuditSuggestion(
                    id=_suggestion_id("contradiction", ordered_ids),
                    kind="contradiction",
                    action="review_contradiction",
                    fact_ids=ordered_ids,
                    confidence=0.88,
                    reason=(
                        "Opposite preference/update claims in the same project "
                        "and category; review which fact should supersede the other "
                        "or be marked stale."
                    ),
                    evidence=[fact.content for fact in ordered],
                )
            )
    return suggestions


def _duplicate_signal(left: Fact, right: Fact) -> _DuplicateSignal | None:
    if left.project != right.project or left.category != right.category:
        return None
    if _reversed_choice(left, right):
        return None

    if _normalized_content(left.content) == _normalized_content(right.content):
        return _DuplicateSignal(score=0.99, reason="normalized content matches exactly")

    left_tokens = _tokens(left.content)
    right_tokens = _tokens(right.content)
    if len(left_tokens) < 3 or len(right_tokens) < 3:
        return None
    shared = left_tokens & right_tokens
    if len(shared) < 4:
        return None
    union = left_tokens | right_tokens
    jaccard = len(shared) / len(union)
    containment = len(shared) / min(len(left_tokens), len(right_tokens))
    tag_overlap = bool(set(left.tags) & set(right.tags))

    if jaccard >= 0.58:
        return _DuplicateSignal(
            score=min(0.95, 0.68 + jaccard / 3),
            reason=f"high token overlap ({jaccard:.0%} Jaccard)",
        )
    if containment >= 0.75:
        return _DuplicateSignal(
            score=min(0.90, 0.62 + containment / 4),
            reason=f"one fact mostly contains the other ({containment:.0%} overlap)",
        )
    if tag_overlap and jaccard >= 0.42:
        return _DuplicateSignal(
            score=min(0.86, 0.60 + jaccard / 3),
            reason=f"shared tags plus token overlap ({jaccard:.0%} Jaccard)",
        )
    return None


def _stale_reason(
    fact: Fact,
    *,
    now: datetime,
    stale_after_days: int,
) -> str | None:
    if fact.expires_at is not None and _aware(fact.expires_at) <= now:
        return f"fact expired at {fact.expires_at.date().isoformat()}"

    observed_at = _aware(fact.observed_at)
    age_days = (now - observed_at).days
    if age_days >= stale_after_days and _TEMPORAL_STALE_RE.search(fact.content):
        return (
            f"time-bound wording is {age_days} days old "
            f"(threshold {stale_after_days} days)"
        )

    for match in _DATED_WINDOW_RE.finditer(fact.content):
        try:
            date = datetime.fromisoformat(match.group("date")).date()
        except ValueError:
            continue
        if date < now.date():
            return f"time-bound date {date.isoformat()} has passed"
    return None


def _is_expired(fact: Fact, now: datetime) -> bool:
    return fact.expires_at is not None and _aware(fact.expires_at) <= now


def _reversed_choice(left: Fact, right: Fact) -> bool:
    left_claim = _choice_claim(left.content)
    right_claim = _choice_claim(right.content)
    if left_claim is None or right_claim is None:
        return False
    if left_claim.winner != right_claim.loser:
        return False
    if left_claim.loser != right_claim.winner:
        return False
    return _contexts_compatible(left_claim.context, right_claim.context)


def _choice_claim(content: str) -> _ChoiceClaim | None:
    for pattern in _PREFERENCE_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        winner = _choice_phrase(match.group("winner"))
        loser = _choice_phrase(match.group("loser"))
        if not winner or not loser:
            continue
        context = frozenset(_tokens(match.group("context") or ""))
        return _ChoiceClaim(winner=winner, loser=loser, context=context)
    return None


def _choice_phrase(value: str) -> str:
    words = _tokens(value)
    return " ".join(sorted(words))


def _contexts_compatible(
    left: frozenset[str],
    right: frozenset[str],
) -> bool:
    return not left or not right or bool(left & right)


def _tokens(content: str) -> set[str]:
    normalized = content.lower().replace("_", " ").replace("-", " ")
    return {
        _stem(token)
        for token in _TOKEN_RE.findall(normalized)
        if token not in _STOPWORDS and len(token) > 1
    }


def _normalized_content(content: str) -> str:
    return " ".join(content.lower().split())


def _order_for_review(facts: Sequence[Fact]) -> list[Fact]:
    return sorted(
        facts,
        key=lambda fact: (
            -fact.confidence,
            -_aware(fact.updated_at).timestamp(),
            -len(fact.content),
            fact.id,
        ),
    )


def _suggested_merge_content(facts: Sequence[Fact]) -> str:
    return max(
        facts, key=lambda fact: (len(_tokens(fact.content)), len(fact.content))
    ).content


def _duplicate_reason(
    fact_ids: Sequence[str],
    signals: dict[tuple[str, str], _DuplicateSignal],
) -> str:
    reasons = [
        signal.reason
        for pair, signal in signals.items()
        if pair[0] in fact_ids and pair[1] in fact_ids
    ]
    if not reasons:
        return "facts appear to repeat the same memory"
    return "; ".join(sorted(set(reasons)))


def _duplicate_confidence(
    fact_ids: Sequence[str],
    signals: dict[tuple[str, str], _DuplicateSignal],
) -> float:
    scores = [
        signal.score
        for pair, signal in signals.items()
        if pair[0] in fact_ids and pair[1] in fact_ids
    ]
    return sum(scores) / len(scores) if scores else 0.80


def _merge_command(fact_ids: Sequence[str], content: str) -> str:
    quoted_ids = " ".join(shlex.quote(fact_id) for fact_id in fact_ids)
    return (
        f"engram merge-memories {quoted_ids} "
        f"--content {shlex.quote(content)} "
        "--reason 'memory audit duplicate suggestion'"
    )


def _stale_command(fact_id: str, reason: str) -> str:
    return f"engram mark-stale {shlex.quote(fact_id)} --reason {shlex.quote(reason)}"


def _suggestion_id(kind: str, fact_ids: Sequence[str]) -> str:
    digest = hashlib.sha1(f"{kind}:{':'.join(sorted(fact_ids))}".encode()).hexdigest()
    return f"{kind[:4]}-{digest[:10]}"


def _kind_order(kind: SuggestionKind) -> int:
    return {"duplicate": 0, "stale": 1, "contradiction": 2}[kind]


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
