"""Lexical search over memory cards: tokenizer plus a BM25F-style index.

Recall, write-time reconciliation, and upkeep clustering all rank facts
through this index. BM25's inverse document frequency is what keeps a
project name that appears in a thousand facts from outscoring the one rare
term that actually identifies the answer.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

import snowballstemmer  # type: ignore[import-untyped]

from engram.core.models import Durability, Fact, FactCategory

TOKEN_RE = re.compile(r"[a-z0-9]+")

# English function words carry no retrieval signal. Domain verbs ("run",
# "use", "set") are deliberately kept.
STOPWORDS = frozenset(
    """
    a about above after again against all also an and any are as at be because
    been before being below between both but by can could did do does doing done
    down during each either else etc few for from further had has have having
    here how however i if in into is it its itself just like likes may me might more most
    must my no nor not now of off on once only or other our out over own per
    rather same shall should so some such than that the their them then there
    these they this those through thus to too under until up upon us very via
    was we were what when where whether which while who whom whose why will with
    within without would yet you your
    """.split()
)

# Query-side vocabulary bridges for common paraphrases in memory lookups.
QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "credential": ("secret", "token"),
    "credentials": ("secret", "token"),
    "database": ("warehouse", "snowflake"),
    "db": ("database", "warehouse"),
    "dedupe": ("deduplicate", "duplicate"),
    "dependency": ("package",),
    "dependencies": ("package",),
    "install": ("package",),
    "memories": ("memory", "fact"),
    "memory": ("fact",),
    "shell": ("terminal", "zsh"),
    "squad": ("team",),
    "ui": ("frontend",),
    "prefer": ("preference",),
    "preferences": ("prefer",),
    "gotcha": ("pitfall",),
    "gotchas": ("pitfall",),
}

# BM25 parameters. k1 saturates repeated terms; b normalizes card length.
_K1 = 1.2
_B = 0.75
_ALIAS_WEIGHT = 0.4
# Field weights: curated retrieval metadata counts like content; scope and
# category words are weak evidence.
_FIELD_WEIGHTS = (1.0, 1.0, 0.8, 0.3)

# Time-bound memories fade: half-life in days for ephemeral cards and events.
EPHEMERAL_HALF_LIFE_DAYS = 30.0
_MIN_DECAY = 0.25
SUSPECT_WEIGHT = 0.6

_stemmer = snowballstemmer.stemmer("english")


@lru_cache(maxsize=65536)
def stem(word: str) -> str:
    return _stemmer.stemWord(word)


def raw_tokens(text: str) -> list[str]:
    normalized = text.lower().replace("_", " ").replace("-", " ")
    return TOKEN_RE.findall(normalized)


def tokenize(text: str) -> list[str]:
    """Stemmed content terms plus adjacent-pair bigram terms, in order."""
    words = [stem(token) for token in raw_tokens(text) if token not in STOPWORDS]
    bigrams = [f"{left}_{right}" for left, right in zip(words, words[1:])]
    return words + bigrams


def term_set(text: str) -> set[str]:
    """Stemmed unigrams only; for overlap heuristics outside ranking."""
    return {
        stem(token)
        for token in raw_tokens(text)
        if token not in STOPWORDS and len(token) > 1
    }


@dataclass(frozen=True)
class SearchHit:
    fact: Fact
    score: float
    # Share of the query's IDF mass that this fact matched (0..1).
    coverage: float


@dataclass(frozen=True)
class _Doc:
    fact: Fact
    length: float
    freqs: dict[str, float]


@lru_cache(maxsize=32768)
def _field_counts(
    fact_id: str,
    updated_at: datetime,
    content: str,
    hint_text: str,
    tag_text: str,
    scope_text: str,
) -> tuple[tuple[str, float], ...]:
    del fact_id, updated_at  # cache key only
    freqs: defaultdict[str, float] = defaultdict(float)
    length = 0.0
    for weight, text in zip(_FIELD_WEIGHTS, (content, hint_text, tag_text, scope_text)):
        terms = tokenize(text)
        length += weight * len(terms)
        for term in terms:
            freqs[term] += weight
    return (("", length), *freqs.items())


def _doc_for(fact: Fact) -> _Doc:
    counts = _field_counts(
        fact.id,
        fact.updated_at,
        fact.content,
        " ".join([fact.memory_key, *fact.retrieval_hints]),
        " ".join(fact.tags),
        " ".join(filter(None, [fact.project or "", fact.category.value])),
    )
    length = counts[0][1]
    return _Doc(fact=fact, length=length, freqs=dict(counts[1:]))


@dataclass(frozen=True)
class _QueryTerm:
    weight: float
    # Query unigram this term stands for (itself, or the word it aliases);
    # None for bigrams, which rank but do not count toward coverage.
    origin: str | None


def _query_terms(query: str) -> dict[str, _QueryTerm]:
    terms: dict[str, _QueryTerm] = {}
    for term in tokenize(query):
        terms[term] = _QueryTerm(1.0, None if "_" in term else term)
    for token in raw_tokens(query):
        if token in STOPWORDS:
            continue
        origin = stem(token)
        for alias in QUERY_ALIASES.get(token, ()):
            for term in tokenize(alias):
                terms.setdefault(term, _QueryTerm(_ALIAS_WEIGHT, origin))
    return terms


def freshness_weight(fact: Fact, now: datetime) -> float:
    """Down-weight time-bound and suspect cards; durable cards never decay."""
    weight = 1.0
    if fact.durability is Durability.ephemeral or fact.category is FactCategory.event:
        observed = fact.observed_at
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - observed).total_seconds() / 86400)
        weight *= max(_MIN_DECAY, 0.5 ** (age_days / EPHEMERAL_HALF_LIFE_DAYS))
    if fact.suspect_reason:
        weight *= SUSPECT_WEIGHT
    return weight


class SearchIndex:
    """Inverted BM25F-style index over a fixed set of facts."""

    def __init__(self, facts: Sequence[Fact]):
        self._docs = [_doc_for(fact) for fact in facts]
        self._postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for idx, doc in enumerate(self._docs):
            for term, freq in doc.freqs.items():
                self._postings[term].append((idx, freq))
        total = sum(doc.length for doc in self._docs)
        self._avg_length = total / len(self._docs) if self._docs else 1.0
        count = len(self._docs)
        self._idf = {
            term: math.log(1 + (count - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self._postings.items()
        }

    def __len__(self) -> int:
        return len(self._docs)

    @property
    def facts(self) -> list[Fact]:
        return [doc.fact for doc in self._docs]

    def idf(self, term: str) -> float:
        # Unseen terms get the maximum IDF so coverage reflects the miss.
        count = len(self._docs)
        return self._idf.get(term, math.log(1 + (count + 0.5) / 0.5))

    def search(
        self,
        query: str,
        *,
        accept: Callable[[Fact], bool] | None = None,
        limit: int | None = None,
        now: datetime | None = None,
    ) -> list[SearchHit]:
        """Rank facts matching ``query``; only positive-scoring facts return."""
        terms = _query_terms(query)
        if not terms or not self._docs:
            return []
        now = now or datetime.now(timezone.utc)
        # Coverage is measured over the query's own words; an alias match
        # credits the word it stands for.
        origins = {term.origin for term in terms.values() if term.origin}
        mass = sum(self.idf(origin) for origin in origins)
        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, set[str]] = defaultdict(set)
        for text, term in terms.items():
            posting = self._postings.get(text)
            if not posting:
                continue
            idf = self._idf[text]
            for idx, freq in posting:
                doc = self._docs[idx]
                norm = _K1 * (1 - _B + _B * doc.length / self._avg_length)
                scores[idx] += term.weight * idf * freq * (_K1 + 1) / (freq + norm)
                if term.origin:
                    matched[idx].add(term.origin)

        hits: list[SearchHit] = []
        for idx, score in scores.items():
            fact = self._docs[idx].fact
            if accept is not None and not accept(fact):
                continue
            matched_mass = sum(self.idf(origin) for origin in matched[idx])
            coverage = min(1.0, matched_mass / mass) if mass else 0.0
            hits.append(
                SearchHit(
                    fact=fact,
                    score=score * freshness_weight(fact, now),
                    coverage=coverage,
                )
            )
        hits.sort(
            key=lambda hit: (hit.score, hit.fact.updated_at.timestamp()),
            reverse=True,
        )
        return hits[:limit] if limit else hits

    def neighbors(
        self,
        fact: Fact,
        *,
        accept: Callable[[Fact], bool] | None = None,
        limit: int = 20,
    ) -> list[SearchHit]:
        """Facts lexically closest to ``fact`` (excluding itself)."""
        query = " ".join([fact.memory_key, fact.content, *fact.retrieval_hints])
        return [
            hit
            for hit in self.search(query, accept=accept, limit=limit + 1)
            if hit.fact.id != fact.id
        ][:limit]
