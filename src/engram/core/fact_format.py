"""Render facts as dated card lines for recall output and model context."""

from engram.core.models import Durability, Fact


def content_signature(text: str) -> str:
    """Normalize fact content for exact-match deduplication."""
    return " ".join(text.lower().split())


def fact_flags(fact: Fact) -> list[str]:
    """Short lifecycle labels shown next to a fact in recall output."""
    flags: list[str] = []
    if fact.durability is Durability.ephemeral:
        flags.append("time-bound")
    if fact.expires_at is not None:
        flags.append(f"expires {fact.expires_at.date().isoformat()}")
    if fact.suspect_reason:
        flags.append(f"unverified: {fact.suspect_reason}")
    return flags


def format_observed(fact: Fact) -> str:
    """The card's date, as ``oldest..newest`` when it carries older claims."""
    newest = fact.observed_at.date()
    oldest = fact.oldest_claim_at.date()
    if oldest < newest:
        return f"{oldest.isoformat()}..{newest.isoformat()}"
    return newest.isoformat()


def format_fact_line(fact: Fact) -> str:
    """One dated card line: ``[category · project · YYYY-MM-DD] content (id: …)``.

    A card that carries claims older than its latest update shows the range
    ``YYYY-MM-DD..YYYY-MM-DD``.
    """
    meta = [fact.category.value]
    if fact.project:
        meta.append(fact.project)
    meta.append(format_observed(fact))
    line = f"[{' · '.join(meta)}] {fact.content}"
    flags = fact_flags(fact)
    if flags:
        line += f" ({'; '.join(flags)})"
    return f"{line} (id: {fact.id})"


def format_facts_for_llm(facts: list[Fact]) -> str:
    """Format facts as a numbered, dated list for LLM context."""
    if not facts:
        return "(no facts stored)"
    lines = []
    for i, fact in enumerate(facts, 1):
        extra = ""
        if fact.memory_key:
            extra += f" [key: {fact.memory_key}]"
        if fact.supersedes:
            extra += f" [supersedes: {fact.supersedes}]"
        lines.append(f"{i}.{extra} {format_fact_line(fact)}")
    return "\n".join(lines)
