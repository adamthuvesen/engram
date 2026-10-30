"""Shared eligibility policy for materialized fact snapshots."""

from datetime import datetime, timezone

from engram.core.models import MIN_ACTIVE_CONFIDENCE, Fact, FactLifecycle


def is_fact_eligible(
    fact: Fact,
    now: datetime | None = None,
    *,
    include_stale: bool = False,
    min_confidence: float = MIN_ACTIVE_CONFIDENCE,
) -> bool:
    """Return whether a fact can participate in active memory workflows."""
    now = now or datetime.now(timezone.utc)
    return (
        fact.lifecycle is FactLifecycle.active
        and fact.confidence >= min_confidence
        and (fact.expires_at is None or fact.expires_at >= now)
        and (include_stale or not fact.stale)
    )
