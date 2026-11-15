"""Collapse a materialized event log into its minimal event sequence."""

from engram.core.models import EventType, Fact, FactEvent, FactLifecycle


def compaction_created_event(fact_id: str, fact: Fact) -> FactEvent:
    return FactEvent(
        event_type=EventType.created,
        fact_id=fact_id,
        timestamp=fact.created_at,
        actor="compaction",
        payload=fact.model_dump(),
    )


def _compaction_forgotten_event(fact_id: str, fact: Fact) -> FactEvent:
    return FactEvent(
        event_type=EventType.forgotten,
        fact_id=fact_id,
        timestamp=fact.updated_at,
        actor="compaction",
        payload={"reason": "compacted_tombstone"},
    )


def compacted_event_log(
    materialized: dict[str, tuple[Fact, bool]],
    *,
    keep_tombstones: bool,
) -> tuple[list[FactEvent], int]:
    compacted: list[FactEvent] = []
    tombstones = 0
    for fact_id, (fact, is_active) in materialized.items():
        if not is_active:
            if not keep_tombstones:
                continue
            compacted.append(compaction_created_event(fact_id, fact))
            compacted.append(
                FactEvent(
                    event_type=EventType.superseded,
                    fact_id=fact_id,
                    timestamp=fact.updated_at,
                    payload={"superseded_by": fact.superseded_by},
                )
                if fact.lifecycle is FactLifecycle.superseded
                else _compaction_forgotten_event(fact_id, fact)
            )
            tombstones += 1
        else:
            compacted.append(compaction_created_event(fact_id, fact))
    return compacted, tombstones
