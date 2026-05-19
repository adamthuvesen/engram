"""Tests for the append-only event-log model: event ids and replay."""

from datetime import datetime, timedelta, timezone

import pytest

from engram.models import (
    EVENT_LOG_META_VERSION,
    EventLogMeta,
    EventType,
    Fact,
    FactCategory,
    FactEvent,
    materialize_events,
    new_event_id,
    replay_fact,
)


def _make_fact(**overrides) -> Fact:
    defaults: dict = {
        "id": "fact_a",
        "category": FactCategory.preference,
        "content": "Uses polars",
    }
    defaults.update(overrides)
    return Fact(**defaults)


def _created_event(fact: Fact, ts: datetime | None = None) -> FactEvent:
    return FactEvent(
        event_type=EventType.created,
        fact_id=fact.id,
        timestamp=ts or fact.created_at,
        payload=fact.model_dump(),
    )


def _at(seconds: int) -> datetime:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return base + timedelta(seconds=seconds)


# --- new_event_id ------------------------------------------------------------


def test_new_event_id_is_strictly_monotonic_in_process():
    ids = [new_event_id() for _ in range(100)]
    assert ids == sorted(ids), "event ids must be strictly increasing"
    assert len(set(ids)) == 100, "event ids must be unique"


def test_new_event_id_is_lexicographically_orderable():
    a = new_event_id()
    b = new_event_id()
    # Lexicographic comparison must agree with issue order.
    assert a < b


# --- replay: create-only -----------------------------------------------------


def test_replay_create_only_returns_active_fact():
    fact = _make_fact()
    result, is_active = replay_fact([_created_event(fact)])
    assert result is not None
    assert result.id == "fact_a"
    assert result.content == "Uses polars"
    assert is_active is True


def test_replay_empty_event_list_returns_none():
    result, is_active = replay_fact([])
    assert result is None
    assert is_active is False


def test_replay_without_created_returns_none():
    edit = FactEvent(
        event_type=EventType.edited,
        fact_id="fact_a",
        timestamp=_at(0),
        payload={"content": "no create"},
    )
    result, is_active = replay_fact([edit])
    assert result is None
    assert is_active is False


def test_replay_mixed_fact_ids_raises():
    a = _make_fact(id="fact_a")
    b = _make_fact(id="fact_b")
    with pytest.raises(ValueError, match="multiple fact_ids"):
        replay_fact([_created_event(a), _created_event(b)])


# --- replay: create + edit ---------------------------------------------------


def test_replay_create_then_edit_applies_update():
    fact = _make_fact(content="v1")
    events = [
        _created_event(fact, ts=_at(0)),
        FactEvent(
            event_type=EventType.edited,
            fact_id="fact_a",
            timestamp=_at(1),
            payload={"content": "v2"},
        ),
    ]
    result, is_active = replay_fact(events)
    assert result is not None
    assert result.content == "v2"
    assert is_active is True
    assert result.updated_at == _at(1)


def test_replay_edit_only_touches_editable_fields():
    fact = _make_fact()
    original_id = fact.id
    original_created_at = fact.created_at
    events = [
        _created_event(fact, ts=_at(0)),
        FactEvent(
            event_type=EventType.edited,
            fact_id="fact_a",
            timestamp=_at(1),
            payload={"id": "stolen", "created_at": _at(99).isoformat(), "content": "v2"},
        ),
    ]
    result, _ = replay_fact(events)
    assert result is not None
    assert result.id == original_id, "id must be immutable from edited events"
    assert result.created_at == original_created_at
    assert result.content == "v2"


def test_replay_edit_coerces_iso_datetime_payload():
    fact = _make_fact()
    expires = _at(5_000).isoformat()
    events = [
        _created_event(fact, ts=_at(0)),
        FactEvent(
            event_type=EventType.edited,
            fact_id="fact_a",
            timestamp=_at(1),
            payload={"expires_at": expires},
        ),
    ]
    result, _ = replay_fact(events)
    assert result is not None
    assert result.expires_at == _at(5_000)


# --- replay: create + forget [+ restore] -------------------------------------


def test_replay_create_then_forget_yields_inactive():
    fact = _make_fact()
    events = [
        _created_event(fact, ts=_at(0)),
        FactEvent(
            event_type=EventType.forgotten,
            fact_id="fact_a",
            timestamp=_at(1),
            payload={"reason": "obsolete"},
        ),
    ]
    result, is_active = replay_fact(events)
    assert result is not None
    assert is_active is False


def test_replay_forget_then_restore_returns_active():
    fact = _make_fact()
    events = [
        _created_event(fact, ts=_at(0)),
        FactEvent(
            event_type=EventType.forgotten,
            fact_id="fact_a",
            timestamp=_at(1),
            payload={},
        ),
        FactEvent(
            event_type=EventType.restored,
            fact_id="fact_a",
            timestamp=_at(2),
            payload={},
        ),
    ]
    result, is_active = replay_fact(events)
    assert result is not None
    assert is_active is True


def test_replay_restore_then_forget_is_inactive_again():
    fact = _make_fact()
    events = [
        _created_event(fact, ts=_at(0)),
        FactEvent(event_type=EventType.forgotten, fact_id="fact_a", timestamp=_at(1)),
        FactEvent(event_type=EventType.restored, fact_id="fact_a", timestamp=_at(2)),
        FactEvent(event_type=EventType.forgotten, fact_id="fact_a", timestamp=_at(3)),
    ]
    result, is_active = replay_fact(events)
    assert result is not None
    assert is_active is False


def test_replay_create_edit_forget():
    fact = _make_fact(content="v1")
    events = [
        _created_event(fact, ts=_at(0)),
        FactEvent(
            event_type=EventType.edited,
            fact_id="fact_a",
            timestamp=_at(1),
            payload={"content": "v2"},
        ),
        FactEvent(event_type=EventType.forgotten, fact_id="fact_a", timestamp=_at(2)),
    ]
    result, is_active = replay_fact(events)
    assert result is not None
    assert result.content == "v2", "edits before forget must still apply"
    assert is_active is False


# --- replay: stale toggle ----------------------------------------------------


def test_replay_stale_then_unstale():
    fact = _make_fact()
    events = [
        _created_event(fact, ts=_at(0)),
        FactEvent(
            event_type=EventType.stale,
            fact_id="fact_a",
            timestamp=_at(1),
            payload={"reason": "outdated"},
        ),
    ]
    result, is_active = replay_fact(events)
    assert result is not None
    assert result.stale is True
    assert result.stale_reason == "outdated"
    assert is_active is True, "stale facts remain active in the lifecycle sense"

    events.append(
        FactEvent(event_type=EventType.unstale, fact_id="fact_a", timestamp=_at(2))
    )
    result, _ = replay_fact(events)
    assert result is not None
    assert result.stale is False
    assert result.stale_reason == ""


# --- replay: conflicting concurrent edits ------------------------------------


def test_replay_later_timestamp_wins_for_same_field():
    fact = _make_fact(content="v1")
    a = FactEvent(
        event_type=EventType.edited,
        fact_id="fact_a",
        timestamp=_at(10),
        payload={"content": "from-machine-a"},
        actor="machine-a",
    )
    b = FactEvent(
        event_type=EventType.edited,
        fact_id="fact_a",
        timestamp=_at(20),
        payload={"content": "from-machine-b"},
        actor="machine-b",
    )
    # Events arrive in arbitrary on-disk order; replay must sort them.
    result, _ = replay_fact([_created_event(fact, ts=_at(0)), b, a])
    assert result is not None
    assert result.content == "from-machine-b"


def test_replay_exact_timestamp_tie_broken_by_event_id():
    fact = _make_fact(content="v1")
    same_ts = _at(10)
    low = FactEvent(
        event_id="aaa_low",
        event_type=EventType.edited,
        fact_id="fact_a",
        timestamp=same_ts,
        payload={"content": "low"},
    )
    high = FactEvent(
        event_id="zzz_high",
        event_type=EventType.edited,
        fact_id="fact_a",
        timestamp=same_ts,
        payload={"content": "high"},
    )
    result, _ = replay_fact([_created_event(fact, ts=_at(0)), high, low])
    assert result is not None
    assert result.content == "high", (
        "lexicographically greater event_id MUST win on exact-timestamp tie"
    )


def test_replay_both_conflicting_events_remain_for_audit():
    # The events list given to replay is the full audit trail; no event is
    # dropped. This test asserts the calling layer's data preservation
    # contract: both conflicting events are in the list, both are visible.
    fact = _make_fact(content="v1")
    a = FactEvent(
        event_type=EventType.edited,
        fact_id="fact_a",
        timestamp=_at(10),
        payload={"content": "from-a"},
    )
    b = FactEvent(
        event_type=EventType.edited,
        fact_id="fact_a",
        timestamp=_at(20),
        payload={"content": "from-b"},
    )
    events = [_created_event(fact, ts=_at(0)), a, b]
    assert len(events) == 3, "audit log preserves both conflicting edits"
    result, _ = replay_fact(events)
    assert result is not None
    assert result.content == "from-b"


# --- replay: supersession ----------------------------------------------------


def test_replay_superseded_fact_becomes_inactive():
    old = _make_fact(id="fact_old", content="v1")
    new_id = "fact_new"
    events = [
        _created_event(old, ts=_at(0)),
        FactEvent(
            event_type=EventType.superseded,
            fact_id="fact_old",
            timestamp=_at(1),
            payload={"superseded_by": new_id},
        ),
    ]
    result, is_active = replay_fact(events)
    assert result is not None
    assert is_active is False, "superseded facts must drop out of active recall"
    assert result.content == "v1", "content remains for audit"


def test_materialize_supersession_chain():
    old = _make_fact(id="fact_old", content="v1")
    new = _make_fact(id="fact_new", content="v2", supersedes="fact_old")
    events = [
        _created_event(old, ts=_at(0)),
        FactEvent(
            event_type=EventType.superseded,
            fact_id="fact_old",
            timestamp=_at(1),
            payload={"superseded_by": "fact_new"},
        ),
        _created_event(new, ts=_at(1)),
    ]
    materialized = materialize_events(events)
    assert set(materialized) == {"fact_old", "fact_new"}
    old_fact, old_active = materialized["fact_old"]
    new_fact, new_active = materialized["fact_new"]
    assert old_active is False
    assert new_active is True
    assert new_fact.supersedes == "fact_old"


# --- event-log meta ----------------------------------------------------------


def test_event_log_meta_default_version():
    meta = EventLogMeta()
    assert meta.meta == EVENT_LOG_META_VERSION
    assert meta.meta == "event-log-v1"
