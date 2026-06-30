"""Tests for the Textual dashboard — data layer, filtering, actions, and UI."""

from datetime import datetime, timedelta, timezone

import pytest
from textual.widgets import DataTable, TabbedContent

from engram.core.config import get_settings
from engram.dashboard.app import EngramDashboard
from engram.dashboard.constants import MIN_ACTIVE_CONFIDENCE, NO_PROJECT_LABEL
from engram.dashboard.data import (
    content_hash_for,
    format_age,
    format_bytes,
    format_confidence,
    format_timestamp,
    load_dashboard_data,
)
from engram.dashboard.tables import (
    filter_facts_by_text,
    handle_table_key,
    next_sort_state,
    short_cell,
)
from engram.core.models import CandidateStatus, Fact, FactCategory, MemoryCandidate
from engram.storage.store import FactStore


# ── Fixtures ──


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    s = FactStore(data_dir=tmp_path)
    yield s
    get_settings.cache_clear()


def _fact(
    content="test",
    category=FactCategory.project,
    project="engram",
    confidence=1.0,
    **kw,
):
    return Fact(
        category=category, content=content, project=project, confidence=confidence, **kw
    )


def _candidate(
    content="suggestion", project="engram", status=CandidateStatus.pending, **kw
):
    return MemoryCandidate(
        category=FactCategory.project,
        content=content,
        project=project,
        status=status,
        why_store="testing",
        **kw,
    )


# ── Data layer tests ──


def test_load_empty_store(store):
    data = load_dashboard_data(store)
    assert data.total == 0
    assert data.active_count == 0
    assert data.forgotten_count == 0
    assert data.expired_count == 0


def test_active_facts_counted(store):
    store.append_facts([_fact("one"), _fact("two"), _fact("three")])
    data = load_dashboard_data(store)
    assert data.total == 3
    assert data.active_count == 3
    assert data.forgotten_count == 0


def test_forgotten_facts_classified(store):
    store.append_facts([_fact("active"), _fact("gone", confidence=0.05)])
    data = load_dashboard_data(store)
    assert data.active_count == 1
    assert data.forgotten_count == 1
    assert len(data.forgotten_facts) == 1
    assert data.forgotten_facts[0].content == "gone"


def test_expired_facts_classified(store):
    past = datetime.now(timezone.utc) - timedelta(days=1)
    store.append_facts([_fact("active"), _fact("old", expires_at=past)])
    data = load_dashboard_data(store)
    assert data.active_count == 1
    assert data.expired_count == 1


def test_stale_facts_excluded_from_active_rollups(store):
    store.append_facts(
        [
            _fact("active", category=FactCategory.preference, project="alpha"),
            _fact(
                "stale",
                category=FactCategory.preference,
                project="alpha",
                stale=True,
            ),
        ]
    )
    data = load_dashboard_data(store)

    assert data.total == 2
    assert data.active_count == 1
    assert [fact.content for fact in data.active_facts] == ["active"]
    assert data.by_category["preference"] == 1
    assert data.by_project["alpha"] == 1

    health = data.project_health["alpha"]
    assert health.total == 2
    assert health.active == 1
    assert health.forgotten == 0
    assert health.expired == 0
    assert health.categories["preference"] == 1


def test_category_distribution(store):
    store.append_facts(
        [
            _fact("a", category=FactCategory.preference),
            _fact("b", category=FactCategory.preference),
            _fact("c", category=FactCategory.event),
        ]
    )
    data = load_dashboard_data(store)
    assert data.by_category["preference"] == 2
    assert data.by_category["event"] == 1


def test_project_distribution(store):
    store.append_facts(
        [
            _fact("a", project="alpha"),
            _fact("b", project="alpha"),
            _fact("c", project=None),
        ]
    )
    data = load_dashboard_data(store)
    assert data.by_project["alpha"] == 2
    assert data.by_project[NO_PROJECT_LABEL] == 1


def test_project_health(store):
    store.append_facts(
        [
            _fact("a", project="demo"),
            _fact("b", project="demo", confidence=0.05),
        ]
    )
    data = load_dashboard_data(store)
    health = data.project_health["demo"]
    assert health.active == 1
    assert health.forgotten == 1
    assert health.total == 2


def test_candidate_counts(store):
    store.append_candidates(
        [
            _candidate("pending one"),
            _candidate("approved", status=CandidateStatus.approved),
        ]
    )
    data = load_dashboard_data(store)
    assert data.pending_count == 1
    assert len(data.candidates) == 2


def test_sparkline_length(store):
    store.append_facts([_fact("x")])
    data = load_dashboard_data(store)
    assert len(data.activity_30d) == 30
    assert len(data.activity_7d) == 7


# ── Content hash tests ──


def test_content_hash_changes_on_write(store):
    store.append_facts([_fact("first")])
    hash1 = content_hash_for(store)
    store.append_facts([_fact("second")])
    hash2 = content_hash_for(store)
    assert hash1 != hash2


def test_content_hash_stable_when_unchanged(store):
    store.append_facts([_fact("stable")])
    assert content_hash_for(store) == content_hash_for(store)


# ── Formatting tests ──


def test_format_age_days():
    dt = datetime.now(timezone.utc) - timedelta(days=5)
    assert format_age(dt) == "5d"


def test_format_age_hours():
    dt = datetime.now(timezone.utc) - timedelta(hours=3)
    assert format_age(dt) == "3h"


def test_format_age_minutes():
    dt = datetime.now(timezone.utc) - timedelta(minutes=15)
    assert format_age(dt) == "15m"


def test_format_age_now():
    dt = datetime.now(timezone.utc) - timedelta(seconds=5)
    assert format_age(dt) == "now"


def test_format_bytes_kb():
    assert format_bytes(2048) == "2.0 KB"


def test_format_bytes_zero():
    assert format_bytes(0) == "0 B"


def test_format_timestamp_includes_utc():
    dt = datetime(2026, 3, 15, 14, 30, tzinfo=timezone.utc)
    result = format_timestamp(dt)
    assert "2026-03-15 14:30 UTC" in result
    assert "ago)" in result


def test_format_confidence_green():
    result = format_confidence(0.9)
    assert "$success" in result


def test_format_confidence_yellow():
    result = format_confidence(0.6)
    assert "$warning" in result


def test_format_confidence_red():
    result = format_confidence(0.3)
    assert "$error" in result


def test_short_cell_truncates_only_when_needed():
    assert short_cell("short", 10) == "short"
    assert short_cell("longer than ten", 10) == "longer tha..."


def test_filter_facts_by_text_matches_fact_fields():
    fact = _fact("Uses textual dashboard", tags=["terminal"], project="engram")
    fact.id = "fact-filter-test"

    assert filter_facts_by_text([fact], "textual") == [fact]
    assert filter_facts_by_text([fact], "terminal") == [fact]
    assert filter_facts_by_text([fact], "engram") == [fact]
    assert filter_facts_by_text([fact], "fact-filter-test") == [fact]
    assert filter_facts_by_text([fact], "engram", include_project=False) == []


def test_next_sort_state_cycles_column_or_direction():
    columns = ["created_at", "category"]

    assert next_sort_state(columns, 0, True, reverse=False) == (1, True, "category")
    assert next_sort_state(columns, 1, True, reverse=True) == (1, False, "category")


def test_space_on_empty_table_is_handled_without_selection():
    class Event:
        key = "space"
        prevented = False

        def prevent_default(self):
            self.prevented = True

    event = Event()
    table = DataTable()
    selected_ids: set[str] = set()

    handled = handle_table_key(
        event,
        focused=table,
        table=table,
        selected_ids=selected_ids,
        visible_ids=[],
        refresh_table=lambda: pytest.fail("empty table should not refresh"),
        cycle_sort=lambda reverse: pytest.fail("space should not sort"),
    )

    assert handled is True
    assert event.prevented is True
    assert selected_ids == set()


# ── Constants tests ──


def test_min_confidence_threshold():
    assert MIN_ACTIVE_CONFIDENCE == 0.1


def test_no_project_label():
    assert NO_PROJECT_LABEL == "(none)"


def test_dashboard_refresh_screen_logs_failures(store, monkeypatch, caplog):
    class BrokenScreen:
        def refresh_data(self, data):
            raise RuntimeError("refresh failed")

    app = EngramDashboard()
    monkeypatch.setattr(app, "query_one", lambda *args, **kwargs: BrokenScreen())

    with caplog.at_level("ERROR", logger="engram.dashboard.app"):
        app._refresh_screen("overview")

    assert "Unable to refresh dashboard tab overview" in caplog.text


# ── UI tests ──


@pytest.mark.anyio
async def test_dashboard_tab_content_uses_available_height(store):
    store.append_facts([_fact("Dashboard smoke test fact")])

    app = EngramDashboard()
    async with app.run_test(size=(140, 40)) as pilot:
        overview_pane = app.query_one("#overview")
        assert overview_pane.region.height > 10

        tabs = app.query_one("#tabs", TabbedContent)
        tabs.active = "facts"
        await pilot.pause()

        facts_pane = app.query_one("#facts")
        facts_table = app.query_one("#facts-table", DataTable)

        assert facts_pane.region.height > 10
        assert facts_table.region.height > 5
        assert facts_table.row_count == 1


@pytest.mark.anyio
async def test_forget_and_restore(store):
    store.append_facts([_fact("to forget")])
    data = load_dashboard_data(store)
    fact_id = data.active_facts[0].id

    app = EngramDashboard()
    async with app.run_test(size=(140, 40)) as pilot:
        # Forget directly through store action
        app._store.forget(fact_id, reason="test")
        app._force_refresh()
        await pilot.pause()

        # Fact should now be forgotten
        assert app._data.forgotten_count == 1
        assert app._data.active_count == 0

        # Restore it
        app._store.restore_fact(fact_id)
        app._force_refresh()
        await pilot.pause()

        assert app._data.active_count == 1
        assert app._data.forgotten_count == 0


@pytest.mark.anyio
async def test_ctrl_z_undo_restores_forgotten_fact(store):
    store.append_facts([_fact("to undo", confidence=0.8)])
    fact_id = load_dashboard_data(store).active_facts[0].id

    app = EngramDashboard()
    async with app.run_test(size=(140, 40)) as pilot:
        app._pending_forget = [fact_id]
        app._on_forget_confirmed(True)
        await pilot.pause()

        assert app._data.active_count == 0
        assert app._data.forgotten_count == 1

        app.key_ctrl_z()
        await pilot.pause()

        assert app._data.active_count == 1
        assert app._data.forgotten_count == 0
        assert store.load_active_facts()[0].id == fact_id


@pytest.mark.anyio
async def test_help_screen_opens(store):
    app = EngramDashboard()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("question_mark")
        await pilot.pause()
        # Help screen should be pushed
        assert len(app.screen_stack) > 1
        await pilot.press("escape")
        await pilot.pause()
        assert len(app.screen_stack) == 1
