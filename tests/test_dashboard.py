"""Tests for the Textual dashboard — data layer, filtering, actions, and UI."""

from datetime import datetime, timedelta, timezone

import pytest
from textual.widgets import DataTable, TabbedContent

from engram.config import get_settings
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
from engram.models import CandidateStatus, Fact, FactCategory, MemoryCandidate
from engram.store import FactStore


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
    assert "#788c5d" in result


def test_format_confidence_yellow():
    result = format_confidence(0.6)
    assert "#eda100" in result


def test_format_confidence_red():
    result = format_confidence(0.3)
    assert "#f7768e" in result


# ── Constants tests ──


def test_min_confidence_threshold():
    assert MIN_ACTIVE_CONFIDENCE == 0.1


def test_no_project_label():
    assert NO_PROJECT_LABEL == "(none)"


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
async def test_forgotten_tab_exists(store):
    store.append_facts([_fact("gone", confidence=0.05)])

    app = EngramDashboard()
    async with app.run_test(size=(140, 40)) as pilot:
        tabs = app.query_one("#tabs", TabbedContent)
        tabs.active = "forgotten"
        await pilot.pause()

        forgotten_table = app.query_one("#forgotten-table", DataTable)
        assert forgotten_table.row_count == 1


@pytest.mark.anyio
async def test_forget_and_undo(store):
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
        app.action_restore_facts([fact_id])
        await pilot.pause()

        assert app._data.active_count == 1
        assert app._data.forgotten_count == 0


@pytest.mark.anyio
async def test_candidates_tab_shows_pending(store):
    store.append_candidates([_candidate("test suggestion")])

    app = EngramDashboard()
    async with app.run_test(size=(140, 40)) as pilot:
        tabs = app.query_one("#tabs", TabbedContent)
        tabs.active = "candidates"
        await pilot.pause()

        cand_table = app.query_one("#cand-table", DataTable)
        assert cand_table.row_count == 1


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
