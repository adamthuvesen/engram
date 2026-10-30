"""Tests for the Textual dashboard — data layer, filtering, actions, and UI."""

import asyncio
from datetime import datetime, timedelta, timezone
import threading

import pytest
from textual.widgets import DataTable, Static, TabbedContent

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
        await app.workers.wait_for_complete()
        await pilot.pause()
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
        await app.workers.wait_for_complete()
        # Forget directly through store action
        store.forget(fact_id, reason="test")
        app._force_refresh()
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Fact should now be forgotten
        assert app._data.forgotten_count == 1
        assert app._data.active_count == 0

        # Restore it
        store.restore_fact(fact_id)
        app._force_refresh()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app._data.active_count == 1
        assert app._data.forgotten_count == 0


@pytest.mark.anyio
async def test_ctrl_z_undo_restores_forgotten_fact(store):
    store.append_facts([_fact("to undo", confidence=0.8)])
    fact_id = load_dashboard_data(store).active_facts[0].id

    app = EngramDashboard()
    async with app.run_test(size=(140, 40)) as pilot:
        await app.workers.wait_for_complete()
        app._pending_forget = [fact_id]
        app._on_forget_confirmed(True)
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app._data.active_count == 0
        assert app._data.forgotten_count == 1

        app.key_ctrl_z()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app._data.active_count == 1
        assert app._data.forgotten_count == 0
        assert store.load_active_facts()[0].id == fact_id


@pytest.mark.anyio
async def test_help_screen_opens(store):
    app = EngramDashboard()
    async with app.run_test(size=(140, 40)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.press("question_mark")
        await pilot.pause()
        # Help screen should be pushed
        assert len(app.screen_stack) > 1
        await pilot.press("escape")
        await pilot.pause()
        assert len(app.screen_stack) == 1


def test_superseded_facts_have_distinct_lifecycle_counts(store):
    original = _fact("old policy")
    store.append_facts([original])
    store.correct_fact(original.id, "new policy")
    data = load_dashboard_data(store)
    assert data.active_count == 1
    assert data.forgotten_count == 0
    assert data.superseded_count == 1
    assert data.project_health["engram"].superseded == 1


@pytest.mark.anyio
async def test_dashboard_refreshes_on_expiry_without_file_changes(store, monkeypatch):
    from engram.dashboard import app as app_module
    from engram.dashboard import data as data_module

    now = datetime.now(timezone.utc)
    store.append_facts([_fact("expires soon", expires_at=now + timedelta(minutes=1))])
    app = EngramDashboard()
    async with app.run_test(size=(140, 40)):
        await app.workers.wait_for_complete()
        before = content_hash_for(store)
        assert app._data.active_count == 1

        class Later(datetime):
            @classmethod
            def now(cls, tz=None):
                return now + timedelta(minutes=2)

        monkeypatch.setattr(app_module, "datetime", Later)
        monkeypatch.setattr(data_module, "datetime", Later)
        app._refresh_data()
        await app.workers.wait_for_complete()
        assert content_hash_for(store) == before
        assert app._data.active_count == 0
        assert app._data.expired_count == 1


@pytest.mark.anyio
async def test_undo_only_restores_facts_this_action_forgot(store):
    from engram.core.models import FactLifecycle

    active = _fact("active")
    forgotten = _fact("already forgotten")
    replaced = _fact("already replaced")
    store.append_facts([active, forgotten, replaced])
    store.forget(forgotten.id)
    store.correct_fact(replaced.id, "replacement")
    app = EngramDashboard()
    async with app.run_test(size=(140, 40)) as pilot:
        await app.workers.wait_for_complete()
        app._pending_forget = [
            forgotten.id,
            replaced.id,
            active.id,
            active.id,
            "unknown",
        ]
        app._on_forget_confirmed(True)
        await app.workers.wait_for_complete()
        assert app._undo_stack == [active.id]
        app.key_ctrl_z()
        await app.workers.wait_for_complete()
        await pilot.pause()
        states = {fact.id: fact.lifecycle for fact in store.load_facts()}
        assert states[active.id] is FactLifecycle.active
        assert states[forgotten.id] is FactLifecycle.forgotten
        assert states[replaced.id] is FactLifecycle.superseded


@pytest.mark.anyio
async def test_timeline_uses_required_plotext_widget(store):
    from textual_plotext import PlotextPlot

    app = EngramDashboard()
    async with app.run_test(size=(140, 40)):
        assert isinstance(app.query_one("#timeline-chart"), PlotextPlot)


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["startup", "refresh", "forget", "undo"])
async def test_dashboard_responds_while_store_lock_is_held(
    store, monkeypatch, operation
):
    from engram.storage.transactions import locked_store

    fact = _fact("locked store")
    store.append_facts([fact])
    app = EngramDashboard()
    assert app._store is None
    assert "Loading memory" in app._status_text()
    held = threading.Event()
    release = threading.Event()
    released = threading.Event()
    attempted = threading.Event()

    def hold_lock():
        with locked_store(store.data_dir):
            held.set()
            release.wait(10)
        released.set()

    method_name = {
        "startup": "recover_transactions",
        "refresh": "load_facts",
        "forget": "forget",
        "undo": "restore_fact",
    }[operation]
    original = getattr(FactStore, method_name)

    def blocked_operation(self, *args, **kwargs):
        attempted.set()
        return original(self, *args, **kwargs)

    holder = threading.Thread(target=hold_lock)
    if operation == "startup":
        monkeypatch.setattr(FactStore, method_name, blocked_operation)
        holder.start()
        assert await asyncio.to_thread(held.wait, 2)
    try:
        async with app.run_test(size=(140, 40)) as pilot:
            if operation != "startup":
                await app.workers.wait_for_complete()
                if operation == "undo":
                    app._pending_forget = [fact.id]
                    app._on_forget_confirmed(True)
                    await app.workers.wait_for_complete()
                monkeypatch.setattr(FactStore, method_name, blocked_operation)
                holder.start()
                assert await asyncio.to_thread(held.wait, 2)
                if operation == "refresh":
                    app._force_refresh()
                elif operation == "forget":
                    app._pending_forget = [fact.id]
                    app._on_forget_confirmed(True)
                else:
                    app.key_ctrl_z()
            assert await asyncio.to_thread(attempted.wait, 2)
            assert len(app.workers) == 1
            for _ in range(5):
                app._refresh_data()
            assert len(app.workers) <= 2
            await pilot.press("t", "3")
            assert app.theme == "engram-light"
            assert app.query_one("#tabs", TabbedContent).active == "facts"
            assert not released.is_set()
            if operation == "forget":
                app.key_ctrl_z()
            release.set()
            await app.workers.wait_for_complete()
            assert app._data.active_count == 1
            assert app._data.forgotten_count == 0
    finally:
        release.set()
        if holder.ident is not None:
            await asyncio.to_thread(holder.join, 2)
        assert not holder.is_alive()


@pytest.mark.anyio
async def test_failed_refresh_preserves_snapshot_and_retries_unchanged_hash(
    store, monkeypatch
):
    from engram.dashboard import app as app_module

    store.append_facts([_fact("first")])
    app = EngramDashboard()
    async with app.run_test(size=(140, 40)):
        await app.workers.wait_for_complete()
        before = app._data
        old_hash = app._last_hash
        store.append_facts([_fact("second")])
        original = app_module.load_dashboard_data
        attempts = 0

        def fail_once(store):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("read denied")
            return original(store)

        monkeypatch.setattr(app_module, "load_dashboard_data", fail_once)
        app._refresh_data()
        await app.workers.wait_for_complete()
        assert app._data is before
        assert app._last_hash == old_hash
        status = str(app.query_one("#status-bar", Static).render())
        assert "read denied" in status
        assert "Press r to retry" in status
        app._refresh_data()
        await app.workers.wait_for_complete()
        assert attempts == 2
        assert app._data.active_count == 2
        assert app._storage_error is None


@pytest.mark.anyio
async def test_startup_failure_can_retry(store, monkeypatch):
    store.append_facts([_fact("recoverable")])
    original = FactStore.recover_transactions

    def fail_recovery(self):
        raise OSError("cannot open store")

    monkeypatch.setattr(FactStore, "recover_transactions", fail_recovery)
    app = EngramDashboard()
    async with app.run_test(size=(140, 40)) as pilot:
        await app.workers.wait_for_complete()
        assert app._store is None
        assert app._last_hash is None
        assert "cannot open store" in app._status_text()
        monkeypatch.setattr(FactStore, "recover_transactions", original)
        await pilot.press("r")
        await app.workers.wait_for_complete()
        assert app._data.active_count == 1
        assert app._storage_error is None


@pytest.mark.anyio
async def test_partial_forget_and_failed_undo_keep_successful_undo_entries(
    store, monkeypatch
):
    first, second = _fact("first"), _fact("second")
    store.append_facts([first, second])
    original_forget = FactStore.forget
    original_restore = FactStore.restore_fact

    def fail_second(self, fact_id, **kwargs):
        if fact_id == second.id:
            raise OSError("forget denied")
        return original_forget(self, fact_id, **kwargs)

    def fail_restore(self, fact_id):
        raise OSError("restore denied")

    app = EngramDashboard()
    async with app.run_test(size=(140, 40)):
        await app.workers.wait_for_complete()
        monkeypatch.setattr(FactStore, "forget", fail_second)
        app._pending_forget = [first.id, second.id]
        app._on_forget_confirmed(True)
        await app.workers.wait_for_complete()
        assert app._undo_stack == [first.id]
        assert app._data.active_count == 1
        assert "forget denied" in app._status_text()
        monkeypatch.setattr(FactStore, "restore_fact", fail_restore)
        app.key_ctrl_z()
        await app.workers.wait_for_complete()
        assert app._undo_stack == [first.id]
        assert "restore denied" in app._status_text()
        monkeypatch.setattr(FactStore, "restore_fact", original_restore)
        app.key_ctrl_z()
        await app.workers.wait_for_complete()
        assert app._undo_stack == []
        assert app._data.active_count == 2


@pytest.mark.anyio
async def test_forget_keeps_undo_when_snapshot_reload_fails(store, monkeypatch):
    from engram.dashboard import app as app_module

    fact = _fact("undo after read failure")
    store.append_facts([fact])
    original = app_module.load_dashboard_data

    def fail_read(store):
        raise OSError("snapshot unavailable")

    app = EngramDashboard()
    async with app.run_test(size=(140, 40)):
        await app.workers.wait_for_complete()
        before = app._data
        monkeypatch.setattr(app_module, "load_dashboard_data", fail_read)
        app._pending_forget = [fact.id]
        app._on_forget_confirmed(True)
        await app.workers.wait_for_complete()
        assert app._data is before
        assert app._undo_stack == [fact.id]
        assert store.load_active_facts() == []
        assert "snapshot unavailable" in app._status_text()
        monkeypatch.setattr(app_module, "load_dashboard_data", original)
        app.key_ctrl_z()
        await app.workers.wait_for_complete()
        assert app._data.active_count == 1
        assert app._undo_stack == []
        assert app._storage_error is None


@pytest.mark.anyio
async def test_requested_undo_survives_expiry_while_refresh_owns_storage(
    store, monkeypatch
):
    from engram.dashboard import app as app_module

    fact = _fact("queued undo")
    store.append_facts([fact])
    forget_started = threading.Event()
    release_forget = threading.Event()
    refresh_started = threading.Event()
    release_refresh = threading.Event()
    original_forget = FactStore.forget
    original_load = app_module.load_dashboard_data
    read_count = 0

    def blocked_forget(self, fact_id, **kwargs):
        forget_started.set()
        assert release_forget.wait(5)
        return original_forget(self, fact_id, **kwargs)

    def blocked_refresh(store):
        nonlocal read_count
        read_count += 1
        if read_count == 2:
            refresh_started.set()
            assert release_refresh.wait(5)
        return original_load(store)

    app = EngramDashboard()
    async with app.run_test(size=(140, 40)):
        await app.workers.wait_for_complete()
        monkeypatch.setattr(FactStore, "forget", blocked_forget)
        monkeypatch.setattr(app_module, "load_dashboard_data", blocked_refresh)
        try:
            app._pending_forget = [fact.id]
            app._on_forget_confirmed(True)
            assert await asyncio.to_thread(forget_started.wait, 2)
            app._force_refresh()
            await asyncio.sleep(0)
            app.key_ctrl_z()
            app.key_ctrl_z()
            release_forget.set()
            assert await asyncio.to_thread(refresh_started.wait, 2)
            assert app._undo_stack == [fact.id]
            # Deliver the expiry callback while refresh still owns storage.
            app._expire_undo()
            assert app._undo_stack == [fact.id]
            release_refresh.set()
            await app.workers.wait_for_complete()
            assert app._undo_stack == []
            assert app._data.active_count == 1
            assert [item.id for item in store.load_active_facts()] == [fact.id]
        finally:
            release_forget.set()
            release_refresh.set()


@pytest.mark.anyio
@pytest.mark.parametrize("cancel_before_start", [True, False])
async def test_cancelled_undo_does_not_prevent_expiry(
    store, monkeypatch, cancel_before_start
):
    from textual.worker import WorkerCancelled

    fact = _fact("cancelled undo")
    store.append_facts([fact])
    app = EngramDashboard()
    async with app.run_test(size=(140, 40)):
        await app.workers.wait_for_complete()
        app._pending_forget = [fact.id]
        app._on_forget_confirmed(True)
        await app.workers.wait_for_complete()
        scheduled_callbacks = []
        original_set_timer = app.set_timer

        def record_timer(delay, callback, **kwargs):
            scheduled_callbacks.append(callback)
            return original_set_timer(delay, callback, **kwargs)

        monkeypatch.setattr(app, "set_timer", record_timer)
        async with app._storage_lock:
            app.key_ctrl_z()
            worker = next(w for w in app.workers if w.name == "_undo_worker")
            if not cancel_before_start:
                await asyncio.sleep(0)
            app._expire_undo()
            assert app._undo_stack == [fact.id]
            assert scheduled_callbacks
            worker.cancel()
            with pytest.raises(WorkerCancelled):
                await worker.wait()
            scheduled_callbacks[-1]()
            assert app._undo_stack == []
        assert store.load_active_facts() == []
