"""Forgotten facts screen — browse and restore soft-deleted facts."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Input

from engram.dashboard.data import DashboardData, format_age, shorten_project
from engram.dashboard.tables import (
    filter_facts_by_text,
    focus_nav_from_top_row,
    handle_table_key,
    item_by_id,
    next_sort_state,
    schedule_filter_timer,
    short_cell,
    sort_by_column,
)
from engram.dashboard.widgets.fact_detail import FactDetail
from engram.core.models import Fact


class ForgottenScreen(Container):
    """Browse forgotten facts with option to restore."""

    SORT_COLUMNS = ["updated_at", "category", "project", "content"]

    def __init__(self, data: DashboardData) -> None:
        super().__init__()
        self._data = data
        self._filtered_facts: list[Fact] = list(data.forgotten_facts)
        self._selected_fact: Fact | None = None
        self._filter_text: str = ""
        self._selected_ids: set[str] = set()
        self._search_timer = None
        self._sort_index: int = 0
        self._sort_reverse: bool = True

    def compose(self) -> ComposeResult:
        with Horizontal(id="forgotten-filter-bar", classes="filter-bar"):
            yield Input(placeholder="Search forgotten facts...", id="forgotten-search")

        with Horizontal(id="forgotten-body", classes="screen-body"):
            with Vertical(id="forgotten-table-container", classes="table-container"):
                yield DataTable(id="forgotten-table", cursor_type="row")
            yield FactDetail(id="forgotten-detail", classes="detail-pane")

    def on_mount(self) -> None:
        table = self.query_one("#forgotten-table", DataTable)
        table.add_columns(
            "", "id", "category", "project", "content", "tags", "forgotten"
        )
        self._populate_table()

    def _populate_table(self) -> None:
        table = self.query_one("#forgotten-table", DataTable)
        table.clear()
        for fact in self._filtered_facts:
            sel = "✓" if fact.id in self._selected_ids else " "
            table.add_row(
                sel,
                fact.id,
                fact.category.value,
                shorten_project(fact.project),
                short_cell(fact.content, 55),
                ", ".join(fact.tags[:3]),
                format_age(fact.updated_at),
                key=fact.id,
            )

    def _apply_filters(self) -> None:
        facts = filter_facts_by_text(self._data.forgotten_facts, self._filter_text)
        self._filtered_facts = facts
        self._populate_table()

    def _schedule_filter(self) -> None:
        self._search_timer = schedule_filter_timer(
            self, self._search_timer, self._apply_filters
        )

    @on(Input.Changed, "#forgotten-search")
    def on_search(self, event: Input.Changed) -> None:
        self._filter_text = event.value
        self._schedule_filter()

    @on(DataTable.RowSelected, "#forgotten-table")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key and event.row_key.value:
            fact = self._find_fact(event.row_key.value)
            if fact:
                self._selected_fact = fact
                detail = self.query_one("#forgotten-detail", FactDetail)
                detail.add_class("visible")
                detail.update_fact(fact)

    def _find_fact(self, fact_id: str) -> Fact | None:
        return item_by_id(self._data.forgotten_facts, fact_id)

    def _get_action_targets(self) -> list[str]:
        """Multi-select takes priority; otherwise act on the cursor row."""
        if self._selected_ids:
            return list(self._selected_ids)
        if self._selected_fact:
            return [self._selected_fact.id]
        return []

    def on_key(self, event) -> None:
        table = self.query_one("#forgotten-table", DataTable)
        focused = self.app.focused

        if focus_nav_from_top_row(self.app, table, event):
            return

        if event.key == "down" and isinstance(focused, Input):
            event.prevent_default()
            table.focus()
            return

        if event.key == "slash" and not isinstance(focused, Input):
            event.prevent_default()
            self.query_one("#forgotten-search", Input).focus()
            return

        if handle_table_key(
            event,
            focused=focused,
            table=table,
            selected_ids=self._selected_ids,
            visible_ids=(fact.id for fact in self._filtered_facts),
            refresh_table=self._populate_table,
            cycle_sort=self._cycle_sort,
        ):
            return

    def _cycle_sort(self, reverse: bool) -> None:
        self._sort_index, self._sort_reverse, col = next_sort_state(
            self.SORT_COLUMNS,
            self._sort_index,
            self._sort_reverse,
            reverse=reverse,
        )
        sort_by_column(self._filtered_facts, col, reverse=self._sort_reverse)
        self._populate_table()

    def key_escape(self) -> None:
        focused = self.app.focused
        if isinstance(focused, Input):
            self.query_one("#forgotten-table", DataTable).focus()
            return
        detail = self.query_one("#forgotten-detail", FactDetail)
        detail.remove_class("visible")
        self._selected_fact = None

    def key_u(self) -> None:
        if isinstance(self.app.focused, Input):
            return
        targets = self._get_action_targets()
        if targets:
            self.app.action_restore_facts(targets)
            self._selected_ids -= set(targets)

    def refresh_data(self, data: DashboardData) -> None:
        self._data = data
        self._apply_filters()
