"""Fact browser screen — filterable, sortable table with detail pane."""

from typing import TYPE_CHECKING, cast

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.timer import Timer
from textual.widgets import DataTable, Input, Select

from engram.dashboard.constants import NO_PROJECT_LABEL
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
from engram.core.models import Fact, FactCategory

if TYPE_CHECKING:
    from engram.dashboard.app import EngramDashboard


class FactsScreen(Container):
    """Browse, filter, sort, and inspect facts."""

    SORT_COLUMNS = ["created_at", "category", "project", "content", "confidence"]

    def __init__(self, data: DashboardData) -> None:
        super().__init__()
        self._data = data
        self._filtered_facts: list[Fact] = list(data.active_facts)
        self._selected_fact: Fact | None = None
        self._filter_category: str | None = None
        self._filter_project: str | None = None
        self._filter_text: str = ""
        self._selected_ids: set[str] = set()
        self._sort_index: int = 0
        self._sort_reverse: bool = True
        self._search_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        cat_options = [(c.value, c.value) for c in FactCategory]
        proj_options = (
            [(p, p) for p in self._data.projects] if self._data.projects else []
        )

        with Horizontal(id="filter-bar", classes="filter-bar"):
            yield Input(placeholder="Search facts...", id="search-input")
            yield Select(
                [("All categories", "all")] + cat_options,
                value="all",
                id="cat-filter",
                allow_blank=False,
            )
            yield Select(
                [("All projects", "all")] + proj_options,
                value="all",
                id="proj-filter",
                allow_blank=False,
            )

        with Horizontal(id="facts-body", classes="screen-body"):
            with Vertical(id="facts-table-container", classes="table-container"):
                yield DataTable(id="facts-table", cursor_type="row")
            yield FactDetail(id="detail-pane", classes="detail-pane")

    def on_mount(self) -> None:
        table = self.query_one("#facts-table", DataTable)
        table.add_columns(
            "", "id", "category", "project", "content", "conf", "tags", "created"
        )
        self._populate_table()

    def _populate_table(self) -> None:
        table = self.query_one("#facts-table", DataTable)
        table.clear()
        for fact in self._filtered_facts:
            sel = "✓" if fact.id in self._selected_ids else " "
            table.add_row(
                sel,
                fact.id,
                fact.category.value,
                shorten_project(fact.project),
                short_cell(fact.content, 55),
                f"{fact.confidence:.0%}",
                ", ".join(fact.tags[:3]),
                format_age(fact.created_at),
                key=fact.id,
            )

    def _apply_filters(self) -> None:
        facts = list(self._data.active_facts)

        if self._filter_category and self._filter_category != "all":
            facts = [f for f in facts if f.category.value == self._filter_category]
        if self._filter_project and self._filter_project != "all":
            target = (
                None
                if self._filter_project == NO_PROJECT_LABEL
                else self._filter_project
            )
            facts = [f for f in facts if f.project == target]
        facts = filter_facts_by_text(facts, self._filter_text)

        col = self.SORT_COLUMNS[self._sort_index]
        sort_by_column(facts, col, reverse=self._sort_reverse)

        self._filtered_facts = facts
        # Selection refers to filtered rows; reset when the set changes.
        self._selected_ids.clear()
        self._populate_table()

    def _schedule_filter(self) -> None:
        self._search_timer = schedule_filter_timer(
            self, self._search_timer, self._apply_filters
        )

    @on(Input.Changed, "#search-input")
    def on_search(self, event: Input.Changed) -> None:
        self._filter_text = event.value
        self._schedule_filter()

    @on(Select.Changed, "#cat-filter")
    def on_cat_filter(self, event: Select.Changed) -> None:
        self._filter_category = str(event.value)
        self._apply_filters()

    @on(Select.Changed, "#proj-filter")
    def on_proj_filter(self, event: Select.Changed) -> None:
        self._filter_project = str(event.value)
        self._apply_filters()

    @on(DataTable.RowSelected, "#facts-table")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key and event.row_key.value:
            fact = self._find_fact(event.row_key.value)
            if fact:
                self._selected_fact = fact
                detail = self.query_one("#detail-pane", FactDetail)
                detail.add_class("visible")
                detail.update_fact(fact)

    def _find_fact(self, fact_id: str) -> Fact | None:
        return item_by_id(self._data.active_facts, fact_id)

    def _get_action_targets(self) -> list[str]:
        if self._selected_ids:
            return list(self._selected_ids)
        if self._selected_fact:
            return [self._selected_fact.id]
        return []

    def _cycle_sort(self, reverse: bool) -> None:
        self._sort_index, self._sort_reverse, col = next_sort_state(
            self.SORT_COLUMNS,
            self._sort_index,
            self._sort_reverse,
            reverse=reverse,
        )
        sort_by_column(self._filtered_facts, col, reverse=self._sort_reverse)
        self._populate_table()
        direction = "▼" if self._sort_reverse else "▲"
        self.app.notify(f"Sort: {col} {direction}", severity="information")

    def on_key(self, event) -> None:
        table = self.query_one("#facts-table", DataTable)
        focused = self.app.focused

        if focus_nav_from_top_row(self.app, table, event):
            return

        if event.key == "down" and isinstance(focused, (Select, Input)):
            event.prevent_default()
            table.focus()
            return

        if isinstance(focused, Select) and event.key == "left":
            event.prevent_default()
            if focused.id == "proj-filter":
                self.query_one("#cat-filter", Select).focus()
            elif focused.id == "cat-filter":
                self.query_one("#search-input", Input).focus()
            return
        if isinstance(focused, Select) and event.key == "right":
            event.prevent_default()
            if focused.id == "cat-filter":
                self.query_one("#proj-filter", Select).focus()
            return

        if isinstance(focused, Input) and event.key == "right":
            if focused.cursor_position == len(focused.value):
                event.prevent_default()
                self.query_one("#cat-filter", Select).focus()
                return

        if event.key == "slash" and not isinstance(focused, Input):
            event.prevent_default()
            self.query_one("#search-input", Input).focus()
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

    def key_escape(self) -> None:
        focused = self.app.focused
        if isinstance(focused, (Select, Input)):
            self.query_one("#facts-table", DataTable).focus()
            return
        table = self.query_one("#facts-table", DataTable)
        if focused is table:
            detail = self.query_one("#detail-pane", FactDetail)
            detail.remove_class("visible")
            self._selected_fact = None

    def key_f(self) -> None:
        if isinstance(self.app.focused, (Input, Select)):
            return
        targets = self._get_action_targets()
        if targets:
            app = cast("EngramDashboard", self.app)
            app.action_forget_facts(targets)
            self._selected_ids -= set(targets)

    def refresh_data(self, data: DashboardData) -> None:
        self._data = data
        self._apply_filters()
