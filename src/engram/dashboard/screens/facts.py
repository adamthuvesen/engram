"""Fact browser screen — filterable, sortable table with detail pane."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Input, Label, Select

from engram.dashboard.constants import NO_PROJECT_LABEL, SEARCH_DEBOUNCE_S
from engram.dashboard.data import DashboardData, format_age
from engram.dashboard.widgets.fact_detail import FactDetail
from engram.models import Fact, FactCategory


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
        self._search_timer = None

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

        yield Label(
            "↑↓ nav  Enter detail  / search  s sort  Space select  f forget",
            classes="hint",
        )

    def on_mount(self) -> None:
        table = self.query_one("#facts-table", DataTable)
        table.add_columns(
            "", "ID", "Category", "Project", "Content", "Conf", "Tags", "Created"
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
                fact.project or NO_PROJECT_LABEL,
                fact.content[:55] + ("..." if len(fact.content) > 55 else ""),
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
        if self._filter_text:
            q = self._filter_text.lower()
            facts = [
                f
                for f in facts
                if q in f.content.lower()
                or q in " ".join(f.tags).lower()
                or q in (f.project or "").lower()
                or q in f.id
            ]

        # Apply current sort
        col = self.SORT_COLUMNS[self._sort_index]
        facts.sort(key=lambda f: getattr(f, col) or "", reverse=self._sort_reverse)

        self._filtered_facts = facts
        # Clear selection on filter change
        self._selected_ids.clear()
        self._populate_table()

    def _schedule_filter(self) -> None:
        if self._search_timer:
            self._search_timer.stop()
        self._search_timer = self.set_timer(SEARCH_DEBOUNCE_S, self._apply_filters)

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
        for f in self._data.active_facts:
            if f.id == fact_id:
                return f
        return None

    def _get_action_targets(self) -> list[str]:
        if self._selected_ids:
            return list(self._selected_ids)
        if self._selected_fact:
            return [self._selected_fact.id]
        return []

    def _cycle_sort(self, reverse: bool) -> None:
        if reverse:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_index = (self._sort_index + 1) % len(self.SORT_COLUMNS)
        col = self.SORT_COLUMNS[self._sort_index]
        self._filtered_facts.sort(
            key=lambda f: getattr(f, col) or "",
            reverse=self._sort_reverse,
        )
        self._populate_table()
        direction = "▼" if self._sort_reverse else "▲"
        self.app.notify(f"Sort: {col} {direction}", severity="information")

    def on_key(self, event) -> None:
        table = self.query_one("#facts-table", DataTable)
        focused = self.app.focused

        if focused is table and event.key == "up" and table.cursor_row == 0:
            event.prevent_default()
            try:
                from textual.widgets import Tabs

                self.app.query_one(Tabs).focus()
            except Exception:
                pass
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

        # Space toggles selection
        if focused is table and event.key == "space":
            event.prevent_default()
            row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
            if row_key.value:
                fid = row_key.value
                if fid in self._selected_ids:
                    self._selected_ids.discard(fid)
                else:
                    self._selected_ids.add(fid)
                self._populate_table()
            return

        if focused is table and event.key == "ctrl+a":
            event.prevent_default()
            self._selected_ids = {f.id for f in self._filtered_facts}
            self._populate_table()
            return

        if focused is table and event.key == "ctrl+d":
            event.prevent_default()
            self._selected_ids.clear()
            self._populate_table()
            return

        # Sort
        if focused is table and event.key == "s":
            event.prevent_default()
            self._cycle_sort(reverse=False)
            return
        if focused is table and event.key == "S":
            event.prevent_default()
            self._cycle_sort(reverse=True)
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
            self.app.action_forget_facts(targets)
            self._selected_ids -= set(targets)

    def refresh_data(self, data: DashboardData) -> None:
        self._data = data
        self._apply_filters()
