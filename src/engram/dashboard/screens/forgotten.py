"""Forgotten facts screen — browse and restore soft-deleted facts."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Input, Label

from engram.dashboard.constants import NO_PROJECT_LABEL, SEARCH_DEBOUNCE_S
from engram.dashboard.data import DashboardData, format_age
from engram.dashboard.widgets.fact_detail import FactDetail
from engram.models import Fact


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

        yield Label(
            "↑↓ nav  Enter detail  / search  s sort  Space select  u restore",
            classes="hint",
        )

    def on_mount(self) -> None:
        table = self.query_one("#forgotten-table", DataTable)
        table.add_columns(
            "", "ID", "Category", "Project", "Content", "Tags", "Forgotten"
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
                fact.project or NO_PROJECT_LABEL,
                fact.content[:55] + ("..." if len(fact.content) > 55 else ""),
                ", ".join(fact.tags[:3]),
                format_age(fact.updated_at),
                key=fact.id,
            )

    def _apply_filters(self) -> None:
        facts = list(self._data.forgotten_facts)
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
        self._filtered_facts = facts
        self._populate_table()

    def _schedule_filter(self) -> None:
        if self._search_timer:
            self._search_timer.stop()
        self._search_timer = self.set_timer(SEARCH_DEBOUNCE_S, self._apply_filters)

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
        for f in self._data.forgotten_facts:
            if f.id == fact_id:
                return f
        return None

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

        if focused is table and event.key == "up" and table.cursor_row == 0:
            event.prevent_default()
            try:
                from textual.widgets import Tabs

                self.app.query_one(Tabs).focus()
            except Exception:
                pass
            return

        if event.key == "down" and isinstance(focused, Input):
            event.prevent_default()
            table.focus()
            return

        if event.key == "slash" and not isinstance(focused, Input):
            event.prevent_default()
            self.query_one("#forgotten-search", Input).focus()
            return

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

        if focused is table and event.key == "s":
            event.prevent_default()
            self._cycle_sort(reverse=False)
            return
        if focused is table and event.key == "S":
            event.prevent_default()
            self._cycle_sort(reverse=True)
            return

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
