"""Category detail screen — pushed from overview drill-down, with search."""

from collections import Counter

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Static

from engram.dashboard.constants import NO_PROJECT_LABEL
from engram.dashboard.data import (
    DashboardData,
    format_age,
    get_facts_for_category,
    shorten_project,
)
from engram.dashboard.tables import (
    filter_facts_by_text,
    item_by_id,
    schedule_filter_timer,
    short_cell,
)
from engram.dashboard.widgets.fact_detail import FactDetail
from engram.core.models import Fact


class CategoryDetailScreen(ModalScreen):
    """Detail view for a single category, pushed as a modal."""

    BINDINGS = [("escape", "dismiss", "Back")]

    def __init__(self, category: str, data: DashboardData) -> None:
        super().__init__()
        self._category = category
        self._data = data
        self._all_facts = get_facts_for_category(data, category)
        self._filtered_facts = list(self._all_facts)
        self._filter_text: str = ""
        self._search_timer = None

    def compose(self) -> ComposeResult:
        container = Vertical(id="cat-detail-container")
        container.border_title = f"category › {self._category}"
        with container:
            with Horizontal(id="cat-stats-row"):
                with Vertical(classes="cat-stat"):
                    yield Static(f"Facts: [b]{len(self._all_facts)}[/b]")
                with Vertical(classes="cat-stat"):
                    projects = Counter(
                        f.project or NO_PROJECT_LABEL for f in self._all_facts
                    )
                    top = projects.most_common(3)
                    yield Static(
                        "Top projects: "
                        + ", ".join(f"[b]{p}[/b] ({c})" for p, c in top)
                        if top
                        else "No projects"
                    )
                with Vertical(classes="cat-stat"):
                    if self._all_facts:
                        newest = max(self._all_facts, key=lambda f: f.created_at)
                        yield Static(
                            f"Newest: [b]{format_age(newest.created_at)} ago[/b]"
                        )
                    else:
                        yield Static("Newest: —")

            yield Input(placeholder="Search category facts...", id="cat-search-input")

            with Horizontal(id="cat-facts-body"):
                with Vertical(id="cat-facts-table-wrap"):
                    yield DataTable(id="cat-detail-table", cursor_type="row")
                yield FactDetail(id="cat-fact-detail", classes="detail-pane")

    def on_mount(self) -> None:
        table = self.query_one("#cat-detail-table", DataTable)
        table.add_columns("id", "project", "content", "confidence", "tags", "created")
        self._populate_table()
        table.focus()

    def _populate_table(self) -> None:
        table = self.query_one("#cat-detail-table", DataTable)
        table.clear()
        for f in self._filtered_facts:
            table.add_row(
                f.id,
                shorten_project(f.project),
                short_cell(f.content, 55),
                f"{f.confidence:.0%}",
                ", ".join(f.tags[:3]),
                format_age(f.created_at),
                key=f.id,
            )

    def _apply_filters(self) -> None:
        facts = filter_facts_by_text(self._all_facts, self._filter_text)
        self._filtered_facts = facts
        self._populate_table()

    def _schedule_filter(self) -> None:
        self._search_timer = schedule_filter_timer(
            self, self._search_timer, self._apply_filters
        )

    @on(Input.Changed, "#cat-search-input")
    def on_search(self, event: Input.Changed) -> None:
        self._filter_text = event.value
        self._schedule_filter()

    @on(DataTable.RowSelected, "#cat-detail-table")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key and event.row_key.value:
            fact = self._find_fact(event.row_key.value)
            if fact:
                detail = self.query_one("#cat-fact-detail", FactDetail)
                detail.add_class("visible")
                detail.update_fact(fact)

    def _find_fact(self, fact_id: str) -> Fact | None:
        return item_by_id(self._all_facts, fact_id)

    def on_key(self, event) -> None:
        focused = self.app.focused
        if event.key == "slash" and not isinstance(focused, Input):
            event.prevent_default()
            self.query_one("#cat-search-input", Input).focus()
            return
        if isinstance(focused, Input) and event.key == "down":
            event.prevent_default()
            self.query_one("#cat-detail-table", DataTable).focus()
            return
