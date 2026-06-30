"""Projects screen — list projects, drill into detail with search."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Input, Label, Static

from engram.dashboard.data import (
    DashboardData,
    ProjectHealth,
    format_age,
    get_facts_for_project,
)
from engram.dashboard.tables import (
    filter_facts_by_text,
    focus_nav_from_top_row,
    item_by_id,
    next_sort_state,
    schedule_filter_timer,
    short_cell,
)
from engram.dashboard.widgets.fact_detail import FactDetail
from engram.core.models import Fact


class ProjectDetailView(Container):
    """Drill-in view for a single project with search."""

    def __init__(
        self, project: str, health: ProjectHealth, data: DashboardData
    ) -> None:
        super().__init__()
        self._project = project
        self._health = health
        self._data = data
        self._all_facts = get_facts_for_project(data, project)
        self._filtered_facts = list(self._all_facts)
        self._filter_text: str = ""
        self._search_timer = None

    def compose(self) -> ComposeResult:
        h = self._health
        with Horizontal(id="proj-detail-header"):
            with Vertical(classes="proj-stat-box"):
                yield Label(f"[b]{self._project}[/b]")
                yield Static(
                    f"Active: [b]{h.active}[/b]  Forgotten: [b]{h.forgotten}[/b]  Expired: [b]{h.expired}[/b]"
                )
            with Vertical(classes="proj-stat-box"):
                yield Label("[b]Health[/b]")
                ratio = h.active / h.total if h.total > 0 else 0
                yield Static(f"Active ratio: [b]{ratio:.0%}[/b]")
                yield Static(f"Supersessions: [b]{h.supersession_depth}[/b]")
            with Vertical(classes="proj-stat-box"):
                yield Label("[b]Age[/b]")
                yield Static(
                    f"Oldest: [b]{format_age(h.oldest)}[/b]"
                    if h.oldest
                    else "Oldest: —"
                )
                yield Static(
                    f"Newest: [b]{format_age(h.newest)}[/b]"
                    if h.newest
                    else "Newest: —"
                )
            with Vertical(classes="proj-stat-box"):
                yield Label("[b]Categories[/b]")
                for cat, count in sorted(h.categories.items(), key=lambda x: -x[1])[:4]:
                    yield Static(f"{cat}: [b]{count}[/b]")

        with Horizontal(id="proj-search-bar", classes="filter-bar"):
            yield Input(placeholder="Search project facts...", id="proj-search-input")

        with Horizontal(id="proj-facts-body"):
            with Vertical(id="proj-facts-table-wrap"):
                yield DataTable(id="proj-detail-table", cursor_type="row")
            yield FactDetail(id="proj-fact-detail", classes="detail-pane")

    def on_mount(self) -> None:
        table = self.query_one("#proj-detail-table", DataTable)
        table.add_columns("id", "category", "content", "confidence", "tags", "created")
        self._populate_table()
        table.focus()

    def _populate_table(self) -> None:
        table = self.query_one("#proj-detail-table", DataTable)
        table.clear()
        for f in self._filtered_facts:
            table.add_row(
                f.id,
                f.category.value,
                short_cell(f.content, 55),
                f"{f.confidence:.0%}",
                ", ".join(f.tags[:3]),
                format_age(f.created_at),
                key=f.id,
            )

    def _apply_filters(self) -> None:
        facts = filter_facts_by_text(
            self._all_facts, self._filter_text, include_project=False
        )
        self._filtered_facts = facts
        self._populate_table()

    def _schedule_filter(self) -> None:
        self._search_timer = schedule_filter_timer(
            self, self._search_timer, self._apply_filters
        )

    @on(Input.Changed, "#proj-search-input")
    def on_search(self, event: Input.Changed) -> None:
        self._filter_text = event.value
        self._schedule_filter()

    @on(DataTable.RowSelected, "#proj-detail-table")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key and event.row_key.value:
            fact = self._find_fact(event.row_key.value)
            if fact:
                detail = self.query_one("#proj-fact-detail", FactDetail)
                detail.add_class("visible")
                detail.update_fact(fact)

    def _find_fact(self, fact_id: str) -> Fact | None:
        return item_by_id(self._all_facts, fact_id)

    def refresh_data(self, data: DashboardData, health: ProjectHealth) -> None:
        self._data = data
        self._health = health
        self._all_facts = get_facts_for_project(data, self._project)
        self._apply_filters()

    def on_key(self, event) -> None:
        table = self.query_one("#proj-detail-table", DataTable)
        focused = self.app.focused

        if focus_nav_from_top_row(self.app, table, event):
            return

        if event.key == "down" and isinstance(focused, Input):
            event.prevent_default()
            table.focus()
            return

        if event.key == "slash" and not isinstance(focused, Input):
            event.prevent_default()
            self.query_one("#proj-search-input", Input).focus()
            return

        if event.key == "escape":
            # Close detail pane before letting parent handle back navigation.
            try:
                detail = self.query_one("#proj-fact-detail", FactDetail)
                if detail.has_class("visible"):
                    event.prevent_default()
                    detail.remove_class("visible")
                    return
            except Exception:
                pass
            if isinstance(focused, Input):
                event.prevent_default()
                table.focus()
                return


class ProjectsScreen(Container):
    """List all projects with health indicators, drill into detail."""

    SORT_COLUMNS = ["active", "forgotten", "expired", "total", "name"]

    def __init__(self, data: DashboardData) -> None:
        super().__init__()
        self._data = data
        self._detail_view: ProjectDetailView | None = None
        self._sort_index: int = 0
        self._sort_reverse: bool = True

    def compose(self) -> ComposeResult:
        with Vertical(id="proj-list-view"):
            yield Label(
                "PROJECTS  [dim]↑↓ Enter to drill in  s sort[/]",
                classes="panel-title",
                id="proj-list-header",
            )
            yield DataTable(id="proj-overview-table", cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one("#proj-overview-table", DataTable)
        table.add_columns(
            "project",
            "active",
            "forgotten",
            "expired",
            "health",
            "supersessions",
            "age",
        )
        self._populate_table()

    def _get_sorted_projects(self) -> list[tuple[str, ProjectHealth]]:
        items = list(self._data.project_health.items())
        col = self.SORT_COLUMNS[self._sort_index]
        items.sort(
            key=lambda x: getattr(x[1], col) if hasattr(x[1], col) else x[0],
            reverse=self._sort_reverse,
        )
        return items

    def _populate_table(self) -> None:
        table = self.query_one("#proj-overview-table", DataTable)
        table.clear()
        for proj_name, health in self._get_sorted_projects():
            ratio = health.active / health.total if health.total > 0 else 0
            age = format_age(health.oldest) if health.oldest else "—"
            table.add_row(
                proj_name,
                str(health.active),
                str(health.forgotten),
                str(health.expired),
                f"{ratio:.0%}",
                str(health.supersession_depth),
                age,
                key=proj_name,
            )

    @on(DataTable.RowSelected, "#proj-overview-table")
    def on_project_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key and event.row_key.value:
            self._show_detail(event.row_key.value)

    def _show_detail(self, project: str) -> None:
        health = self._data.project_health.get(project)
        if not health:
            return
        list_view = self.query_one("#proj-list-view")
        list_view.display = False
        self._detail_view = ProjectDetailView(project, health, self._data)
        self.mount(self._detail_view)

    def _cycle_sort(self, reverse: bool) -> None:
        self._sort_index, self._sort_reverse, col = next_sort_state(
            self.SORT_COLUMNS,
            self._sort_index,
            self._sort_reverse,
            reverse=reverse,
        )
        self._populate_table()
        direction = "▼" if self._sort_reverse else "▲"
        self.app.notify(f"Sort: {col} {direction}", severity="information")

    def on_key(self, event) -> None:
        if self._detail_view and event.key == "escape":
            event.prevent_default()
            self._close_detail()
            return

        if self._detail_view:
            return

        table = self.query_one("#proj-overview-table", DataTable)
        focused = self.app.focused

        if focus_nav_from_top_row(self.app, table, event):
            return

        if focused is table and event.key == "s":
            event.prevent_default()
            self._cycle_sort(reverse=False)
            return
        if focused is table and event.key == "S":
            event.prevent_default()
            self._cycle_sort(reverse=True)
            return

    def _close_detail(self) -> None:
        if self._detail_view:
            self._detail_view.remove()
            self._detail_view = None
            self.query_one("#proj-list-view").display = True
            try:
                self.query_one("#proj-overview-table", DataTable).focus()
            except Exception:
                pass

    def key_escape(self) -> None:
        self._close_detail()

    def refresh_data(self, data: DashboardData) -> None:
        self._data = data
        if self._detail_view:
            health = self._data.project_health.get(self._detail_view._project)
            if health:
                self._detail_view.refresh_data(data, health)
            else:
                self._close_detail()
        else:
            self._populate_table()
