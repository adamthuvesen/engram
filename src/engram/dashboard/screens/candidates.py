"""Candidate review screen — approve/reject with multi-select and sorting."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Label, Select

from engram.dashboard.constants import NO_PROJECT_LABEL
from engram.dashboard.data import DashboardData
from engram.dashboard.widgets.fact_detail import FactDetail
from engram.models import CandidateStatus, MemoryCandidate


class CandidatesScreen(Container):
    """Review and manage memory candidates."""

    SORT_COLUMNS = ["created_at", "status", "category", "project", "content"]

    def __init__(self, data: DashboardData) -> None:
        super().__init__()
        self._data = data
        self._filter_status: str = "pending"
        self._filtered: list[MemoryCandidate] = list(data.pending_candidates)
        self._selected: MemoryCandidate | None = None
        self._selected_ids: set[str] = set()
        self._sort_index: int = 0
        self._sort_reverse: bool = True

    def compose(self) -> ComposeResult:
        with Horizontal(id="cand-header", classes="filter-bar"):
            yield Select(
                [
                    ("Pending", "pending"),
                    ("Approved", "approved"),
                    ("Rejected", "rejected"),
                    ("All", "all"),
                ],
                value="pending",
                id="cand-status-filter",
                allow_blank=False,
            )
            pending = len(self._data.pending_candidates)
            total = len(self._data.candidates)
            yield Label(
                f"{pending} pending / {total} total",
                classes="cand-stats",
            )

        with Horizontal(id="cand-body", classes="screen-body"):
            with Vertical(id="cand-table-container", classes="table-container"):
                yield DataTable(id="cand-table", cursor_type="row")
            yield FactDetail(id="cand-detail", classes="detail-pane")

        yield Label(
            "↑↓ nav  Enter detail  s sort  Space select  a approve  r reject",
            classes="hint",
        )

    def on_mount(self) -> None:
        table = self.query_one("#cand-table", DataTable)
        table.add_columns(
            "", "ID", "Status", "Category", "Project", "Content", "Why Store"
        )
        self._populate_table()

    def _populate_table(self) -> None:
        table = self.query_one("#cand-table", DataTable)
        table.clear()
        for c in self._filtered:
            sel = "✓" if c.id in self._selected_ids else " "
            status_icon = {"pending": "⏳", "approved": "✅", "rejected": "❌"}.get(
                c.status.value, "?"
            )
            table.add_row(
                sel,
                c.id,
                f"{status_icon} {c.status.value}",
                c.category.value,
                c.project or NO_PROJECT_LABEL,
                c.content[:45] + ("..." if len(c.content) > 45 else ""),
                c.why_store[:25] + ("..." if len(c.why_store) > 25 else ""),
                key=c.id,
            )

    def _apply_filter(self) -> None:
        if self._filter_status == "all":
            self._filtered = list(self._data.candidates)
        else:
            target = CandidateStatus(self._filter_status)
            self._filtered = [c for c in self._data.candidates if c.status == target]

        col = self.SORT_COLUMNS[self._sort_index]
        self._filtered.sort(
            key=lambda c: str(getattr(c, col) or ""),
            reverse=self._sort_reverse,
        )
        self._selected_ids.clear()
        self._populate_table()

    @on(Select.Changed, "#cand-status-filter")
    def on_status_filter(self, event: Select.Changed) -> None:
        self._filter_status = str(event.value)
        self._apply_filter()

    @on(DataTable.RowSelected, "#cand-table")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key and event.row_key.value:
            candidate = self._find_candidate(event.row_key.value)
            if candidate:
                self._selected = candidate
                detail = self.query_one("#cand-detail", FactDetail)
                detail.add_class("visible")
                detail.update_fact(candidate)

    def _find_candidate(self, cid: str) -> MemoryCandidate | None:
        for c in self._data.candidates:
            if c.id == cid:
                return c
        return None

    def _get_pending_targets(self) -> list[str]:
        if self._selected_ids:
            return [
                cid
                for cid in self._selected_ids
                if any(
                    c.id == cid and c.status == CandidateStatus.pending
                    for c in self._data.candidates
                )
            ]
        if self._selected and self._selected.status == CandidateStatus.pending:
            return [self._selected.id]
        return []

    def _cycle_sort(self, reverse: bool) -> None:
        if reverse:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_index = (self._sort_index + 1) % len(self.SORT_COLUMNS)
        col = self.SORT_COLUMNS[self._sort_index]
        self._filtered.sort(
            key=lambda c: str(getattr(c, col) or ""),
            reverse=self._sort_reverse,
        )
        self._populate_table()
        direction = "▼" if self._sort_reverse else "▲"
        self.app.notify(f"Sort: {col} {direction}", severity="information")

    def on_key(self, event) -> None:
        table = self.query_one("#cand-table", DataTable)
        focused = self.app.focused

        if focused is table and event.key == "up" and table.cursor_row == 0:
            event.prevent_default()
            try:
                from textual.widgets import Tabs

                self.app.query_one(Tabs).focus()
            except Exception:
                pass
            return

        if isinstance(focused, Select) and event.key == "down":
            event.prevent_default()
            table.focus()
            return

        if focused is table and event.key == "space":
            event.prevent_default()
            row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
            if row_key.value:
                cid = row_key.value
                if cid in self._selected_ids:
                    self._selected_ids.discard(cid)
                else:
                    self._selected_ids.add(cid)
                self._populate_table()
            return

        if focused is table and event.key == "ctrl+a":
            event.prevent_default()
            self._selected_ids = {c.id for c in self._filtered}
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

    def key_escape(self) -> None:
        focused = self.app.focused
        if isinstance(focused, Select):
            self.query_one("#cand-table", DataTable).focus()
            return
        detail = self.query_one("#cand-detail", FactDetail)
        detail.remove_class("visible")
        self._selected = None

    def key_a(self) -> None:
        if isinstance(self.app.focused, Select):
            return
        targets = self._get_pending_targets()
        if targets:
            self.app.action_approve_candidates(targets)
            self._selected_ids -= set(targets)

    def key_r(self) -> None:
        if isinstance(self.app.focused, Select):
            return
        targets = self._get_pending_targets()
        if targets:
            self.app.action_reject_candidates(targets)
            self._selected_ids -= set(targets)

    def refresh_data(self, data: DashboardData) -> None:
        self._data = data
        self._apply_filter()
