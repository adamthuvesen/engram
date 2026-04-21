"""Engram terminal dashboard — main app."""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.timer import Timer
from textual.widgets import Footer, Header, Label, TabbedContent, TabPane

from engram.dashboard.constants import REFRESH_INTERVAL, UNDO_WINDOW_S
from engram.dashboard.data import DashboardData, content_hash_for, load_dashboard_data
from engram.dashboard.screens.candidates import CandidatesScreen
from engram.dashboard.screens.category import CategoryDetailScreen
from engram.dashboard.screens.facts import FactsScreen
from engram.dashboard.screens.forgotten import ForgottenScreen
from engram.dashboard.screens.help import FOOTER_HINTS, HelpScreen
from engram.dashboard.screens.overview import OverviewScreen
from engram.dashboard.screens.projects import ProjectsScreen
from engram.dashboard.screens.timeline import TimelineScreen
from engram.store import FactStore

CSS_PATH = Path(__file__).parent / "styles" / "dashboard.tcss"

# ── Anthropic brand palettes ──

ANTHROPIC_DARK = Theme(
    name="anthropic-dark",
    primary="#d97757",
    secondary="#6a9bcc",
    accent="#d97757",
    foreground="#f5f4ed",
    background="#1a1816",
    surface="#2a2520",
    panel="#3d3630",
    warning="#eda100",
    error="#f7768e",
    success="#788c5d",
    dark=True,
    variables={
        "stat-active": "#788c5d",
        "stat-forgotten": "#f7768e",
        "stat-expired": "#eda100",
        "stat-pending": "#6a9bcc",
        "bar-category": "#d97757",
        "bar-project": "#788c5d",
    },
)

ANTHROPIC_LIGHT = Theme(
    name="anthropic-light",
    primary="#c15f3c",
    secondary="#4a7aaa",
    accent="#c15f3c",
    foreground="#141413",
    background="#f5f4ed",
    surface="#ffffff",
    panel="#e8e6dc",
    warning="#b88600",
    error="#c15050",
    success="#5a7040",
    dark=False,
    variables={
        "stat-active": "#5a7040",
        "stat-forgotten": "#c15050",
        "stat-expired": "#b88600",
        "stat-pending": "#4a7aaa",
        "bar-category": "#c15f3c",
        "bar-project": "#5a7040",
    },
)


class EngramDashboard(App):
    """Engram memory dashboard."""

    TITLE = "engram"
    SUB_TITLE = "memory dashboard"
    CSS_PATH = CSS_PATH

    BINDINGS = [
        Binding("1", "show_tab('overview')", "Overview", show=False),
        Binding("2", "show_tab('timeline')", "Timeline", show=False),
        Binding("3", "show_tab('facts')", "Facts", show=False),
        Binding("4", "show_tab('candidates')", "Candidates", show=False),
        Binding("5", "show_tab('projects')", "Projects", show=False),
        Binding("6", "show_tab('forgotten')", "Forgotten", show=False),
        Binding("t", "toggle_theme", "Theme", show=False),
        Binding("question_mark", "show_help", "Help", show=False),
        Binding("q", "quit", "Quit", show=False),
        Binding("ctrl+p", "command_palette", show=False),
    ]

    ENABLE_COMMAND_PALETTE = True

    def __init__(self) -> None:
        super().__init__()
        self.register_theme(ANTHROPIC_DARK)
        self.register_theme(ANTHROPIC_LIGHT)
        self.theme = "anthropic-dark"
        self._store = FactStore()
        self._data: DashboardData = load_dashboard_data(self._store)
        self._last_hash: int = self._data.content_hash
        self._undo_stack: list[tuple[str, float]] = []  # (fact_id, old_confidence)
        self._undo_timer: Timer | None = None
        self._dirty_tabs: set[str] = set()
        self._pending_forget: list[str] = []
        self._pending_reject: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        d = self._data
        with TabbedContent(initial="overview", id="tabs"):
            with TabPane(f"Overview ({d.active_count})", id="overview"):
                yield OverviewScreen(d)
            with TabPane("Timeline", id="timeline"):
                yield TimelineScreen(d)
            with TabPane(f"Facts ({d.active_count})", id="facts"):
                yield FactsScreen(d)
            with TabPane(f"Candidates ({d.pending_count})", id="candidates"):
                yield CandidatesScreen(d)
            with TabPane(f"Projects ({len(d.projects)})", id="projects"):
                yield ProjectsScreen(d)
            with TabPane(f"Forgotten ({d.forgotten_count})", id="forgotten"):
                yield ForgottenScreen(d)
        yield Label("? help  1-6 tabs  t theme  q quit", id="dynamic-footer")
        yield Footer()

    TAB_FOCUS = {
        "overview": "#cat-list",
        "timeline": None,
        "facts": "#facts-table",
        "candidates": "#cand-table",
        "projects": "#proj-overview-table",
        "forgotten": "#forgotten-table",
    }

    def on_mount(self) -> None:
        self.set_interval(REFRESH_INTERVAL, self._refresh_data)
        self.set_timer(0.1, self._focus_tabs)

    def _focus_tabs(self) -> None:
        try:
            from textual.widgets import Tabs

            self.query_one(Tabs).focus()
        except Exception:
            pass

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        tab_id = event.pane.id or ""
        # Refresh dirty tabs when they become active
        if tab_id in self._dirty_tabs:
            self._dirty_tabs.discard(tab_id)
            self._refresh_screen(tab_id)
        self.set_timer(0.05, self._focus_tabs)
        # Update dynamic footer
        self._update_footer_hint()

    def on_descendant_focus(self, event) -> None:
        self._update_footer_hint()

    def _update_footer_hint(self) -> None:
        try:
            footer = self.query_one("#dynamic-footer", Label)
            focused = self.focused
            if focused and focused.id and focused.id in FOOTER_HINTS:
                footer.update(FOOTER_HINTS[focused.id])
            elif focused:
                # Check by widget type
                from textual.widgets import Input, Select

                if isinstance(focused, Input):
                    footer.update(FOOTER_HINTS["search-input"])
                elif isinstance(focused, Select):
                    footer.update(FOOTER_HINTS["filter"])
                else:
                    footer.update(FOOTER_HINTS["default"])
            else:
                footer.update(FOOTER_HINTS["default"])
        except Exception:
            pass

    def on_key(self, event) -> None:
        from textual.widgets import Tabs

        if isinstance(self.focused, Tabs) and event.key == "down":
            event.prevent_default()
            active = self.query_one("#tabs", TabbedContent).active
            self._focus_tab_widget(active)

    def _focus_tab_widget(self, tab_id: str) -> None:
        selector = self.TAB_FOCUS.get(tab_id)
        if selector:
            try:
                self.query_one(selector).focus()
            except Exception:
                pass

    def _refresh_data(self) -> None:
        # Smart refresh: skip if data hasn't changed
        current_hash = content_hash_for(self._store)
        if current_hash == self._last_hash:
            return
        self._last_hash = current_hash

        self._data = load_dashboard_data(self._store)

        # Only refresh the active tab; mark others dirty
        active_tab = self.query_one("#tabs", TabbedContent).active
        all_tabs = {
            "overview",
            "timeline",
            "facts",
            "candidates",
            "projects",
            "forgotten",
        }
        self._dirty_tabs |= all_tabs - {active_tab}
        self._refresh_screen(active_tab)
        self._update_tab_labels()
        self.sub_title = f"memory dashboard — {self._data.active_count} facts"

    def _refresh_screen(self, tab_id: str) -> None:
        screen_map = {
            "overview": OverviewScreen,
            "timeline": TimelineScreen,
            "facts": FactsScreen,
            "candidates": CandidatesScreen,
            "projects": ProjectsScreen,
            "forgotten": ForgottenScreen,
        }
        cls = screen_map.get(tab_id)
        if cls:
            try:
                self.query_one(cls).refresh_data(self._data)
            except Exception:
                pass

    def _update_tab_labels(self) -> None:
        d = self._data
        labels = {
            "overview": f"Overview ({d.active_count})",
            "facts": f"Facts ({d.active_count})",
            "candidates": f"Candidates ({d.pending_count})",
            "projects": f"Projects ({len(d.projects)})",
            "forgotten": f"Forgotten ({d.forgotten_count})",
        }
        try:
            tabs_widget = self.query_one("#tabs", TabbedContent)
            for tab_id, label in labels.items():
                tab = tabs_widget.get_tab(f"--content-tab-{tab_id}")
                if tab:
                    tab.label = label
        except Exception:
            pass

    def action_show_tab(self, tab_id: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab_id

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_toggle_theme(self) -> None:
        if self.theme == "anthropic-dark":
            self.theme = "anthropic-light"
        else:
            self.theme = "anthropic-dark"

    def action_show_category(self, category: str) -> None:
        self.push_screen(CategoryDetailScreen(category, self._data))

    def action_show_project(self, project: str) -> None:
        self.action_show_tab("projects")
        try:
            self.query_one(ProjectsScreen)._show_detail(project)
        except Exception:
            pass

    # ── Fact actions with confirmation ──

    def action_forget_fact(self, fact_id: str) -> None:
        self._forget_facts([fact_id])

    def action_forget_facts(self, fact_ids: list[str]) -> None:
        self._forget_facts(fact_ids)

    def _forget_facts(self, fact_ids: list[str]) -> None:
        count = len(fact_ids)
        self._pending_forget = fact_ids
        self.push_screen(
            ConfirmScreen(f"Forget {count} fact{'s' if count > 1 else ''}?"),
            callback=self._on_forget_confirmed,
        )

    def _on_forget_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            self._pending_forget = []
            return
        fact_ids = self._pending_forget
        self._pending_forget = []
        undo_entries: list[tuple[str, float]] = []
        for fid in fact_ids:
            # Find original confidence before forgetting
            for f in self._data.all_facts:
                if f.id == fid:
                    undo_entries.append((fid, f.confidence))
                    break
            self._store.forget(fid, reason="Forgotten via dashboard")
        self._undo_stack = undo_entries
        self._schedule_undo_expiry()
        self._force_refresh()
        count = len(fact_ids)
        self.notify(
            f"Forgot {count} fact{'s' if count > 1 else ''}. Ctrl+Z to undo ({int(UNDO_WINDOW_S)}s)",
            severity="warning",
        )

    def _schedule_undo_expiry(self) -> None:
        if self._undo_timer:
            self._undo_timer.stop()
        self._undo_timer = self.set_timer(UNDO_WINDOW_S, self._expire_undo)

    def _expire_undo(self) -> None:
        self._undo_stack.clear()

    def key_ctrl_z(self) -> None:
        if not self._undo_stack:
            self.notify("Nothing to undo", severity="information")
            return
        count = 0
        for fid, old_confidence in self._undo_stack:
            self._store.update_fact(fid, confidence=old_confidence)
            count += 1
        self._undo_stack.clear()
        if self._undo_timer:
            self._undo_timer.stop()
        self._force_refresh()
        self.notify(
            f"Restored {count} fact{'s' if count > 1 else ''}", severity="information"
        )

    def action_approve_candidate(self, candidate_id: str) -> None:
        self.action_approve_candidates([candidate_id])

    def action_approve_candidates(self, candidate_ids: list[str]) -> None:
        self._store.approve_candidates(candidate_ids)
        self._force_refresh()
        count = len(candidate_ids)
        self.notify(
            f"Approved {count} candidate{'s' if count > 1 else ''}",
            severity="information",
        )

    def action_reject_candidate(self, candidate_id: str) -> None:
        self.action_reject_candidates([candidate_id])

    def action_reject_candidates(self, candidate_ids: list[str]) -> None:
        count = len(candidate_ids)
        if count > 1:
            self._pending_reject = candidate_ids
            self.push_screen(
                ConfirmScreen(f"Reject {count} candidates?"),
                callback=self._on_reject_confirmed,
            )
        else:
            self._do_reject(candidate_ids)

    def _on_reject_confirmed(self, confirmed: bool) -> None:
        if confirmed:
            self._do_reject(self._pending_reject)
        self._pending_reject = []

    def _do_reject(self, candidate_ids: list[str]) -> None:
        self._store.reject_candidates(candidate_ids, reason="Rejected via dashboard")
        self._force_refresh()
        count = len(candidate_ids)
        self.notify(
            f"Rejected {count} candidate{'s' if count > 1 else ''}", severity="warning"
        )

    def action_restore_facts(self, fact_ids: list[str]) -> None:
        for fid in fact_ids:
            self._store.update_fact(fid, confidence=1.0)
        self._force_refresh()
        count = len(fact_ids)
        self.notify(
            f"Restored {count} fact{'s' if count > 1 else ''}", severity="information"
        )

    def _force_refresh(self) -> None:
        """Force a full data reload, bypassing the hash check."""
        self._data = load_dashboard_data(self._store)
        self._last_hash = self._data.content_hash
        active_tab = self.query_one("#tabs", TabbedContent).active
        all_tabs = {
            "overview",
            "timeline",
            "facts",
            "candidates",
            "projects",
            "forgotten",
        }
        self._dirty_tabs |= all_tabs - {active_tab}
        self._refresh_screen(active_tab)
        self._update_tab_labels()
        self.sub_title = f"memory dashboard — {self._data.active_count} facts"


class ConfirmScreen(ModalScreen[bool]):
    """Simple yes/no confirmation dialog."""

    BINDINGS = [
        ("y", "confirm", "Yes"),
        ("n", "cancel", "No"),
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical

        with Vertical(id="confirm-box"):
            yield Label(self._message, classes="confirm-msg")
            yield Label("[b]y[/b] yes  /  [b]n[/b] no", classes="confirm-hint")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


def main() -> None:
    app = EngramDashboard()
    app.run()


if __name__ == "__main__":
    main()
