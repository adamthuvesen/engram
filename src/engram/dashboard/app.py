"""Engram terminal dashboard — main app."""

from collections.abc import Callable
from dataclasses import dataclass
import logging
from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.timer import Timer
from textual.widgets import (
    Input,
    Label,
    OptionList,
    Select,
    Static,
    TabbedContent,
    TabPane,
)
from textual.widgets.option_list import Option

from engram.dashboard.constants import REFRESH_INTERVAL, UNDO_WINDOW_S
from engram.dashboard.data import (
    DashboardData,
    content_hash_for,
    format_bytes,
    load_dashboard_data,
)
from engram.dashboard.screens.category import CategoryDetailScreen
from engram.dashboard.screens.facts import FactsScreen
from engram.dashboard.screens.help import FOOTER_HINTS, HelpScreen
from engram.dashboard.screens.overview import OverviewScreen
from engram.dashboard.screens.projects import ProjectsScreen
from engram.dashboard.screens.timeline import TimelineScreen
from engram.storage.store import FactStore

CSS_PATH = Path(__file__).parent / "styles" / "dashboard.tcss"

logger = logging.getLogger(__name__)


# Linear/Vercel-inspired palette: cool near-monochrome base, hairline borders,
# a single indigo accent. Dark is the default; `t` toggles to light.
ENGRAM_DARK = Theme(
    name="engram-dark",
    primary="#7c87f5",
    secondary="#4cb782",
    accent="#7c87f5",
    foreground="#f7f8f8",
    background="#08090a",
    surface="#101113",
    panel="#17181b",
    warning="#e0a82e",
    error="#e5484d",
    success="#4cb782",
    dark=True,
    variables={
        "card-border": "#26282d",
        "text-muted": "#8a8f98",
        "stat-active": "#4cb782",
        "stat-forgotten": "#e5484d",
        "stat-expired": "#e0a82e",
        "stat-pending": "#7c87f5",
        "bar-category": "#7c87f5",
        "bar-project": "#4cb782",
    },
)

ENGRAM_LIGHT = Theme(
    name="engram-light",
    primary="#5e6ad2",
    secondary="#2f9e6b",
    accent="#5e6ad2",
    foreground="#0a0a0a",
    background="#ffffff",
    surface="#fafafa",
    panel="#f1f1f2",
    warning="#b7791f",
    error="#d23f44",
    success="#2f9e6b",
    dark=False,
    variables={
        "card-border": "#e6e6e8",
        "text-muted": "#6b7280",
        "stat-active": "#2f9e6b",
        "stat-forgotten": "#d23f44",
        "stat-expired": "#b7791f",
        "stat-pending": "#5e6ad2",
        "bar-category": "#5e6ad2",
        "bar-project": "#2f9e6b",
    },
)


@dataclass(frozen=True)
class DashboardSection:
    id: str
    title: str
    screen: type
    focus: str | None = None
    count: Callable[[DashboardData], int] | None = None


SECTIONS = (
    DashboardSection(
        "overview", "overview", OverviewScreen, "#cat-list", lambda d: d.active_count
    ),
    DashboardSection(
        "projects",
        "projects",
        ProjectsScreen,
        "#proj-overview-table",
        lambda d: len(d.projects),
    ),
    DashboardSection(
        "facts", "facts", FactsScreen, "#facts-table", lambda d: d.active_count
    ),
    DashboardSection("timeline", "timeline", TimelineScreen),
)
SECTION_BY_ID = {section.id: section for section in SECTIONS}
SECTION_INDEX = {section.id: index for index, section in enumerate(SECTIONS)}
SECTION_IDS = frozenset(SECTION_BY_ID)


class EngramDashboard(App):
    """Engram memory dashboard."""

    TITLE = "engram"
    SUB_TITLE = "memory dashboard"
    CSS_PATH = CSS_PATH

    BINDINGS = [
        *[
            Binding(str(index), f"show_tab('{section.id}')", section.title, show=False)
            for index, section in enumerate(SECTIONS, start=1)
        ],
        Binding("t", "toggle_theme", "Theme", show=False),
        Binding("question_mark", "show_help", "Help", show=False),
        Binding("q", "quit", "Quit", show=False),
        Binding("ctrl+p", "command_palette", show=False),
    ]

    ENABLE_COMMAND_PALETTE = True

    def __init__(self) -> None:
        super().__init__()
        self.register_theme(ENGRAM_DARK)
        self.register_theme(ENGRAM_LIGHT)
        self.theme = "engram-dark"
        self._store = FactStore()
        self._data: DashboardData = load_dashboard_data(self._store)
        self._last_hash: int = self._data.content_hash
        self._undo_stack: list[str] = []
        self._undo_timer: Timer | None = None
        self._dirty_tabs: set[str] = set()
        self._pending_forget: list[str] = []

    def compose(self) -> ComposeResult:
        d = self._data
        with Vertical(id="frame"):
            yield Static(self._status_text(), id="status-bar")
            with Horizontal(id="main"):
                with Vertical(id="sidebar"):
                    yield Static("sections", id="sidebar-title")
                    yield OptionList(*self._nav_options(), id="nav")
                with TabbedContent(initial=SECTIONS[0].id, id="tabs"):
                    for section in SECTIONS:
                        with TabPane(section.title, id=section.id):
                            yield section.screen(d)
            yield Label(FOOTER_HINTS["default"], id="dynamic-footer")

    def _nav_options(self) -> list[Option]:
        d = self._data
        options = []
        for i, section in enumerate(SECTIONS, start=1):
            count = section.count(d) if section.count else None
            suffix = f"  [dim]{count}[/]" if count is not None else ""
            options.append(
                Option(f"[dim]{i}[/]  {section.title}{suffix}", id=f"nav-{section.id}")
            )
        return options

    def _status_text(self) -> str:
        d = self._data
        sep = "  [dim]·[/]  "
        return (
            f"[b]engram[/]  [$accent]memory[/]{sep}"
            f"{d.active_count:,} facts{sep}"
            f"{len(d.projects)} projects{sep}"
            f"{d.pending_count} pending{sep}"
            f"{d.forgotten_count} forgotten{sep}"
            f"{format_bytes(d.storage_bytes)}"
        )

    def _set_screen_titles(self) -> None:
        for section in SECTIONS:
            try:
                self.query_one(section.screen).border_title = section.title
            except Exception:
                logger.debug(
                    "Unable to set border title for %s", section.id, exc_info=True
                )

    def on_mount(self) -> None:
        self._set_screen_titles()
        self.set_interval(REFRESH_INTERVAL, self._refresh_data)
        self.set_timer(0.1, self.focus_nav)

    def focus_nav(self) -> None:
        try:
            self.query_one("#nav", OptionList).focus()
        except Exception:
            logger.debug("Unable to focus nav rail", exc_info=True)

    def _focus_content(self) -> None:
        active = self.query_one("#tabs", TabbedContent).active
        self._focus_tab_widget(active)

    @on(OptionList.OptionHighlighted, "#nav")
    def on_nav_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        oid = event.option_id or ""
        if oid.startswith("nav-"):
            tabs = self.query_one("#tabs", TabbedContent)
            sid = oid[4:]
            if tabs.active != sid:
                tabs.active = sid

    @on(OptionList.OptionSelected, "#nav")
    def on_nav_selected(self, event: OptionList.OptionSelected) -> None:
        self._focus_content()

    def _sync_nav_highlight(self, tab_id: str) -> None:
        try:
            nav = self.query_one("#nav", OptionList)
            idx = SECTION_INDEX[tab_id]
            if nav.highlighted != idx:
                nav.highlighted = idx
        except Exception:
            logger.debug("Unable to sync nav highlight", exc_info=True)

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        tab_id = event.pane.id or ""
        if tab_id in self._dirty_tabs:
            self._dirty_tabs.discard(tab_id)
            self._refresh_screen(tab_id)
        self._sync_nav_highlight(tab_id)
        self.set_timer(0.05, self.focus_nav)
        self._update_footer_hint()

    def on_descendant_focus(self, event) -> None:
        self._update_footer_hint()

    def _update_footer_hint(self) -> None:
        try:
            footer = self.query_one("#dynamic-footer", Label)
            footer.update(FOOTER_HINTS[self._footer_hint_key()])
        except Exception:
            logger.debug("Unable to update dashboard footer hint", exc_info=True)

    def _footer_hint_key(self) -> str:
        focused = self.focused
        if focused and focused.id and focused.id in FOOTER_HINTS:
            return focused.id
        if isinstance(focused, Input):
            return "search-input"
        if isinstance(focused, Select):
            return "filter"
        return "default"

    def on_key(self, event) -> None:
        nav = self.query_one("#nav", OptionList)
        if self.focused is nav and event.key == "right":
            event.prevent_default()
            self._focus_content()

    def _focus_tab_widget(self, tab_id: str) -> None:
        section = SECTION_BY_ID.get(tab_id)
        if section and section.focus:
            try:
                self.query_one(section.focus).focus()
            except Exception:
                logger.debug(
                    "Unable to focus widget for dashboard tab %s", tab_id, exc_info=True
                )

    def _refresh_data(self) -> None:
        current_hash = content_hash_for(self._store)
        if current_hash == self._last_hash:
            return
        self._last_hash = current_hash

        self._data = load_dashboard_data(self._store)
        self._apply_data_to_tabs()

    def _refresh_screen(self, tab_id: str) -> None:
        section = SECTION_BY_ID.get(tab_id)
        if section:
            try:
                self.query_one(section.screen).refresh_data(self._data)
            except Exception:
                logger.exception("Unable to refresh dashboard tab %s", tab_id)

    def _update_nav_labels(self) -> None:
        try:
            nav = self.query_one("#nav", OptionList)
            current = nav.highlighted
            nav.clear_options()
            nav.add_options(self._nav_options())
            if current is not None:
                nav.highlighted = current
        except Exception:
            logger.exception("Unable to update nav labels")

    def action_show_tab(self, tab_id: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab_id

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_toggle_theme(self) -> None:
        self.theme = "engram-light" if self.theme == "engram-dark" else "engram-dark"

    def action_show_category(self, category: str) -> None:
        self.push_screen(CategoryDetailScreen(category, self._data))

    def action_show_project(self, project: str) -> None:
        self.action_show_tab("projects")
        try:
            self.query_one(ProjectsScreen)._show_detail(project)
        except Exception:
            logger.exception("Unable to show project detail for %s", project)

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
        known_fact_ids = {fact.id for fact in self._data.all_facts}
        undo_entries: list[str] = []
        for fid in fact_ids:
            if fid in known_fact_ids:
                undo_entries.append(fid)
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
        restored = 0
        for fid in self._undo_stack:
            if self._store.restore_fact(fid):
                restored += 1
        self._undo_stack.clear()
        if self._undo_timer:
            self._undo_timer.stop()
        self._force_refresh()
        if restored == 0:
            self.notify("Nothing to undo", severity="information")
            return
        self.notify(
            f"Restored {restored} fact{'s' if restored > 1 else ''}",
            severity="information",
        )

    def _force_refresh(self) -> None:
        """Force a full data reload, bypassing the hash check."""
        self._data = load_dashboard_data(self._store)
        self._last_hash = self._data.content_hash
        self._apply_data_to_tabs()

    def _apply_data_to_tabs(self) -> None:
        active_tab = self.query_one("#tabs", TabbedContent).active
        self._dirty_tabs |= SECTION_IDS - {active_tab}
        self._refresh_screen(active_tab)
        self._update_nav_labels()
        try:
            self.query_one("#status-bar", Static).update(self._status_text())
        except Exception:
            logger.debug("Unable to update status bar", exc_info=True)


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
