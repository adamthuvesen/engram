"""Help modal — shows all keybindings organized by context."""

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, Static

# Keybinding registry — shared source of truth for help and dynamic footer.
# (key, context, description)
KEYBINDINGS: list[tuple[str, str, str]] = [
    # Global
    ("1-4", "Global", "Switch tabs"),
    ("t", "Global", "Toggle theme"),
    ("?", "Global", "Show help"),
    ("q", "Global", "Quit"),
    ("Ctrl+P", "Global", "Command palette"),
    # Tables
    ("↑/↓", "Tables", "Navigate rows"),
    ("Enter", "Tables", "View detail"),
    ("Esc", "Tables", "Close detail / go back"),
    ("/", "Tables", "Focus search"),
    ("s", "Tables", "Cycle sort column"),
    ("S", "Tables", "Reverse sort"),
    ("Space", "Tables", "Toggle selection"),
    ("Ctrl+A", "Tables", "Select all visible"),
    ("Ctrl+D", "Tables", "Deselect all"),
    # Facts
    ("f", "Facts", "Forget selected"),
    ("Ctrl+Z", "Facts", "Undo last forget"),
]

# Compact hints for the dynamic footer, keyed by widget context.
# Keys are accent-highlighted; groups separated by " | ", items by " · ".
_GLOBAL = "   [dim]|[/]   [$accent]t[/] theme · [$accent]?[/] help · [$accent]q[/] quit"


def _hint(items: str) -> str:
    return items + _GLOBAL


FOOTER_HINTS: dict[str, str] = {
    "facts-table": _hint(
        "[$accent]↑↓[/] nav · [$accent]↵[/] detail · [$accent]/[/] search · "
        "[$accent]s[/] sort · [$accent]space[/] select · [$accent]f[/] forget"
    ),
    "proj-overview-table": _hint(
        "[$accent]↑↓[/] nav · [$accent]↵[/] drill in · [$accent]s[/] sort"
    ),
    "proj-detail-table": _hint(
        "[$accent]↑↓[/] nav · [$accent]↵[/] detail · [$accent]/[/] search · "
        "[$accent]esc[/] back"
    ),
    "cat-detail-table": _hint(
        "[$accent]↑↓[/] nav · [$accent]↵[/] detail · [$accent]/[/] search"
    ),
    "search-input": _hint(
        "type to filter · [$accent]↓[/] or [$accent]tab[/] to table · "
        "[$accent]esc[/] close"
    ),
    "filter": _hint("[$accent]← →[/] switch filters · [$accent]↓[/] to table"),
    "default": "[$accent]1-4[/] tabs · [$accent]t[/] theme · [$accent]?[/] help · [$accent]q[/] quit",
}


class HelpScreen(ModalScreen):
    """Full keybinding reference, pushed as modal."""

    BINDINGS = [("escape", "dismiss", "Close"), ("question_mark", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-container"):
            yield Label("KEYBOARD SHORTCUTS", classes="help-title")
            with VerticalScroll():
                current_section = ""
                for key, context, desc in KEYBINDINGS:
                    if context != current_section:
                        current_section = context
                        yield Static(f"\n[b]{context}[/b]", classes="help-section")
                    yield Static(
                        f"  [bold $accent]{key:<12}[/]  {desc}", classes="help-row"
                    )
            yield Label("[dim]Press Esc or ? to close[/]", classes="help-hint")
