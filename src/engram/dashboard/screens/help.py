"""Help modal — shows all keybindings organized by context."""

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, Static

# Keybinding registry — shared source of truth for help and dynamic footer.
# (key, context, description)
KEYBINDINGS: list[tuple[str, str, str]] = [
    # Global
    ("1-6", "Global", "Switch tabs"),
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
    # Candidates
    ("a", "Candidates", "Approve selected"),
    ("r", "Candidates", "Reject selected"),
    # Forgotten
    ("u", "Forgotten", "Unforgive / restore selected"),
]

# Compact hints for the dynamic footer, keyed by widget context.
FOOTER_HINTS: dict[str, str] = {
    "facts-table": "↑↓ nav  Enter detail  / search  s sort  Space select  f forget",
    "cand-table": "↑↓ nav  Enter detail  s sort  Space select  a approve  r reject",
    "forgotten-table": "↑↓ nav  Enter detail  s sort  Space select  u restore",
    "proj-overview-table": "↑↓ nav  Enter drill in  s sort",
    "proj-detail-table": "↑↓ nav  Enter detail  / search  Esc back",
    "cat-detail-table": "↑↓ nav  Enter detail  / search",
    "search-input": "Type to filter  ↓ or Tab to table  Esc close",
    "filter": "← → switch filters  ↓ to table",
    "default": "? help  1-6 tabs  t theme  q quit",
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
