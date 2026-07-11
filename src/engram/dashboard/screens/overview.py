"""Overview screen — stats, category/project bars, activity sparkline."""

import logging
from typing import TYPE_CHECKING, cast

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Label, OptionList, Sparkline, Static
from textual.widgets.option_list import Option

from engram.dashboard.data import DashboardData, format_bytes

if TYPE_CHECKING:
    from engram.dashboard.app import EngramDashboard

logger = logging.getLogger(__name__)


class OverviewScreen(VerticalScroll):
    """Main overview with stat cards, distributions, and activity sparkline."""

    def __init__(self, data: DashboardData, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def compose(self) -> ComposeResult:
        d = self._data

        yield Static(self._stat_strip_text(), id="stat-strip")

        with Horizontal(id="distributions"):
            with Vertical(id="cat-panel", classes="dist-panel"):
                yield Label("categories [dim]↑↓ enter[/]", classes="panel-title")
                yield OptionList(
                    *self._bar_options(d.by_category, "cat", "bar-category"),
                    id="cat-list",
                )
            with Vertical(id="proj-panel", classes="dist-panel"):
                yield Label("projects [dim]↑↓ enter[/]", classes="panel-title")
                yield OptionList(
                    *self._bar_options(d.by_project, "proj", "bar-project"),
                    id="proj-list",
                )

        with Vertical(id="activity-panel"):
            yield Label("activity [dim]· last 30 days[/]", classes="panel-title")
            spark_data = (
                d.activity_30d if any(v > 0 for v in d.activity_30d) else [0, 0.1]
            )
            yield Sparkline(spark_data, summary_function=max)
            total_30d = int(sum(d.activity_30d))
            total_7d = int(sum(d.activity_7d))
            avg = total_30d / 30 if total_30d else 0
            yield Static(
                f"[b]{total_30d}[/b] facts (30d)  ·  "
                f"[b]{total_7d}[/b] facts (7d)  ·  "
                f"[b]{avg:.1f}[/b]/day avg  ·  "
                f"{format_bytes(d.storage_bytes)}",
                classes="activity-stats",
            )

    def _stat_strip_text(self) -> str:
        d = self._data
        sep = "   [dim]·[/]   "
        return (
            f"[dim]total[/] [b]{d.total:,}[/]{sep}"
            f"[dim]active[/] [b $stat-active]{d.active_count:,}[/]{sep}"
            f"[dim]forgotten[/] [b $stat-forgotten]{d.forgotten_count}[/]{sep}"
            f"[dim]expired[/] [b $stat-expired]{d.expired_count}[/]{sep}"
            f"[dim]pending[/] [b $stat-pending]{d.pending_count}[/]"
        )

    def _get_theme_color(self, var_name: str) -> str:
        try:
            variables = self.app.get_css_variables()
            return variables.get(var_name, "#7c87f5")
        except Exception:
            logger.debug(
                "Unable to read dashboard theme color %s", var_name, exc_info=True
            )
            return "#7c87f5"

    def _bar_options(
        self, dist: dict[str, int], prefix: str, color_var: str
    ) -> list[Option]:
        color = self._get_theme_color(color_var)
        if not dist:
            return [Option("[dim](empty)[/]")]
        max_val = max(dist.values()) or 1
        options = []
        for name, count in dist.items():
            bar_len = max(1, int(14 * count / max_val))
            bar = "▇" * bar_len
            label = f"{name[:13]:<13} [{color}]{bar}[/] {count}"
            options.append(Option(label, id=f"{prefix}-{name}"))
        return options

    def on_key(self, event) -> None:
        focused = self.app.focused
        if not isinstance(focused, OptionList):
            return

        if event.key == "up" and (
            focused.highlighted is None or focused.highlighted == 0
        ):
            event.prevent_default()
            app = cast("EngramDashboard", self.app)
            app.focus_nav()
            return

        if event.key in ("left", "right"):
            event.prevent_default()
            try:
                if focused.id == "cat-list":
                    self.query_one("#proj-list", OptionList).focus()
                else:
                    self.query_one("#cat-list", OptionList).focus()
            except Exception:
                logger.debug("Unable to switch overview list focus", exc_info=True)

    @on(OptionList.OptionSelected, "#cat-list")
    def on_category_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id and event.option.id.startswith("cat-"):
            app = cast("EngramDashboard", self.app)
            app.action_show_category(event.option.id[4:])

    @on(OptionList.OptionSelected, "#proj-list")
    def on_project_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id and event.option.id.startswith("proj-"):
            app = cast("EngramDashboard", self.app)
            app.action_show_project(event.option.id[5:])

    def refresh_data(self, data: DashboardData) -> None:
        self._data = data
        try:
            self.query_one("#stat-strip", Static).update(self._stat_strip_text())
        except Exception:
            logger.exception("Unable to refresh overview stat strip")
