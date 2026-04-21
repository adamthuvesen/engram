"""Overview screen — stats, category/project bars, activity sparkline."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Label, OptionList, Sparkline, Static
from textual.widgets.option_list import Option

from engram.dashboard.data import DashboardData, format_bytes
from engram.dashboard.widgets.stat_card import StatCard


class OverviewScreen(VerticalScroll):
    """Main overview with stat cards, distributions, and activity sparkline."""

    def __init__(self, data: DashboardData, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = data

    def compose(self) -> ComposeResult:
        d = self._data

        with Horizontal(id="stats-row"):
            yield StatCard("Total", d.total, d.activity_30d, id="stat-total")
            yield StatCard("Active", d.active_count, id="stat-active")
            yield StatCard("Forgotten", d.forgotten_count, id="stat-forgotten")
            yield StatCard("Expired", d.expired_count, id="stat-expired")
            yield StatCard("Pending", d.pending_count, id="stat-candidates")

        with Horizontal(id="distributions"):
            with Vertical(id="cat-panel", classes="dist-panel"):
                yield Label("Categories [dim]↑↓ Enter[/]", classes="panel-title")
                yield OptionList(
                    *self._bar_options(d.by_category, "cat", "bar-category"),
                    id="cat-list",
                )
            with Vertical(id="proj-panel", classes="dist-panel"):
                yield Label("Projects [dim]↑↓ Enter[/]", classes="panel-title")
                yield OptionList(
                    *self._bar_options(d.by_project, "proj", "bar-project"),
                    id="proj-list",
                )

        with Vertical(id="activity-panel"):
            yield Label("Activity — last 30 days", classes="panel-title")
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

    def _get_theme_color(self, var_name: str) -> str:
        try:
            variables = self.app.get_css_variables()
            return variables.get(var_name, "#d97757")
        except Exception:
            return "#d97757"

    def _bar_options(
        self, dist: dict[str, int], prefix: str, color_var: str
    ) -> list[Option]:
        color = self._get_theme_color(color_var)
        if not dist:
            return [Option("[dim](empty)[/]")]
        max_val = max(dist.values()) or 1
        options = []
        for name, count in dist.items():
            bar_len = max(1, int(24 * count / max_val))
            bar = "▇" * bar_len
            label = f"{name[:14]:<14}  [{color}]{bar}[/] {count}"
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
            try:
                from textual.widgets import Tabs

                self.app.query_one(Tabs).focus()
            except Exception:
                pass
            return

        if event.key in ("left", "right"):
            event.prevent_default()
            try:
                if focused.id == "cat-list":
                    self.query_one("#proj-list", OptionList).focus()
                else:
                    self.query_one("#cat-list", OptionList).focus()
            except Exception:
                pass

    @on(OptionList.OptionSelected, "#cat-list")
    def on_category_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id and event.option.id.startswith("cat-"):
            self.app.action_show_category(event.option.id[4:])

    @on(OptionList.OptionSelected, "#proj-list")
    def on_project_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id and event.option.id.startswith("proj-"):
            self.app.action_show_project(event.option.id[5:])

    def on_theme_changed(self) -> None:
        try:
            cat_list = self.query_one("#cat-list", OptionList)
            cat_list.clear_options()
            cat_list.add_options(
                self._bar_options(self._data.by_category, "cat", "bar-category")
            )
            proj_list = self.query_one("#proj-list", OptionList)
            proj_list.clear_options()
            proj_list.add_options(
                self._bar_options(self._data.by_project, "proj", "bar-project")
            )
        except Exception:
            pass

    def refresh_data(self, data: DashboardData) -> None:
        self._data = data
        try:
            self.query_one("#stat-total", StatCard).set_value(data.total)
            self.query_one("#stat-active", StatCard).set_value(data.active_count)
            self.query_one("#stat-forgotten", StatCard).set_value(data.forgotten_count)
            self.query_one("#stat-expired", StatCard).set_value(data.expired_count)
            self.query_one("#stat-candidates", StatCard).set_value(data.pending_count)
            self.query_one("#stat-total", StatCard).update_spark(data.activity_30d)
        except Exception:
            pass
