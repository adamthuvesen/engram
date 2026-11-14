"""Timeline screen — temporal charts of fact activity."""

import logging

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Label, Sparkline, Static
from textual_plotext import PlotextPlot

from engram.dashboard.data import DashboardData

logger = logging.getLogger(__name__)


class TimelineScreen(Container):
    def __init__(self, data: DashboardData) -> None:
        super().__init__()
        self._data = data

    def compose(self) -> ComposeResult:
        yield PlotextPlot(id="timeline-chart")

        with Horizontal(id="sparkline-row"):
            with Vertical(classes="spark-panel"):
                yield Label("7-DAY ACTIVITY", classes="panel-title")
                yield Sparkline(self._data.activity_7d or [0], summary_function=max, id="spark-7d")
            with Vertical(classes="spark-panel"):
                yield Label("30-DAY ACTIVITY", classes="panel-title")
                yield Sparkline(
                    self._data.activity_30d or [0], summary_function=max, id="spark-30d"
                )
            with Vertical(classes="spark-panel"):
                yield Label("SUMMARY", classes="panel-title")
                total_7d = int(sum(self._data.activity_7d)) if self._data.activity_7d else 0
                total_30d = int(sum(self._data.activity_30d)) if self._data.activity_30d else 0
                yield Static(f"Last 7 days:  [b]{total_7d}[/b] facts created")
                yield Static(f"Last 30 days: [b]{total_30d}[/b] facts created")
                if total_7d > 0 and total_30d > 0:
                    avg_daily = total_30d / 30
                    yield Static(f"Daily avg:    [b]{avg_daily:.1f}[/b] facts/day")

    def on_mount(self) -> None:
        self._draw_chart()

    def _is_dark(self) -> bool:
        try:
            return self.app.current_theme.dark
        except Exception:
            logger.debug("Unable to read dashboard theme brightness", exc_info=True)
            return True

    def _draw_chart(self) -> None:
        try:
            plot_widget = self.query_one(PlotextPlot)
        except Exception:
            logger.debug("Unable to find timeline chart widget", exc_info=True)
            return

        plt = plot_widget.plt
        plt.clear_figure()

        if self._is_dark():
            plt.theme("dark")
            bar_color = (124, 135, 245)  # #7c87f5 — indigo accent
            forgot_color = (76, 183, 130)  # #4cb782 — green (contrast)
        else:
            plt.theme("clear")
            bar_color = (94, 106, 210)  # #5e6ad2 — indigo accent
            forgot_color = (47, 158, 107)  # #2f9e6b — green (contrast)

        plt.title("Facts Created Over Time")

        dates = sorted(self._data.daily_created.keys())
        if not dates:
            plt.title("No data yet")
            plot_widget.refresh()
            return

        created_vals = [self._data.daily_created.get(d, 0) for d in dates]
        x_indices = [float(index) for index in range(len(dates))]
        plt.bar(x_indices, created_vals, label="created", color=bar_color)

        if self._data.daily_forgotten:
            forgot_vals = [self._data.daily_forgotten.get(d, 0) for d in dates]
            plt.bar(x_indices, forgot_vals, label="forgotten", color=forgot_color)

        step = max(1, len(dates) // 10)
        tick_indices = x_indices[::step]
        tick_labels = [dates[int(i)][5:] for i in tick_indices]  # MM-DD
        plt.xticks(tick_indices, tick_labels)
        plt.ylabel("count")

        plot_widget.refresh()

    def refresh_data(self, data: DashboardData) -> None:
        self._data = data
        try:
            self.query_one("#spark-7d", Sparkline).data = data.activity_7d or [0]
            self.query_one("#spark-30d", Sparkline).data = data.activity_30d or [0]
        except Exception:
            logger.exception("Unable to refresh timeline sparklines")
        self._draw_chart()
