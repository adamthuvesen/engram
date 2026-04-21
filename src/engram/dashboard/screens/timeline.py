"""Timeline screen — temporal charts of fact activity."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Label, Sparkline, Static

from engram.dashboard.data import DashboardData

HAS_PLOTEXT = True
try:
    from textual_plotext import PlotextPlot
except ImportError:
    HAS_PLOTEXT = False


class TimelineScreen(Container):
    """Shows fact creation/activity over time."""

    def __init__(self, data: DashboardData) -> None:
        super().__init__()
        self._data = data

    def compose(self) -> ComposeResult:
        if HAS_PLOTEXT:
            yield PlotextPlot(id="timeline-chart")
        else:
            with Vertical(id="timeline-chart"):
                yield Label("ACTIVITY OVER TIME", classes="panel-title")
                yield Static("[dim]Install textual-plotext for full charts[/]")
                yield Sparkline(self._data.activity_30d or [0], summary_function=max)

        with Horizontal(id="sparkline-row"):
            with Vertical(classes="spark-panel"):
                yield Label("7-DAY ACTIVITY", classes="panel-title")
                yield Sparkline(
                    self._data.activity_7d or [0], summary_function=max, id="spark-7d"
                )
            with Vertical(classes="spark-panel"):
                yield Label("30-DAY ACTIVITY", classes="panel-title")
                yield Sparkline(
                    self._data.activity_30d or [0], summary_function=max, id="spark-30d"
                )
            with Vertical(classes="spark-panel"):
                yield Label("SUMMARY", classes="panel-title")
                total_7d = (
                    int(sum(self._data.activity_7d)) if self._data.activity_7d else 0
                )
                total_30d = (
                    int(sum(self._data.activity_30d)) if self._data.activity_30d else 0
                )
                yield Static(f"Last 7 days:  [b]{total_7d}[/b] facts created")
                yield Static(f"Last 30 days: [b]{total_30d}[/b] facts created")
                if total_7d > 0 and total_30d > 0:
                    avg_daily = total_30d / 30
                    yield Static(f"Daily avg:    [b]{avg_daily:.1f}[/b] facts/day")

    def on_mount(self) -> None:
        if HAS_PLOTEXT:
            self._draw_chart()

    def _is_dark(self) -> bool:
        try:
            return self.app.current_theme.dark
        except Exception:
            return True

    def _draw_chart(self) -> None:
        if not HAS_PLOTEXT:
            return
        try:
            plot_widget = self.query_one(PlotextPlot)
        except Exception:
            return

        plt = plot_widget.plt
        plt.clear_figure()

        if self._is_dark():
            plt.theme("dark")
            bar_color = (217, 119, 87)  # #d97757 — Anthropic coral
            forgot_color = (120, 140, 93)  # #788c5d — Anthropic green (contrast)
        else:
            plt.theme("clear")
            bar_color = (193, 95, 60)  # #c15f3c — Anthropic terra cotta
            forgot_color = (90, 112, 64)  # #5a7040 — Anthropic dark green

        plt.title("Facts Created Over Time")

        dates = sorted(self._data.daily_created.keys())
        if not dates:
            plt.title("No data yet")
            plot_widget.refresh()
            return

        created_vals = [self._data.daily_created.get(d, 0) for d in dates]
        x_indices = list(range(len(dates)))
        plt.bar(x_indices, created_vals, label="created", color=bar_color)

        if self._data.daily_forgotten:
            forgot_vals = [self._data.daily_forgotten.get(d, 0) for d in dates]
            plt.bar(x_indices, forgot_vals, label="forgotten", color=forgot_color)

        step = max(1, len(dates) // 10)
        tick_indices = x_indices[::step]
        tick_labels = [dates[i][5:] for i in tick_indices]  # MM-DD
        plt.xticks(tick_indices, tick_labels)
        plt.ylabel("count")

        plot_widget.refresh()

    def on_theme_changed(self) -> None:
        if HAS_PLOTEXT:
            self._draw_chart()

    def refresh_data(self, data: DashboardData) -> None:
        self._data = data
        try:
            self.query_one("#spark-7d", Sparkline).data = data.activity_7d or [0]
            self.query_one("#spark-30d", Sparkline).data = data.activity_30d or [0]
        except Exception:
            pass
        if HAS_PLOTEXT:
            self._draw_chart()
