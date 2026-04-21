"""Stat card widget — compact number + label + optional sparkline."""

from textual.app import ComposeResult
from textual.widgets import Label, Sparkline, Static


class StatCard(Static):
    """Compact stat display with colored value."""

    def __init__(
        self,
        label: str,
        value: int = 0,
        spark_data: list[float] | None = None,
        color: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label.upper()
        self._value = value
        self._spark_data = spark_data
        self._color = color

    def compose(self) -> ComposeResult:
        yield Label(self._label, classes="stat-label")
        val_widget = Static(f"[b]{self._value:,}[/b]", classes="stat-value")
        if self._color:
            val_widget.styles.color = self._color
        yield val_widget
        if self._spark_data and any(v > 0 for v in self._spark_data):
            yield Sparkline(
                self._spark_data, summary_function=max, classes="stat-spark"
            )

    def set_value(self, value: int) -> None:
        self._value = value
        try:
            self.query_one(".stat-value", Static).update(f"[b]{value:,}[/b]")
        except Exception:
            pass

    def update_spark(self, data: list[float]) -> None:
        self._spark_data = data
        try:
            spark = self.query_one(Sparkline)
            spark.data = data
        except Exception:
            pass
