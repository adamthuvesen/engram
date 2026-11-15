"""The one place that reads the clock. Tests replace `now`."""

from datetime import UTC, datetime


def now() -> datetime:
    """The current time, timezone-aware in UTC."""
    return datetime.now(UTC)
