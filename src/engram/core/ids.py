"""The one place that makes fresh ids. Tests replace `new_id`."""

from uuid import uuid4


def new_id() -> str:
    """A fresh random id: 12 hex characters."""
    return uuid4().hex[:12]
