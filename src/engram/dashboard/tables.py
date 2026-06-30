"""Shared table behavior for the Textual dashboard."""

from collections.abc import Callable, Iterable
from typing import Protocol, TypeVar

from textual.timer import Timer
from textual.widgets import DataTable

from engram.dashboard.constants import SEARCH_DEBOUNCE_S
from engram.core.models import Fact


class HasId(Protocol):
    id: str


T = TypeVar("T")
ItemT = TypeVar("ItemT", bound=HasId)


def short_cell(value: str, limit: int) -> str:
    """Keep wide text columns readable without changing detail-pane content."""
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def item_by_id(items: Iterable[ItemT], item_id: str) -> ItemT | None:
    for item in items:
        if item.id == item_id:
            return item
    return None


def filter_facts_by_text(
    facts: Iterable[Fact], query: str, *, include_project: bool = True
) -> list[Fact]:
    q = query.lower()
    if not q:
        return list(facts)
    return [
        fact
        for fact in facts
        if q in fact.content.lower()
        or q in " ".join(fact.tags).lower()
        or q in fact.id
        or (include_project and q in (fact.project or "").lower())
    ]


def schedule_filter_timer(
    owner, timer: Timer | None, callback: Callable[[], None]
) -> Timer:
    if timer:
        timer.stop()
    return owner.set_timer(SEARCH_DEBOUNCE_S, callback)


def next_sort_state(
    columns: list[str], sort_index: int, sort_reverse: bool, *, reverse: bool
) -> tuple[int, bool, str]:
    if reverse:
        sort_reverse = not sort_reverse
    else:
        sort_index = (sort_index + 1) % len(columns)
    return sort_index, sort_reverse, columns[sort_index]


def sort_by_column(items: list[T], column: str, *, reverse: bool) -> None:
    items.sort(key=lambda item: getattr(item, column) or "", reverse=reverse)


def cursor_row_id(table: DataTable) -> str | None:
    if table.row_count == 0 or not table.is_valid_coordinate(table.cursor_coordinate):
        return None
    row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
    return str(row_key.value) if row_key.value else None


def toggle_cursor_selection(table: DataTable, selected_ids: set[str]) -> bool:
    item_id = cursor_row_id(table)
    if not item_id:
        return False
    if item_id in selected_ids:
        selected_ids.discard(item_id)
    else:
        selected_ids.add(item_id)
    return True


def handle_table_key(
    event,
    *,
    focused,
    table: DataTable,
    selected_ids: set[str],
    visible_ids: Iterable[str],
    refresh_table: Callable[[], None],
    cycle_sort: Callable[[bool], None],
) -> bool:
    if focused is not table:
        return False

    if event.key == "space":
        event.prevent_default()
        if toggle_cursor_selection(table, selected_ids):
            refresh_table()
        return True
    if event.key == "ctrl+a":
        event.prevent_default()
        selected_ids.clear()
        selected_ids.update(visible_ids)
        refresh_table()
        return True
    if event.key == "ctrl+d":
        event.prevent_default()
        selected_ids.clear()
        refresh_table()
        return True
    if event.key == "s":
        event.prevent_default()
        cycle_sort(False)
        return True
    if event.key == "S":
        event.prevent_default()
        cycle_sort(True)
        return True
    return False


def focus_nav_from_top_row(app, table: DataTable, event) -> bool:
    if app.focused is table and event.key == "up" and table.cursor_row == 0:
        event.prevent_default()
        app.focus_nav()
        return True
    return False
