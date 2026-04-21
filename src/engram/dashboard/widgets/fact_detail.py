"""Fact detail panel — shows full metadata for a selected fact."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from engram.dashboard.data import format_confidence, format_timestamp
from engram.models import Fact, MemoryCandidate


class FactDetail(Static):
    """Renders full metadata for a single fact."""

    def __init__(self, fact: Fact | MemoryCandidate | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._fact = fact

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            if self._fact:
                yield from self._render_fact(self._fact)
            else:
                yield Static("Select a row to view details", classes="detail-title")

    def _render_fact(self, f: Fact | MemoryCandidate):
        yield Static("FACT DETAIL", classes="detail-title")
        yield Static(f"[b]ID:[/b]          {f.id}", classes="detail-row")
        yield Static(f"[b]Category:[/b]    {f.category.value}", classes="detail-row")
        yield Static(
            f"[b]Project:[/b]     {f.project or '(none)'}", classes="detail-row"
        )
        yield Static(
            f"[b]Confidence:[/b]  {format_confidence(f.confidence)}",
            classes="detail-row",
        )
        yield Static(f"[b]Source:[/b]      {f.source}", classes="detail-row")
        yield Static(
            f"[b]Evidence:[/b]    {f.evidence_kind.value}", classes="detail-row"
        )
        if f.source_ref:
            yield Static(f"[b]Source Ref:[/b]  {f.source_ref}", classes="detail-row")
        yield Static(
            f"[b]Tags:[/b]        {', '.join(f.tags) or '(none)'}", classes="detail-row"
        )
        yield Static(
            f"[b]Created:[/b]     {format_timestamp(f.created_at)}",
            classes="detail-row",
        )
        yield Static(
            f"[b]Updated:[/b]     {format_timestamp(f.updated_at)}",
            classes="detail-row",
        )
        if f.effective_at:
            yield Static(
                f"[b]Effective:[/b]  {format_timestamp(f.effective_at)}",
                classes="detail-row",
            )
        if f.expires_at:
            yield Static(
                f"[b]Expires:[/b]    {format_timestamp(f.expires_at)}",
                classes="detail-row",
            )
        if f.supersedes:
            yield Static(f"[b]Supersedes:[/b] {f.supersedes}", classes="detail-row")
        if f.why_store:
            yield Static(f"[b]Why Store:[/b]  {f.why_store}", classes="detail-row")
        if isinstance(f, MemoryCandidate):
            yield Static(f"[b]Status:[/b]     {f.status.value}", classes="detail-row")
            if f.review_note:
                yield Static(
                    f"[b]Review:[/b]     {f.review_note}", classes="detail-row"
                )
        yield Static("")
        yield Static("[b]Content:[/b]", classes="detail-row")
        yield Static(f.content, classes="detail-content")

    def update_fact(self, fact: Fact | MemoryCandidate | None) -> None:
        self._fact = fact
        self.remove_children()
        if fact:
            self.mount_all(list(self._render_fact(fact)))
        else:
            self.mount(Static("Select a row to view details", classes="detail-title"))
