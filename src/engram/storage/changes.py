"""Fact mutation batches and the results the store reports for them."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from engram.core.models import Fact


@dataclass
class ChangeSet:
    """A batch of fact mutations appended to the event log atomically.

    ``supersede`` maps an existing fact ID to the ID of the fact replacing it
    (usually one of ``new_facts``). ``stale`` maps IDs to a retirement reason.
    ``edits`` maps IDs to editable-field updates. ``expected`` maps target IDs
    to the ``updated_at`` the caller based its decision on; a target changed
    since then (edited, marked stale, ...) is skipped.
    """

    new_facts: list[Fact] = field(default_factory=list)
    supersede: dict[str, str] = field(default_factory=dict)
    stale: dict[str, str] = field(default_factory=dict)
    unstale: list[str] = field(default_factory=list)
    edits: dict[str, dict[str, Any]] = field(default_factory=dict)
    expected: dict[str, datetime] = field(default_factory=dict)
    require_eligible: bool = False
    reason: str = ""
    actor: str = "engram"

    def __post_init__(self) -> None:
        ids = [fact.id for fact in self.new_facts]
        if len(ids) != len(set(ids)):
            raise ValueError("ChangeSet contains duplicate new fact IDs")
        groups = [set(self.supersede), set(self.stale), set(self.edits)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("ChangeSet contains contradictory mutations for one fact")
        if set(ids) & (groups[0] | groups[1] | groups[2]):
            raise ValueError("ChangeSet cannot create and mutate the same fact")
        if set(self.unstale) & (set(self.stale) | set(self.supersede) | set(ids)):
            raise ValueError("ChangeSet contains contradictory lifecycle changes")
        if any(old == new for old, new in self.supersede.items()):
            raise ValueError("ChangeSet cannot supersede a fact with itself")


@dataclass
class ChangeResult:
    created: list[Fact] = field(default_factory=list)
    superseded: dict[str, str] = field(default_factory=dict)
    staled: dict[str, str] = field(default_factory=dict)
    edited: list[str] = field(default_factory=list)
    unstaled: list[str] = field(default_factory=list)
    # Target ID -> why the change was not applied.
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def applied(self) -> bool:
        return bool(self.created or self.superseded or self.staled or self.edited or self.unstaled)


@dataclass
class ApprovalResult:
    """Authoritative candidate decisions from one locked approval transaction."""

    facts: list[Fact] = field(default_factory=list)
    approved: dict[str, str] = field(default_factory=dict)
    edited: list[str] = field(default_factory=list)
    invalid_edit_ids: list[str] = field(default_factory=list)
    unresolved: dict[str, str] = field(default_factory=dict)
