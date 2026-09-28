"""Report models shared by the upkeep steps."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class UpkeepStep(str, Enum):
    """Upkeep steps, in the order ``run_upkeep`` executes them."""

    projects = "projects"
    verify = "verify"
    consolidate = "consolidate"
    briefs = "briefs"


ALL_STEPS: tuple[UpkeepStep, ...] = tuple(UpkeepStep)
LLM_STEPS = frozenset({UpkeepStep.consolidate, UpkeepStep.briefs})


class StepReport(BaseModel):
    """What one upkeep step examined, changed (or would change), and failed on."""

    model_config = ConfigDict(extra="forbid")

    step: UpkeepStep
    # Non-empty when the step did not run at all.
    skipped: str = ""
    counts: dict[str, int] = Field(default_factory=dict)
    # Action ("staled", "superseded", "edited", ...) -> affected fact IDs.
    fact_ids: dict[str, list[str]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    llm_calls: int = 0

    def add(self, action: str, *fact_ids: str) -> None:
        self.fact_ids.setdefault(action, []).extend(fact_ids)

    def bump(self, counter: str, amount: int = 1) -> None:
        self.counts[counter] = self.counts.get(counter, 0) + amount


class UpkeepReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    started_at: datetime
    finished_at: datetime | None = None
    dry_run: bool = False
    project: str | None = None
    steps: list[StepReport] = Field(default_factory=list)

    def step(self, step: UpkeepStep) -> StepReport | None:
        return next((report for report in self.steps if report.step is step), None)


_MAX_LISTED_NOTES = 10


def format_upkeep_report(report: UpkeepReport) -> str:
    """Human-readable summary: one block per step, notes truncated."""
    scope = report.project or "all projects"
    mode = "dry run (nothing applied)" if report.dry_run else "applied"
    lines = [f"Upkeep for {scope}: {mode}"]
    for step in report.steps:
        if step.skipped:
            lines.append(f"- {step.step.value}: skipped ({step.skipped})")
            continue
        parts = [f"{name} {count}" for name, count in step.counts.items()]
        parts += [
            f"{action} {len(ids)}" for action, ids in step.fact_ids.items() if ids
        ]
        if step.llm_calls:
            parts.append(f"llm calls {step.llm_calls}")
        lines.append(f"- {step.step.value}: {', '.join(parts) or 'nothing to do'}")
        for note in step.notes[:_MAX_LISTED_NOTES]:
            lines.append(f"    {note}")
        if len(step.notes) > _MAX_LISTED_NOTES:
            lines.append(f"    … {len(step.notes) - _MAX_LISTED_NOTES} more")
        for error in step.errors:
            lines.append(f"    error: {error}")
    return "\n".join(lines)
