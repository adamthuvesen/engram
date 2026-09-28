"""Upkeep: keep memory current and compact without anyone asking.

``run_upkeep`` runs these steps in order, each appending ordinary events to
the log (so every change is reversible):

1. ``projects``: canonicalize fact project names (paths -> repo names).
2. ``verify``: check fact anchors against the project's git checkout; retire
   facts whose anchors all vanished, flag facts missing some (no LLM).
3. ``consolidate``: LLM-merge clusters of related cards, retire junk.
4. ``briefs``: refresh each project's orientation card.

The LLM steps are skipped when no provider key is configured. One run at a
time per data directory: ``upkeep.lock`` is taken non-blocking and a
concurrent run returns a report with every step skipped.

Multi-machine sync: the lock is per machine. Two synced machines can each
consolidate the same cluster before syncing, and the union-merged event log
then holds both replacement cards (their sources superseded twice). Upkeep
does not deduplicate those; the next consolidation of that project sees two
near-identical cards and merges them. Enable background upkeep on one machine
to avoid the churn.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engram.core.config import ensure_openai_api_key, get_settings
from engram.core.models import Fact
from engram.core.projects import canonical_project, record_project_root, resolve_project
from engram.maintenance.briefs import project_brief, run_briefs
from engram.maintenance.consolidate import BRIEF_KEY, run_consolidation
from engram.maintenance.upkeep_report import (
    ALL_STEPS,
    LLM_STEPS,
    StepReport,
    UpkeepReport,
    UpkeepStep,
    format_upkeep_report,
)
from engram.maintenance.upkeep_state import (
    StateLine,
    append_state,
    load_state,
    upkeep_lock,
)
from engram.maintenance.verify import verify_facts
from engram.storage.store import AsyncFactStore, ChangeSet

logger = logging.getLogger(__name__)


def llm_available() -> bool:
    return ensure_openai_api_key() is not None


def _canonicalize_projects(
    facts: list[Fact],
    *,
    data_dir: Path,
    project: str | None,
    dry_run: bool,
    report: StepReport,
) -> ChangeSet:
    changes = ChangeSet(reason="upkeep: canonical project names", actor="engram:upkeep")
    resolved: dict[str, tuple[str | None, Path | None]] = {}
    known = {
        fact.project
        for fact in facts
        if fact.project and not fact.project.startswith(("/", "~"))
    }
    for fact in facts:
        if fact.project is None:
            continue
        if fact.project not in resolved:
            resolved[fact.project] = resolve_project(fact.project, known=known)
        name, root = resolved[fact.project]
        if project is not None and name != project:
            continue
        if name and name != fact.project:
            changes.edits[fact.id] = {"project": name}
            changes.expected[fact.id] = fact.updated_at
            report.add("edited", fact.id)
    for raw, (name, root) in sorted(resolved.items()):
        if name and name != raw:
            report.notes.append(f"{raw!r} -> {name!r}")
        if name and root is not None and not dry_run:
            record_project_root(data_dir, name, root)
    return changes


async def _run_projects(
    store: AsyncFactStore, project: str | None, dry_run: bool, report: StepReport
) -> None:
    facts = await store.load_active_facts(include_stale=True)
    changes = await asyncio.to_thread(
        _canonicalize_projects,
        facts,
        data_dir=store.data_dir,
        project=project,
        dry_run=dry_run,
        report=report,
    )
    if changes.edits and not dry_run:
        result = await store.apply_changes(changes)
        report.bump("skipped_changed", len(result.skipped))


async def _run_verify(
    store: AsyncFactStore, project: str | None, dry_run: bool, report: StepReport
) -> None:
    settings = get_settings()
    facts = [
        fact
        for fact in await store.load_active_facts(
            project=project, include_global=False, include_stale=True
        )
        if fact.memory_key != BRIEF_KEY
    ]
    plan = await asyncio.to_thread(
        verify_facts,
        facts,
        data_dir=store.data_dir,
        search_roots=list(settings.repo_search_roots),
        report=report,
    )
    if dry_run:
        return
    for fact_id in plan.unstale:
        await store.unmark_stale(fact_id)
    changes = plan.changes
    if changes.stale or changes.edits:
        result = await store.apply_changes(changes)
        report.bump("skipped_changed", len(result.skipped))


async def run_upkeep(
    store: AsyncFactStore,
    *,
    project: str | None = None,
    steps: Sequence[UpkeepStep] = ALL_STEPS,
    dry_run: bool = False,
    now: datetime | None = None,
    full: bool = False,
) -> UpkeepReport:
    """Run the requested upkeep steps (always in canonical order).

    ``dry_run`` computes every change without applying it; the LLM steps still
    call the LLM so their proposals can be reviewed. ``full`` ignores the
    incremental consolidation state and reprocesses every project.
    """
    now = now or datetime.now(timezone.utc)
    project = canonical_project(project)
    report = UpkeepReport(started_at=now, dry_run=dry_run, project=project)
    ordered = [step for step in ALL_STEPS if step in steps]
    with upkeep_lock(store.data_dir) as acquired:
        if not acquired:
            report.steps = [
                StepReport(step=step, skipped="another upkeep run is in progress")
                for step in ordered
            ]
        else:
            if not dry_run and project is None and set(ordered) == set(ALL_STEPS):
                # Recorded at start so a server restarted mid-run waits too.
                await asyncio.to_thread(
                    append_state, store.data_dir, [StateLine(kind="run", at=now)]
                )
            await _run_steps(store, report, ordered, project, dry_run, now, full)
    report.finished_at = datetime.now(timezone.utc)
    return report


async def _run_steps(
    store: AsyncFactStore,
    report: UpkeepReport,
    steps: list[UpkeepStep],
    project: str | None,
    dry_run: bool,
    now: datetime,
    full: bool,
) -> None:
    has_llm = any(step in LLM_STEPS for step in steps) and llm_available()
    for step in steps:
        step_report = StepReport(step=step)
        report.steps.append(step_report)
        if step in LLM_STEPS and not has_llm:
            step_report.skipped = "no LLM API key configured"
            continue
        try:
            if step is UpkeepStep.projects:
                await _run_projects(store, project, dry_run, step_report)
            elif step is UpkeepStep.verify:
                await _run_verify(store, project, dry_run, step_report)
            elif step is UpkeepStep.consolidate:
                await run_consolidation(
                    store,
                    project=project,
                    report=step_report,
                    dry_run=dry_run,
                    now=now,
                    full=full,
                )
            else:
                await run_briefs(
                    store,
                    project=project,
                    report=step_report,
                    dry_run=dry_run,
                    now=now,
                )
        except Exception as exc:  # noqa: BLE001 - later steps still run
            logger.exception("upkeep step %s failed", step.value)
            step_report.errors.append(f"step failed: {exc}")


def seconds_until_due(
    data_dir: Path, *, interval: float, minimum: float, now: datetime | None = None
) -> float:
    """Seconds until the next background run: ``interval`` after the last
    recorded full run, but never sooner than ``minimum``."""
    last = load_state(data_dir).last_run_at
    if last is None:
        return minimum
    now = now or datetime.now(timezone.utc)
    due = last + timedelta(seconds=interval)
    return max(minimum, (due - now).total_seconds())


async def upkeep_loop(
    get_store: Callable[[], AsyncFactStore],
    *,
    interval: float,
    initial_delay: float = 120.0,
) -> None:
    """Run ``run_upkeep`` every ``interval`` seconds until cancelled.

    The schedule survives restarts: each wait runs until ``interval`` after
    the last full run recorded in the maintenance state, and is never shorter
    than ``initial_delay``. Without an LLM key only the local steps (projects, verify) do work.
    Failures are logged and the loop keeps going.
    """
    while True:
        try:
            store = get_store()
            delay = await asyncio.to_thread(
                seconds_until_due,
                store.data_dir,
                interval=interval,
                minimum=initial_delay,
            )
            await asyncio.sleep(delay)
            report = await run_upkeep(store)
            logger.info("%s", format_upkeep_report(report))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the background loop must not die
            logger.exception("upkeep run failed")
            await asyncio.sleep(max(initial_delay, 1.0))


__all__ = [
    "ALL_STEPS",
    "StepReport",
    "UpkeepReport",
    "UpkeepStep",
    "format_upkeep_report",
    "llm_available",
    "project_brief",
    "run_upkeep",
    "seconds_until_due",
    "upkeep_loop",
]
