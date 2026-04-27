"""Data loading and aggregation for the dashboard."""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from engram.dashboard.constants import MIN_ACTIVE_CONFIDENCE, NO_PROJECT_LABEL
from engram.models import CandidateStatus, Fact, MemoryCandidate
from engram.store import FactStore


@dataclass
class ProjectHealth:
    name: str
    total: int = 0
    active: int = 0
    forgotten: int = 0
    expired: int = 0
    categories: dict[str, int] = field(default_factory=dict)
    oldest: datetime | None = None
    newest: datetime | None = None
    supersession_depth: int = 0


@dataclass
class DashboardData:
    """Precomputed aggregations for the dashboard."""

    all_facts: list[Fact] = field(default_factory=list)
    active_facts: list[Fact] = field(default_factory=list)
    forgotten_facts: list[Fact] = field(default_factory=list)
    expired_facts: list[Fact] = field(default_factory=list)
    candidates: list[MemoryCandidate] = field(default_factory=list)
    pending_candidates: list[MemoryCandidate] = field(default_factory=list)

    total: int = 0
    active_count: int = 0
    forgotten_count: int = 0
    expired_count: int = 0
    pending_count: int = 0
    storage_bytes: int = 0

    by_category: dict[str, int] = field(default_factory=dict)
    by_project: dict[str, int] = field(default_factory=dict)

    # Daily counts keyed by YYYY-MM-DD; activity_* are dense per-day series.
    daily_created: dict[str, int] = field(default_factory=dict)
    daily_forgotten: dict[str, int] = field(default_factory=dict)
    daily_expired: dict[str, int] = field(default_factory=dict)
    activity_30d: list[float] = field(default_factory=list)
    activity_7d: list[float] = field(default_factory=list)

    project_health: dict[str, ProjectHealth] = field(default_factory=dict)
    categories: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)

    # File-stat-based hash; used to skip refreshes when storage hasn't changed.
    content_hash: int = 0


def content_hash_for(store: FactStore) -> int:
    """Quick content hash based on file sizes + mtimes — no full read needed."""
    h = 0
    for path in (store.facts_path, store.candidates_path):
        if path.exists():
            stat = path.stat()
            h ^= hash((stat.st_size, stat.st_mtime_ns))
    return h


def load_dashboard_data(store: FactStore | None = None) -> DashboardData:
    """Load and compute all dashboard aggregations."""
    store = store or FactStore()
    now = datetime.now(timezone.utc)
    data = DashboardData()
    data.content_hash = content_hash_for(store)

    data.all_facts = store.load_facts()
    data.candidates = store.load_candidates()
    data.pending_candidates = [
        c for c in data.candidates if c.status == CandidateStatus.pending
    ]

    for fact in data.all_facts:
        is_forgotten = fact.confidence < MIN_ACTIVE_CONFIDENCE
        is_expired = fact.expires_at is not None and fact.expires_at < now
        if is_forgotten:
            data.forgotten_facts.append(fact)
        elif is_expired:
            data.expired_facts.append(fact)
        else:
            data.active_facts.append(fact)

    data.total = len(data.all_facts)
    data.active_count = len(data.active_facts)
    data.forgotten_count = len(data.forgotten_facts)
    data.expired_count = len(data.expired_facts)
    data.pending_count = len(data.pending_candidates)
    data.storage_bytes = (
        store.facts_path.stat().st_size if store.facts_path.exists() else 0
    )

    cat_counter = Counter(f.category.value for f in data.active_facts)
    data.by_category = dict(cat_counter.most_common())
    proj_counter = Counter(f.project or NO_PROJECT_LABEL for f in data.active_facts)
    data.by_project = dict(proj_counter.most_common())
    data.categories = list(data.by_category.keys())
    data.projects = list(data.by_project.keys())

    data.daily_created = _daily_counts(data.all_facts, key=lambda f: f.created_at)
    data.daily_forgotten = _daily_counts(
        data.forgotten_facts, key=lambda f: f.updated_at
    )
    data.daily_expired = _daily_counts(
        data.expired_facts, key=lambda f: f.expires_at or f.updated_at
    )

    data.activity_30d = _sparkline_data(data.daily_created, days=30, now=now)
    data.activity_7d = _sparkline_data(data.daily_created, days=7, now=now)

    _compute_project_health(data, now)

    return data


def get_facts_for_category(data: DashboardData, category: str) -> list[Fact]:
    return [f for f in data.active_facts if f.category.value == category]


def get_facts_for_project(data: DashboardData, project: str) -> list[Fact]:
    target = None if project == NO_PROJECT_LABEL else project
    return [f for f in data.active_facts if f.project == target]


def format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def format_age(dt: datetime) -> str:
    delta = datetime.now(timezone.utc) - dt
    if delta.days > 365:
        return f"{delta.days // 365}y"
    if delta.days > 30:
        return f"{delta.days // 30}mo"
    if delta.days > 0:
        return f"{delta.days}d"
    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours}h"
    mins = delta.seconds // 60
    if mins > 0:
        return f"{mins}m"
    return "now"


def format_timestamp(dt: datetime) -> str:
    return f"{dt:%Y-%m-%d %H:%M} UTC ({format_age(dt)} ago)"


def format_confidence(conf: float) -> str:
    pct = f"{conf:.0%}"
    if conf >= 0.8:
        return f"[#788c5d]{pct}[/]"
    if conf >= 0.5:
        return f"[#eda100]{pct}[/]"
    return f"[#f7768e]{pct}[/]"


def _daily_counts(facts: list[Fact], key) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for f in facts:
        dt = key(f)
        if dt:
            counter[dt.strftime("%Y-%m-%d")] += 1
    return dict(sorted(counter.items()))


def _sparkline_data(daily: dict[str, int], days: int, now: datetime) -> list[float]:
    result = []
    for i in range(days, 0, -1):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        result.append(float(daily.get(day, 0)))
    return result


def _compute_project_health(data: DashboardData, now: datetime) -> None:
    all_by_project: dict[str, list[Fact]] = {}
    for f in data.all_facts:
        key = f.project or NO_PROJECT_LABEL
        all_by_project.setdefault(key, []).append(f)

    for proj_name, facts in all_by_project.items():
        health = ProjectHealth(name=proj_name)
        supersedes_ids = set()
        for f in facts:
            is_forgotten = f.confidence < MIN_ACTIVE_CONFIDENCE
            is_expired = f.expires_at is not None and f.expires_at < now
            if is_forgotten:
                health.forgotten += 1
            elif is_expired:
                health.expired += 1
            else:
                health.active += 1
                cat = f.category.value
                health.categories[cat] = health.categories.get(cat, 0) + 1
            health.total += 1
            if health.oldest is None or f.created_at < health.oldest:
                health.oldest = f.created_at
            if health.newest is None or f.created_at > health.newest:
                health.newest = f.created_at
            if f.supersedes:
                supersedes_ids.add(f.supersedes)

        health.supersession_depth = len(supersedes_ids)
        data.project_health[proj_name] = health
