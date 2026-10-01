"""Tests for upkeep: project canonicalization, anchor verification,
consolidation, and project briefs. The LLM is always mocked."""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engram.core.config import get_settings
from engram.core.models import Durability, Fact, FactCategory
from engram.core.projects import PROJECT_ROOTS_FILE, record_project_root
from engram.maintenance import briefs, consolidate
from engram.maintenance.consolidate import (
    MAX_CARD_CHARS,
    ConsolidatedCard,
    ConsolidationResponse,
    RetiredInputCard,
    partition_clusters,
)
from engram.maintenance.upkeep import (
    UpkeepStep,
    format_upkeep_report,
    project_brief,
    run_upkeep,
)
from engram.maintenance.upkeep import seconds_until_due
from engram.maintenance.upkeep_state import (
    STATE_FILE,
    StateLine,
    append_state,
    load_state,
    upkeep_lock,
)
from engram.maintenance.verify import derive_anchors, normalize_anchor
from engram.storage.search import SearchIndex
from engram.storage.store import AsyncFactStore, FactStore, format_fact_line

Responder = Callable[[list[dict]], ConsolidationResponse]


def _fact(fact_id: str, content: str, project: str | None = "widget", **kw) -> Fact:
    return Fact(
        id=fact_id,
        category=kw.pop("category", FactCategory.convention),
        content=content,
        project=project,
        **kw,
    )


def _store(tmp_path: Path, facts: list[Fact]) -> AsyncFactStore:
    store = FactStore(data_dir=tmp_path / "data")
    store.append_facts(facts)
    return AsyncFactStore(store)


def _by_id(store: AsyncFactStore) -> dict[str, Fact]:
    return {fact.id: fact for fact in store.sync_store.load_facts()}


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _cluster_cards(prompt: str) -> list[dict]:
    body = prompt.split("CLUSTER (oldest first):\n", 1)[1]
    return json.loads(body.split("\n\nCORRECTION", 1)[0])


@pytest.fixture(autouse=True)
def _isolated_repo_search(tmp_path: Path, monkeypatch):
    """Never walk the developer's home directory looking for repositories."""
    monkeypatch.setenv(
        "ENGRAM_REPO_SEARCH_ROOTS", json.dumps([str(tmp_path / "repos")])
    )
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def llm(monkeypatch):
    """Route consolidation and brief calls to in-test responders."""
    calls: dict[str, list[str]] = {"consolidate": [], "brief": []}
    state: dict[str, Responder | None] = {"responder": None}

    async def fake_consolidate(prompt, system, response_model, **kwargs):
        calls["consolidate"].append(prompt)
        responder = state["responder"]
        assert responder is not None
        return responder(_cluster_cards(prompt))

    async def fake_brief(prompt, system, response_model, **kwargs):
        calls["brief"].append(prompt)
        return briefs.BriefResponse(
            content=f"brief #{len(calls['brief'])}", retrieval_hints=["overview"]
        )

    monkeypatch.setattr(consolidate, "complete_model", fake_consolidate)
    monkeypatch.setattr(briefs, "complete_model", fake_brief)
    monkeypatch.setattr("engram.maintenance.upkeep.llm_available", lambda: True)

    def use(responder: Responder) -> None:
        state["responder"] = responder

    return calls, use


def _keep_all(cards: list[dict]) -> ConsolidationResponse:
    return ConsolidationResponse(
        cards=[
            ConsolidatedCard(
                source_ids=[card["id"]],
                memory_key=card["memory_key"],
                content=card["content"],
                category=FactCategory(card["category"]),
                durability=Durability(card["durability"]),
                anchors=card["anchors"],
                retrieval_hints=["hint"],
                tags=["tag"],
            )
            for card in cards
        ],
        retire=[],
    )


def _upkeep(store: AsyncFactStore, *steps: UpkeepStep, **kw):
    return asyncio.run(run_upkeep(store, steps=steps, **kw))


# --- projects ----------------------------------------------------------------


def test_projects_step_canonicalizes_names_and_records_roots(tmp_path: Path):
    repo = tmp_path / "repos" / "widget"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    store = _store(
        tmp_path,
        [
            _fact("a", "uses pnpm", project="Widget"),
            _fact("b", "uses ruff", project=str(repo)),
            _fact("c", "prefers vim", project=None),
            _fact("d", "already fine", project="widget"),
        ],
    )

    report = _upkeep(store, UpkeepStep.projects)

    facts = _by_id(store)
    assert {facts[i].project for i in "abd"} == {"widget"}
    assert facts["c"].project is None
    assert sorted(report.steps[0].fact_ids["edited"]) == ["a", "b"]
    roots = json.loads((store.data_dir / PROJECT_ROOTS_FILE).read_text())
    assert roots == {"widget": str(repo.resolve())}


# --- verify ------------------------------------------------------------------


def test_anchor_classification_and_derivation():
    assert normalize_anchor("src/app.py:12") == ("path", "src/app.py")
    assert normalize_anchor("snapshot.ts") == ("path", "snapshot.ts")
    assert normalize_anchor("snapshot.events") == ("symbol", "events")
    assert normalize_anchor("FactStore.apply_changes()") == ("symbol", "apply_changes")
    assert normalize_anchor("~/dotfiles/x.md") is None
    assert normalize_anchor("https://x.io/a.md") is None
    assert normalize_anchor("run") is None
    assert derive_anchors(
        "Sorted in src/renderer/lib/snapshot.ts (see `config.toml`), not "
        "~/.claude/x.md, https://a.io/b.md, and/or `snapshot.events`."
    ) == ["src/renderer/lib/snapshot.ts", "config.toml"]


@pytest.fixture
def widget_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repos" / "widget"
    (repo / "src" / "renderer" / "lib").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("def render_widget():\n    pass\n")
    (repo / "src" / "renderer" / "lib" / "snapshot.ts").write_text("export {}\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_verify_stales_flags_clears_and_records_anchors(
    tmp_path: Path, widget_repo: Path
):
    store = _store(
        tmp_path,
        [
            _fact("gone", "All gone.", anchors=["src/gone.py", "drop_table"]),
            _fact("partial", "Partly.", anchors=["src/app.py", "src/removed.py"]),
            _fact(
                "restored",
                "Back again.",
                anchors=["src/app.py", "render_widget"],
                suspect_reason="missing: src/app.py",
            ),
            _fact(
                "forever",
                "Evergreen.",
                anchors=["src/gone.py"],
                durability=Durability.evergreen,
            ),
            _fact("derived", "Events sort newest-first in lib/snapshot.ts."),
            _fact("derived_gone", "The loader lives in src/loader.py."),
            _fact("plain", "No anchors at all."),
            _fact("table_only", "Uses a warehouse table.", anchors=["DIM_USERS_V2"]),
            _fact("bare_file", "Reads config.toml.", anchors=["config.toml"]),
        ],
    )
    record_project_root(store.data_dir, "widget", widget_repo)

    report = _upkeep(store, UpkeepStep.verify).steps[0]

    facts = _by_id(store)
    assert facts["gone"].stale
    assert facts["gone"].stale_reason == "anchors missing: src/gone.py, drop_table"
    assert not facts["derived_gone"].stale  # prose-derived: flag only
    # A missing symbol alone may be a database object, not code: flag only.
    assert not facts["table_only"].stale
    assert facts["table_only"].suspect_reason == "missing: DIM_USERS_V2"
    assert not facts["bare_file"].stale  # bare names may live outside the repo
    assert facts["derived_gone"].suspect_reason == "missing: src/loader.py"
    assert not facts["partial"].stale
    assert facts["partial"].suspect_reason == "missing: src/removed.py"
    assert facts["restored"].suspect_reason == ""
    assert not facts["forever"].stale
    assert facts["derived"].anchors == ["lib/snapshot.ts"]
    touched = {fact_id for ids in report.fact_ids.values() for fact_id in ids}
    assert "plain" not in touched and "forever" not in touched
    assert report.counts["projects_with_repo"] == 1
    assert report.fact_ids["staled"] == ["gone"]

    # Once recorded, a verified anchor is explicit: losing it retires the fact.
    (widget_repo / "src" / "renderer" / "lib" / "snapshot.ts").unlink()
    _git(widget_repo, "commit", "-qam", "drop snapshot")
    _upkeep(store, UpkeepStep.verify)
    assert _by_id(store)["derived"].stale

    # Idempotent: a second pass changes nothing.
    before = store.facts_path.read_bytes()
    _upkeep(store, UpkeepStep.verify)
    assert store.facts_path.read_bytes() == before


def test_verify_respects_branches_and_restores_its_own_stale_facts(
    tmp_path: Path, widget_repo: Path
):
    # A feature branch adds a file and a symbol that main does not have.
    _git(widget_repo, "checkout", "-qb", "feature")
    (widget_repo / "src" / "feature.py").write_text("def ship_feature():\n    pass\n")
    _git(widget_repo, "add", ".")
    _git(widget_repo, "commit", "-qm", "feature")
    _git(widget_repo, "checkout", "-q", "-")
    store = _store(
        tmp_path,
        [
            _fact("branch_path", "On a branch.", anchors=["src/feature.py"]),
            _fact("branch_symbol", "On a branch.", anchors=["ship_feature"]),
            _fact("comes_back", "Returns later.", anchors=["src/later.py"]),
            _fact("user_stale", "User retired.", anchors=["src/app.py"]),
        ],
    )
    store.sync_store.mark_stale("user_stale", "no longer relevant")
    record_project_root(store.data_dir, "widget", widget_repo)

    report = _upkeep(store, UpkeepStep.verify).steps[0]

    facts = _by_id(store)
    assert not facts["branch_path"].stale and not facts["branch_symbol"].stale
    assert facts["branch_path"].suspect_reason == (
        "missing: src/feature.py (only on a branch)"
    )
    assert facts["branch_symbol"].suspect_reason.endswith("(only on a branch)")
    assert facts["comes_back"].stale
    assert report.fact_ids["staled"] == ["comes_back"]

    (widget_repo / "src" / "later.py").write_text("x = 1\n")
    _git(widget_repo, "add", ".")
    _git(widget_repo, "commit", "-qm", "later")
    report = _upkeep(store, UpkeepStep.verify).steps[0]

    facts = _by_id(store)
    assert not facts["comes_back"].stale
    assert facts["user_stale"].stale  # only verify's own retirements come back
    assert report.fact_ids["restored"] == ["comes_back"]


# --- consolidate -------------------------------------------------------------


def test_partition_clusters_is_disjoint_and_seeded_by_new_facts():
    topics = ["snowflake warehouse", "pnpm install", "ruff lint", "docker compose"]
    facts = [
        _fact(
            f"f{i:02d}",
            f"{topics[i % 4]} detail number {i}",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i),
        )
        for i in range(60)
    ]
    index = SearchIndex(facts)

    clusters = partition_clusters(facts, index, is_new=lambda fact: True)
    ids = [fact.id for cluster in clusters for fact in cluster]
    assert len(ids) == len(set(ids)) == 60
    assert all(len(cluster) <= 25 for cluster in clusters)

    only_new = partition_clusters(facts, index, is_new=lambda fact: fact.id == "f59")
    assert len(only_new) == 1 and only_new[0][0].id == "f59"
    assert partition_clusters(facts, index, is_new=lambda fact: False) == []


def test_consolidation_merges_and_retires_atomically(tmp_path: Path, llm):
    calls, use = llm
    store = _store(
        tmp_path,
        [
            _fact("a", "Widget deploys with make deploy.", memory_key="deploy"),
            _fact("b", "Widget deploy needs VPN.", memory_key="deploy-vpn"),
            _fact("c", "Ran make check, 158 tests green.", memory_key="tests"),
        ],
    )
    use(
        lambda cards: ConsolidationResponse(
            cards=[
                ConsolidatedCard(
                    source_ids=["a", "b"],
                    memory_key="widget-deploy",
                    content="Widget deploys with make deploy, which needs the VPN.",
                    category=FactCategory.workflow,
                    durability=Durability.durable,
                    anchors=[],
                    retrieval_hints=["how to deploy widget"],
                    tags=["deploy"],
                )
            ],
            retire=[RetiredInputCard(id="c", reason="test outcome")],
        )
    )

    report = _upkeep(store, UpkeepStep.consolidate).steps[0]

    facts = _by_id(store)
    assert facts["a"].confidence == 0.0 and facts["b"].confidence == 0.0
    assert facts["c"].stale and facts["c"].stale_reason == "consolidation: test outcome"
    (new_id,) = report.fact_ids["created"]
    merged = facts[new_id]
    assert merged.supersedes == "a"
    assert merged.consolidates == ["a", "b"]
    assert merged.project == "widget"
    assert merged.source == "engram:consolidate"
    assert merged.retrieval_hints == ["how to deploy widget"]
    assert len(calls["consolidate"]) == 1
    assert (store.data_dir / STATE_FILE).exists()


def _card(source_ids: list[str], key: str, content: str) -> ConsolidatedCard:
    return ConsolidatedCard(
        source_ids=source_ids,
        memory_key=key,
        content=content,
        category=FactCategory.workflow,
        durability=Durability.durable,
        anchors=[],
        retrieval_hints=["hint"],
        tags=["tag"],
    )


def test_consolidation_splits_a_card_holding_independent_claims(tmp_path: Path, llm):
    _, use = llm
    store = _store(
        tmp_path,
        [
            _fact(
                "big",
                "Widget deploys with make deploy. Widget logs live in Datadog.",
                consolidates=["ancient"],
            )
        ],
    )
    use(
        lambda cards: ConsolidationResponse(
            cards=[
                _card(["big"], "widget-deploy", "Widget deploys with make deploy."),
                _card(["big"], "widget-logs", "Widget logs live in Datadog."),
            ],
            retire=[],
        )
    )

    report = _upkeep(store, UpkeepStep.consolidate).steps[0]

    facts = _by_id(store)
    deploy, logs = (facts[fact_id] for fact_id in report.fact_ids["created"])
    assert facts["big"].confidence == 0.0
    assert report.fact_ids["superseded"] == ["big"]
    assert report.counts["split_cards"] == 1
    assert {deploy.memory_key, logs.memory_key} == {"widget-deploy", "widget-logs"}
    # Both halves keep the lineage of the card they came from.
    assert deploy.consolidates == logs.consolidates == ["big", "ancient"]
    assert deploy.supersedes == logs.supersedes == "big"


def test_split_that_repeats_claims_is_rejected(tmp_path: Path, llm):
    calls, use = llm
    store = _store(tmp_path, [_fact("a", "A one."), _fact("b", "B two.")])
    answers = iter(
        [
            # Identical pieces.
            [_card(["a"], "k1", "A one."), _card(["a"], "k2", "A one.")],
            # One piece is the whole source; the other merges it again.
            [_card(["a"], "k1", "A one."), _card(["a", "b"], "k2", "A one. B two.")],
        ]
    )

    def responder(cards: list[dict]) -> ConsolidationResponse:
        return ConsolidationResponse(cards=next(answers), retire=[])

    use(responder)
    report = _upkeep(store, UpkeepStep.consolidate).steps[0]

    assert "two cards have the same content" in calls["consolidate"][1]
    assert "split cards repeated whole in one piece: a" in report.errors[0]
    assert all(fact.confidence == 1.0 for fact in _by_id(store).values())


def test_oversized_merge_is_rejected_but_an_untouched_long_card_is_kept(
    tmp_path: Path, llm
):
    calls, use = llm
    long_text = "Widget fact. " * (MAX_CARD_CHARS // 13 + 1)
    store = _store(
        tmp_path,
        [_fact("a", "one"), _fact("b", "two"), _fact("long", long_text.strip())],
    )
    answers = iter([[_card(["a", "b"], "bloated", "x" * (MAX_CARD_CHARS + 1))], None])

    def responder(cards: list[dict]) -> ConsolidationResponse:
        kept = _keep_all(cards)
        merged = next(answers)
        if merged is None:
            return kept
        untouched = [card for card in kept.cards if card.content == long_text.strip()]
        return ConsolidationResponse(cards=[*merged, *untouched], retire=[])

    use(responder)
    report = _upkeep(store, UpkeepStep.consolidate).steps[0]

    assert len(calls["consolidate"]) == 2
    assert f"cards longer than {MAX_CARD_CHARS} characters" in calls["consolidate"][1]
    # Only the merge is named: the long source card copied verbatim is allowed.
    assert "independent cards: bloated (1201))" in calls["consolidate"][1]
    assert report.errors == []
    assert _by_id(store)["long"].confidence == 1.0


def test_merged_card_keeps_the_age_of_its_oldest_claim(tmp_path: Path, llm):
    _, use = llm
    old = datetime(2026, 3, 22, tzinfo=timezone.utc)
    new = datetime(2026, 10, 1, tzinfo=timezone.utc)
    store = _store(
        tmp_path,
        [
            _fact("a", "Widget deploys with make deploy.", observed_at=old),
            _fact("b", "Widget deploy needs VPN.", observed_at=new),
        ],
    )
    use(
        lambda cards: ConsolidationResponse(
            cards=[_card(["a", "b"], "widget-deploy", "Deploy needs make and VPN.")],
            retire=[],
        )
    )

    report = _upkeep(store, UpkeepStep.consolidate).steps[0]

    merged = _by_id(store)[report.fact_ids["created"][0]]
    assert merged.observed_at == new
    assert merged.first_observed_at == old
    assert "2026-03-22..2026-10-01" in format_fact_line(merged)


def test_unchanged_single_source_card_is_edited_in_place(tmp_path: Path, llm):
    _, use = llm
    observed = datetime.now(timezone.utc) - timedelta(days=5)
    store = _store(
        tmp_path,
        [_fact("keep", "Widget staging is down this week.", observed_at=observed)],
    )

    def mark_ephemeral(cards: list[dict]) -> ConsolidationResponse:
        response = _keep_all(cards)
        response.cards[0].durability = Durability.ephemeral
        return response

    use(mark_ephemeral)
    report = _upkeep(store, UpkeepStep.consolidate).steps[0]

    fact = _by_id(store)["keep"]
    assert fact.confidence == 1.0
    assert fact.durability is Durability.ephemeral
    assert fact.retrieval_hints == ["hint"]
    # The TTL starts when upkeep classifies the card, never in the past.
    assert fact.expires_at is not None
    assert fact.expires_at > datetime.now(timezone.utc) + timedelta(days=44)
    assert report.fact_ids["edited"] == ["keep"]
    assert "created" not in report.fact_ids or not report.fact_ids["created"]


def test_coverage_error_retries_once_then_applies(tmp_path: Path, llm):
    calls, use = llm
    store = _store(tmp_path, [_fact("a", "one"), _fact("b", "two")])
    answers = iter(
        [
            ConsolidationResponse(cards=[], retire=[]),  # drops both ids
            None,
        ]
    )

    def responder(cards: list[dict]) -> ConsolidationResponse:
        return next(answers) or _keep_all(cards)

    use(responder)
    report = _upkeep(store, UpkeepStep.consolidate).steps[0]

    assert len(calls["consolidate"]) == 2
    assert "CORRECTION" in calls["consolidate"][1]
    assert "input ids not accounted for: a, b" in calls["consolidate"][1]
    assert report.errors == []


def test_persistently_invalid_cluster_is_retried_then_abandoned(tmp_path: Path, llm):
    calls, use = llm
    store = _store(tmp_path, [_fact("a", "one"), _fact("b", "two")])
    before = store.facts_path.read_bytes()

    def double_use(cards: list[dict]) -> ConsolidationResponse:
        response = _keep_all(cards)
        response.retire = [RetiredInputCard(id="a", reason="dup")]
        return response

    use(double_use)
    report = _upkeep(store, UpkeepStep.consolidate).steps[0]

    assert len(calls["consolidate"]) == 2
    assert store.facts_path.read_bytes() == before
    assert "ids both kept and retired: a" in report.errors[0]
    # The project's timestamp advances; only the failed seed is retried.
    state = load_state(store.data_dir).projects["widget"]
    assert state.retry_seeds == {"a": 1}

    _upkeep(store, UpkeepStep.consolidate)
    assert load_state(store.data_dir).projects["widget"].retry_seeds == {"a": 2}
    report = _upkeep(store, UpkeepStep.consolidate).steps[0]
    assert any("gave up" in note for note in report.notes)
    assert load_state(store.data_dir).projects["widget"].retry_seeds == {}
    assert len(calls["consolidate"]) == 6

    _upkeep(store, UpkeepStep.consolidate)
    assert len(calls["consolidate"]) == 6  # abandoned, nothing new


def test_retry_seed_is_dropped_once_its_cluster_succeeds(tmp_path: Path, llm):
    calls, use = llm
    store = _store(tmp_path, [_fact("a", "one"), _fact("b", "two")])
    answers = iter([ConsolidationResponse(cards=[], retire=[])] * 2)
    use(lambda cards: next(answers, None) or _keep_all(cards))

    _upkeep(store, UpkeepStep.consolidate)
    assert load_state(store.data_dir).projects["widget"].retry_seeds == {"a": 1}
    _upkeep(store, UpkeepStep.consolidate)
    assert load_state(store.data_dir).projects["widget"].retry_seeds == {}
    assert len(calls["consolidate"]) == 3


def test_concurrent_write_aborts_cluster_and_is_seen_next_run(tmp_path: Path, llm):
    calls, use = llm
    store = _store(
        tmp_path, [_fact("a", "one"), _fact("g", "gadget", project="gadget")]
    )

    def edit_during_call(cards: list[dict]) -> ConsolidationResponse:
        if len(calls["consolidate"]) == 1:
            # Another writer changes a source and adds a fact mid-run.
            store.sync_store.mark_stale("a", "user retired it")
            store.sync_store.append_facts([_fact("late", "late", project="gadget")])
        return _keep_all(cards)

    use(edit_during_call)
    report = _upkeep(store, UpkeepStep.consolidate).steps[0]

    assert _by_id(store)["a"].stale_reason == "user retired it"
    assert any("changed concurrently" in error for error in report.errors)
    calls["consolidate"].clear()
    _upkeep(store, UpkeepStep.consolidate)
    prompts = "".join(calls["consolidate"])
    assert '"late"' in prompts  # written after the snapshot: seen now


def test_run_upkeep_skips_when_another_run_holds_the_lock(tmp_path: Path):
    store = _store(tmp_path, [_fact("a", "one")])
    with upkeep_lock(store.data_dir) as acquired:
        assert acquired
        report = _upkeep(store, *UpkeepStep)
    assert [step.skipped for step in report.steps] == [
        "another upkeep run is in progress"
    ] * 4
    assert load_state(store.data_dir).last_run_at is None

    report = _upkeep(store, *UpkeepStep)
    assert not report.step(UpkeepStep.verify).skipped
    assert load_state(store.data_dir).last_run_at == report.started_at


def test_background_schedule_survives_restarts(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    now = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    assert seconds_until_due(data_dir, interval=3600, minimum=120, now=now) == 120
    append_state(data_dir, [StateLine(kind="run", at=now - timedelta(minutes=10))])
    assert seconds_until_due(data_dir, interval=3600, minimum=120, now=now) == 3000
    append_state(data_dir, [StateLine(kind="run", at=now - timedelta(hours=5))])
    # The older line does not win over the newer one.
    assert seconds_until_due(data_dir, interval=3600, minimum=120, now=now) == 3000


def test_consolidation_never_mixes_projects(tmp_path: Path, llm):
    calls, use = llm
    store = _store(
        tmp_path,
        [
            _fact("w1", "shared words deploy", project="widget"),
            _fact("w2", "shared words deploy again", project="widget"),
            _fact("g1", "shared words deploy", project="gadget"),
            _fact("n1", "shared words deploy", project=None),
        ],
    )
    use(_keep_all)
    _upkeep(store, UpkeepStep.consolidate)

    # Prompts carry handles, not IDs; each cluster must hold one scope only.
    scopes = sorted(
        (prompt.split("\n", 1)[0], len(_cluster_cards(prompt)))
        for prompt in calls["consolidate"]
    )
    assert scopes == [
        ("PROJECT SCOPE: (global)", 1),
        ("PROJECT SCOPE: gadget", 1),
        ("PROJECT SCOPE: widget", 2),
    ]


def test_incremental_state_skips_unchanged_projects(tmp_path: Path, llm):
    calls, use = llm
    store = _store(
        tmp_path,
        [
            _fact("w1", "widget fact", project="widget"),
            _fact("g1", "gadget fact", project="gadget"),
        ],
    )
    use(_keep_all)
    _upkeep(store, UpkeepStep.consolidate)
    assert len(calls["consolidate"]) == 2

    # Nothing changed: no calls. The run's own metadata edits don't count.
    report = _upkeep(store, UpkeepStep.consolidate).steps[0]
    assert len(calls["consolidate"]) == 2
    assert report.counts["projects_unchanged"] == 2

    store.sync_store.append_facts([_fact("w2", "new widget fact", project="widget")])
    _upkeep(store, UpkeepStep.consolidate)
    assert len(calls["consolidate"]) == 3
    last = calls["consolidate"][-1]
    assert "new widget fact" in last and "gadget fact" not in last

    _upkeep(store, UpkeepStep.consolidate, full=True)
    assert len(calls["consolidate"]) == 5


# --- briefs ------------------------------------------------------------------


def test_brief_is_written_then_replaced_when_facts_change(tmp_path: Path, llm):
    calls, _ = llm
    store = _store(tmp_path, [_fact(f"f{i}", f"widget fact {i}") for i in range(5)])
    store.sync_store.append_facts([_fact("tiny", "one fact", project="gadget")])

    _upkeep(store, UpkeepStep.briefs)
    first = asyncio.run(project_brief(store, "widget"))
    assert first is not None and first.content == "brief #1"
    assert first.tags == ["brief"] and first.source == "engram:brief"
    assert asyncio.run(project_brief(store, "gadget")) is None  # < 5 facts

    _upkeep(store, UpkeepStep.briefs)
    assert len(calls["brief"]) == 1  # unchanged project, no new brief

    store.sync_store.append_facts([_fact("f5", "widget fact 5")])
    _upkeep(store, UpkeepStep.briefs)
    second = asyncio.run(project_brief(store, "widget"))
    assert second is not None and second.content == "brief #2"
    assert second.supersedes == first.id
    assert _by_id(store)[first.id].confidence == 0.0


def test_brief_refresh_supersedes_every_duplicate(tmp_path: Path, llm):
    calls, _ = llm
    dupes = [
        _fact(f"brief{i}", f"old brief {i}", memory_key="project-brief")
        for i in range(2)
    ]
    store = _store(tmp_path, [_fact(f"f{i}", f"widget fact {i}") for i in range(5)])
    store.sync_store.append_facts(dupes)

    _upkeep(store, UpkeepStep.briefs)

    facts = _by_id(store)
    assert facts["brief0"].confidence == 0.0 and facts["brief1"].confidence == 0.0
    current = asyncio.run(project_brief(store, "widget"))
    assert current is not None and current.content == "brief #1"


# --- whole run ---------------------------------------------------------------


def test_dry_run_writes_nothing(tmp_path: Path, widget_repo: Path, llm):
    calls, use = llm
    store = _store(
        tmp_path,
        [
            _fact(f"f{i}", f"Widget fact {i} in src/gone{i}.py", project="Widget")
            for i in range(6)
        ],
    )
    use(
        lambda cards: ConsolidationResponse(
            cards=[],
            retire=[RetiredInputCard(id=card["id"], reason="junk") for card in cards],
        )
    )
    before = sorted(path.name for path in store.data_dir.iterdir())
    log = store.facts_path.read_bytes()

    report = _upkeep(store, *UpkeepStep, dry_run=True)

    assert store.facts_path.read_bytes() == log
    after = sorted(path.name for path in store.data_dir.iterdir())
    assert [name for name in after if not name.endswith(".lock")] == [
        name for name in before if not name.endswith(".lock")
    ]
    assert report.dry_run
    assert len(report.step(UpkeepStep.projects).fact_ids["edited"]) == 6
    assert len(report.step(UpkeepStep.consolidate).fact_ids["staled"]) == 6
    assert calls["brief"]
    assert "dry run" in format_upkeep_report(report)


def test_llm_steps_skip_without_key(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("engram.maintenance.upkeep.llm_available", lambda: False)
    store = _store(tmp_path, [_fact("a", "one")])

    report = _upkeep(store, *UpkeepStep)

    assert report.step(UpkeepStep.consolidate).skipped
    assert report.step(UpkeepStep.briefs).skipped
    assert not report.step(UpkeepStep.verify).skipped
