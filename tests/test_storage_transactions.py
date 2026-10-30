"""Storage invariants across interruptions and offline snapshot merges."""

import asyncio
import json
import shutil
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from engram.core.fact_state import is_fact_eligible
from engram.core.models import (
    CandidateStatus,
    EventLogMeta,
    Fact,
    FactCategory,
    FactEvent,
    FactLifecycle,
    MemoryCandidate,
    StoreTransaction,
    TransactionStatus,
    materialize_events,
)
from engram.storage import store as store_module
from engram.storage import sync as sync_module
from engram.storage.store import ChangeSet, FactStore, StoreIntegrityError


def fact(fact_id: str, **fields) -> Fact:
    return Fact(
        id=fact_id,
        category=FactCategory.workflow,
        content=f"Memory {fact_id}",
        **fields,
    )


def candidate(candidate_id: str) -> MemoryCandidate:
    return MemoryCandidate(
        id=candidate_id,
        category=FactCategory.workflow,
        content=f"Candidate {candidate_id}",
    )


def test_nested_snapshots_and_search_results_cannot_mutate_store(tmp_path):
    store = FactStore(tmp_path)
    original = fact(
        "one",
        tags=["original"],
        brief_source_revisions={"source": datetime.now(timezone.utc)},
    )
    store.append_facts([original])
    first = store.load_facts()[0]
    first.content = "Changed"
    first.tags.append("extra")
    assert first.brief_source_revisions is not None
    first.brief_source_revisions.clear()
    hit = store.search_facts("Memory")[0]
    hit.fact.tags.clear()
    index = store.search_index()
    index.facts[0].content = "Changed index"
    loaded = store.load_facts()[0]
    assert loaded.content == original.content
    assert loaded.tags == ["original"]
    assert loaded.brief_source_revisions == original.brief_source_revisions
    assert store.search_facts("Memory")[0].fact.content == original.content


def test_restore_only_reverses_forgetting_and_preserves_confidence(tmp_path):
    store = FactStore(tmp_path)
    store.append_facts([fact("forgotten", confidence=0.6), fact("replaced")])
    store.forget("forgotten")
    assert store.restore_fact("forgotten").confidence == 0.6
    replacement = store.correct_fact("replaced", "Replacement memory")
    assert replacement is not None
    assert store.restore_fact("replaced") is None
    replaced = next(row for row in store.load_facts() if row.id == "replaced")
    assert replaced.lifecycle is FactLifecycle.superseded
    assert replaced.confidence == 0.0
    assert not is_fact_eligible(replaced)
    raw, active = materialize_events(store._load_all_events())["replaced"]
    assert raw.confidence == 1.0
    assert not active
    store.compact_event_log()
    assert store.restore_fact("replaced") is None


@pytest.mark.parametrize(
    "fields",
    [{"confidence": 2.0}, {"confidence": -0.1}, {"id": "other"}, {"tags": "text"}],
)
def test_invalid_edit_never_creates_intent_or_changes_bytes(tmp_path, fields):
    store = FactStore(tmp_path)
    store.append_facts([fact("one")])
    before = store.facts_path.read_bytes()
    with pytest.raises(ValidationError):
        store.update_fact("one", **fields)
    assert store.facts_path.read_bytes() == before
    assert not store.intent_path.exists()


@pytest.mark.parametrize(
    "fields",
    [
        {"supersede": {"one": "two"}, "stale": {"one": "old"}},
        {"supersede": {"one": "one"}},
        {"stale": {"one": "old"}, "unstale": ["one"]},
    ],
)
def test_contradictory_changes_are_rejected(fields):
    with pytest.raises(ValueError, match="ChangeSet"):
        ChangeSet(**fields)


def test_expected_read_guard_aborts_new_fact_creation(tmp_path):
    store = FactStore(tmp_path)
    source = fact("source")
    store.append_facts([source])
    store.update_fact("source", content="Changed")
    result = store.apply_changes(
        ChangeSet(new_facts=[fact("brief")], expected={source.id: source.updated_at}),
        all_or_nothing=True,
    )
    assert result.skipped == {"source": "changed since it was read"}
    assert {row.id for row in store.load_facts()} == {"source"}


def test_recovery_reuses_exact_events_and_replays_twice(tmp_path, monkeypatch):
    store = FactStore(tmp_path)
    original = fact("old")
    store.append_facts([original])
    append = store._append_events_raw
    captured: list[FactEvent] = []

    def interrupted(events):
        captured.extend(events)
        append(events[:1])
        raise OSError("interrupted append")

    monkeypatch.setattr(store, "_append_events_raw", interrupted)
    with pytest.raises(OSError, match="interrupted append"):
        store.apply_changes(
            ChangeSet(new_facts=[fact("new")], supersede={"old": "new"})
        )
    prepared = json.loads(store.intent_path.read_text())
    assert [row["event_id"] for row in prepared] == [
        event.event_id for event in captured
    ]
    reader = FactStore(tmp_path)
    with pytest.raises(RuntimeError, match="recover_transactions"):
        reader.load_facts()
    reader.recover_transactions()
    after = reader.facts_path.read_bytes()
    reader.recover_transactions()
    assert reader.facts_path.read_bytes() == after
    assert [row.id for row in reader.load_active_facts()] == ["new"]
    by_id = {
        event.event_id: event.model_dump(mode="json")
        for event in reader._load_all_events()
    }
    for row in prepared:
        assert by_id[row["event_id"]] == row


def test_partial_json_tail_remains_visible_until_explicit_repair(tmp_path, monkeypatch):
    store = FactStore(tmp_path)
    store.append_facts([fact("one")])

    def interrupted(events):
        with store.facts_path.open("a") as stream:
            stream.write(events[0].model_dump_json()[:40])
        raise OSError("partial row")

    monkeypatch.setattr(store, "_append_events_raw", interrupted)
    with pytest.raises(OSError, match="partial row"):
        store.append_facts([fact("two")])
    reader = FactStore(tmp_path)
    with pytest.raises(StoreIntegrityError, match="invalid event record"):
        reader.recover_transactions()
    assert reader.intent_path.exists()
    assert reader.repair()["facts_corrupt"] == 1
    reader.recover_transactions()
    assert {row.id for row in reader.load_facts()} == {"one", "two"}
    assert len(reader._load_all_events()) == 2


def test_reader_waits_for_complete_batch(tmp_path, monkeypatch):
    writer = FactStore(tmp_path)
    reader = FactStore(tmp_path)
    writer.append_facts([fact("old")])
    partial = threading.Event()
    release = threading.Event()
    read_done = threading.Event()
    observed = []
    errors = []
    append = writer._append_events_raw

    def pause_between_events(events):
        append(events[:1])
        partial.set()
        assert release.wait(2)
        append(events[1:])

    def write():
        try:
            writer.apply_changes(
                ChangeSet(new_facts=[fact("new")], supersede={"old": "new"})
            )
        except BaseException as exc:
            errors.append(exc)

    def read():
        observed.extend(reader.load_active_facts())
        read_done.set()

    monkeypatch.setattr(writer, "_append_events_raw", pause_between_events)
    writing = threading.Thread(target=write)
    writing.start()
    assert partial.wait(2)
    reading = threading.Thread(target=read)
    reading.start()
    try:
        assert not read_done.wait(0.05)
    finally:
        release.set()
        writing.join(2)
        reading.join(2)
    assert errors == []
    assert [row.id for row in observed] == ["new"]


def test_readonly_open_and_reads_create_no_files(tmp_path):
    path = tmp_path / "absent"
    store = FactStore.open_readonly(path)
    assert store.load_facts() == []
    assert store.load_candidates() == []
    assert not path.exists()


def test_unknown_prepared_transaction_stays_unresolved(tmp_path):
    store = FactStore(tmp_path)
    transaction = StoreTransaction(type="unknown", status=TransactionStatus.prepared)
    store._append_transaction(transaction)
    before = store.transaction_log_path.read_bytes()
    reopened = FactStore(tmp_path)
    assert reopened.transaction_log_path.read_bytes() == before
    with pytest.raises(ValueError, match="Unsupported transaction"):
        reopened.recover_transactions()
    assert reopened.transaction_log_path.read_bytes() == before
    assert [row.id for row in reopened._pending_transactions()] == [transaction.id]


def test_legacy_approval_recovery_keeps_original_fact_identity(tmp_path):
    store = FactStore(tmp_path)
    row = candidate("one")
    store.append_candidates([row])
    promoted = fact("legacy-random-id")
    legacy = StoreTransaction(
        type="approve_candidates",
        status=TransactionStatus.prepared,
        candidate_ids=[row.id],
        new_facts=[promoted],
        candidate_updates={row.id: {"status": "approved"}},
    )
    store._append_transaction(legacy)
    with pytest.raises(RuntimeError, match="recover_transactions"):
        FactStore(tmp_path).load_facts()
    store.recover_transactions()
    after = store.facts_path.read_bytes()
    store.recover_transactions()
    assert store.facts_path.read_bytes() == after
    assert [row.id for row in store.load_facts()] == [promoted.id]
    assert store.load_candidates()[0].status is CandidateStatus.approved


def union_file(name: str, base: Path, left: Path, right: Path, target: Path):
    result = subprocess.run(
        [
            "git",
            "merge-file",
            "--union",
            "--stdout",
            str(left / name),
            str(base / name),
            str(right / name),
        ],
        capture_output=True,
        check=True,
    )
    (target / name).write_bytes(result.stdout)


def offline_stores(tmp_path):
    base = tmp_path / "base"
    store = FactStore(base)
    store.append_candidates([candidate("one"), candidate("two")])
    store.facts_path.write_text(
        EventLogMeta().model_dump_json(exclude_none=True) + "\n"
    )
    left = tmp_path / "left"
    right = tmp_path / "right"
    shutil.copytree(base, left)
    shutil.copytree(base, right)
    merged = tmp_path / "merged"
    merged.mkdir()
    return base, FactStore(left), FactStore(right), FactStore(merged)


def test_union_merge_disjoint_approval_and_rejection_materializes_once(tmp_path):
    base, left, right, merged = offline_stores(tmp_path)
    left.approve_candidates(["one"])
    right.reject_candidates(["two"], "not useful")
    for name in ("facts.jsonl", "candidates.jsonl"):
        union_file(name, base, left.data_dir, right.data_dir, merged.data_dir)
    rows = {row.id: row for row in merged.load_candidates()}
    assert len(rows) == 2
    assert rows["one"].status is CandidateStatus.approved
    assert rows["two"].status is CandidateStatus.rejected
    assert merged.load_candidates(status=CandidateStatus.pending) == []
    assert merged.approve_candidates(["one", "two"]) == []
    assert len(merged.load_facts()) == 1


def test_same_revision_offline_approvals_converge_to_one_fact(tmp_path):
    base, left, right, merged = offline_stores(tmp_path)
    left_fact = left.approve_candidates(["one"])[0]
    right_fact = right.approve_candidates(["one"])[0]
    assert left_fact.id == right_fact.id
    assert left_fact.model_dump() == right_fact.model_dump()
    for name in ("facts.jsonl", "candidates.jsonl"):
        union_file(name, base, left.data_dir, right.data_dir, merged.data_dir)
    assert len(merged.load_facts()) == 1
    assert len(merged.load_candidates(status=CandidateStatus.approved)) == 1
    assert merged.approve_candidates(["one"]) == []


def test_conflicting_offline_decisions_are_visible_and_not_pending(tmp_path):
    base, left, right, merged = offline_stores(tmp_path)
    left.approve_candidates(["one"])
    right.reject_candidates(["one"], "rejected offline")
    union_file("candidates.jsonl", base, left.data_dir, right.data_dir, merged.data_dir)
    conflicted = merged.load_candidates(status=CandidateStatus.conflicted)
    assert len(conflicted) == 1
    assert conflicted[0].id == "one"
    assert merged.approve_candidates(["one"]) == []


def test_pending_dedup_reads_entire_queue_and_allows_rejected_reproposal(tmp_path):
    store = FactStore(tmp_path)
    rows = [candidate(str(number)) for number in range(250)]
    store.append_candidates(rows)
    assert (
        store.append_pending_candidates(
            [rows[0].model_copy(update={"id": "duplicate"})]
        )
        == []
    )
    store.reject_candidates([rows[0].id])
    assert (
        len(store.append_pending_candidates([rows[0].model_copy(update={"id": "new"})]))
        == 1
    )


def test_cached_search_excludes_newly_expired_fact(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    store = FactStore(tmp_path)
    store.append_facts([fact("one", expires_at=now + timedelta(minutes=1))])
    assert store.search_facts("Memory")

    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return now + timedelta(minutes=2)

    monkeypatch.setattr(store_module, "datetime", Later)
    assert store.search_facts("Memory") == []


@pytest.mark.parametrize("method", ["purge", "compact_event_log"])
def test_synced_store_refuses_history_rewrites(tmp_path, method):
    store = FactStore(tmp_path)
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="coordinated offline"):
        getattr(store, method)()


def test_data_snapshot_commits_only_allowlisted_files(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    git("init", "-b", "main")
    # Git identity applies to subprocesses invoked by the sync implementation too.
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    (tmp_path / "seed").write_text("seed")
    git("add", "seed")
    git("commit", "-m", "seed")
    FactStore(tmp_path).append_facts([fact("one")])
    (tmp_path / "unrelated").write_text("unrelated")
    git("add", "unrelated")
    sync_module._commit_data_snapshot(tmp_path, 5)
    assert git("show", "--pretty=", "--name-only", "HEAD").strip() == "facts.jsonl"
    assert git("diff", "--cached", "--name-only").strip() == "unrelated"


@pytest.mark.asyncio
async def test_cancelled_sync_waits_for_worker_before_returning(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    stopped = threading.Event()

    def blocked_sync(*args, **kwargs):
        started.set()
        assert release.wait(2)
        stopped.set()
        return {"status": "ok"}

    monkeypatch.setattr(sync_module, "sync", blocked_sync)
    task = asyncio.create_task(sync_module.run_sync(tmp_path, timeout=1))
    await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


@pytest.mark.parametrize(
    "row", ["[]", "null", "42", '{"event_type":"edited","fact_id":"one","payload":[]}']
)
def test_valid_json_with_invalid_event_shape_has_actionable_integrity_error(
    tmp_path, row
):
    store = FactStore(tmp_path)
    store.append_facts([fact("one")])
    with store.facts_path.open("a") as stream:
        stream.write(row + "\n")
    with pytest.raises(StoreIntegrityError) as error:
        store.load_facts()
    assert error.value.line == 3
    readonly = FactStore.open_readonly(tmp_path)
    assert len(readonly.load_facts()) == 1
    assert store.repair()["facts_corrupt"] == 1
    assert len(store.load_facts()) == 1


@pytest.mark.parametrize("category", [[], {}])
def test_malformed_edit_category_is_diagnosable_and_repairable(tmp_path, category):
    from engram.maintenance.doctor import _check_jsonl_integrity

    store = FactStore(tmp_path)
    store.append_facts([fact("one")])
    with store.facts_path.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "event_type": "edited",
                    "fact_id": "one",
                    "payload": {"category": category},
                }
            )
            + "\n"
        )
    with pytest.raises(StoreIntegrityError) as error:
        store.load_facts()
    assert error.value.line == 3
    issues = []
    assert _check_jsonl_integrity(store.facts_path, "facts", FactEvent, issues) == 1
    assert len(issues) == 1
    assert store.repair()["facts_corrupt"] == 1
    assert store.load_facts()[0].category is FactCategory.workflow


def test_first_writer_headers_union_merge_without_corruption(tmp_path):
    from engram.maintenance.doctor import _check_jsonl_integrity

    left = FactStore(tmp_path / "left")
    right = FactStore(tmp_path / "right")
    left.append_facts([fact("left")])
    right.append_facts([fact("right")])
    assert (
        left.facts_path.read_text().splitlines()[0]
        == right.facts_path.read_text().splitlines()[0]
    )
    base = tmp_path / "empty"
    base.write_text("")
    merged = subprocess.run(
        [
            "git",
            "merge-file",
            "--union",
            "-p",
            str(left.facts_path),
            str(base),
            str(right.facts_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    left.facts_path.write_text(merged.stdout)
    assert left.facts_path.read_text().splitlines()[0] == '{"meta":"event-log-v1"}'
    assert {row.id for row in left.load_facts()} == {"left", "right"}
    issues = []
    assert _check_jsonl_integrity(left.facts_path, "facts", FactEvent, issues) == 2
    assert not issues
    assert left.repair()["facts_corrupt"] == 0


def test_legacy_duplicate_v1_sentinels_are_not_corrupt_rows(tmp_path):
    from engram.maintenance.doctor import _check_jsonl_integrity

    store = FactStore(tmp_path)
    store.append_facts([fact("one")])
    before = store.facts_path.read_bytes()
    with store.facts_path.open("a") as stream:
        stream.write('{"meta":"event-log-v1","migrated_at":"2025-01-02T00:00:00Z"}\n')
    store.append_facts([fact("two")])
    assert store.facts_path.read_bytes().startswith(before)
    assert len(store.load_facts()) == 2
    issues = []
    assert _check_jsonl_integrity(store.facts_path, "facts", FactEvent, issues) == 2
    assert not issues
    assert store.repair()["facts_corrupt"] == 0


@pytest.mark.parametrize(
    "header", ['{"meta":"event-log-v2"}', "{}", '{"created_at":"2025-01-02T00:00:00Z"}']
)
def test_unknown_or_missing_header_marker_is_rejected(tmp_path, header):
    store = FactStore(tmp_path)
    store.facts_path.write_text(header + "\n")
    with pytest.raises(StoreIntegrityError):
        store.load_facts()
    with pytest.raises(ValueError, match="sentinel"):
        store.repair()
    assert store.facts_path.read_text() == header + "\n"


def test_correction_preserves_full_source_metadata(tmp_path):
    from engram.core.models import Durability

    now = datetime.now(timezone.utc)
    original = fact(
        "one",
        durability=Durability.ephemeral,
        expires_at=now + timedelta(days=2),
        effective_at=now - timedelta(days=1),
        anchors=["src/main.py"],
        suspect_reason="missing symbol",
        first_observed_at=now - timedelta(days=10),
        source_group_id="source",
        consolidates=["ancestor"],
        retrieval_hints=["hint"],
        brief_source_revisions={"ancestor": now},
    )
    store = FactStore(tmp_path)
    store.append_facts([original])
    replacement = store.correct_fact("one", "Correct content", reason="Correction")
    assert replacement is not None
    overridden = {
        "id",
        "content",
        "created_at",
        "updated_at",
        "observed_at",
        "supersedes",
        "why_store",
    }
    assert replacement.model_dump(exclude=overridden) == original.model_dump(
        exclude=overridden
    )
    assert (
        next(row for row in store.load_facts() if row.id == replacement.id).model_dump()
        == replacement.model_dump()
    )


def test_approval_edits_are_owned_by_journal_and_recovered_exactly(
    tmp_path, monkeypatch
):
    store = FactStore(tmp_path)
    row = candidate("one")
    store.append_candidates([row])

    def interrupt(transaction):
        raise OSError("simulated crash after prepared journal")

    monkeypatch.setattr(store, "_apply_approval_transaction", interrupt)
    with pytest.raises(OSError):
        store.approve_candidates_result(
            [row.id],
            edits={row.id: "Edited content"},
        )
    journal = store._pending_transactions()[0]
    assert (
        journal.new_facts[0].content
        == journal.candidates[0].content
        == "Edited content"
    )
    recovered = FactStore(tmp_path)
    assert recovered.recover_transactions() == 1
    approved = recovered.load_candidates()[0]
    assert approved.status is CandidateStatus.approved
    assert approved.content == recovered.load_facts()[0].content == "Edited content"


def test_losing_approval_cannot_edit_already_approved_candidate(tmp_path):
    store = FactStore(tmp_path)
    row = candidate("one")
    store.append_candidates([row])
    first = store.approve_candidates_result([row.id])
    result = store.approve_candidates_result([row.id], edits={row.id: "Losing edit"})
    assert result.edited == []
    assert result.approved == {}
    assert list(result.unresolved) == [row.id]
    assert store.load_candidates()[0].content == first.facts[0].content
    assert store.load_facts()[0].content == row.content


def test_approval_reports_transaction_owned_applied_edits(tmp_path):
    store = FactStore(tmp_path)
    row = candidate("one")
    store.append_candidates([row])
    approved = store.approve_candidates_result([row.id], edits={row.id: "Final edit"})
    assert approved.edited == [row.id]
    assert approved.approved == {row.id: approved.facts[0].id}
    assert (
        store.load_candidates()[0].content == approved.facts[0].content == "Final edit"
    )


@pytest.mark.parametrize(
    "change", ["edit", "expire", "stale", "incomplete", "forgotten"]
)
def test_approval_keeps_changed_replacement_sources_unresolved(tmp_path, change):
    source = fact("source")
    store = FactStore(tmp_path)
    store.append_facts([source])
    row = candidate("replacement").model_copy(
        update={
            "replaces": [source.id],
            "replaces_expected": {}
            if change == "incomplete"
            else {source.id: source.updated_at},
        }
    )
    store.append_candidates([row])
    if change == "edit":
        store.update_fact(source.id, content="Changed source")
    elif change == "expire":
        store.update_fact(
            source.id, expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
        )
    elif change == "stale":
        store.mark_stale(source.id)
    elif change == "forgotten":
        store.forget(source.id)
    result = store.approve_candidates_result([row.id], edits={row.id: "Must not leak"})
    assert result.approved == {} and result.edited == []
    assert list(result.unresolved) == [row.id]
    assert store.load_candidates()[0].status is CandidateStatus.pending
    assert store.load_candidates()[0].content == row.content
    assert len(store.load_facts()) == 1


def test_legacy_promotion_hash_ignores_only_new_absent_dependency_field():
    import hashlib
    from engram.storage.candidates import candidate_promotion_id

    row = candidate("legacy")
    legacy = row.model_dump(
        mode="json",
        exclude={"status", "review_note", "updated_at", "replaces_expected"},
    )
    revision = hashlib.sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected = hashlib.sha256(f"candidate:{row.id}:{revision}".encode()).hexdigest()[
        :24
    ]
    assert candidate_promotion_id(row) == expected


@pytest.mark.parametrize(
    "fields",
    [
        {"stale": True},
        {"expires_at": datetime(2000, 1, 1, tzinfo=timezone.utc)},
        {"confidence": 0.05},
    ],
)
def test_required_dependencies_must_be_eligible_without_revision_change(
    tmp_path, fields
):
    source = fact("source", **fields)
    store = FactStore(tmp_path)
    store.append_facts([source])
    result = store.apply_changes(
        ChangeSet(
            new_facts=[fact("new")],
            expected={source.id: source.updated_at},
            require_eligible=True,
        ),
        all_or_nothing=True,
    )
    assert result.skipped == {source.id: "not an eligible fact"}
    assert len(store.load_facts()) == 1


def test_stale_restoration_keeps_default_changeset_behavior(tmp_path):
    source = fact("source", stale=True)
    store = FactStore(tmp_path)
    store.append_facts([source])
    result = store.apply_changes(
        ChangeSet(unstale=[source.id], expected={source.id: source.updated_at}),
        all_or_nothing=True,
    )
    assert result.unstaled == [source.id]


@pytest.mark.parametrize("filename", ["recall_log.jsonl", "maintenance_state.jsonl"])
def test_sync_validation_rejects_every_allowlisted_log(tmp_path, filename):
    (tmp_path / filename).write_text('{"invalid":true}\n')
    with pytest.raises(StoreIntegrityError) as error:
        sync_module._validate_sync_snapshot(tmp_path)
    assert error.value.path.name == filename


@pytest.mark.parametrize(
    "returncode,stdout", [(1, ""), (0, "invalid"), (0, ""), (0, "-1")]
)
def test_commit_count_errors_are_typed(tmp_path, monkeypatch, returncode, stdout):
    monkeypatch.setattr(
        sync_module,
        "_run_git",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], returncode, stdout, "failure"
        ),
    )
    with pytest.raises(sync_module.SyncError) as error:
        sync_module._count_commits("origin/main..HEAD", tmp_path, 5)
    assert error.value.code == "git_count_failed"


def test_empty_remote_inspection_uses_real_local_git_without_push(tmp_path):
    subprocess.run(
        ["git", "init", "--bare", str(tmp_path / "remote.git")],
        capture_output=True,
        check=True,
    )
    assert sync_module._remote_is_empty(tmp_path, str(tmp_path / "remote.git"), 5)
    with pytest.raises(sync_module.SyncError) as error:
        sync_module._remote_is_empty(tmp_path, str(tmp_path / "missing.git"), 5)
    assert error.value.code == "git_remote_inspection_failed"


@pytest.mark.parametrize("remote_refs", ["", "abcd\trefs/heads/other\n"])
def test_sync_bootstraps_only_verified_empty_remote_without_real_push(
    tmp_path, monkeypatch, remote_refs
):
    calls = []
    monkeypatch.setattr(sync_module, "_ensure_git_available", lambda: None)
    monkeypatch.setattr(sync_module, "_ensure_repo", lambda *args: None)
    monkeypatch.setattr(
        sync_module, "_resolve_remote_and_branch", lambda *args: ("origin", "main")
    )
    monkeypatch.setattr(sync_module, "_ensure_managed_repo_setup", lambda *args: None)
    monkeypatch.setattr(sync_module, "_commit_data_snapshot", lambda *args: None)

    def git(args, **kwargs):
        calls.append(args)
        if args[0] == "ls-remote":
            return subprocess.CompletedProcess(args, 0, remote_refs, "")
        if args[0] == "fetch":
            return subprocess.CompletedProcess(args, 1, "", "missing main")
        if args[0] == "rev-list":
            return subprocess.CompletedProcess(args, 0, "2\n", "")
        # The push path is intercepted here. This test never executes git push.
        return subprocess.CompletedProcess(args, 0, "localhead\n", "")

    monkeypatch.setattr(sync_module, "_run_git", git)
    if remote_refs:
        with pytest.raises(sync_module.SyncError) as error:
            sync_module._sync_locked(tmp_path)
        assert error.value.code == "git_fetch_failed"
        assert not any(args[0] == "push" for args in calls)
    else:
        result = sync_module._sync_locked(tmp_path)
        assert result["pushed_commits"] == 2
        assert ["rev-list", "--count", "HEAD"] in calls
        assert ["push", "origin", "main"] in calls
        assert not any(args[0] in {"fetch", "pull"} for args in calls)


@pytest.mark.asyncio
async def test_operation_losing_approval_reports_no_edits_or_candidate_mutation(
    tmp_path, monkeypatch
):
    from engram import operations
    from engram.storage.store import AsyncFactStore

    store = FactStore(tmp_path)
    row = candidate("one")
    store.append_candidates([row])
    facade = AsyncFactStore(store)
    approve = facade.approve_candidates_result

    async def competing_approval(*args, **kwargs):
        store.approve_candidates([row.id])
        return await approve(*args, **kwargs)

    monkeypatch.setattr(facade, "approve_candidates_result", competing_approval)
    result = await operations.approve_candidates(
        [row.id], edits={row.id: "Losing edit"}, store=facade
    )
    assert result.exit_code == operations.EXIT_NOT_FOUND
    assert store.load_candidates()[0].content == row.content
    assert store.load_facts()[0].content == row.content


@pytest.mark.parametrize("filename", ["recall_log.jsonl", "maintenance_state.jsonl"])
def test_sync_rejects_corruption_introduced_by_integration_before_publication(
    tmp_path, monkeypatch, filename
):
    calls = []
    monkeypatch.setattr(sync_module, "_ensure_git_available", lambda: None)
    monkeypatch.setattr(sync_module, "_ensure_repo", lambda *args: None)
    monkeypatch.setattr(
        sync_module, "_resolve_remote_and_branch", lambda *args: ("origin", "main")
    )
    monkeypatch.setattr(sync_module, "_ensure_managed_repo_setup", lambda *args: None)
    monkeypatch.setattr(sync_module, "_commit_data_snapshot", lambda *args: None)

    def git(args, **kwargs):
        calls.append(args)
        if args[0] == "push":
            raise AssertionError("Corrupt integration must never reach publication")
        if args[0] == "pull":
            (tmp_path / filename).write_text('{"invalid":true}\n')
        return subprocess.CompletedProcess(args, 0, "localhead\n", "")

    monkeypatch.setattr(sync_module, "_run_git", git)
    with pytest.raises(StoreIntegrityError) as error:
        sync_module._sync_locked(tmp_path)
    assert error.value.path.name == filename
    assert ["pull", "--rebase", "origin", "main"] in calls


@pytest.mark.parametrize(
    "fields",
    [
        {"stale": True},
        {"expires_at": datetime(2000, 1, 1, tzinfo=timezone.utc)},
        {"confidence": 0.05},
    ],
)
def test_approval_rejects_ineligible_sources_with_matching_revisions(tmp_path, fields):
    source = fact("source", **fields)
    store = FactStore(tmp_path)
    store.append_facts([source])
    row = candidate("replacement").model_copy(
        update={
            "replaces": [source.id],
            "replaces_expected": {source.id: source.updated_at},
        }
    )
    store.append_candidates([row])
    result = store.approve_candidates_result([row.id])
    assert not result.facts
    assert list(result.unresolved) == [row.id]


@pytest.mark.parametrize("guarded", [True, False])
def test_replacement_approval_accepts_matching_sources_and_legacy_proposals(
    tmp_path, guarded
):
    source = fact("source")
    store = FactStore(tmp_path)
    store.append_facts([source])
    row = candidate("replacement").model_copy(
        update={
            "replaces": [source.id],
            "replaces_expected": {source.id: source.updated_at} if guarded else None,
        }
    )
    store.append_candidates([row])
    if not guarded:
        store.update_fact(source.id, content="Legacy proposals do not check revisions")
    result = store.approve_candidates_result([row.id])
    assert result.approved == {row.id: result.facts[0].id}
    assert not result.unresolved
    loaded = {item.id: item for item in store.load_facts()}
    assert loaded[source.id].superseded_by == result.facts[0].id


def test_repair_preserves_absent_header_date_and_real_legacy_date(tmp_path):
    for dirname, header in [
        ("canonical", '{"meta":"event-log-v1"}'),
        ("legacy", '{"meta":"event-log-v1","migrated_at":"2025-01-02T00:00:00Z"}'),
    ]:
        store = FactStore(tmp_path / dirname)
        store.facts_path.write_text(header + "\ninvalid row\n")
        assert store.repair()["facts_corrupt"] == 1
        saved = json.loads(store.facts_path.read_text())
        if dirname == "canonical":
            assert saved == {"meta": "event-log-v1"}
        else:
            assert saved["created_at"] == "2025-01-02T00:00:00Z"


@pytest.mark.parametrize(
    "status",
    [
        None,
        CandidateStatus.approved,
        CandidateStatus.rejected,
        CandidateStatus.conflicted,
    ],
)
def test_unavailable_edit_aborts_entire_approval_before_journal(tmp_path, status):
    store = FactStore(tmp_path)
    valid = candidate("valid")
    rows = [valid]
    if status is not None:
        rows.append(candidate("unavailable").model_copy(update={"status": status}))
    store.append_candidates(rows)
    before = store.candidates_path.read_bytes()
    result = store.approve_candidates_result(
        ["valid", "unavailable"],
        edits={"valid": "Must not change", "unavailable": "Unavailable edit"},
    )
    assert result.invalid_edit_ids == ["unavailable"]
    assert not result.facts and not result.edited and not result.approved
    assert set(result.unresolved) == {"valid", "unavailable"}
    assert store.candidates_path.read_bytes() == before
    assert not store.facts_path.exists()
    assert not store.transaction_log_path.exists()


@pytest.mark.parametrize("guarded", [True, False])
def test_overlapping_replacement_group_conflicts_but_independent_approval_succeeds(
    tmp_path, guarded
):
    store = FactStore(tmp_path)
    one, two = fact("source-one"), fact("source-two")
    store.append_facts([one, two])
    rows = []
    for candidate_id, sources in [
        ("a", [one]),
        ("b", [one, two]),
        ("c", [two]),
        ("independent", []),
    ]:
        rows.append(
            candidate(candidate_id).model_copy(
                update={
                    "replaces": [source.id for source in sources],
                    "replaces_expected": {
                        source.id: source.updated_at for source in sources
                    }
                    if guarded
                    else None,
                }
            )
        )
    store.append_candidates(rows)
    result = store.approve_candidates_result(
        ["a", "b", "c", "independent"],
        edits={"a": "Conflicted edit", "independent": "Approved edit"},
    )
    assert list(result.approved) == ["independent"]
    assert result.edited == ["independent"]
    assert set(result.unresolved) == {"a", "b", "c"}
    for candidate_id in ["a", "b"]:
        assert "source-one" in result.unresolved[candidate_id]
        assert "a, b" in result.unresolved[candidate_id]
    for candidate_id in ["b", "c"]:
        assert "source-two" in result.unresolved[candidate_id]
        assert "b, c" in result.unresolved[candidate_id]
    saved = {row.id: row for row in store.load_candidates()}
    assert saved["a"].content == "Candidate a"
    assert all(
        saved[candidate_id].status is CandidateStatus.pending
        for candidate_id in ["a", "b", "c"]
    )
    assert saved["independent"].status is CandidateStatus.approved
    assert saved["independent"].content == "Approved edit"
    facts = {row.id: row for row in store.load_facts()}
    assert set(facts) == {one.id, two.id, result.approved["independent"]}
    assert facts[one.id].lifecycle is FactLifecycle.active
    assert facts[two.id].lifecycle is FactLifecycle.active


def test_overlapping_supersedes_candidates_preserve_list_wrapper(tmp_path):
    source = fact("source")
    store = FactStore(tmp_path)
    store.append_facts([source])
    store.append_candidates(
        [
            candidate(candidate_id).model_copy(
                update={
                    "supersedes": source.id,
                    "replaces_expected": {source.id: source.updated_at},
                }
            )
            for candidate_id in ["a", "b"]
        ]
    )
    before = store.facts_path.read_bytes()
    assert store.approve_candidates(["a", "b"]) == []
    assert store.facts_path.read_bytes() == before
    assert all(row.status is CandidateStatus.pending for row in store.load_candidates())
    assert not store.transaction_log_path.exists()


@pytest.mark.asyncio
async def test_operation_exposes_competing_replacement_conflicts(tmp_path):
    from engram import operations

    source = fact("source")
    store = FactStore(tmp_path)
    store.append_facts([source])
    store.append_candidates(
        [
            candidate(candidate_id).model_copy(
                update={
                    "replaces": [source.id],
                    "replaces_expected": {source.id: source.updated_at},
                }
            )
            for candidate_id in ["a", "b"]
        ]
        + [candidate("independent")]
    )
    result = await operations.approve_candidates(["a", "b", "independent"], store=store)
    assert result.envelope.status.value == "partial"
    assert result.envelope.data["applied_ids"] == ["independent"]
    assert result.envelope.data["unresolved_ids"] == ["a", "b"]
    assert len(result.envelope.errors) == 2
    for error in result.envelope.errors:
        assert error.code.value == "conflict"
        assert "source" in error.message and "a, b" in error.message
