"""Tests for git-backed sync of the Engram data directory.

These tests stand up a local bare repository on the filesystem and exercise
the full ``engram sync`` code path: pull-rebase-push, conflict-free append
merges across two machines, error handling for missing remote / missing git,
and the ``.engram-sync-state`` artifact.

Skip the entire module if ``git`` is not on PATH — we shell out rather than
embed a library, and the system git is the only required dependency.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from engram.models import Fact, FactCategory
from engram.store import FactStore
from engram.sync import SYNC_STATE_FILENAME, SyncError, read_sync_state, sync

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git binary not available"
)


def _git(*args: str, cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _make_bare_repo(tmp: Path) -> Path:
    bare = tmp / "remote.git"
    _git("init", "--bare", "-b", "main", str(bare), cwd=tmp)
    return bare


def _init_clone(tmp: Path, name: str, bare: Path) -> Path:
    """Initialize a working clone of ``bare``.

    The first clone seeds the bare remote with an init commit; subsequent
    clones ``git clone`` from the bare remote so all peers share history.
    """
    clone = tmp / name
    # Has the bare remote already been seeded by an earlier clone?
    refs = subprocess.run(
        ["git", "ls-remote", str(bare)],
        capture_output=True,
        text=True,
        check=False,
    )
    already_seeded = bool(refs.stdout.strip())

    if already_seeded:
        _git("clone", "-b", "main", str(bare), str(clone), cwd=tmp)
        _git("config", "user.email", "engram@example.com", cwd=clone)
        _git("config", "user.name", "engram-test", cwd=clone)
        return clone

    clone.mkdir()
    _git("init", "-b", "main", str(clone), cwd=tmp)
    _git("remote", "add", "origin", str(bare), cwd=clone)
    _git("config", "user.email", "engram@example.com", cwd=clone)
    _git("config", "user.name", "engram-test", cwd=clone)
    (clone / "README.md").write_text("engram-test\n")
    _git("add", "README.md", cwd=clone)
    _git("commit", "-m", "init", cwd=clone)
    _git("push", "-u", "origin", "main", cwd=clone)
    return clone


def _commit_all(repo: Path, message: str) -> None:
    _git("add", "-A", cwd=repo)
    _git("commit", "-m", message, cwd=repo)


# ---------------------------------------------------------------------------
# Setup validation
# ---------------------------------------------------------------------------


def test_sync_errors_when_not_a_git_repo(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with pytest.raises(SyncError) as exc:
        sync(data_dir)
    assert exc.value.code == "not_a_git_repo"


def test_sync_errors_when_no_remote_configured(tmp_path: Path):
    repo = tmp_path / "data"
    repo.mkdir()
    _git("init", "-b", "main", str(repo), cwd=tmp_path)
    _git("config", "user.email", "engram@example.com", cwd=repo)
    _git("config", "user.name", "engram-test", cwd=repo)
    (repo / "seed").write_text("x")
    _git("add", "seed", cwd=repo)
    _git("commit", "-m", "init", cwd=repo)

    with pytest.raises(SyncError) as exc:
        sync(repo)
    assert exc.value.code == "no_remote_configured"


def test_sync_errors_when_git_missing(monkeypatch, tmp_path: Path):
    repo = tmp_path / "data"
    repo.mkdir()

    monkeypatch.setattr("engram.sync.shutil.which", lambda _: None)
    with pytest.raises(SyncError) as exc:
        sync(repo)
    assert exc.value.code == "git_not_found"


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_sync_no_op_when_already_up_to_date(tmp_path: Path):
    bare = _make_bare_repo(tmp_path)
    clone = _init_clone(tmp_path, "alice", bare)

    # First sync writes the managed .gitignore + .gitattributes setup commits.
    first = sync(clone)
    assert first["status"] == "ok"
    assert first["pushed_commits"] >= 1

    # Second sync, with nothing new, is a true no-op.
    second = sync(clone)
    assert second["status"] == "ok"
    assert second["pulled_commits"] == 0
    assert second["pushed_commits"] == 0
    state = read_sync_state(clone)
    assert state is not None
    assert state["status"] == "ok"
    assert (clone / SYNC_STATE_FILENAME).exists()


def test_sync_pushes_local_commits(tmp_path: Path):
    bare = _make_bare_repo(tmp_path)
    clone = _init_clone(tmp_path, "alice", bare)

    # Burn the .gitattributes setup commit first so we measure only the
    # user's actual append.
    sync(clone)

    store = FactStore(data_dir=clone)
    store.append_facts(
        [Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="x")]
    )
    _commit_all(clone, "add a fact")

    result = sync(clone)
    assert result["status"] == "ok"
    assert result["pushed_commits"] == 1
    assert result["pulled_commits"] == 0


def test_sync_pulls_remote_commits(tmp_path: Path):
    bare = _make_bare_repo(tmp_path)
    alice = _init_clone(tmp_path, "alice", bare)
    bob = _init_clone(tmp_path, "bob", bare)

    # Alice writes a fact and pushes.
    alice_store = FactStore(data_dir=alice)
    alice_store.append_facts(
        [Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="hello")]
    )
    _commit_all(alice, "alice fact")
    sync(alice)

    # Bob syncs — should pull Alice's commit (bob has nothing of its own).
    result = sync(bob)
    assert result["status"] == "ok"
    assert result["pulled_commits"] >= 1
    bob_store = FactStore(data_dir=bob)
    assert {f.id for f in bob_store.load_facts()} == {"aaaaaaaaaaaa"}


def test_sync_two_machines_disjoint_facts_merge_via_append(tmp_path: Path):
    """Two clones each append a distinct event after a shared starting point;
    after both sync, both materialize the union of facts via git's line-level
    three-way merge of the append-only event log.

    The realistic precondition this test mirrors: machine A initializes the
    store and pushes the first ``facts.jsonl``. Machine B clones the repo
    (so ``facts.jsonl`` already exists). Both then append independently and
    sync. Concurrent *creation* of ``facts.jsonl`` from an empty base is a
    separate "both added" git case handled by enabling ``merge=union`` via
    ``.gitattributes`` in the data dir — covered in the docs setup steps.
    """
    bare = _make_bare_repo(tmp_path)
    alice = _init_clone(tmp_path, "alice", bare)

    # Alice initializes the event log and pushes — this is the shared starting
    # point both peers diverge from.
    alice_store = FactStore(data_dir=alice)
    alice_store.append_facts(
        [Fact(id="000000000000", category=FactCategory.preference, content="seed")]
    )
    _commit_all(alice, "alice seed event")
    sync(alice)

    # Bob clones from the bare remote AFTER alice's seed push — so bob already
    # has alice's ``facts.jsonl`` when he starts.
    bob = _init_clone(tmp_path, "bob", bare)

    # Alice appends another event + pushes.
    alice_store.append_facts(
        [Fact(id="aaaaaaaaaaaa", category=FactCategory.preference, content="alice")]
    )
    _commit_all(alice, "alice fact")
    sync(alice)

    # Bob appends independently, then syncs (pulls + rebases + pushes).
    bob_store = FactStore(data_dir=bob)
    bob_store.append_facts(
        [Fact(id="bbbbbbbbbbbb", category=FactCategory.preference, content="bob")]
    )
    _commit_all(bob, "bob fact")
    bob_result = sync(bob)
    assert bob_result["status"] == "ok"
    assert bob_result["pulled_commits"] >= 1

    # Alice syncs again — picks up bob's pushed commit.
    alice_result = sync(alice)
    assert alice_result["pulled_commits"] >= 1

    expected_ids = {"000000000000", "aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert {f.id for f in FactStore(data_dir=alice).load_facts()} == expected_ids
    assert {f.id for f in FactStore(data_dir=bob).load_facts()} == expected_ids


# ---------------------------------------------------------------------------
# Sync state file
# ---------------------------------------------------------------------------


def test_sync_state_is_json_and_carries_completed_at(tmp_path: Path):
    bare = _make_bare_repo(tmp_path)
    clone = _init_clone(tmp_path, "alice", bare)
    sync(clone)
    state_path = clone / SYNC_STATE_FILENAME
    payload = json.loads(state_path.read_text())
    assert "completed_at" in payload
    assert payload["status"] == "ok"


def test_sync_state_unchanged_on_failure(monkeypatch, tmp_path: Path):
    """If sync fails, a prior sync-state file must be left intact."""
    bare = _make_bare_repo(tmp_path)
    clone = _init_clone(tmp_path, "alice", bare)
    initial = sync(clone)
    state_before = (clone / SYNC_STATE_FILENAME).read_text()

    # Force the next sync to fail by pointing the remote at a bogus URL.
    _git("remote", "set-url", "origin", str(tmp_path / "does-not-exist"), cwd=clone)
    with pytest.raises(SyncError):
        sync(clone, timeout=5.0)

    # Sync state file must not have been overwritten.
    assert (clone / SYNC_STATE_FILENAME).read_text() == state_before
    assert initial["status"] == "ok"


# ---------------------------------------------------------------------------
# Compaction sentinel
# ---------------------------------------------------------------------------


def test_sync_skips_when_compaction_sentinel_present(tmp_path: Path):
    from engram.sync import COMPACTION_SENTINEL_FILENAME

    bare = _make_bare_repo(tmp_path)
    clone = _init_clone(tmp_path, "alice", bare)
    (clone / COMPACTION_SENTINEL_FILENAME).write_text("in-progress")
    result = sync(clone)
    assert result["status"] == "skipped"
    assert result["reason"] == "compaction_in_progress"


# ---------------------------------------------------------------------------
# Auto-sync background loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_sync_loop_runs_sync_on_interval(tmp_path: Path):
    """The loop should call ``sync`` after each interval and surface results
    via ``on_result``.
    """
    import asyncio

    from engram.sync import SyncError, auto_sync_loop

    bare = _make_bare_repo(tmp_path)
    clone = _init_clone(tmp_path, "alice", bare)
    # Burn the setup-commit push first so subsequent syncs are fast no-ops.
    sync(clone)

    calls: list[object] = []
    task = asyncio.create_task(
        auto_sync_loop(clone, interval=0.05, timeout=5.0, on_result=calls.append)
    )

    # Each tick performs real git ops; give the loop a generous window to
    # accumulate at least two ticks.
    await asyncio.sleep(1.0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(calls) >= 2
    for value in calls:
        assert isinstance(value, (dict, SyncError))


@pytest.mark.asyncio
async def test_auto_sync_loop_continues_after_failure(tmp_path: Path, monkeypatch):
    """A sync error mid-loop must not stop the loop."""
    import asyncio

    from engram import sync as sync_module
    from engram.sync import SyncError, auto_sync_loop

    bare = _make_bare_repo(tmp_path)
    clone = _init_clone(tmp_path, "alice", bare)
    sync(clone)  # burn setup-commit push

    call_count = {"n": 0}
    real_sync = sync_module.sync

    def failing_sync(data_dir, *, timeout=30.0, skip_if_compacting=True):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise SyncError(code="injected", message="boom")
        return real_sync(
            data_dir, timeout=timeout, skip_if_compacting=skip_if_compacting
        )

    monkeypatch.setattr("engram.sync.sync", failing_sync)

    results: list[object] = []
    task = asyncio.create_task(
        auto_sync_loop(clone, interval=0.1, timeout=5.0, on_result=results.append)
    )
    await asyncio.sleep(0.45)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert any(isinstance(r, SyncError) for r in results)
    assert any(isinstance(r, dict) for r in results)
