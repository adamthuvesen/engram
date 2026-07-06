"""Git-backed sync for the Engram data directory.

The Engram data directory (``~/.engram/data/`` by default) is intentionally
JSONL-only, so cross-machine sync is just a thin wrapper over the ``git``
binary. We shell out rather than embedding a git library — the system ``git``
is portable, well-tested, and the user already knows how to debug it.

This module is import-safe: it does not invoke git at import time. All git
calls happen inside :func:`sync` and return a structured result the caller
(CLI, MCP tool, or background task) can inspect.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


SYNC_STATE_FILENAME = ".engram-sync-state"
COMPACTION_SENTINEL_FILENAME = ".engram-compaction-in-progress"
DEFAULT_GIT_TIMEOUT_SECONDS = 30.0


# Git merge driver to use for the event-log files. ``union`` makes git
# concatenate both versions of a three-way merge instead of producing a
# conflict, which is exactly the right semantic for an append-only event
# log where every line is independently meaningful.
GITATTRIBUTES_MARKER = "# engram-sync: managed merge attributes"
GITATTRIBUTES_LINES = [
    GITATTRIBUTES_MARKER,
    "facts.jsonl merge=union",
    "ingestion_log.jsonl merge=union",
    "recall_log.jsonl merge=union",
    "transactions.jsonl merge=union",
    "candidates.jsonl merge=union",
]

# Files engram writes inside the data dir that should NEVER be tracked by
# git: inter-process lock sidecars and per-machine sync state.
GITIGNORE_MARKER = "# engram-sync: managed ignores"
GITIGNORE_PATTERNS = [
    "*.lock",
    ".engram-sync-state",
    ".engram-compaction-in-progress",
]
GITIGNORE_LINES = [GITIGNORE_MARKER, *GITIGNORE_PATTERNS]


class SyncError(Exception):
    """Raised for sync failures that should surface as a non-zero exit."""

    def __init__(self, code: str, message: str, *, git_stderr: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.git_stderr = git_stderr


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    timeout: float,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a git subcommand with a timeout, capturing stdout/stderr as text."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _require_git_success(
    proc: subprocess.CompletedProcess[str],
    *,
    code: str,
    message: str,
) -> None:
    if proc.returncode != 0:
        raise SyncError(code=code, message=message, git_stderr=proc.stderr)


def _ensure_git_available() -> None:
    if shutil.which("git") is None:
        raise SyncError(
            code="git_not_found",
            message=(
                "`git` binary not found on PATH. Install git and re-run "
                "(see https://git-scm.com/downloads)."
            ),
        )


def _ensure_repo(data_dir: Path) -> None:
    if not (data_dir / ".git").exists():
        raise SyncError(
            code="not_a_git_repo",
            message=(
                f"{data_dir} is not a git repository. Run "
                f"`git -C {data_dir} init` and configure a remote first."
            ),
        )


def _resolve_remote_and_branch(data_dir: Path, timeout: float) -> tuple[str, str]:
    """Determine the remote name and current branch for the data dir repo.

    The remote name is fixed to ``origin`` if present (the convention for
    single-remote stores); otherwise the first remote returned by
    ``git remote`` is used. The branch is read from
    ``git symbolic-ref --short HEAD``, which fails on a detached HEAD.
    """
    remotes_proc = _run_git(["remote"], cwd=data_dir, timeout=timeout)
    if remotes_proc.returncode != 0:
        raise SyncError(
            code="git_remote_failed",
            message="Failed to list git remotes for the data directory.",
            git_stderr=remotes_proc.stderr,
        )
    remote_names = [name for name in remotes_proc.stdout.splitlines() if name.strip()]
    if not remote_names:
        raise SyncError(
            code="no_remote_configured",
            message=(
                "No git remote configured. Run "
                f"`git -C {data_dir} remote add origin <url>` and try again."
            ),
        )
    remote = "origin" if "origin" in remote_names else remote_names[0]

    branch_proc = _run_git(
        ["symbolic-ref", "--short", "HEAD"], cwd=data_dir, timeout=timeout
    )
    if branch_proc.returncode != 0:
        raise SyncError(
            code="detached_head",
            message=(
                "HEAD is detached or not on a branch. Checkout a branch in "
                f"{data_dir} before running sync."
            ),
            git_stderr=branch_proc.stderr,
        )
    branch = branch_proc.stdout.strip()
    if not branch:
        raise SyncError(
            code="detached_head",
            message="Could not resolve current branch.",
        )
    return remote, branch


def _count_commits(spec: str, cwd: Path, timeout: float) -> int:
    """Return the number of commits in ``git rev-list ... --count <spec>``."""
    proc = _run_git(["rev-list", "--count", spec], cwd=cwd, timeout=timeout)
    if proc.returncode != 0:
        return 0
    try:
        return int(proc.stdout.strip() or "0")
    except ValueError:
        return 0


def _write_sync_state(data_dir: Path, payload: dict[str, Any]) -> None:
    state_path = data_dir / SYNC_STATE_FILENAME
    state_path.write_text(json.dumps(payload, indent=2) + "\n")


def read_sync_state(data_dir: Path) -> dict[str, Any] | None:
    """Read the last-known sync state, or ``None`` if no sync has occurred."""
    state_path = data_dir / SYNC_STATE_FILENAME
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def is_compaction_in_progress(data_dir: Path) -> bool:
    """Return True when a compaction sentinel file is present in the data dir."""
    return (data_dir / COMPACTION_SENTINEL_FILENAME).exists()


def _ensure_managed_file(
    data_dir: Path,
    *,
    timeout: float,
    relative_path: str,
    marker: str,
    lines: list[str],
    commit_message: str,
) -> bool:
    """Helper: append a managed block to ``relative_path`` if missing.

    Returns True if a new commit was created.
    """
    target = data_dir / relative_path
    existing = target.read_text() if target.exists() else ""
    if marker in existing:
        return False

    new_block = "\n".join(lines) + "\n"
    if existing and not existing.endswith("\n"):
        existing += "\n"
    target.write_text(existing + new_block)

    add_proc = _run_git(["add", relative_path], cwd=data_dir, timeout=timeout)
    if add_proc.returncode != 0:
        if existing:
            target.write_text(existing)
        else:
            target.unlink(missing_ok=True)
        raise SyncError(
            code="managed_file_stage_failed",
            message=f"Failed to stage managed {relative_path}.",
            git_stderr=add_proc.stderr,
        )

    commit_proc = _run_git(
        ["commit", "-m", commit_message, "--", relative_path],
        cwd=data_dir,
        timeout=timeout,
    )
    if commit_proc.returncode != 0:
        raise SyncError(
            code="managed_file_commit_failed",
            message=f"Failed to commit managed {relative_path}.",
            git_stderr=commit_proc.stderr,
        )
    logger.info("engram-sync: wrote managed %s", relative_path)
    return True


def _commit_untrack(data_dir: Path, paths: list[str], timeout: float) -> bool:
    """Commit removal of tracked local-only files without other staged changes."""
    head_proc = _run_git(["rev-parse", "HEAD"], cwd=data_dir, timeout=timeout)
    _require_git_success(
        head_proc,
        code="untrack_locks_commit_failed",
        message="Failed to resolve HEAD before untracking engram state files.",
    )
    head_before = head_proc.stdout.strip()

    head_paths_proc = _run_git(
        ["ls-tree", "-r", "--name-only", "HEAD", "--", *paths],
        cwd=data_dir,
        timeout=timeout,
    )
    _require_git_success(
        head_paths_proc,
        code="untrack_locks_commit_failed",
        message="Failed to inspect tracked engram state files.",
    )
    head_paths = [p for p in head_paths_proc.stdout.splitlines() if p.strip()]
    new_commit = ""
    if head_paths:
        with tempfile.TemporaryDirectory(prefix="engram-sync-index-") as tmp:
            env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
            read_proc = _run_git(
                ["read-tree", "HEAD"], cwd=data_dir, timeout=timeout, env=env
            )
            _require_git_success(
                read_proc,
                code="untrack_locks_commit_failed",
                message="Failed to prepare temporary git index.",
            )
            remove_proc = _run_git(
                ["update-index", "--force-remove", "--", *head_paths],
                cwd=data_dir,
                timeout=timeout,
                env=env,
            )
            _require_git_success(
                remove_proc,
                code="untrack_locks_commit_failed",
                message="Failed to stage engram state file removals.",
            )
            tree_proc = _run_git(["write-tree"], cwd=data_dir, timeout=timeout, env=env)
            _require_git_success(
                tree_proc,
                code="untrack_locks_commit_failed",
                message="Failed to write temporary git tree.",
            )
            commit_proc = _run_git(
                [
                    "commit-tree",
                    tree_proc.stdout.strip(),
                    "-p",
                    head_before,
                    "-m",
                    "engram-sync: untrack lock and state files",
                ],
                cwd=data_dir,
                timeout=timeout,
            )
            _require_git_success(
                commit_proc,
                code="untrack_locks_commit_failed",
                message="Failed to commit untrack of engram lock/state files.",
            )
            new_commit = commit_proc.stdout.strip()

    rm_proc = _run_git(
        ["rm", "--cached", "--ignore-unmatch", *paths],
        cwd=data_dir,
        timeout=timeout,
    )
    _require_git_success(
        rm_proc,
        code="untrack_locks_failed",
        message="Failed to untrack engram lock/state files.",
    )

    if new_commit:
        update_ref_proc = _run_git(
            ["update-ref", "HEAD", new_commit, head_before],
            cwd=data_dir,
            timeout=timeout,
        )
        _require_git_success(
            update_ref_proc,
            code="untrack_locks_commit_failed",
            message="Failed to update HEAD after untracking engram state files.",
        )
        return True
    return False


def _untrack_lock_files(data_dir: Path, timeout: float) -> bool:
    """Remove any lock / state files that were previously tracked.

    Returns True if a new "remove tracked junk" commit was created.
    """
    tracked = _run_git(
        ["ls-files", *GITIGNORE_PATTERNS],
        cwd=data_dir,
        timeout=timeout,
    )
    if tracked.returncode != 0 or not tracked.stdout.strip():
        return False
    paths = [p for p in tracked.stdout.splitlines() if p.strip()]
    committed = _commit_untrack(data_dir, paths, timeout)
    logger.info("engram-sync: untracked %d lock/state file(s)", len(paths))
    return committed


def _ensure_managed_repo_setup(data_dir: Path, timeout: float) -> bool:
    """Ensure ``.gitattributes`` and ``.gitignore`` are set up for sync.

    Writes managed blocks if missing, untracks any lock/state files that
    crept into history before the .gitignore was in place. Returns True if
    any commit was created. Idempotent.
    """
    changed = False
    changed |= _ensure_managed_file(
        data_dir,
        timeout=timeout,
        relative_path=".gitignore",
        marker=GITIGNORE_MARKER,
        lines=GITIGNORE_LINES,
        commit_message="engram-sync: ignore lock and state files",
    )
    changed |= _untrack_lock_files(data_dir, timeout)
    changed |= _ensure_managed_file(
        data_dir,
        timeout=timeout,
        relative_path=".gitattributes",
        marker=GITATTRIBUTES_MARKER,
        lines=GITATTRIBUTES_LINES,
        commit_message="engram-sync: configure union merge for event log",
    )
    return changed


def sync(
    data_dir: Path,
    *,
    timeout: float = DEFAULT_GIT_TIMEOUT_SECONDS,
    skip_if_compacting: bool = True,
) -> dict[str, Any]:
    """Pull, rebase, and push the data directory against its configured remote.

    Returns a structured result on success. Raises :class:`SyncError` on any
    failure. The wall-clock elapsed time is always present in the result as
    ``took_ms``. Callers that want JSON output can pass the dict straight to
    ``json.dumps``.
    """
    started = time.monotonic()

    _ensure_git_available()
    _ensure_repo(data_dir)
    if skip_if_compacting and is_compaction_in_progress(data_dir):
        return {
            "status": "skipped",
            "reason": "compaction_in_progress",
            "pulled_commits": 0,
            "pushed_commits": 0,
            "took_ms": int((time.monotonic() - started) * 1000),
        }

    remote, branch = _resolve_remote_and_branch(data_dir, timeout)

    # Ensure ``.gitignore`` and ``.gitattributes`` are set up for sync — and
    # any pre-existing tracked lock/state files are untracked. This creates
    # one-shot setup commits on first sync; idempotent thereafter.
    _ensure_managed_repo_setup(data_dir, timeout)

    # Snapshot the current HEAD so we can count pulled commits afterwards.
    head_before_proc = _run_git(["rev-parse", "HEAD"], cwd=data_dir, timeout=timeout)
    head_before = head_before_proc.stdout.strip()

    fetch_proc = _run_git(["fetch", remote, branch], cwd=data_dir, timeout=timeout)
    if fetch_proc.returncode != 0:
        raise SyncError(
            code="git_fetch_failed",
            message=f"git fetch {remote} {branch} failed.",
            git_stderr=fetch_proc.stderr,
        )

    pull_proc = _run_git(
        ["pull", "--rebase", remote, branch], cwd=data_dir, timeout=timeout
    )
    if pull_proc.returncode != 0:
        raise SyncError(
            code="git_pull_failed",
            message=f"git pull --rebase {remote} {branch} failed.",
            git_stderr=pull_proc.stderr,
        )

    head_after_pull_proc = _run_git(
        ["rev-parse", "HEAD"], cwd=data_dir, timeout=timeout
    )
    head_after_pull = head_after_pull_proc.stdout.strip()
    pulled_commits = (
        0
        if head_before == head_after_pull
        else _count_commits(f"{head_before}..{head_after_pull}", data_dir, timeout)
    )

    pushed_commits = _count_commits(f"{remote}/{branch}..HEAD", data_dir, timeout)

    push_proc = _run_git(["push", remote, branch], cwd=data_dir, timeout=timeout)
    if push_proc.returncode != 0:
        raise SyncError(
            code="git_push_failed",
            message=f"git push {remote} {branch} failed.",
            git_stderr=push_proc.stderr,
        )

    took_ms = int((time.monotonic() - started) * 1000)
    result = {
        "status": "ok",
        "remote": remote,
        "branch": branch,
        "pulled_commits": pulled_commits,
        "pushed_commits": pushed_commits,
        "took_ms": took_ms,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_sync_state(data_dir, result)
    logger.info(
        "engram sync ok: pulled=%d pushed=%d (%s/%s) in %d ms",
        pulled_commits,
        pushed_commits,
        remote,
        branch,
        took_ms,
    )
    return result


async def auto_sync_loop(
    data_dir: Path,
    *,
    interval: float,
    timeout: float,
    on_result: Callable[[Any], None] | None = None,
) -> None:
    """Run ``sync`` every ``interval`` seconds until the task is cancelled.

    Errors are caught and reported via ``on_result`` so a single failure
    does not stop the loop. Cancellation is honored cooperatively.
    """
    logger.info(
        "engram-sync auto-loop started (interval=%.1fs, timeout=%.1fs)",
        interval,
        timeout,
    )
    try:
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            try:
                result = await asyncio.to_thread(sync, data_dir, timeout=timeout)
                if on_result is not None:
                    on_result(result)
            except SyncError as exc:
                logger.warning(
                    "engram-sync auto-loop: sync failed (%s): %s",
                    exc.code,
                    exc.message,
                )
                if on_result is not None:
                    on_result(exc)
            except Exception as exc:  # noqa: BLE001 — auto-loop must not die
                logger.exception("engram-sync auto-loop unexpected error: %s", exc)
                if on_result is not None:
                    on_result(SyncError(code="unexpected", message=str(exc)))
    except asyncio.CancelledError:
        logger.info("engram-sync auto-loop cancelled")
        raise


async def run_final_sync(
    data_dir: Path,
    *,
    timeout: float,
) -> dict[str, Any] | SyncError:
    """Run one synchronous sync on shutdown. Never raises."""
    try:
        return await asyncio.to_thread(sync, data_dir, timeout=timeout)
    except SyncError as exc:
        logger.warning(
            "engram-sync shutdown sync failed (%s): %s", exc.code, exc.message
        )
        return exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("engram-sync shutdown sync unexpected: %s", exc)
        return SyncError(code="shutdown_unexpected", message=str(exc))
