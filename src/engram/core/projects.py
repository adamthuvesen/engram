"""Project scope canonicalization and repository discovery.

Agents pass a project as a bare name ("argmax") or as their working directory
("/Users/me/dev/argmax", or a git worktree of it). Both must land on the same
scope, so every project value is canonicalized at the boundary: paths resolve
to their git repository's name, names are trimmed and lowercased.

Resolved repository roots are cached per machine in ``projects.local.json``
(excluded from sync, since paths differ between machines). Upkeep uses them to
check fact anchors against the current code.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Collection
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOTS_FILE = "projects.local.json"
_GIT_TIMEOUT_SECONDS = 5.0
_REPO_SEARCH_DEPTH = 3


def _looks_like_path(value: str) -> bool:
    return value.startswith(("/", "~", "./", "../"))


def _git(path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def repo_root_for_path(path: Path) -> Path | None:
    """Return the main checkout root for ``path``, following git worktrees."""
    if not path.exists():
        return None
    common_dir = _git(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if common_dir:
        common = Path(common_dir)
        # A normal repo or worktree shares ``<main checkout>/.git``.
        if common.name == ".git":
            return common.parent
    toplevel = _git(path, "rev-parse", "--show-toplevel")
    return Path(toplevel) if toplevel else None


def normalize_project_name(value: str) -> str:
    """Canonical spelling of a bare project name."""
    return value.strip().lower()


def resolve_project(
    value: str | None, *, known: Collection[str] = ()
) -> tuple[str | None, Path | None]:
    """Resolve a caller-supplied project into ``(canonical_name, repo_root)``.

    ``repo_root`` is only known when the caller passed a path inside a git
    repository. For a path that no longer exists (e.g. a deleted worktree),
    ``known`` project names are matched against its components.
    """
    if value is None:
        return None, None
    text = value.strip()
    if not text:
        return None, None
    if not _looks_like_path(text):
        return normalize_project_name(text), None

    path = Path(os.path.expanduser(text))
    root = repo_root_for_path(path)
    if root is not None:
        return normalize_project_name(root.name), root
    if not path.exists() and known:
        name = _known_project_in_path(path, known)
        if name is not None:
            return name, None
    return normalize_project_name(path.name or text), None


def _known_project_in_path(path: Path, known: Collection[str]) -> str | None:
    """Match a known project against path parts: exact first, then prefix.

    Worktree directories are often named ``<repo>-<suffix>``, so the longest
    known name followed by ``-`` wins when no part matches exactly.
    """
    names = {normalize_project_name(name) for name in known if name}
    # Innermost component first: "~/dev/personal/forecaster-wt" is forecaster,
    # not the "personal" directory that holds it.
    for part in (normalize_project_name(part) for part in reversed(path.parts)):
        if part in names:
            return part
        prefixed = [name for name in names if part.startswith(f"{name}-")]
        if prefixed:
            return max(prefixed, key=len)
    return None


def canonical_project(value: str | None) -> str | None:
    """Canonical project name for ``value`` (no registry side effects)."""
    return resolve_project(value)[0]


def load_project_roots(data_dir: Path) -> dict[str, Path]:
    path = data_dir / PROJECT_ROOTS_FILE
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        logger.warning("Ignoring unreadable %s", path)
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(name): Path(root)
        for name, root in raw.items()
        if isinstance(root, str) and root
    }


def record_project_root(data_dir: Path, name: str, root: Path) -> None:
    """Remember where ``name`` lives on this machine."""
    roots = load_project_roots(data_dir)
    if roots.get(name) == root:
        return
    roots[name] = root
    path = data_dir / PROJECT_ROOTS_FILE
    fd, tmp = tempfile.mkstemp(dir=data_dir, prefix=".projects-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(
                {key: str(value) for key, value in sorted(roots.items())}, fh, indent=2
            )
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def register_project(data_dir: Path, value: str | None) -> str | None:
    """Canonicalize ``value`` and record its repository root when known."""
    name, root = resolve_project(value)
    if name and root is not None:
        try:
            record_project_root(data_dir, name, root)
        except OSError:
            logger.warning("Could not record repo root for %s", name, exc_info=True)
    return name


def find_repo_root(
    name: str,
    *,
    data_dir: Path,
    search_roots: list[Path],
) -> Path | None:
    """Locate the repository for project ``name`` on this machine.

    Checks the recorded roots first, then looks for a git checkout named
    ``name`` up to three levels under each search root.
    """
    recorded = load_project_roots(data_dir).get(name)
    if recorded is not None and (recorded / ".git").exists():
        return recorded
    for base in search_roots:
        base = Path(os.path.expanduser(str(base)))
        found = _search_repo(base, name, _REPO_SEARCH_DEPTH)
        if found is not None:
            return found
    return None


def _search_repo(base: Path, name: str, depth: int) -> Path | None:
    if depth < 0 or not base.is_dir():
        return None
    candidate = base / name
    if (candidate / ".git").exists():
        return candidate
    if depth == 0:
        return None
    try:
        children = sorted(
            child
            for child in base.iterdir()
            if child.is_dir()
            and not child.name.startswith(".")
            and child.name not in {"node_modules", "Library", "Applications"}
            and not (child / ".git").exists()
        )
    except OSError:
        return None
    for child in children:
        found = _search_repo(child, name, depth - 1)
        if found is not None:
            return found
    return None
