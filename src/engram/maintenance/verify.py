"""Anchor verification: catch memories whose code moved or vanished.

A fact's anchors are the repo-relative paths and code symbols it depends on.
When every anchor is gone from the project's repository the fact is retired as
stale; when only some are gone it is flagged suspect (still recallable, but
down-weighted and labelled). No LLM is involved.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from engram.core.models import Durability, Fact
from engram.core.projects import find_repo_root
from engram.maintenance.upkeep_report import StepReport
from engram.storage.store import ChangeSet

logger = logging.getLogger(__name__)

# Suspect reasons written by this step start with this prefix, so a later run
# only clears flags it set itself.
MISSING_PREFIX = "missing: "
# Stale reasons written by this step; such facts are re-checked and restored
# when their anchors come back.
STALE_PREFIX = "anchors missing: "
_GIT_TIMEOUT_SECONDS = 10.0

# Extensions that make a token unambiguously a file path. ``snapshot.events``
# is a symbol access, ``snapshot.ts`` is a file.
FILE_EXTENSIONS = frozenset(
    """
    c cc cfg cjs conf cpp cs css csv env go gradle graphql h hcl hpp html ini
    ipynb java jinja jl js json jsonl jsx kt lock lua md mdx mjs php plist proto
    ps1 py pyi rb rs sass scss sh sql svelte swift tf toml ts tsx txt vue xml
    yaml yml zsh
    """.split()
)

AnchorKind = Literal["path", "symbol"]

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_LOCATION_SUFFIX_RE = re.compile(r"(?::\d+(?::\d+)?|#L\d+(?:-L?\d+)?)$")
_BACKTICK_RE = re.compile(r"`([^`\s]+)`")
_TOKEN_STRIP = "\"'`()[]{}<>,;:!?*"
_MIN_SYMBOL_LENGTH = 4


def _extension(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[-1].lower() if "." in base.lstrip(".") else ""


def normalize_anchor(anchor: str) -> tuple[AnchorKind, str] | None:
    """Classify an anchor as a checkable repo path or code symbol.

    Returns ``None`` for anchors that cannot be checked against a repository
    (URLs, absolute or home paths, globs, prose).
    """
    text = _LOCATION_SUFFIX_RE.sub("", anchor.strip().strip("`").rstrip("."))
    if not text or "://" in text or text.startswith(("/", "~", "$")):
        return None
    if any(char in text for char in "*{}<> \t$"):
        return None
    text = text.removeprefix("./")
    if "/" in text or _extension(text) in FILE_EXTENSIONS:
        return "path", text.rstrip("/")
    text = text.removesuffix("()")
    if not _IDENTIFIER_RE.fullmatch(text):
        return None
    symbol = text.rsplit(".", 1)[-1]
    if len(symbol) < _MIN_SYMBOL_LENGTH:
        return None
    return "symbol", symbol


def derive_anchors(content: str) -> list[str]:
    """Conservatively pull file-path anchors out of a fact's prose.

    Only tokens containing ``/`` with a known file extension, or backticked
    bare file names (``name.ext``) qualify. Home, absolute, and URL paths are
    ignored because they cannot be checked against the project's repository.
    """
    slashed = [
        token.strip(_TOKEN_STRIP).rstrip(".")
        for token in content.split()
        if "/" in token and not token.lstrip(_TOKEN_STRIP).startswith("@")
    ]
    backticked = [token for token in _BACKTICK_RE.findall(content) if "/" not in token]
    found: list[str] = []
    for token in [*slashed, *backticked]:
        normalized = normalize_anchor(token)
        if (
            normalized
            and normalized[0] == "path"
            and _extension(normalized[1]) in FILE_EXTENSIONS
        ):
            found.append(normalized[1])
    return list(dict.fromkeys(found))


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


AnchorStatus = Literal["present", "branch", "missing"]


def _suffixes(files: list[str]) -> set[str]:
    """Every trailing sub-path of every file and directory in ``files``, so
    "lib/snapshot.ts" matches "src/renderer/lib/snapshot.ts"."""
    directories = {
        entry.rsplit("/", depth)[0]
        for entry in files
        for depth in range(1, entry.count("/") + 1)
    }
    found: set[str] = set()
    for entry in [*files, *directories]:
        parts = entry.split("/")
        found.update("/".join(parts[start:]) for start in range(len(parts)))
    return found


def _listing(result: subprocess.CompletedProcess[str] | None) -> list[str] | None:
    if result is None or result.returncode != 0:
        return None
    return [entry for entry in result.stdout.split("\0") if entry]


@dataclass
class RepoCheck:
    """Answers "does this anchor still exist?" for one repository, cached.

    An anchor missing from the checkout may still live on an unmerged local
    branch (worktrees share ``refs/heads``); those only count as missing when
    no local branch head has them either.
    """

    root: Path
    _checkout: set[str] | None = None
    _branches: set[str] | None = None
    _trees: list[str] | None = None
    _failed: bool = False
    _symbols: dict[str, AnchorStatus | None] = field(default_factory=dict)

    def _branch_trees(self) -> list[str]:
        if self._trees is None:
            result = _run_git(
                self.root, "for-each-ref", "--format=%(tree)", "refs/heads"
            )
            ok = result is not None and result.returncode == 0
            trees = result.stdout.split() if result is not None and ok else []
            self._trees = sorted(set(trees))
        return self._trees

    def _load_paths(self) -> None:
        if self._checkout is not None or self._failed:
            return
        files = _listing(_run_git(self.root, "ls-files", "-z"))
        if files is None:
            self._failed = True
            return
        self._checkout = _suffixes(files)
        branch_files: set[str] = set()
        for tree in self._branch_trees():
            listed = _listing(
                _run_git(self.root, "ls-tree", "-r", "--name-only", "-z", tree)
            )
            branch_files.update(listed or [])
        self._branches = _suffixes(sorted(branch_files))

    def path_status(self, path: str) -> AnchorStatus | None:
        """``None`` when git could not list the repository."""
        if (self.root / path).exists():
            return "present"
        self._load_paths()
        if self._checkout is None or self._branches is None:
            return None
        if path in self._checkout:
            return "present"
        return "branch" if path in self._branches else "missing"

    def _grep(self, symbol: str, *trees: str) -> bool | None:
        result = _run_git(self.root, "grep", "-q", "-F", "-w", "-e", symbol, *trees)
        if result is None or result.returncode not in (0, 1):
            return None
        return result.returncode == 0

    def symbol_status(self, symbol: str) -> AnchorStatus | None:
        """``None`` when git could not answer (the anchor is then ignored)."""
        if symbol not in self._symbols:
            status: AnchorStatus | None
            in_checkout = self._grep(symbol)
            if in_checkout is None or in_checkout:
                status = "present" if in_checkout else None
            else:
                trees = self._branch_trees()
                on_branch = self._grep(symbol, *trees) if trees else False
                status = (
                    None if on_branch is None else "branch" if on_branch else "missing"
                )
            self._symbols[symbol] = status
        return self._symbols[symbol]

    def status(self, kind: AnchorKind, value: str) -> AnchorStatus | None:
        return self.path_status(value) if kind == "path" else self.symbol_status(value)


@dataclass(frozen=True)
class AnchorVerdict:
    fact: Fact
    present: list[str]
    # Gone from the checkout but still on some local branch head.
    on_branch: list[str]
    missing: list[str]
    derived: bool


def check_fact(fact: Fact, repo: RepoCheck) -> AnchorVerdict | None:
    """Check a fact's anchors (or anchors derived from its prose)."""
    derived = not fact.anchors
    anchors = derive_anchors(fact.content) if derived else fact.anchors
    found: dict[AnchorStatus, list[str]] = {"present": [], "branch": [], "missing": []}
    for anchor in anchors:
        normalized = normalize_anchor(anchor)
        if normalized is None:
            continue
        status = repo.status(*normalized)
        if status is not None:
            found[status].append(anchor)
    if not any(found.values()):
        return None
    return AnchorVerdict(
        fact=fact,
        present=found["present"],
        on_branch=found["branch"],
        missing=found["missing"],
        derived=derived,
    )


@dataclass
class VerifyPlan:
    changes: ChangeSet
    # Facts this step once retired whose anchors came back.
    unstale: list[str] = field(default_factory=list)


def _is_file_path(anchor: str) -> bool:
    """A repo-relative file path: has a directory part and a file extension.

    Bare names (``config.toml``, ``.claude.json``) often live outside the repo.
    """
    normalized = normalize_anchor(anchor)
    return (
        normalized is not None
        and normalized[0] == "path"
        and "/" in normalized[1]
        and _extension(normalized[1]) in FILE_EXTENSIONS
    )


def _missing_reason(verdict: AnchorVerdict) -> str:
    gone = [*verdict.missing, *(f"{a} (only on a branch)" for a in verdict.on_branch)]
    return MISSING_PREFIX + ", ".join(gone)


def plan_verification(verdicts: list[AnchorVerdict], report: StepReport) -> VerifyPlan:
    """Turn anchor verdicts into stale / suspect / restore / anchor changes."""
    plan = VerifyPlan(
        changes=ChangeSet(reason="upkeep: anchor verification", actor="engram:verify")
    )
    changes = plan.changes
    for verdict in verdicts:
        fact = verdict.fact
        gone_everywhere = not verdict.present and not verdict.on_branch
        if fact.stale:
            if gone_everywhere:
                continue
            # Restored facts are unstaled first, so no ``expected`` check here.
            plan.unstale.append(fact.id)
            report.add("restored", fact.id)
            if verdict.missing or verdict.on_branch:
                changes.edits[fact.id] = {"suspect_reason": _missing_reason(verdict)}
            continue
        # Paths guessed from prose are often runtime files or another repo's
        # paths, and bare symbols or extensionless paths are often database
        # objects or external IDs, so those alone only ever flag a fact.
        # Retiring takes explicit anchors, all gone everywhere, at least one
        # of them a repo-relative file path.
        if (
            gone_everywhere
            and not verdict.derived
            and any(_is_file_path(anchor) for anchor in verdict.missing)
        ):
            reason = STALE_PREFIX + ", ".join(verdict.missing)
            changes.stale[fact.id] = reason
            changes.expected[fact.id] = fact.updated_at
            report.add("staled", fact.id)
            report.notes.append(f"stale {fact.id}: {reason}")
            continue
        fields: dict[str, object] = {}
        if verdict.missing or verdict.on_branch:
            reason = _missing_reason(verdict)
            if fact.suspect_reason != reason:
                fields["suspect_reason"] = reason
                report.add("suspect", fact.id)
                report.notes.append(f"suspect {fact.id}: {reason}")
        else:
            if fact.suspect_reason.startswith(MISSING_PREFIX):
                fields["suspect_reason"] = ""
                report.add("cleared", fact.id)
            if verdict.derived:
                fields["anchors"] = verdict.present
                report.add("anchored", fact.id)
        if fields:
            changes.edits[fact.id] = fields
            changes.expected[fact.id] = fact.updated_at
    return plan


def verify_facts(
    facts: list[Fact],
    *,
    data_dir: Path,
    search_roots: list[Path],
    report: StepReport,
) -> VerifyPlan:
    """Check project-scoped facts against their repositories (blocking).

    ``facts`` may include stale facts; only those this step retired
    (``stale_reason`` starting with ``STALE_PREFIX``) are re-checked.
    """
    by_project: dict[str, list[Fact]] = {}
    for fact in facts:
        if not fact.project or fact.durability is Durability.evergreen:
            continue
        if fact.stale and not fact.stale_reason.startswith(STALE_PREFIX):
            continue
        by_project.setdefault(fact.project, []).append(fact)

    verdicts: list[AnchorVerdict] = []
    for project, project_facts in sorted(by_project.items()):
        checkable = [
            fact
            for fact in project_facts
            if fact.anchors or derive_anchors(fact.content)
        ]
        if not checkable:
            continue
        root = find_repo_root(project, data_dir=data_dir, search_roots=search_roots)
        if root is None:
            report.bump("projects_without_repo")
            continue
        report.bump("projects_with_repo")
        repo = RepoCheck(root)
        for fact in checkable:
            verdict = check_fact(fact, repo)
            if verdict is not None:
                verdicts.append(verdict)
        report.bump("facts_checked", len(checkable))
    return plan_verification(verdicts, report)


__all__ = [
    "MISSING_PREFIX",
    "STALE_PREFIX",
    "RepoCheck",
    "VerifyPlan",
    "check_fact",
    "derive_anchors",
    "normalize_anchor",
    "plan_verification",
    "verify_facts",
]
