"""Source evidence attached to evaluation reports."""

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from engram.recall.retriever import SELECTOR_VERSION

Execution = Literal["deterministic", "provider", "caller_supplied"]


class ReportProvenance(BaseModel):
    dataset_sha256: str
    selector_version: str = SELECTOR_VERSION
    code_revision: str | None = None
    working_tree_dirty: bool | None = None
    execution: Execution = "deterministic"
    live_status: str = "skipped: provider disabled"


def _source_revision() -> tuple[str | None, bool | None]:
    if shutil.which("git") is None:
        return None, None
    source = Path(__file__).resolve()
    root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=source.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if root_result.returncode:
        return None, None
    root = Path(root_result.stdout.strip()).resolve()
    if source != root / "src" / "engram" / "evaluation" / "report.py":
        return None, None
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "src/engram/__init__.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode:
        return None, None
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        revision.stdout.strip() if revision.returncode == 0 else None,
        bool(status.stdout) if status.returncode == 0 else None,
    )


def provenance_for_bytes(
    data: bytes, *, execution: Execution = "deterministic"
) -> ReportProvenance:
    revision, dirty = _source_revision()
    live_status = {
        "deterministic": "skipped: provider disabled",
        "provider": "enabled explicitly",
        "caller_supplied": "unknown: caller supplied outputs without execution evidence",
    }[execution]
    return ReportProvenance(
        dataset_sha256=hashlib.sha256(data).hexdigest(),
        code_revision=revision,
        working_tree_dirty=dirty,
        execution=execution,
        live_status=live_status,
    )


def report_provenance(
    path: Path, *, live: bool = False, supplied: bool = False
) -> ReportProvenance:
    execution: Execution = (
        "provider" if live else "caller_supplied" if supplied else "deterministic"
    )
    return provenance_for_bytes(path.read_bytes(), execution=execution)


def print_provenance(evidence: ReportProvenance) -> None:
    print(f"Dataset SHA256: {evidence.dataset_sha256}")
    print(
        f"Selector: {evidence.selector_version}. Code revision: {evidence.code_revision or 'unavailable'}"
    )
    print(f"Working tree dirty: {evidence.working_tree_dirty}. Live: {evidence.live_status}")
