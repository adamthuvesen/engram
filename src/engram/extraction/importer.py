"""Import existing Claude Code memories into the Engram fact store."""

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from engram.core.config import get_settings
from engram.core.projects import canonical_project
from engram.core.structured_outputs import UnresolvedDecision
from engram.extraction.observer import ingest
from engram.storage.async_store import AsyncFactStore, run_sync
from engram.storage.store import FactStore

logger = logging.getLogger(__name__)


class ImportFileResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    project: str | None
    status: Literal["ok", "partial", "error"]
    facts_extracted: int = 0
    unresolved: list[UnresolvedDecision] = Field(default_factory=list)
    error: str | None = None


async def import_claude_code_memories(
    store: FactStore | AsyncFactStore | None = None,
) -> dict[str, Any]:
    """Walk ~/.claude/projects/*/memory/*.md and import as structured facts.

    Returns summary of what was imported.
    """
    if store is None:
        store = AsyncFactStore(await run_sync(FactStore))
    projects_dir = get_settings().claude_projects_dir

    if not projects_dir.exists():
        return {"error": f"Claude projects directory not found: {projects_dir}"}

    memory_files = list(projects_dir.glob("*/memory/*.md"))
    # Exclude MEMORY.md index files
    memory_files = [f for f in memory_files if f.name != "MEMORY.md"]

    if not memory_files:
        return {
            "imported_files": 0,
            "total_facts": 0,
            "details": [],
            "message": "No memory files found to import",
        }

    total_facts = 0
    imported_files: list[ImportFileResult] = []
    failed_files = 0

    for memory_file in memory_files:
        project_name = canonical_project(_clean_project_name(memory_file.parent.parent.name))
        try:
            content, metadata = await run_sync(_parse_memory_file, memory_file)
            if not content.strip():
                raise ValueError("memory file has no content")
            source = f"claude_code:{memory_file.relative_to(projects_dir)}"
            result = await ingest(
                content=f"Memory type: {metadata.get('type', 'unknown')}\n"
                f"Name: {metadata.get('name', 'unknown')}\n"
                f"Description: {metadata.get('description', '')}\n\n{content}",
                source=source,
                project=project_name,
                store=store,
            )
            if result.errors:
                raise ValueError("; ".join(result.errors))
        except Exception as exc:
            failed_files += 1
            imported_files.append(
                ImportFileResult(
                    file=memory_file.name,
                    project=project_name,
                    status="error",
                    error=str(exc),
                )
            )
            continue

        total_facts += len(result.created)
        imported_files.append(
            ImportFileResult(
                file=memory_file.name,
                project=project_name,
                facts_extracted=len(result.created),
                status="partial" if result.unresolved else "ok",
                unresolved=result.unresolved,
            )
        )

    return {
        "imported_files": len(imported_files) - failed_files,
        "failed_files": failed_files,
        "status": "partial"
        if failed_files or any(item.status == "partial" for item in imported_files)
        else "ok",
        "total_facts": total_facts,
        "details": [item.model_dump() for item in imported_files],
    }


def _parse_memory_file(path: Path) -> tuple[str, dict[str, Any]]:
    """Parse a Claude Code memory file with YAML frontmatter.

    Returns (body_content, frontmatter_dict).
    """
    text = path.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) < 3:
            raise ValueError("memory frontmatter has no closing delimiter")
        parsed = yaml.safe_load(parts[1])
        if parsed is not None and not isinstance(parsed, dict):
            raise ValueError("memory frontmatter must be a YAML mapping")
        metadata = parsed or {}
        body = parts[2].strip()

    return body, metadata


def _clean_project_name(dir_name: str) -> str:
    """Convert Claude's mangled project directory name to a readable name.

    Example: "-Users-jdoe-dev-myproject" -> "myproject"
    """
    parts = [p for p in dir_name.split("-") if p]
    if not parts:
        return dir_name

    # Strip an exact home-directory prefix when Claude encodes an absolute path.
    home_parts = [p for p in Path.home().parts if p and p != "/"]
    i = 0
    while i < len(parts) and i < len(home_parts) and parts[i] == home_parts[i]:
        i += 1
    remaining = parts[i:]

    if not remaining:
        return parts[-1]

    # Common local workspace roots. If one is present, the next segment is often
    # an org/workspace folder, so prefer the remainder as the project slug.
    workspace_roots = {"dev", "code", "src", "repos", "projects", "workspace", "work"}
    stripped_workspace_root = False
    if remaining[0] in workspace_roots:
        remaining = remaining[1:]
        stripped_workspace_root = True

    if not remaining:
        return parts[-1]

    if stripped_workspace_root and len(remaining) > 1:
        remaining = remaining[1:]

    return "-".join(remaining) if remaining else parts[-1]
