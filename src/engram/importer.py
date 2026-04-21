"""Import existing Claude Code memories into the Engram fact store."""

import logging
from pathlib import Path

import yaml

from engram.config import get_settings
from engram.observer import extract_facts
from engram.store import FactStore

logger = logging.getLogger(__name__)


async def import_claude_code_memories(store: FactStore | None = None) -> dict:
    """Walk ~/.claude/projects/*/memory/*.md and import as structured facts.

    Returns summary of what was imported.
    """
    store = store or FactStore()
    projects_dir = get_settings().claude_projects_dir

    if not projects_dir.exists():
        return {"error": f"Claude projects directory not found: {projects_dir}"}

    memory_files = list(projects_dir.glob("*/memory/*.md"))
    # Exclude MEMORY.md index files
    memory_files = [f for f in memory_files if f.name != "MEMORY.md"]

    if not memory_files:
        return {"imported": 0, "message": "No memory files found to import"}

    total_facts = 0
    imported_files = []

    for memory_file in memory_files:
        content, metadata = _parse_memory_file(memory_file)
        if not content:
            continue

        # Derive project name from the directory path
        project_dir = memory_file.parent.parent.name
        project_name = _clean_project_name(project_dir)

        source = f"claude_code:{memory_file.relative_to(projects_dir)}"
        logger.info("Importing %s (project: %s)", source, project_name)

        facts = await extract_facts(
            content=f"Memory type: {metadata.get('type', 'unknown')}\n"
            f"Name: {metadata.get('name', 'unknown')}\n"
            f"Description: {metadata.get('description', '')}\n\n"
            f"{content}",
            source=source,
            project=project_name,
            store=store,
        )

        total_facts += len(facts)
        imported_files.append(
            {
                "file": str(memory_file.name),
                "project": project_name,
                "facts_extracted": len(facts),
            }
        )

    return {
        "imported_files": len(imported_files),
        "total_facts": total_facts,
        "details": imported_files,
    }


def _parse_memory_file(path: Path) -> tuple[str, dict]:
    """Parse a Claude Code memory file with YAML frontmatter.

    Returns (body_content, frontmatter_dict).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return "", {}

    # Parse YAML frontmatter
    metadata = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                pass
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
