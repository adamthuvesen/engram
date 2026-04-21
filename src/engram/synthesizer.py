"""Synthesizer — LLM-powered fact consolidation and cleanup."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from engram.config import get_settings
from engram.llm import complete_json
from engram.models import Fact
from engram.store import FactStore, format_facts_for_llm

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM = """You are a knowledge base maintenance agent. You receive a batch of stored facts
and must decide what to do with each one.

Goals:
1. Remove duplicates — if two or more facts say essentially the same thing, merge them into one
2. Remove stale facts — outdated information that is no longer true or useful
3. Improve clarity — rewrite vague or poorly worded facts to be clearer and more specific
4. Keep everything else unchanged

For each fact, return exactly one action:

- "keep": fact is fine as-is, no changes needed
- "remove": fact is stale, outdated, or no longer valuable (provide "reason")
- "rewrite": fact should be rewritten for clarity (provide "new_content", "new_tags", "reason")
- "merge": this fact should absorb one or more other facts (provide "merge_with" list of fact IDs,
  "merged_content", "merged_tags", "reason")
- "merge_source": this fact will be absorbed into another fact (provide "merge_target" fact ID)

Rules:
- Every fact in the batch MUST get exactly one action
- "merge" and "merge_source" must be paired: if fact A has action "merge" with merge_with=[B],
  then fact B must have action "merge_source" with merge_target=A
- Prefer keeping facts over removing when uncertain
- When rewriting, preserve the original meaning — only improve wording and clarity
- merged_content should combine information from all merged facts into one clear statement
- new_tags / merged_tags should be 1-3 lowercase tags

Return JSON:
{"actions": [{"fact_id": "...", "action": "keep|remove|rewrite|merge|merge_source", ...}, ...]}"""


@dataclass
class SynthesisResult:
    """Summary of a synthesis operation."""

    total_analyzed: int = 0
    kept: int = 0
    removed: int = 0
    rewritten: int = 0
    merged_groups: int = 0
    merged_sources: int = 0
    errors: list[str] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)


async def synthesize(
    project: str | None = None,
    dry_run: bool = True,
    store: FactStore | None = None,
) -> SynthesisResult:
    """Analyze and consolidate the fact store.

    Groups facts by (project, category), processes each batch through an LLM
    that decides what to keep, remove, rewrite, or merge.

    Args:
        project: Only process facts for this project. None = all projects.
        dry_run: If True, return proposed changes without applying them.
        store: FactStore instance (uses default if None).

    Returns:
        SynthesisResult with counts and details of all actions.
    """
    store = store or FactStore()
    settings = get_settings()
    batch_size = settings.synthesis_batch_size

    facts = store.load_active_facts(project=project)
    if not facts:
        return SynthesisResult()

    # Group by (project, category) so related facts land in the same batch
    groups: dict[tuple[str | None, str], list[Fact]] = defaultdict(list)
    for fact in facts:
        groups[(fact.project, fact.category.value)].append(fact)

    result = SynthesisResult(total_analyzed=len(facts))

    for group_key, group_facts in groups.items():
        # Chunk into batches
        for i in range(0, len(group_facts), batch_size):
            batch = group_facts[i : i + batch_size]
            batch_actions = await _synthesize_batch(batch, store)

            for action in batch_actions:
                act = action.get("action", "keep")
                if act == "keep":
                    result.kept += 1
                elif act == "remove":
                    result.removed += 1
                    result.details.append(action)
                elif act == "rewrite":
                    result.rewritten += 1
                    result.details.append(action)
                elif act == "merge":
                    result.merged_groups += 1
                    result.details.append(action)
                elif act == "merge_source":
                    result.merged_sources += 1
                else:
                    result.errors.append(
                        f"Unknown action '{act}' for fact {action.get('fact_id')}"
                    )

            if not dry_run:
                _apply_actions(batch_actions, store)

    return result


async def _synthesize_batch(batch: list[Fact], store: FactStore) -> list[dict]:
    """Send a batch of facts to the LLM for analysis."""
    facts_text = format_facts_for_llm(batch)
    prompt = f"Analyze these facts and decide what to do with each one:\n\n{facts_text}"

    try:
        response = await complete_json(prompt=prompt, system=SYNTHESIS_SYSTEM)
    except Exception as e:
        logger.error("Synthesis LLM call failed: %s", e)
        return [{"fact_id": f.id, "action": "keep"} for f in batch]

    actions = response.get("actions", [])

    # Validate: every fact in the batch should have an action
    returned_ids = {a.get("fact_id") for a in actions}
    batch_ids = {f.id for f in batch}

    for missing_id in batch_ids - returned_ids:
        logger.warning(
            "LLM did not return action for fact %s, defaulting to keep", missing_id
        )
        actions.append({"fact_id": missing_id, "action": "keep"})

    # Drop actions for facts not in this batch (hallucinated IDs)
    actions = [a for a in actions if a.get("fact_id") in batch_ids]

    return actions


def _apply_actions(actions: list[dict], store: FactStore) -> None:
    """Apply synthesis actions to the store in a single batch write."""
    updates: dict[str, dict] = {}

    for action in actions:
        fact_id = action.get("fact_id")
        if not fact_id:
            continue

        act = action.get("action", "keep")

        if act == "keep":
            continue

        elif act == "remove":
            updates[fact_id] = {"confidence": 0.0}

        elif act == "rewrite":
            new_content = action.get("new_content")
            if not new_content:
                logger.warning(
                    "Rewrite action for %s missing new_content, skipping", fact_id
                )
                continue
            upd: dict = {"content": new_content}
            if "new_tags" in action:
                upd["tags"] = action["new_tags"]
            updates[fact_id] = upd

        elif act == "merge":
            merged_content = action.get("merged_content")
            if not merged_content:
                logger.warning(
                    "Merge action for %s missing merged_content, skipping", fact_id
                )
                continue
            upd = {"content": merged_content}
            if "merged_tags" in action:
                upd["tags"] = action["merged_tags"]
            # Primary link to first absorbed fact
            merge_with = action.get("merge_with", [])
            if merge_with:
                upd["supersedes"] = merge_with[0]
                # Encode additional absorbed IDs in tags so the full audit trail is preserved
                if len(merge_with) > 1:
                    extra_tags = [f"merged:{fid}" for fid in merge_with[1:]]
                    existing_tags = upd.get("tags", [])
                    upd["tags"] = list(set(existing_tags + extra_tags))
            updates[fact_id] = upd

        elif act == "merge_source":
            updates[fact_id] = {"confidence": 0.0}

    if updates:
        store.batch_update_facts(updates)


def format_synthesis_result(result: SynthesisResult, dry_run: bool) -> str:
    """Format a SynthesisResult as a human-readable string for MCP output."""
    mode = "Preview (dry run)" if dry_run else "Applied"
    lines = [f"## Synthesis {mode}\n"]
    lines.append(f"**Analyzed:** {result.total_analyzed} facts")
    lines.append(
        f"**Kept:** {result.kept} | "
        f"**Removed:** {result.removed} | "
        f"**Rewritten:** {result.rewritten} | "
        f"**Merged:** {result.merged_groups} groups ({result.merged_sources} facts absorbed)"
    )

    if result.errors:
        lines.append(f"\n**Errors:** {len(result.errors)}")
        for err in result.errors:
            lines.append(f"- {err}")

    removals = [d for d in result.details if d.get("action") == "remove"]
    rewrites = [d for d in result.details if d.get("action") == "rewrite"]
    merges = [d for d in result.details if d.get("action") == "merge"]

    if removals:
        lines.append("\n### Removals")
        for r in removals:
            lines.append(f"- `{r['fact_id']}` — {r.get('reason', 'no reason given')}")

    if rewrites:
        lines.append("\n### Rewrites")
        for r in rewrites:
            lines.append(
                f'- `{r["fact_id"]}` → "{r.get("new_content", "?")}" — {r.get("reason", "")}'
            )

    if merges:
        lines.append("\n### Merges")
        for m in merges:
            sources = ", ".join(f"`{s}`" for s in m.get("merge_with", []))
            lines.append(
                f'- `{m["fact_id"]}` + {sources} → "{m.get("merged_content", "?")}" '
                f"— {m.get('reason', '')}"
            )

    if dry_run and (removals or rewrites or merges):
        lines.append("\nRun again with `dry_run=False` to apply these changes.")

    return "\n".join(lines)
