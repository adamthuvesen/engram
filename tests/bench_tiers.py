"""Benchmark: tier selection across realistic queries.

Seeds a store with ~150 facts across multiple projects/categories,
then runs queries of varying complexity and reports:
- Which tier each query lands in
- Prefilter score distributions
- Whether the tier feels right for accuracy

Run: uv run python tests/bench_tiers.py
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engram.models import Fact, FactCategory
from engram.retriever import _select_tier
from engram.store import FactStore

# ---------------------------------------------------------------------------
# Seed data — realistic cross-project facts
# ---------------------------------------------------------------------------

NOW = datetime.now(timezone.utc)
WEEK_AGO = NOW - timedelta(days=3)
MONTH_AGO = NOW - timedelta(days=35)


def _fact(
    content, category=FactCategory.preference, project=None, tags=None, age="recent"
):
    ts = WEEK_AGO if age == "recent" else MONTH_AGO
    return Fact(
        category=category,
        content=content,
        project=project,
        tags=tags or [],
        created_at=ts,
        updated_at=ts,
        observed_at=ts,
    )


FACTS = [
    # Personal info (5)
    _fact(
        "Alex Chen works on the ML platform team at Acme Analytics",
        FactCategory.personal_info,
        tags=["team", "role"],
    ),
    _fact(
        "Alex Chen is a senior machine learning engineer",
        FactCategory.personal_info,
        tags=["role"],
    ),
    _fact(
        "Alex Chen's manager is Jordan Kim", FactCategory.personal_info, tags=["team"]
    ),
    _fact(
        "Alex Chen works remotely from Portland",
        FactCategory.personal_info,
        tags=["location"],
    ),
    _fact(
        "Alex Chen has been at Acme Analytics for 3 years",
        FactCategory.personal_info,
        tags=["tenure"],
    ),
    # Preferences (12)
    _fact(
        "The user prefers polars over pandas for dataframes",
        tags=["python", "polars", "data"],
    ),
    _fact(
        "The user prefers ruff for Python formatting and linting",
        tags=["python", "ruff"],
    ),
    _fact("The user prefers uv for Python package management", tags=["python", "uv"]),
    _fact(
        "The user prefers concise commit messages in conventional format", tags=["git"]
    ),
    _fact(
        "The user prefers TypeScript for frontend work", tags=["typescript", "frontend"]
    ),
    _fact(
        "The user prefers Tailwind CSS over styled-components", tags=["css", "frontend"]
    ),
    _fact(
        "The user prefers FastAPI over Flask for Python APIs", tags=["python", "api"]
    ),
    _fact(
        "The user dislikes verbose AI-generated comments in code", tags=["code-style"]
    ),
    _fact(
        "The user prefers explicit error handling over silent failures",
        tags=["code-style"],
    ),
    _fact(
        "The user prefers dark mode terminals with Anthropic theme", tags=["terminal"]
    ),
    _fact("The user prefers short PR descriptions with bullet points", tags=["github"]),
    _fact(
        "The user prefers vitest over jest for TypeScript testing",
        tags=["testing", "typescript"],
    ),
    # Engram project (10)
    _fact(
        "Engram uses JSONL for persistent storage",
        FactCategory.project,
        "engram",
        ["storage", "jsonl"],
    ),
    _fact(
        "Engram uses litellm for model-agnostic LLM calls",
        FactCategory.project,
        "engram",
        ["llm"],
    ),
    _fact(
        "Engram uses FastMCP 2.x for the MCP server",
        FactCategory.project,
        "engram",
        ["mcp"],
    ),
    _fact(
        "Engram has a terminal dashboard built with Textual",
        FactCategory.project,
        "engram",
        ["dashboard", "tui"],
    ),
    _fact(
        "Engram's retriever uses 3 parallel search agents",
        FactCategory.project,
        "engram",
        ["retrieval"],
    ),
    _fact(
        "Engram extracts facts using structured LLM output",
        FactCategory.project,
        "engram",
        ["extraction"],
    ),
    _fact(
        "Engram stores facts in ~/.engram/data/facts.jsonl",
        FactCategory.project,
        "engram",
        ["storage"],
    ),
    _fact(
        "Engram supports soft-delete via confidence=0.0",
        FactCategory.project,
        "engram",
        ["storage"],
    ),
    _fact(
        "Engram deduplicates facts using content hash + LLM",
        FactCategory.project,
        "engram",
        ["dedup"],
    ),
    _fact(
        "Engram config uses ENGRAM_ env prefix with pydantic-settings",
        FactCategory.project,
        "engram",
        ["config"],
    ),
    # acme-dw project (8)
    _fact(
        "acme-dw uses Snowflake as the data warehouse",
        FactCategory.project,
        "acme-dw",
        ["snowflake", "dbt"],
    ),
    _fact(
        "acme-dw models are organized by domain: marketing, product, finance",
        FactCategory.project,
        "acme-dw",
        ["dbt", "architecture"],
    ),
    _fact(
        "acme-dw uses incremental models for large tables",
        FactCategory.project,
        "acme-dw",
        ["dbt", "performance"],
    ),
    _fact(
        "acme-dw CI runs on GitHub Actions with dbt Cloud",
        FactCategory.project,
        "acme-dw",
        ["ci", "dbt"],
    ),
    _fact(
        "The revenue_summary model in acme-dw is the source of truth for monthly recurring revenue",
        FactCategory.project,
        "acme-dw",
        ["revenue", "metrics"],
    ),
    _fact(
        "acme-dw uses the ANALYTICS_WH warehouse for analyst queries",
        FactCategory.project,
        "acme-dw",
        ["snowflake"],
    ),
    _fact(
        "acme-dw has a staging layer that mirrors raw Snowflake tables",
        FactCategory.project,
        "acme-dw",
        ["dbt", "staging"],
    ),
    _fact(
        "The user_metrics model joins product usage with billing data",
        FactCategory.project,
        "acme-dw",
        ["metrics"],
    ),
    # Decisions (6)
    _fact(
        "Decided to use JSONL over SQLite for engram storage — inspectability wins",
        FactCategory.decision,
        "engram",
        ["architecture"],
    ),
    _fact(
        "Decided to use 3 parallel agents for retrieval instead of single-pass",
        FactCategory.decision,
        "engram",
        ["architecture"],
    ),
    _fact(
        "Decided to use LLM-powered retrieval instead of embeddings",
        FactCategory.decision,
        "engram",
        ["architecture"],
    ),
    _fact(
        "Decided to keep the dashboard terminal-only, no web UI",
        FactCategory.decision,
        "engram",
        ["dashboard"],
    ),
    _fact(
        "Decided to use polars instead of pandas for the ETL pipeline",
        FactCategory.decision,
        "acme-dw",
        ["python", "data"],
    ),
    _fact(
        "Decided to standardize on a single analytics event schema across services",
        FactCategory.decision,
        tags=["analytics"],
    ),
    # Conventions (6)
    _fact(
        "Convention: Python projects use pyproject.toml, never setup.py",
        FactCategory.convention,
        tags=["python"],
    ),
    _fact(
        "Convention: all API endpoints return JSON with snake_case keys",
        FactCategory.convention,
        tags=["api"],
    ),
    _fact(
        "Convention: use pathlib.Path over os.path everywhere",
        FactCategory.convention,
        tags=["python"],
    ),
    _fact(
        "Convention: commit messages follow conventional commits format",
        FactCategory.convention,
        tags=["git"],
    ),
    _fact(
        "Convention: test files are named test_<module>.py",
        FactCategory.convention,
        tags=["testing", "python"],
    ),
    _fact(
        "Convention: environment variables are loaded via pydantic-settings",
        FactCategory.convention,
        tags=["python", "config"],
    ),
    # Pitfalls (5)
    _fact(
        "Pitfall: Snowflake VARIANT columns need explicit casting in dbt",
        FactCategory.pitfall,
        "acme-dw",
        ["snowflake", "dbt"],
    ),
    _fact(
        "Pitfall: litellm silently falls back to gpt-3.5 if model string is wrong",
        FactCategory.pitfall,
        "engram",
        ["llm"],
    ),
    _fact(
        "Pitfall: FastMCP tools must be async even for sync operations",
        FactCategory.pitfall,
        "engram",
        ["mcp"],
    ),
    _fact(
        "Pitfall: pytest-asyncio needs asyncio_mode=auto in pyproject.toml",
        FactCategory.pitfall,
        tags=["testing", "python"],
    ),
    _fact(
        "Pitfall: ruff and black conflict on trailing comma rules",
        FactCategory.pitfall,
        tags=["python", "linting"],
    ),
    # Events (5)
    _fact(
        "The auth middleware was rewritten due to compliance requirements (2026-02)",
        FactCategory.event,
        tags=["security", "compliance"],
    ),
    _fact(
        "Snowflake warehouse migration completed 2026-01-15",
        FactCategory.event,
        "acme-dw",
        ["snowflake"],
    ),
    _fact(
        "Engram v0.1.0 released with core MCP tools 2026-02-27",
        FactCategory.event,
        "engram",
        ["release"],
    ),
    _fact(
        "The team adopted Claude Code as primary coding agent 2025-12",
        FactCategory.event,
        tags=["tooling"],
    ),
    _fact(
        "Mobile release freeze starts 2026-03-05", FactCategory.event, tags=["release"]
    ),
    # Workflow (8)
    _fact(
        "Use 'uv run pytest tests/ -v' to run engram tests",
        FactCategory.workflow,
        "engram",
        ["testing"],
    ),
    _fact(
        "Use 'uv run engram' to start the MCP server",
        FactCategory.workflow,
        "engram",
        ["mcp"],
    ),
    _fact(
        "Use 'snow sql -q' for ad-hoc Snowflake queries",
        FactCategory.workflow,
        "acme-dw",
        ["snowflake"],
    ),
    _fact(
        "Use 'gh pr create' for pull requests, never push directly to main",
        FactCategory.workflow,
        tags=["github"],
    ),
    _fact(
        "Secrets are loaded via 1Password CLI: op read 'op://...'",
        FactCategory.workflow,
        tags=["secrets"],
    ),
    _fact(
        "Use context7 MCP for looking up library documentation",
        FactCategory.workflow,
        tags=["docs"],
    ),
    _fact(
        "Run 'ruff check . --fix && ruff format .' before committing Python",
        FactCategory.workflow,
        tags=["python", "linting"],
    ),
    _fact(
        "Use engram recall at session start to load user context",
        FactCategory.workflow,
        "engram",
        ["memory"],
    ),
    # Assistant info (4)
    _fact(
        "AI agents should be concise and avoid sycophantic language",
        FactCategory.assistant_info,
        tags=["tone"],
    ),
    _fact(
        "AI agents should question assumptions and offer counterpoints",
        FactCategory.assistant_info,
        tags=["behavior"],
    ),
    _fact(
        "AI agents should use engram recall before non-trivial tasks",
        FactCategory.assistant_info,
        tags=["memory"],
    ),
    _fact(
        "AI agents should never push to git without explicit user approval",
        FactCategory.assistant_info,
        tags=["safety"],
    ),
    # More facts to add noise/volume (20 older facts)
    _fact(
        "The user explored using DuckDB for local analytics",
        FactCategory.event,
        tags=["data"],
        age="old",
    ),
    _fact(
        "The data pipeline processes ~50M events per day",
        FactCategory.project,
        "acme-dw",
        ["scale"],
        age="old",
    ),
    _fact(
        "The user set up Grafana dashboards for API latency monitoring",
        FactCategory.event,
        tags=["observability"],
        age="old",
    ),
    _fact(
        "Marketing team requested a cohort analysis model",
        FactCategory.event,
        "acme-dw",
        ["analytics"],
        age="old",
    ),
    _fact(
        "The user prefers minimal Docker images based on python:3.11-slim",
        tags=["docker"],
        age="old",
    ),
    _fact(
        "CI pipeline takes ~8 minutes for the full dbt build",
        FactCategory.project,
        "acme-dw",
        ["ci"],
        age="old",
    ),
    _fact(
        "The user investigated vector databases for semantic search",
        FactCategory.event,
        tags=["search"],
        age="old",
    ),
    _fact(
        "The Next.js app uses App Router with server components",
        FactCategory.project,
        tags=["frontend", "nextjs"],
        age="old",
    ),
    _fact(
        "The user benchmarked polars vs pandas — 10x faster on aggregations",
        FactCategory.event,
        tags=["python", "data"],
        age="old",
    ),
    _fact(
        "The Slack bot uses webhooks for deployment notifications",
        FactCategory.workflow,
        tags=["slack", "deploy"],
        age="old",
    ),
    _fact("The user prefers Warp terminal over iTerm2", tags=["terminal"], age="old"),
    _fact(
        "Database migrations use alembic with auto-generation",
        FactCategory.workflow,
        tags=["database"],
        age="old",
    ),
    _fact(
        "The user set up pre-commit hooks for ruff and mypy",
        FactCategory.workflow,
        tags=["python"],
        age="old",
    ),
    _fact(
        "The team uses Linear for issue tracking",
        FactCategory.workflow,
        tags=["project-management"],
        age="old",
    ),
    _fact(
        "The user configured Claude Code with custom skills and agents",
        FactCategory.workflow,
        tags=["tooling"],
        age="old",
    ),
    _fact(
        "The user investigated Streamlit vs Gradio for quick ML demos",
        FactCategory.event,
        tags=["ml", "ui"],
        age="old",
    ),
    _fact(
        "API rate limiting is handled by FastAPI middleware",
        FactCategory.project,
        tags=["api", "security"],
        age="old",
    ),
    _fact(
        "The user set up GitHub Actions with matrix builds for Python 3.11+",
        FactCategory.workflow,
        tags=["ci", "python"],
        age="old",
    ),
    _fact(
        "The user prefers Pydantic v2 over v1 for data validation",
        tags=["python", "pydantic"],
        age="old",
    ),
    _fact(
        "The team uses Notion for documentation and meeting notes",
        FactCategory.workflow,
        tags=["docs"],
        age="old",
    ),
]


# ---------------------------------------------------------------------------
# Query scenarios with expected complexity
# ---------------------------------------------------------------------------

QUERIES = [
    # TRIVIAL — should be Tier 0 (direct lookup, 1-3 strong matches)
    ("What team does Alex Chen work on?", "trivial"),
    ("What is Alex Chen's role?", "trivial"),
    ("Who is Alex Chen's manager?", "trivial"),
    # SIMPLE — should be Tier 0 or 1 (focused, few matches)
    ("What dataframe library does the user prefer?", "simple"),
    ("What Python formatter does the user use?", "simple"),
    ("What terminal does the user prefer?", "simple"),
    ("What testing framework for TypeScript?", "simple"),
    # MODERATE — should be Tier 1 (needs synthesis but focused)
    ("How should commit messages be formatted?", "moderate"),
    ("What are the Python conventions in this codebase?", "moderate"),
    ("How do I run tests for engram?", "moderate"),
    ("What storage does engram use and why?", "moderate"),
    # COMPLEX — should be Tier 2 (cross-cutting, multiple perspectives)
    ("What are all the architectural decisions made for engram?", "complex"),
    (
        "Tell me about the acme-dw project — architecture, conventions, pitfalls",
        "complex",
    ),
    ("What should I know before contributing to engram?", "complex"),
    ("What's the user's Python development setup and preferences?", "complex"),
    ("What tooling and workflows does the team use?", "complex"),
    ("What pitfalls should I watch out for across all projects?", "complex"),
    # BROAD — should definitely be Tier 2 (vague, wide scope)
    ("What do you know about the user?", "broad"),
    ("Give me all context about this user's projects", "broad"),
    ("What's the current state of everything?", "broad"),
    # CROSS-PROJECT — should be Tier 2
    ("How does the user's data stack work end-to-end?", "cross-project"),
    ("What decisions has the user made about data tooling?", "cross-project"),
    # TEMPORAL — should be Tier 2 (needs temporal reasoning)
    ("What happened recently?", "temporal"),
    ("What events and releases have occurred?", "temporal"),
    # EDGE CASES
    ("xyzzy gibberish nonexistent topic", "no-match"),
    ("snowflake", "single-word"),
    ("python testing conventions pitfalls", "multi-topic"),
]


# ---------------------------------------------------------------------------
# Which tier is ideal for each complexity level?
# ---------------------------------------------------------------------------

IDEAL_TIER = {
    "trivial": {0, 1},  # 0 ideal, 1 acceptable
    "simple": {0, 1, 2},  # depends on score distribution — flat=2 is fine
    "moderate": {1, 2},  # 1 ideal, 2 acceptable
    "complex": {2},  # must be 2
    "broad": {2},  # must be 2
    "cross-project": {2},  # must be 2
    "temporal": {0, 2},  # 2 ideal, but 0 is ok if nothing matches
    "no-match": {0},  # nothing found, no point using LLM
    "single-word": {0, 1, 2},  # could go either way
    "multi-topic": {1, 2},  # probably needs synthesis
}


def main():
    store = FactStore(data_dir=Path(tempfile.mkdtemp()))
    store.append_facts(FACTS)

    print(
        f"Seeded store with {len(FACTS)} facts across "
        f"{len(set(f.project for f in FACTS if f.project))} projects\n"
    )

    from engram.retriever import RELEVANCE_FLOOR

    # Run all queries
    results = []
    for query, complexity in QUERIES:
        scored = store.prefilter_facts(query, limit=200)
        tier = _select_tier(scored)

        all_scores = [s for s, _ in scored if s > 0]
        relevant_scores = [s for s in all_scores if s >= RELEVANCE_FLOOR]
        ideal = IDEAL_TIER[complexity]
        ok = tier in ideal

        # Score gap: ratio of top score to 5th score (signal concentration)
        top = relevant_scores[0] if relevant_scores else 0
        fifth = relevant_scores[4] if len(relevant_scores) > 4 else 0
        gap_ratio = top / fifth if fifth > 0 else float("inf") if top > 0 else 0

        results.append(
            {
                "query": query,
                "complexity": complexity,
                "tier": tier,
                "ideal": ideal,
                "ok": ok,
                "n_total": len(all_scores),
                "n_relevant": len(relevant_scores),
                "top_score": top,
                "gap_ratio": gap_ratio,
                "scores": relevant_scores[:8],
            }
        )

    # Print results
    print(
        f"{'Query':<55} {'Type':<14} {'Tier':>4} {'OK':>4} {'#Rel':>5} {'Top':>4} {'Gap':>5}  Relevant scores"
    )
    print("-" * 145)

    correct = 0
    total = len(results)
    tier_counts = {0: 0, 1: 0, 2: 0}

    for r in results:
        mark = "  ✓" if r["ok"] else "  ✗"
        gap_str = f"{r['gap_ratio']:.1f}" if r["gap_ratio"] != float("inf") else " inf"
        scores_str = str(r["scores"][:6])
        print(
            f"{r['query']:<55} {r['complexity']:<14} T{r['tier']:>2} {mark}  {r['n_relevant']:>4} {r['top_score']:>4} {gap_str:>5}  {scores_str}"
        )
        if r["ok"]:
            correct += 1
        tier_counts[r["tier"]] += 1

    print("-" * 145)
    print(f"\nAccuracy: {correct}/{total} ({correct / total * 100:.0f}%)")
    print("\nTier distribution:")
    for t in (0, 1, 2):
        pct = tier_counts[t] / total * 100
        print(f"  Tier {t}: {tier_counts[t]:>3} ({pct:.0f}%)")

    # Flag problems
    bad = [r for r in results if not r["ok"]]
    if bad:
        print(f"\n⚠ {len(bad)} misclassified queries:")
        for r in bad:
            gap_str = (
                f"{r['gap_ratio']:.1f}" if r["gap_ratio"] != float("inf") else "inf"
            )
            print(
                f'  - [{r["complexity"]}] "{r["query"]}" → Tier {r["tier"]} (want {r["ideal"]})'
            )
            print(
                f"    {r['n_relevant']} relevant, top={r['top_score']}, gap={gap_str}, scores={r['scores'][:5]}"
            )

    return results


if __name__ == "__main__":
    main()
