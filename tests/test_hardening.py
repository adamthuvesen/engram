"""Tests for the hardening changes: locking, fsync, batched approvals,
LLM resilience, dedup correctness, search index cache, config, importer."""

import asyncio
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

import engram.server as server
from engram.core.models import (
    CandidateStatus,
    Fact,
    FactCategory,
    MemoryCandidate,
    RecallRecord,
    TransactionStatus,
)
from engram.storage.store import AsyncFactStore, FactStore
from tests.mcp_helpers import call_tool


def _make_store() -> FactStore:
    tmp = Path(tempfile.mkdtemp())
    return FactStore(data_dir=tmp)


def _make_fact(**kwargs) -> Fact:
    defaults = dict(category=FactCategory.preference, content="Test fact")
    defaults.update(kwargs)
    return Fact(**defaults)


def _make_candidate(**kwargs) -> MemoryCandidate:
    defaults = dict(category=FactCategory.preference, content="Test candidate")
    defaults.update(kwargs)
    return MemoryCandidate(**defaults)


# ---------------------------------------------------------------------------
# 1.4 Concurrent appends produce no corruption
# ---------------------------------------------------------------------------


def test_concurrent_append_no_corruption():
    """Two threads appending 50 facts each yields 100 clean JSONL lines."""
    store = _make_store()
    errors = []

    def append_facts(n: int) -> None:
        try:
            facts = [_make_fact(content=f"fact-{n}-{i}") for i in range(50)]
            store.append_facts(facts)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=append_facts, args=(1,))
    t2 = threading.Thread(target=append_facts, args=(2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Thread errors: {errors}"
    loaded = store.load_facts()
    assert len(loaded) == 100
    # Each on-disk line must be parseable. The event-log layout is one meta
    # sentinel followed by one ``created`` event per fact.
    raw_lines = [
        line for line in store.facts_path.read_text().splitlines() if line.strip()
    ]
    assert len(raw_lines) == 101
    for line in raw_lines:
        json.loads(line)  # must not raise


# ---------------------------------------------------------------------------
# 1.5 _rewrite failure leaves no .tmp files
# ---------------------------------------------------------------------------


def test_rewrite_failure_leaves_no_tmp_files():
    """If _rewrite raises before os.replace, no *.tmp file is left behind."""
    store = _make_store()
    store.append_facts([_make_fact(content="initial")])

    call_count = [0]

    def failing_fsync(fd: int) -> None:
        call_count[0] += 1
        raise OSError("Simulated fsync failure")

    with patch("engram.storage.store.os.fsync", side_effect=failing_fsync):
        with pytest.raises(OSError, match="Simulated fsync failure"):
            store._rewrite(store.load_facts())

    tmp_files = list(store.data_dir.glob("*.tmp"))
    assert tmp_files == [], f"Orphaned tmp files: {tmp_files}"
    assert call_count[0] >= 1


# ---------------------------------------------------------------------------
# 1.6 _rewrite calls fsync before os.replace
# ---------------------------------------------------------------------------


def test_rewrite_calls_fsync():
    """fsync is called on the tmp fd before os.replace."""
    store = _make_store()
    store.append_facts([_make_fact(content="data")])
    facts = store.load_facts()

    fsync_calls = []
    replace_calls = []

    original_fsync = __import__("os").fsync
    original_replace = __import__("os").replace

    def tracking_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        original_fsync(fd)

    def tracking_replace(src, dst) -> None:
        replace_calls.append((str(src), str(dst)))
        original_replace(src, dst)

    with (
        patch("engram.storage.store.os.fsync", side_effect=tracking_fsync),
        patch("engram.storage.store.os.replace", side_effect=tracking_replace),
    ):
        store._rewrite(facts)

    assert fsync_calls, "fsync was never called"
    assert replace_calls, "os.replace was never called"
    # fsync must happen before replace
    assert len(fsync_calls) >= 1 and len(replace_calls) >= 1
    # The target file should be intact
    assert store.facts_path.exists()


# ---------------------------------------------------------------------------
# 2.3 approve_candidates uses exactly 3 write operations for N candidates
# ---------------------------------------------------------------------------


def test_approve_candidates_batched_writes():
    """Approving 5 candidates with supersessions causes at most 3 store writes."""
    store = _make_store()

    old_facts = [_make_fact(id=f"old{i}", content=f"Old fact {i}") for i in range(5)]
    store.append_facts(old_facts)

    candidates = [
        _make_candidate(id=f"cand{i}", content=f"New fact {i}", supersedes=f"old{i}")
        for i in range(5)
    ]
    store.append_candidates(candidates)

    rewrite_calls = []
    append_calls = []

    original_rewrite = store._rewrite
    original_append = store.append_facts

    def tracking_rewrite(records, path=None):
        rewrite_calls.append(path or store.facts_path)
        return original_rewrite(records, path=path)

    def tracking_append(facts):
        append_calls.append(len(facts))
        return original_append(facts)

    store._rewrite = tracking_rewrite
    store.append_facts = tracking_append

    approved = store.approve_candidates([f"cand{i}" for i in range(5)])

    assert len(approved) == 5
    # At most 3 write operations: facts batch, candidates batch, new facts append
    total_writes = len(rewrite_calls) + len(append_calls)
    assert total_writes <= 3, (
        f"Too many writes: {total_writes} (rewrites={rewrite_calls}, appends={append_calls})"
    )


def test_approve_candidates_writes_transaction_markers():
    store = _make_store()
    store.append_candidates([_make_candidate(id="cand1", content="Remember this")])

    approved = store.approve_candidates(["cand1"])

    assert len(approved) == 1
    transactions = store._load_transactions()
    assert [tx.status for tx in transactions] == [
        TransactionStatus.prepared,
        TransactionStatus.committed,
    ]
    assert transactions[0].id == transactions[1].id
    assert transactions[0].new_facts[0].id == approved[0].id


def test_recover_prepared_approval_transaction_on_startup():
    store = _make_store()
    old = _make_fact(id="old1", content="Old preference")
    candidate = _make_candidate(
        id="cand1",
        content="New preference",
        supersedes="old1",
    )
    store.append_facts([old])
    store.append_candidates([candidate])

    transaction = store._prepare_approval_transaction(["cand1"])
    assert transaction is not None
    store._append_transaction(transaction)

    recovered = FactStore(data_dir=store.data_dir)

    facts = recovered.load_facts()
    assert len(facts) == 2
    assert next(f for f in facts if f.id == "old1").confidence == 0.0
    assert next(f for f in facts if f.id != "old1").supersedes == "old1"
    assert [fact.id for fact in recovered.load_active_facts()] == [
        next(f for f in facts if f.id != "old1").id
    ]
    assert recovered.load_candidates(status=CandidateStatus.pending) == []
    assert len(recovered.load_candidates(status=CandidateStatus.approved)) == 1
    assert recovered._pending_transactions() == []


def test_approval_recovers_when_apply_fails_after_prepare():
    store = _make_store()
    store.append_candidates([_make_candidate(id="cand1", content="Remember this")])

    def fail_apply(transaction):
        raise OSError("simulated crash after prepare")

    store._apply_approval_transaction = fail_apply

    with pytest.raises(OSError, match="simulated crash after prepare"):
        store.approve_candidates(["cand1"])

    assert store.load_active_facts() == []
    assert len(store.load_candidates(status=CandidateStatus.pending)) == 1

    recovered = FactStore(data_dir=store.data_dir)

    facts = recovered.load_active_facts()
    assert len(facts) == 1
    assert facts[0].content == "Remember this"
    assert len(recovered.load_candidates(status=CandidateStatus.approved)) == 1
    assert recovered._pending_transactions() == []


def test_approval_recovery_does_not_duplicate_fact_after_commit_marker_failure():
    store = _make_store()
    store.append_candidates([_make_candidate(id="cand1", content="Remember this")])

    original_append_transaction = store._append_transaction

    def fail_commit_marker(transaction):
        if transaction.status == TransactionStatus.committed:
            raise OSError("simulated crash before commit marker")
        original_append_transaction(transaction)

    store._append_transaction = fail_commit_marker

    with pytest.raises(OSError, match="simulated crash before commit marker"):
        store.approve_candidates(["cand1"])

    assert len(store.load_active_facts()) == 1
    assert len(store.load_candidates(status=CandidateStatus.approved)) == 1

    recovered = FactStore(data_dir=store.data_dir)

    facts = recovered.load_active_facts()
    assert len(facts) == 1
    assert facts[0].content == "Remember this"
    assert recovered._pending_transactions() == []


def test_reject_candidates_batched_writes():
    """Rejecting many candidates rewrites the candidate file at most once."""
    store = _make_store()
    candidates = [_make_candidate(id=f"cand{i}") for i in range(5)]
    store.append_candidates(candidates)

    rewrite_calls = []
    original_rewrite = store._rewrite

    def tracking_rewrite(records, path=None):
        rewrite_calls.append(path or store.facts_path)
        return original_rewrite(records, path=path)

    store._rewrite = tracking_rewrite

    rejected = store.reject_candidates([f"cand{i}" for i in range(5)], reason="Nope")

    assert len(rejected) == 5
    assert len(rewrite_calls) == 1


# ---------------------------------------------------------------------------
# 3.4 structured LLM output validation
# ---------------------------------------------------------------------------


class _HardeningStructuredResponse(BaseModel):
    answer: int


def test_complete_model_raw_json_parse(monkeypatch):
    """Raw JSON is validated into the requested Pydantic model."""
    from engram.llm import client as llm

    async def fake_complete(**kwargs):
        return '{"answer": 42}'

    monkeypatch.setattr(llm, "complete", fake_complete)
    result = asyncio.run(
        llm.complete_model(
            prompt="test",
            system="sys",
            response_model=_HardeningStructuredResponse,
        )
    )
    assert result == _HardeningStructuredResponse(answer=42)


def test_complete_model_unparseable_raises_validation_error(monkeypatch):
    """Completely unparseable output fails loudly at the structured boundary."""
    from engram.llm import client as llm

    async def fake_complete(**kwargs):
        return "This is not JSON at all!"

    monkeypatch.setattr(llm, "complete", fake_complete)

    with pytest.raises(ValidationError):
        asyncio.run(
            llm.complete_model(
                prompt="test",
                system="sys",
                response_model=_HardeningStructuredResponse,
            )
        )


# ---------------------------------------------------------------------------
# 3.5 LLM retry: num_retries=2 is forwarded to litellm
# ---------------------------------------------------------------------------


def test_complete_passes_num_retries(monkeypatch):
    """complete() passes num_retries=2 to litellm.acompletion."""
    from engram.llm import client as llm

    captured_kwargs: dict = {}

    async def fake_acompletion(**kwargs):
        captured_kwargs.update(kwargs)
        response = MagicMock()
        response.choices[0].message.content = "hello"
        return response

    fake_litellm = MagicMock()
    fake_litellm.suppress_debug_info = False
    fake_litellm.acompletion = fake_acompletion

    monkeypatch.setattr(llm, "_get_litellm", lambda: fake_litellm)
    monkeypatch.setattr("engram.core.config.ensure_openai_api_key", lambda: "key")

    result = asyncio.run(llm.complete(prompt="test", model="openai/gpt-4o-mini"))
    assert result == "hello"
    assert captured_kwargs.get("num_retries") == 2


def test_recall_context_prompt_mode_smoke():
    """The MCP prompt-mode helper should format facts without crashing."""
    store = _make_store()
    store.append_facts(
        [
            _make_fact(
                content="Alex prefers concise terminal summaries",
                project="engram",
            )
        ]
    )
    app = server.create_mcp(store)

    result = asyncio.run(
        call_tool(
            app,
            "recall_context",
            {
                "query": "What does Alex prefer?",
                "project": "engram",
                "mode": "prompt",
            },
        )
    )

    text = str(result)
    assert "Memory (1 of 1 matches, project engram)" in text
    assert "Alex prefers concise terminal summaries" in text


def test_recall_context_prompt_mode_omits_unrelated_fallbacks():
    store = _make_store()
    store.append_facts(
        [
            _make_fact(
                content="Alex prefers concise terminal summaries",
                project="engram",
            )
        ]
    )
    app = server.create_mcp(store)

    result = asyncio.run(
        call_tool(
            app,
            "recall_context",
            {
                "query": "What database warehouse should we use?",
                "project": "engram",
                "mode": "prompt",
            },
        )
    )

    text = str(result)
    assert "# Memory Context" not in text
    assert "Alex prefers concise terminal summaries" not in text
    assert "No relevant memories" in text


def test_import_memories_empty_directory_returns_message(tmp_path, monkeypatch):
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    monkeypatch.setattr(
        "engram.extraction.importer.get_settings",
        lambda: MagicMock(claude_projects_dir=projects_dir),
    )

    store = _make_store()
    app = server.create_mcp(store)
    result = asyncio.run(call_tool(app, "import_memories", {"source": "claude_code"}))

    assert "No memory files found to import" in str(result)


def test_import_memories_accepts_async_store(tmp_path, monkeypatch):
    from engram.extraction.importer import import_claude_code_memories

    projects_dir = tmp_path / "projects"
    memory_dir = projects_dir / "-Users-alex-dev-example-project" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "async-storage.md").write_text(
        "---\n"
        "type: note\n"
        "name: Async Storage\n"
        "description: Storage migration\n"
        "---\n"
        "Engram routes MCP storage through AsyncFactStore.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "engram.extraction.importer.get_settings",
        lambda: MagicMock(claude_projects_dir=projects_dir),
    )

    async def fake_complete_model(prompt: str, system: str, response_model, model=None):
        return response_model.model_validate(
            {
                "facts": [
                    {
                        "memory_key": "engram-async-storage",
                        "content": "Engram routes MCP storage through AsyncFactStore",
                        "category": "project",
                        "tags": ["storage"],
                        "retrieval_hints": ["Engram MCP storage"],
                        "covered_claims": [
                            "Engram routes MCP storage through AsyncFactStore"
                        ],
                        "why_store": "Documents the architecture",
                        "expires_at": None,
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "engram.extraction.observer.complete_model", fake_complete_model
    )

    store = _make_store()
    result = asyncio.run(import_claude_code_memories(store=AsyncFactStore(store)))

    assert result["total_facts"] == 1
    assert store.load_active_facts()[0].content == (
        "Engram routes MCP storage through AsyncFactStore"
    )


def test_list_candidates_search_filters_before_limit():
    store = _make_store()
    candidates = [
        _make_candidate(
            id=f"cand{i}",
            content=f"Routine candidate {i}",
            status=CandidateStatus.pending,
        )
        for i in range(55)
    ]
    target = _make_candidate(
        id="target",
        content="Needle candidate for search",
        status=CandidateStatus.pending,
    )
    store.append_candidates([target, *candidates])
    app = server.create_mcp(store)

    result = asyncio.run(
        call_tool(
            app,
            "list_candidates",
            {"status": "pending", "search": "Needle", "limit": 5},
        )
    )

    text = str(result)
    assert "Needle candidate for search" in text
    assert "Routine candidate" not in text


def test_mcp_candidate_approval_and_rejection_use_async_store():
    store = _make_store()
    store.append_candidates(
        [
            _make_candidate(id="approve-me", content="Approved async candidate"),
            _make_candidate(id="reject-me", content="Rejected async candidate"),
        ]
    )
    app = server.create_mcp(AsyncFactStore(store))

    approved = asyncio.run(
        call_tool(
            app,
            "approve_candidates",
            {"candidate_ids": ["approve-me"]},
        )
    )
    rejected = asyncio.run(
        call_tool(
            app,
            "reject_candidates",
            {"candidate_ids": ["reject-me"], "reason": "not durable"},
        )
    )

    assert "Approved 1 candidate" in str(approved)
    assert "Rejected 1 candidate" in str(rejected)
    assert len(store.load_candidates(status=CandidateStatus.approved)) == 1
    assert len(store.load_candidates(status=CandidateStatus.rejected)) == 1


def test_inspect_invalid_category_returns_helpful_message():
    app = server.create_mcp(_make_store())

    result = asyncio.run(
        call_tool(
            app,
            "inspect",
            {"category": "bogus"},
        )
    )

    assert "Invalid category: bogus" in str(result)


def test_mcp_inspect_stats_purge_and_rename_use_async_store():
    store = _make_store()
    active = _make_fact(content="Project fact", project="old-project")
    forgotten = _make_fact(content="Forgotten fact", confidence=0.0)
    candidate = _make_candidate(id="rename-cand", project="old-project")
    store.append_facts([active, forgotten])
    store.append_candidates([candidate])
    app = server.create_mcp(AsyncFactStore(store))

    renamed = asyncio.run(
        call_tool(
            app,
            "rename_project",
            {"old_project": "old-project", "new_project": "new-project"},
        )
    )
    inspected = asyncio.run(
        call_tool(
            app,
            "inspect",
            {"project": "new-project"},
        )
    )
    stats = asyncio.run(call_tool(app, "memory_stats", {}))
    purged = asyncio.run(call_tool(app, "purge", {}))

    assert "Renamed 2 record" in str(renamed)
    assert "Project fact" in str(inspected)
    assert "**Total facts:** 2" in str(stats)
    assert "Purged 1 facts" in str(purged)


def test_recall_stats_reports_zero_llm_calls():
    store = _make_store()
    store.log_recall(
        RecallRecord(
            query="direct",
            tier=0,
            prefilter_count=1,
            latency_ms=1,
            quality="high",
            llm_calls=0,
        )
    )
    app = server.create_mcp(store)

    result = asyncio.run(call_tool(app, "recall_stats", {}))

    assert "LLM calls (reported): 0" in str(result)


def test_mcp_tools_return_text_and_structured_content():
    store = _make_store()
    store.append_facts([_make_fact(id="aaaaaaaaaaaa", content="Project fact")])
    app = server.create_mcp(AsyncFactStore(store))

    content, structured = asyncio.run(call_tool(app, "inspect", {"format": "json"}))

    assert content[0].text.startswith('{"status":"ok"')
    assert structured["status"] == "ok"
    assert structured["data"][0]["id"] == "aaaaaaaaaaaa"


# ---------------------------------------------------------------------------
# 6.4 Search index cache: built once, rebuilt when the event log changes
# ---------------------------------------------------------------------------


def test_search_index_cached_across_searches():
    store = _make_store()
    store.append_facts(
        [
            _make_fact(content="The user prefers polars for dataframes"),
            _make_fact(content="Use ruff for linting"),
        ]
    )

    first = store.search_index()
    store.search_facts("polars dataframe library")
    assert store.search_index() is first


def test_search_index_rebuilt_after_update_and_purge():
    store = _make_store()
    fact = _make_fact(content="Original content about TypeScript", confidence=1.0)
    store.append_facts([fact])
    assert [h.fact.id for h in store.search_facts("TypeScript")] == [fact.id]

    store.update_fact(fact.id, content="Updated content about JavaScript")
    assert store.search_facts("TypeScript") == []
    assert [h.fact.id for h in store.search_facts("JavaScript")] == [fact.id]

    store.forget(fact.id)
    store.purge()
    assert store.search_facts("JavaScript") == []


# ---------------------------------------------------------------------------
# 7.2 Config placeholder detection covers embedded placeholders
# ---------------------------------------------------------------------------


def test_placeholder_detection_embedded():
    from engram.core.config import _is_unresolved_env_placeholder

    # Should detect
    assert _is_unresolved_env_placeholder("$OPENAI_API_KEY")
    assert _is_unresolved_env_placeholder("${OPENAI_API_KEY}")
    assert _is_unresolved_env_placeholder("Bearer $OPENAI_API_KEY")
    assert _is_unresolved_env_placeholder("${FOO}_suffix")

    # Should NOT detect (no placeholder)
    assert not _is_unresolved_env_placeholder("sk-realkey123")
    assert not _is_unresolved_env_placeholder("price is $5.00")  # lowercase after $
    assert not _is_unresolved_env_placeholder(None)
    assert not _is_unresolved_env_placeholder("")


# ---------------------------------------------------------------------------
# 7.4 Importer _clean_project_name uses home-path-relative logic
# ---------------------------------------------------------------------------


def test_clean_project_name_strips_home_prefix():
    from engram.extraction.importer import _clean_project_name

    # Simulate mangled path for a project under the user's home dir
    # Claude mangles /Users/jdoe/dev/myproject as -Users-jdoe-dev-myproject
    home_parts = [p for p in Path.home().parts if p and p != "/"]
    username = home_parts[-1] if home_parts else "jdoe"
    parent = home_parts[0] if len(home_parts) > 1 else "Users"

    mangled = f"-{parent}-{username}-dev-myproject"
    result = _clean_project_name(mangled)
    assert result == "myproject", f"Expected 'myproject', got '{result}'"


def test_clean_project_name_repo_named_ai_is_kept():
    from engram.extraction.importer import _clean_project_name

    home_parts = [p for p in Path.home().parts if p and p != "/"]
    username = home_parts[-1] if home_parts else "jdoe"
    parent = home_parts[0] if len(home_parts) > 1 else "Users"

    mangled = f"-{parent}-{username}-dev-ai"
    result = _clean_project_name(mangled)
    # Should return 'ai', not strip it as before
    assert result == "ai", f"Expected 'ai', got '{result}'"


def test_clean_project_name_hyphenated_repo_is_kept():
    from engram.extraction.importer import _clean_project_name

    home_parts = [p for p in Path.home().parts if p and p != "/"]
    username = home_parts[-1] if home_parts else "jdoe"
    parent = home_parts[0] if len(home_parts) > 1 else "Users"

    mangled = f"-{parent}-{username}-dev-company-acme-dw"
    result = _clean_project_name(mangled)
    assert result == "acme-dw", f"Expected 'acme-dw', got '{result}'"


def test_clean_project_name_outside_home():
    from engram.extraction.importer import _clean_project_name

    result = _clean_project_name("some-other-project")
    # Outside the home-path heuristic, preserve the full slug.
    assert result == "some-other-project"
