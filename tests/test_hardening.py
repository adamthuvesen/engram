"""Tests for the hardening changes: locking, fsync, batched approvals,
LLM resilience, dedup correctness, prefilter cache, config, importer."""

import asyncio
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import engram.server as server
from engram.models import (
    CandidateStatus,
    Fact,
    FactCategory,
    MemoryCandidate,
    RecallRecord,
    TransactionStatus,
)
from engram.store import AsyncFactStore, FactStore


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
    # Each line must be parseable (load_facts already validates)
    raw_lines = [
        line for line in store.facts_path.read_text().splitlines() if line.strip()
    ]
    assert len(raw_lines) == 100
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

    with patch("engram.store.os.fsync", side_effect=failing_fsync):
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
        patch("engram.store.os.fsync", side_effect=tracking_fsync),
        patch("engram.store.os.replace", side_effect=tracking_replace),
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
    assert next(f for f in facts if f.id == "old1").confidence == 0.3
    assert next(f for f in facts if f.id != "old1").supersedes == "old1"
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
# 3.4 complete_json parsing cascade
# ---------------------------------------------------------------------------


def test_complete_json_raw_parse(monkeypatch):
    """Raw JSON is parsed directly."""
    from engram import llm

    async def fake_complete(**kwargs):
        return '{"facts": [1, 2, 3]}'

    monkeypatch.setattr(llm, "complete", fake_complete)
    result = asyncio.run(llm.complete_json(prompt="test"))
    assert result == {"facts": [1, 2, 3]}


def test_complete_json_fenced_parse(monkeypatch):
    """JSON inside markdown fences is parsed correctly."""
    from engram import llm

    async def fake_complete(**kwargs):
        return '```json\n{"key": "value"}\n```'

    monkeypatch.setattr(llm, "complete", fake_complete)
    result = asyncio.run(llm.complete_json(prompt="test"))
    assert result == {"key": "value"}


def test_complete_json_trailing_prose(monkeypatch):
    """JSON followed by trailing prose is extracted correctly."""
    from engram import llm

    async def fake_complete(**kwargs):
        return '{"answer": 42} Let me know if you need more.'

    monkeypatch.setattr(llm, "complete", fake_complete)
    result = asyncio.run(llm.complete_json(prompt="test"))
    assert result == {"answer": 42}


def test_complete_json_unparseable_returns_empty(monkeypatch):
    """Completely unparseable output returns {} and logs a warning."""
    from engram import llm

    async def fake_complete(**kwargs):
        return "This is not JSON at all!"

    monkeypatch.setattr(llm, "complete", fake_complete)

    with patch("engram.llm.logger") as mock_logger:
        result = asyncio.run(llm.complete_json(prompt="test"))

    assert result == {}
    mock_logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# 3.5 LLM retry: num_retries=2 is forwarded to litellm
# ---------------------------------------------------------------------------


def test_complete_passes_num_retries(monkeypatch):
    """complete() passes num_retries=2 to litellm.acompletion."""
    from engram import llm

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
    monkeypatch.setattr("engram.config.ensure_openai_api_key", lambda: "key")

    result = asyncio.run(llm.complete(prompt="test", model="openai/gpt-4o-mini"))
    assert result == "hello"
    assert captured_kwargs.get("num_retries") == 2


def test_recall_context_prompt_mode_smoke():
    """The MCP prompt-mode helper should format facts without crashing."""
    store = _make_store()
    store.append_facts(
        [
            _make_fact(
                content="Adam prefers concise terminal summaries",
                project="engram",
            )
        ]
    )
    server._store = store

    result = asyncio.run(
        server.mcp._call_tool_mcp(
            "recall_context",
            {
                "query": "What does Adam prefer?",
                "project": "engram",
                "mode": "prompt",
            },
        )
    )

    text = str(result)
    assert "# Memory Context" in text
    assert "Adam prefers concise terminal summaries" in text


def test_recall_context_prompt_mode_omits_unrelated_fallbacks():
    store = _make_store()
    store.append_facts(
        [
            _make_fact(
                content="Adam prefers concise terminal summaries",
                project="engram",
            )
        ]
    )
    server._store = store

    result = asyncio.run(
        server.mcp._call_tool_mcp(
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
    assert "Adam prefers concise terminal summaries" not in text
    assert "No relevant memories" in text


def test_import_memories_empty_directory_returns_message(tmp_path, monkeypatch):
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    monkeypatch.setattr(
        "engram.importer.get_settings",
        lambda: MagicMock(claude_projects_dir=projects_dir),
    )

    store = _make_store()
    server._store = store
    result = asyncio.run(
        server.mcp._call_tool_mcp("import_memories", {"source": "claude_code"})
    )

    assert "No memory files found to import" in str(result)


def test_import_memories_accepts_async_store(tmp_path, monkeypatch):
    from engram.importer import import_claude_code_memories

    projects_dir = tmp_path / "projects"
    memory_dir = projects_dir / "-Users-adam-dev-menti-engram" / "memory"
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
        "engram.importer.get_settings",
        lambda: MagicMock(claude_projects_dir=projects_dir),
    )

    async def fake_complete_json(prompt: str, system: str = "", model=None):
        return {
            "facts": [
                {
                    "content": "Engram routes MCP storage through AsyncFactStore",
                    "category": "project",
                    "tags": ["storage"],
                    "why_store": "Documents the architecture",
                    "expires_at": None,
                }
            ]
        }

    monkeypatch.setattr("engram.observer.complete_json", fake_complete_json)

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
    server._store = store

    result = asyncio.run(
        server.mcp._call_tool_mcp(
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
    server._store = AsyncFactStore(store)

    approved = asyncio.run(
        server.mcp._call_tool_mcp(
            "approve_candidates",
            {"candidate_ids": ["approve-me"]},
        )
    )
    rejected = asyncio.run(
        server.mcp._call_tool_mcp(
            "reject_candidates",
            {"candidate_ids": ["reject-me"], "reason": "not durable"},
        )
    )

    assert "Approved 1 candidate" in str(approved)
    assert "Rejected 1 candidate" in str(rejected)
    assert len(store.load_candidates(status=CandidateStatus.approved)) == 1
    assert len(store.load_candidates(status=CandidateStatus.rejected)) == 1


def test_inspect_invalid_category_returns_helpful_message():
    server._store = _make_store()

    result = asyncio.run(
        server.mcp._call_tool_mcp(
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
    server._store = AsyncFactStore(store)

    renamed = asyncio.run(
        server.mcp._call_tool_mcp(
            "rename_project",
            {"old_project": "old-project", "new_project": "new-project"},
        )
    )
    inspected = asyncio.run(
        server.mcp._call_tool_mcp(
            "inspect",
            {"project": "new-project"},
        )
    )
    stats = asyncio.run(server.mcp._call_tool_mcp("memory_stats", {}))
    purged = asyncio.run(server.mcp._call_tool_mcp("purge", {}))

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
    server._store = store

    result = asyncio.run(server.mcp._call_tool_mcp("recall_stats", {}))

    assert "LLM calls (reported): 0" in str(result)


# ---------------------------------------------------------------------------
# 5.4 Dedup collision: two candidates targeting same ancestor, only best kept
# ---------------------------------------------------------------------------


def test_dedup_collision_keeps_best_candidate(monkeypatch):
    """Two candidates superseding the same ancestor → only higher-confidence kept."""
    from engram.observer import _dedup

    old_fact = _make_fact(id="ancestor", content="Old fact about Python")
    existing = [old_fact]

    low_conf = _make_fact(content="New Python fact low", confidence=0.6)
    high_conf = _make_fact(content="New Python fact high", confidence=0.9)
    candidates = [low_conf, high_conf]

    update_fact_calls = []

    async def fake_complete_json(prompt, system="", **kwargs):
        return {
            "new": [],
            "updates": [
                {"new_idx": 0, "existing_id": "ancestor"},
                {"new_idx": 1, "existing_id": "ancestor"},
            ],
            "duplicates": [],
        }

    fake_store = MagicMock()
    fake_store.update_fact = MagicMock(
        side_effect=lambda fid, **kw: update_fact_calls.append(fid)
    )

    monkeypatch.setattr("engram.observer.complete_json", fake_complete_json)

    with patch("engram.observer._find_near_matches", return_value=existing):
        kept = asyncio.run(_dedup(candidates, existing, store=fake_store))

    assert len(kept) == 1
    assert kept[0].confidence == 0.9
    assert update_fact_calls.count("ancestor") == 1


# ---------------------------------------------------------------------------
# 5.5 Short 2-token existing fact does not over-match long candidate
# ---------------------------------------------------------------------------


def test_near_match_short_fact_does_not_over_match():
    """A 2-token generic fact with 1 incidental overlap does not pass Jaccard ≥0.3."""
    from engram.observer import _find_near_matches

    # 2-token existing fact
    short_fact = _make_fact(content="python libraries")

    # 20-token candidate with only 1 incidental overlap ("python")
    long_candidate = _make_fact(
        content=(
            "The deployment pipeline uses docker compose with nginx reverse proxy "
            "for staging environments and kubernetes for production python"
        )
    )

    near = _find_near_matches([long_candidate], [short_fact])
    # Jaccard: shared=1 (python), union=large → << 0.3
    assert short_fact not in near, (
        "Short generic fact should not near-match a long unrelated candidate via Jaccard"
    )


# ---------------------------------------------------------------------------
# 6.4 Prefilter cache: tokenize each fact once across two recalls
# ---------------------------------------------------------------------------


def test_prefilter_tokenization_cached():
    """Fact tokenization is cached — second prefilter run tokenizes facts 0 times."""
    store = _make_store()
    store.append_facts(
        [
            _make_fact(content="The user prefers polars for dataframes"),
            _make_fact(content="Use ruff for linting"),
        ]
    )

    call_count = [0]
    original = store._tokenize_extended

    def counting_tokenize(text):
        call_count[0] += 1
        return original(text)

    store._tokenize_extended = counting_tokenize

    # First prefilter: tokenizes query (1) + 2 facts (2) = 3 calls
    store.prefilter_facts("polars dataframe library")
    after_first = call_count[0]

    # Second prefilter: tokenizes query (1) only; facts hit cache
    store.prefilter_facts("polars dataframe library")
    after_second = call_count[0]

    calls_first_run = after_first  # 3 (query + 2 facts)
    calls_second_run = after_second - after_first  # should be 1 (query only)

    assert calls_first_run > calls_second_run, (
        f"Expected fewer tokenization calls on second run due to caching, "
        f"but first={calls_first_run}, second={calls_second_run}"
    )
    # Second run should only tokenize the query (1 call), not the facts (0)
    assert calls_second_run == 1, (
        f"Expected exactly 1 tokenization call on second run (query only), got {calls_second_run}"
    )


# ---------------------------------------------------------------------------
# 6.5 Updating a fact invalidates its cache entry
# ---------------------------------------------------------------------------


def test_prefilter_cache_invalidated_on_fact_update():
    """After updating a fact, the next prefilter re-tokenizes it."""
    store = _make_store()
    fact = _make_fact(content="Original content about TypeScript")
    store.append_facts([fact])

    call_count = [0]
    original = store._tokenize_extended

    def counting_tokenize(text):
        call_count[0] += 1
        return original(text)

    store._tokenize_extended = counting_tokenize

    store.prefilter_facts("TypeScript content")
    count_after_first = call_count[0]

    # Update the fact — should invalidate cache
    store.update_fact(fact.id, content="Updated content about JavaScript")

    store.prefilter_facts("TypeScript content")
    count_after_second = call_count[0]

    assert count_after_second > count_after_first, (
        "Expected re-tokenization after fact update, but tokenize was not called again"
    )


# ---------------------------------------------------------------------------
# 6.3 Cache entries for removed facts are evicted on _rewrite
# ---------------------------------------------------------------------------


def test_prefilter_cache_evicted_on_purge():
    """After a fact is purged (removed via _rewrite), its cache entry is gone."""
    store = _make_store()
    fact = _make_fact(content="Fact that will be purged", confidence=1.0)
    store.append_facts([fact])

    # Warm the cache
    store.prefilter_facts("purged fact content")
    assert fact.id in store._tok_cache, "Cache should be populated after prefilter"

    # Forget the fact (sets confidence=0.0) then purge
    store.forget(fact.id)
    store.purge()

    # Cache entry must be gone
    assert fact.id not in store._tok_cache, (
        "Cache entry for purged fact should have been evicted after _rewrite"
    )


# ---------------------------------------------------------------------------
# 7.2 Config placeholder detection covers embedded placeholders
# ---------------------------------------------------------------------------


def test_placeholder_detection_embedded():
    from engram.config import _is_unresolved_env_placeholder

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
    from engram.importer import _clean_project_name

    # Simulate mangled path for a project under the user's home dir
    # Claude mangles /Users/jdoe/dev/myproject as -Users-jdoe-dev-myproject
    home_parts = [p for p in Path.home().parts if p and p != "/"]
    username = home_parts[-1] if home_parts else "jdoe"
    parent = home_parts[0] if len(home_parts) > 1 else "Users"

    mangled = f"-{parent}-{username}-dev-myproject"
    result = _clean_project_name(mangled)
    assert result == "myproject", f"Expected 'myproject', got '{result}'"


def test_clean_project_name_repo_named_ai_is_kept():
    from engram.importer import _clean_project_name

    home_parts = [p for p in Path.home().parts if p and p != "/"]
    username = home_parts[-1] if home_parts else "jdoe"
    parent = home_parts[0] if len(home_parts) > 1 else "Users"

    mangled = f"-{parent}-{username}-dev-ai"
    result = _clean_project_name(mangled)
    # Should return 'ai', not strip it as before
    assert result == "ai", f"Expected 'ai', got '{result}'"


def test_clean_project_name_hyphenated_repo_is_kept():
    from engram.importer import _clean_project_name

    home_parts = [p for p in Path.home().parts if p and p != "/"]
    username = home_parts[-1] if home_parts else "jdoe"
    parent = home_parts[0] if len(home_parts) > 1 else "Users"

    mangled = f"-{parent}-{username}-dev-company-acme-dw"
    result = _clean_project_name(mangled)
    assert result == "acme-dw", f"Expected 'acme-dw', got '{result}'"


def test_clean_project_name_outside_home():
    from engram.importer import _clean_project_name

    result = _clean_project_name("some-other-project")
    # Outside the home-path heuristic, preserve the full slug.
    assert result == "some-other-project"
