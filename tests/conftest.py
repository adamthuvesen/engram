"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _no_llm_key(monkeypatch):
    """Run every test as if no LLM provider key were configured.

    Recall spends an LLM call only when ``_llm_available()`` is true (answer
    mode, or cards mode when no hit clears the relevance bar), and that check
    auto-loads keys from the developer's key cache — without this pin, any
    such recall in a test would make a real, billed LLM call on a keyed
    machine. Tests that exercise those paths monkeypatch ``_llm_available``
    back to True and mock ``complete_with_usage`` / ``complete_model``.
    """
    monkeypatch.setattr("engram.recall.retriever._llm_available", lambda: False)
    # Upkeep's LLM steps (consolidate, briefs) check the same key cache.
    monkeypatch.setattr("engram.maintenance.upkeep.llm_available", lambda: False)
    monkeypatch.setenv("ENGRAM_MAINTENANCE_ENABLED", "false")
