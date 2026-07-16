"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _no_llm_key(monkeypatch):
    """Run every test as if no LLM provider key were configured.

    Zero-hit recall escalates to the tier-1 LLM search only when
    ``_llm_available()`` is true, and that check auto-loads keys from the
    developer's key cache — without this pin, any unmocked zero-hit recall in
    a test would make a real, billed LLM call on a keyed machine. Tests that
    exercise escalation monkeypatch ``_llm_available`` back to True and mock
    ``complete_with_usage``.
    """
    monkeypatch.setattr("engram.recall.retriever._llm_available", lambda: False)
