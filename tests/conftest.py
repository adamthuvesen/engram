"""Shared pytest fixtures."""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live_provider: permits provider calls only with ENGRAM_EVAL_PROVIDER=1",
    )
    config.addinivalue_line("markers", "sync_transport: pushes only to temporary local Git remotes")
    config.addinivalue_line(
        "markers", "subprocess_smoke: starts installed entrypoints; runs in verify only"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--run-sync-transport",
        action="store_true",
        default=False,
        help="Allow sync tests that push to temporary local Git remotes.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-sync-transport"):
        return
    skip = pytest.mark.skip(
        reason="Sync transport requires --run-sync-transport and permission to push to temporary local remotes."
    )
    for item in items:
        if item.get_closest_marker("sync_transport"):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_llm_key(monkeypatch, request):
    """Disable key discovery and reject unmocked provider connections.

    Live provider tests need both the explicit marker and environment opt-in.
    Unit tests can replace the provider factory or bind an injected fake client.
    """

    def reject_provider_connection():
        raise AssertionError("Real provider calls are disabled in tests. Inject a fake client.")

    live_allowed = (
        request.node.get_closest_marker("live_provider")
        and os.environ.get("ENGRAM_EVAL_PROVIDER") == "1"
    )
    if not live_allowed:
        monkeypatch.setattr("engram.llm.client._client", reject_provider_connection)
    if not live_allowed:
        monkeypatch.setattr("engram.llm.client.ensure_openai_api_key", lambda: None)
    # Upkeep's LLM steps (consolidate, briefs) check the same key cache.
    monkeypatch.setattr("engram.maintenance.upkeep.llm_available", lambda: False)
    monkeypatch.setenv("ENGRAM_MAINTENANCE_ENABLED", "false")
