"""Offline verification, evaluation injection, and source boundary contracts."""

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from engram.evaluation.fixtures import EvalFactSpec, EvalFixture, run_fixture
from engram.evaluation.recall import DATASET_PATH, evaluate
from engram.tooling.boundaries import check_source
from engram.tooling.reference import configuration_markdown, update_references
from engram.tooling.smoke import child_environment, run_smoke
from engram.tooling.verify import EXCLUDED, CheckResult, VerificationReport, commands
from tests.conftest import pytest_collection_modifyitems


def test_default_verification_excludes_billable_and_push_profiles():
    for profile in ("check", "verify"):
        command = next(command for command in commands(profile) if "pytest" in command)
        assert command[-1].startswith("not live_provider and not sync_transport and ")
    explicit = commands("sync-transport")[0]
    assert explicit[-3:] == ["-m", "sync_transport", "--run-sync-transport"]


def test_wrong_marker_expression_cannot_enable_git_push_fixtures():
    marked = SimpleNamespace(
        get_closest_marker=lambda name: name == "sync_transport",
        add_marker=lambda value: applied.append(value),
    )
    unmarked = SimpleNamespace(
        get_closest_marker=lambda name: False,
        add_marker=lambda value: pytest.fail("Safe unit test was skipped."),
    )
    applied = []
    pytest_collection_modifyitems(SimpleNamespace(getoption=lambda name: False), [marked, unmarked])
    assert len(applied) == 1
    assert "--run-sync-transport" in applied[0].kwargs["reason"]
    applied.clear()
    pytest_collection_modifyitems(SimpleNamespace(getoption=lambda name: True), [marked])
    assert applied == []


def test_excluded_profiles_are_listed_without_changing_status():
    report = VerificationReport(
        profile="verify",
        checks=[CheckResult(name="pytest", status="passed")],
        excluded=EXCLUDED["verify"],
    )
    assert report.status == "passed"
    assert any("verify-sync" in item for item in report.excluded)
    assert not report.failed
    report.checks.append(CheckResult(name="basedpyright", status="failed"))
    assert report.status == "failed"


@pytest.mark.parametrize("import_from_checkout", [False, True])
def test_package_smoke_isolates_locked_dependencies_and_wheel_import(
    monkeypatch, tmp_path, import_from_checkout
):
    import engram.tooling.verify as verification

    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(tmp_path / "caller-environment"))
    monkeypatch.setenv("VIRTUAL_ENV", str(root / ".venv"))
    monkeypatch.setenv("PYTHONPATH", str(root / "src"))
    calls = []
    environments: list[Path] = []

    def run(command, *, cwd, env=None, **kwargs):
        calls.append((command, cwd, env))
        if command[:2] == ["uv", "build"]:
            dist = Path(command[command.index("--out-dir") + 1])
            dist.mkdir()
            (dist / "engram-0.1.0-py3-none-any.whl").write_bytes(b"fixture wheel")
        if command[:2] == ["uv", "sync"]:
            assert env is not None
            environments.append(Path(env["UV_PROJECT_ENVIRONMENT"]))
        stdout = ""
        if "-c" in command:
            origin = root if import_from_checkout else environments[0]
            stdout = str(origin / "lib/site-packages/engram/__init__.py")
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(verification.subprocess, "run", run)
    report = VerificationReport(profile="package")
    verification.package_smoke(root, report)
    target = environments[0]

    sync, sync_cwd, sync_env = next(call for call in calls if call[0][:2] == ["uv", "sync"])
    assert {"--locked", "--offline", "--no-dev", "--no-install-project"} <= set(sync)
    assert sync[sync.index("--python") + 1] == verification.sys.executable
    assert sync_cwd == root
    assert target.is_absolute() and not target.is_relative_to(root)
    assert target != tmp_path / "caller-environment"
    assert "VIRTUAL_ENV" not in sync_env and "PYTHONPATH" not in sync_env
    assert not any(command[:2] == ["uv", "export"] for command, _, _ in calls)
    install = next(command for command, _, _ in calls if command[:2] == ["uv", "pip"])
    assert install[:3] == ["uv", "pip", "install"]
    assert {"--offline", "--no-deps"} <= set(install)
    assert install[install.index("--python") + 1] == str(target / "bin/python")
    origin_call = next(call for call in calls if "-c" in call[0])
    assert origin_call[0][0] == str(target / "bin/python")
    assert not origin_call[1].is_relative_to(root)
    assert "PYTHONPATH" not in origin_call[2]
    assert "VIRTUAL_ENV" not in origin_call[2]
    assert report.failed is import_from_checkout
    smoke_calls = [call for call in calls if "engram.tooling.smoke" in call[0]]
    if import_from_checkout:
        assert report.checks[-1].name == "installed import origin"
        assert smoke_calls == []
    else:
        assert len(smoke_calls) == 1
        assert smoke_calls[0][0][0] == str(target / "bin/python")
        assert not smoke_calls[0][1].is_relative_to(root)


@pytest.mark.parametrize(
    "source",
    [
        "from engram.storage.store import _is_active_fact\n",
        "import engram.storage.store as storage\nvalue = storage._is_expired_fact\n",
        "from engram.storage.store import FactStore as Store\nasync def read():\n return Store()\n",
        "import engram.storage.store as storage\nasync def read():\n return storage.FactStore()\n",
        "async def read(store):\n return store.sync_store.load_facts()\n",
        "from engram.storage.store import FactStore\nasync def read(d):\n return FactStore.open_readonly(d)\n",
        "import engram.storage.store as s\nasync def read(d):\n return s.FactStore.open_readonly(d)\n",
    ],
)
def test_boundary_rejects_private_policies_and_sync_async_bypasses(source):
    assert check_source(source, "src/engram/operations.py")


def test_boundary_allows_explicit_thread_facade():
    source = "from engram.storage.store import FactStore, run_sync\nasync def read():\n return await run_sync(FactStore)\n"
    assert check_source(source, "src/engram/operations.py") == []
    assert check_source("value = store.sync_store", "src/engram/dashboard/app.py")


def test_config_reference_normalizes_home_and_has_every_setting():
    from engram.core.config import Settings

    text = configuration_markdown()
    assert str(Path.home()) not in text
    assert "~/.engram/data" in text
    for name in Settings.model_fields:
        assert f"ENGRAM_{name.upper()}" in text


def test_generated_references_are_current():
    root = Path(__file__).resolve().parents[1]
    assert asyncio.run(update_references(root, check=True)) == []


def test_eval_provenance_uses_exact_dataset_bytes():
    summary = evaluate()
    assert (
        summary.provenance.dataset_sha256 == hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    )
    assert summary.provenance.live_status.startswith("skipped")


def test_concurrent_fixture_clients_do_not_read_keys_or_mutate_retriever(monkeypatch):
    import engram.recall.retriever as retriever

    original = retriever._llm_available
    monkeypatch.setattr(
        "engram.llm.client.ensure_openai_api_key",
        lambda: pytest.fail("Fixture attempted key discovery."),
    )
    fixture = EvalFixture(
        name="injected",
        query="zebra memory",
        recall_mode="answer",
        facts=[EvalFactSpec(id="aaaaaaaaaaaa", content="zebra memory")],
        mocked_responses=[
            (
                '{"claims":[{"text":"Zebra memory.","fact_ids":["aaaaaaaaaaaa"]}],"quality":"high"}',
                10,
                0,
            )
        ],
        answer_contains=["Zebra memory"],
    )

    async def run():
        return await asyncio.gather(run_fixture(fixture), run_fixture(fixture))

    results = asyncio.run(run())
    assert all(result.passed and result.llm_calls == 1 for result in results)
    assert retriever._llm_available is original


def test_smoke_environment_disables_external_side_effects(monkeypatch, tmp_path):
    monkeypatch.setenv("ENGRAM_DATA_DIR", "/must/not/read")
    monkeypatch.setenv("ENGRAM_SYNC_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("PYTHONPATH", "/must/not/import")
    env = child_environment(tmp_path)
    assert env["ENGRAM_DATA_DIR"] == str(tmp_path)
    assert env["ENGRAM_SYNC_ENABLED"] == env["ENGRAM_MAINTENANCE_ENABLED"] == "false"
    assert "OPENAI_API_KEY" not in env and "PYTHONPATH" not in env


@pytest.mark.subprocess_smoke
def test_mcp_transport_restart_and_packaged_asset_smoke():
    results = asyncio.run(run_smoke(assets=True))
    assert [result.transport for result in results] == ["stdio", "http", "assets"]
    assert "restart persistence" in results[1].checks
    assert all(result.provider == "disabled" for result in results)


@pytest.mark.parametrize("load_error", [False, True])
def test_asset_only_smoke_requires_persisted_fact_in_dashboard(monkeypatch, load_error):
    if load_error:

        def fail_load(store):
            raise OSError("fixture read denied")

        monkeypatch.setattr("engram.dashboard.app.load_dashboard_data", fail_load)
        with pytest.raises(
            RuntimeError, match="Dashboard did not render the persisted smoke fact"
        ) as error:
            asyncio.run(run_smoke(assets=True, transports=False))
        assert "fixture read denied" in str(error.value)
    else:
        results = asyncio.run(run_smoke(assets=True, transports=False))
        assert [result.transport for result in results] == ["assets"]
        assert "persisted dashboard fact" in results[0].checks


def test_report_revision_uses_loaded_source_not_custom_dataset(monkeypatch, tmp_path):
    import engram.evaluation.report as reporting

    source = Path(reporting.__file__).resolve()
    root = source.parents[3]
    calls = []

    def git(command, *, cwd, **kwargs):
        calls.append((command, cwd))
        if "--show-toplevel" in command:
            return SimpleNamespace(returncode=0, stdout=str(root), stderr="")
        if command[-1] == "HEAD":
            return SimpleNamespace(returncode=0, stdout="source-revision", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(reporting.subprocess, "run", git)
    dataset = tmp_path / "other-project" / "dataset.json"
    dataset.parent.mkdir()
    dataset.write_text('{"custom": true}')
    result = reporting.report_provenance(dataset)
    assert result.code_revision == "source-revision"
    assert result.dataset_sha256 == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert calls[0][1] == source.parent
    assert all(cwd != dataset.parent for _, cwd in calls)


def test_report_does_not_attribute_nested_wheel_to_host_repository(monkeypatch, tmp_path):
    import engram.evaluation.report as reporting

    installed = tmp_path / ".venv/lib/site-packages/engram/evaluation/report.py"
    monkeypatch.setattr(reporting, "__file__", str(installed))
    monkeypatch.setattr(
        reporting.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=str(tmp_path), stderr=""),
    )
    assert reporting._source_revision() == (None, None)


def test_supplied_extraction_outputs_have_unknown_live_provenance():
    from engram.evaluation.extraction_quality import evaluate, load_dataset

    outputs = {case.id: case.fixture_extracted_cards for case in load_dataset().cases}
    summary = evaluate(extracted_cards_by_case=outputs)
    assert summary.provenance.execution == "caller_supplied"
    assert summary.provenance.live_status.startswith("unknown")


@pytest.mark.subprocess_smoke
def test_console_entrypoint_smoke_uses_blocked_provider():
    results = asyncio.run(run_smoke(entrypoints=True))
    assert [result.transport for result in results] == ["cli", "stdio", "http"]
    assert all(result.provider == "blocked" for result in results)


def test_tests_cannot_reach_live_provider():
    from engram.llm import client

    with pytest.raises(AssertionError, match="Real provider calls are disabled"):
        client._client()


def citation_problems(agents: str, root: Path) -> list[str]:
    """Every backticked `path` or `path::name` in an Enforced Rules row must resolve."""
    import re

    assert "\n## Enforced Rules\n" in agents, "AGENTS.md lost its ## Enforced Rules section."
    section = agents.split("\n## Enforced Rules\n", 1)[1].split("\n## ", 1)[0]
    rows = [row for row in section.splitlines() if row.startswith("| ")][2:]
    assert rows, "AGENTS.md Enforced Rules table has no rows."
    problems: list[str] = []
    for row in rows:
        cited = re.findall(r"`([^`]+)`", row.split("|")[2])
        if not cited:
            problems.append(f"Enforced Rules row cites no path: {row}")
        for citation in cited:
            path, _, name = citation.partition("::")
            target = root / path
            if not target.is_file():
                problems.append(f"AGENTS.md cites missing path {path}.")
                continue
            if target.name == "justfile":
                defined = rf"^{re.escape(name)}(?:\s[^\n]*)?:"
            else:
                defined = (
                    rf"^\s*(?:(?:async\s+)?def|class)\s+{re.escape(name)}\b|^{re.escape(name)}\s*="
                )
            if name and not re.search(defined, target.read_text(), re.M):
                problems.append(f"AGENTS.md cites {path}::{name}, but {path} does not define it.")
    return problems


def test_enforced_rules_cite_existing_paths():
    root = Path(__file__).resolve().parents[1]
    assert citation_problems((root / "AGENTS.md").read_text(), root) == []


def test_citation_check_finds_rotten_citations():
    root = Path(__file__).resolve().parents[1]
    table = "\n## Enforced Rules\n\n| Rule | Enforced by | Added |\n| --- | --- | --- |\n"
    rows = [
        "| a | nothing cited | d |",
        "| b | `tests/test_tooling.py::test_named_nowhere` | d |",
        "| c | `justfile::check-everything` | d |",
        "| d | `.agents/skills/verify-engram/verify::no_such_name` | d |",
        "| e | `docs/missing.md` | d |",
    ]
    assert citation_problems(table + "\n".join(rows) + "\n", root) == [
        "Enforced Rules row cites no path: | a | nothing cited | d |",
        "AGENTS.md cites tests/test_tooling.py::test_named_nowhere, but tests/test_tooling.py does not define it.",
        "AGENTS.md cites justfile::check-everything, but justfile does not define it.",
        "AGENTS.md cites .agents/skills/verify-engram/verify::no_such_name, but .agents/skills/verify-engram/verify does not define it.",
        "AGENTS.md cites missing path docs/missing.md.",
    ]
