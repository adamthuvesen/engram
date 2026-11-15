"""No file turns strict type checking off, and no file lifts an import ban it does not own."""

import ast
import os
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Any file-level directive: a mode name or a rule override such as reportX=false.
DOWNGRADE = re.compile(r"^\s*#\s*pyright:\s*(?!ignore\b)\S")
# An inline suppression must end with a reason after its rule codes.
UNEXPLAINED = re.compile(r"#\s*(?:noqa:\s*[A-Z0-9, ]+|pyright:\s*ignore(?:\[[^\]]*\])?)\s*$")
BAN_SUPPRESSION = re.compile(r"#\s*(?:ruff:\s*)?noqa:[^#\n]*\bTID251\b", re.IGNORECASE)
# Each per-file TID251 ignore in pyproject.toml, and the banned modules that owner needs.
OWNED_BANS = {
    "src/engram/core/clock.py": set(),
    "src/engram/core/ids.py": {"uuid.uuid4"},
    "src/engram/llm/client.py": {"openai"},
    "tools/smoke.py": {"socket"},
    "tests/test_llm.py": {"openai", "httpx2"},
}


def is_python_file(path: Path) -> bool:
    if path.suffix:
        return path.suffix == ".py"
    with path.open("rb") as handle:
        first_line = handle.readline()
    return first_line.startswith(b"#!") and b"python" in first_line


def listed_python_files(root: Path) -> list[Path]:
    """Tracked files plus untracked ones git does not ignore, so an unstaged file is scanned too."""
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        env=env,
        capture_output=True,
        check=True,
    ).stdout.decode()
    paths = (root / name for name in listing.split("\0") if name)
    return [path for path in paths if path.is_file() and is_python_file(path)]


def downgrades(root: Path) -> list[str]:
    return [
        f"{path.relative_to(root)}:{number}"
        for path in listed_python_files(root)
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1)
        if DOWNGRADE.match(line)
    ]


def test_detector_finds_an_unstaged_downgrade(tmp_path: Path) -> None:
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, env=env, check=True)
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "loose.py").write_text("# pyright: basic\n", encoding="utf-8")
    (tmp_path / "rule.py").write_text("# pyright: reportAssignmentType=false\n")
    (tmp_path / "inline.py").write_text("x: int = 1  # pyright: ignore[reportX]  # why\n")
    assert downgrades(tmp_path) == ["loose.py:1", "rule.py:1"]


def test_no_file_downgrades_type_checking_mode() -> None:
    found = downgrades(ROOT)
    assert not found, (
        "Rule: strict type checking stays on for every file. "
        "File-level `# pyright:` comment turns it off at: " + ", ".join(found) + ". "
        "Fix: delete the comment and fix the type error, or use `# pyright: ignore[rule]  # reason` "
        "on the one line."
    )


def unexplained_suppressions(root: Path) -> list[str]:
    return [
        f"{path.relative_to(root)}:{number}"
        for path in listed_python_files(root)
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1)
        if UNEXPLAINED.search(line)
    ]


def test_every_inline_suppression_gives_a_reason(tmp_path: Path) -> None:
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, env=env, check=True)
    marker = "# " + "noqa: E402"
    (tmp_path / "bare.py").write_text(f"import os  {marker}\n")
    (tmp_path / "why.py").write_text(f"import os  {marker} - after a re-exec\n")
    assert unexplained_suppressions(tmp_path) == ["bare.py:1"]
    found = unexplained_suppressions(ROOT)
    assert not found, (
        "Rule: every inline noqa or pyright ignore ends with a reason. Missing at: "
        + ", ".join(found)
        + ". Fix: append the reason, for example `# noqa: E402 - after a re-exec`."
    )


def ban_suppressions(root: Path) -> list[str]:
    return [
        f"{path.relative_to(root)}:{number}"
        for path in listed_python_files(root)
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1)
        if BAN_SUPPRESSION.search(line)
    ]


def imported_modules(source: str) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def unowned_bans(root: Path, owned: dict[str, set[str]]) -> list[str]:
    lint = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["ruff"]["lint"]
    banned = set(lint["flake8-tidy-imports"]["banned-api"])

    def lifts(codes: list[str]) -> bool:
        return any(code == "ALL" or "TID251".startswith(code) for code in codes)

    per_file = {**lint.get("per-file-ignores", {}), **lint.get("extend-per-file-ignores", {})}
    owners = {path for path, codes in per_file.items() if lifts(codes)}
    problems = [
        f"lint.{key} lifts TID251 for every file"
        for key in ("ignore", "extend-ignore")
        if lifts(lint.get(key, []))
    ]
    if "TID251" not in lint.get("select", []):
        problems.append("lint.select does not select TID251")
    problems += [f"{path}: TID251 ignore has no OWNED_BANS entry" for path in owners - owned.keys()]
    for path in sorted(owners & owned.keys()):
        reported: set[str] = set()
        for module in sorted(imported_modules((root / path).read_text())):
            ban = next((b for b in banned if module == b or module.startswith(b + ".")), None)
            if ban and ban not in owned[path] and ban not in reported:
                reported.add(ban)
                problems.append(f"{path}: imports {module}, but owns only {sorted(owned[path])}")
    return problems


def test_no_file_suppresses_an_import_ban() -> None:
    found = ban_suppressions(ROOT)
    assert not found, (
        "Rule: no noqa on TID251; an import ban has one owner file. Found at: "
        + ", ".join(found)
        + ". Fix: delete the noqa and import through the owner named in the ruff ban message "
        "(engram.llm.client, engram.core.clock, engram.core.ids), or add an owner to "
        "per-file-ignores and OWNED_BANS."
    )


def test_ban_owners_import_only_the_modules_they_own() -> None:
    problems = unowned_bans(ROOT, OWNED_BANS)
    assert problems == [], (
        "Rule: each per-file TID251 ignore in pyproject.toml has an OWNED_BANS entry listing "
        "only the bans that file owns. Problems: " + "; ".join(problems) + ". "
        "Fix: import the banned module through its owner, or update OWNED_BANS and "
        "per-file-ignores together."
    )


def test_network_and_sdk_bans_stay_listed() -> None:
    lint = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["ruff"]["lint"]
    banned = set(lint["flake8-tidy-imports"]["banned-api"])
    required = {
        "openai",
        "httpx",
        "httpx2",
        "requests",
        "urllib.request",
        "http.client",
        "socket",
        "datetime.datetime.now",
        "time.time",
        "uuid.uuid4",
    }
    assert required <= banned, (
        f"Rule: pyproject.toml keeps these bans listed. banned-api lost {sorted(required - banned)}. "
        "Fix: restore the entries in [tool.ruff.lint.flake8-tidy-imports.banned-api], each with a "
        "msg naming the one allowed home."
    )


def test_ban_checks_find_a_suppression_and_a_stray_owner_import(tmp_path: Path) -> None:
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, env=env, check=True)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ruff.lint]\nselect = ["TID251"]\nextend-ignore = ["TID"]\n'
        '[tool.ruff.lint.per-file-ignores]\n"smoke.py" = ["TID251"]\n'
        '[tool.ruff.lint.extend-per-file-ignores]\n"extra.py" = ["TID251"]\n'
        '[tool.ruff.lint.flake8-tidy-imports.banned-api]\n"socket".msg = "x"\n"openai".msg = "x"\n',
        encoding="utf-8",
    )
    (tmp_path / "smoke.py").write_text("import socket\nfrom openai.types import X\n")
    (tmp_path / "quiet.py").write_text("import openai  # " + "noqa: TID251\n")
    assert ban_suppressions(tmp_path) == ["quiet.py:1"]
    assert unowned_bans(tmp_path, {"smoke.py": {"socket"}}) == [
        "lint.extend-ignore lifts TID251 for every file",
        "extra.py: TID251 ignore has no OWNED_BANS entry",
        "smoke.py: imports openai.types, but owns only ['socket']",
    ]


def test_strict_type_settings_hold_their_values() -> None:
    import json

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["basedpyright"]
    assert config["typeCheckingMode"] == "strict"
    assert config["enableTypeIgnoreComments"] is False
    for rule in ("reportUnnecessaryTypeIgnoreComment", "reportIgnoreCommentWithoutRule"):
        assert config[rule] == "error", rule
    verify_project = ROOT / ".agents/skills/verify-engram/pyrightconfig.json"
    assert set(json.loads(verify_project.read_text())) == {"extends", "include"}


SKIP_CALLS = {"pytest.mark.skip", "pytest.mark.xfail", "pytest.skip", "unittest.skip"}
# pytest.skip and unittest.skip take the reason as their first positional argument.
REASON_FIRST_ARGUMENT = {"pytest.skip", "unittest.skip"}


def _dotted(node: ast.expr) -> str:
    names: list[str] = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
    return ".".join(reversed(names))


def _is_text(node: ast.expr) -> bool:
    if isinstance(node, ast.JoinedStr):
        return True
    return (
        isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.strip() != ""
    )


def _has_reason(call: ast.Call, name: str) -> bool:
    given = [keyword.value for keyword in call.keywords if keyword.arg == "reason"]
    if name in REASON_FIRST_ARGUMENT and call.args:
        given.append(call.args[0])
    return any(_is_text(value) for value in given)


def skips_without_reason(source: str) -> list[int]:
    tree = ast.parse(source)
    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _dotted(node.func) in SKIP_CALLS:
            if not _has_reason(node, _dotted(node.func)):
                lines.append(node.lineno)
        elif (
            isinstance(node, ast.Attribute)
            and _dotted(node) in SKIP_CALLS
            and id(node) not in called
        ):
            lines.append(node.lineno)
    return sorted(lines)


def unexplained_skips(root: Path) -> list[str]:
    return [
        f"{path.relative_to(root)}:{line}"
        for path in listed_python_files(root)
        for line in skips_without_reason(path.read_text(errors="replace"))
    ]


def test_every_skip_gives_a_reason(tmp_path: Path) -> None:
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, env=env, check=True)
    (tmp_path / "bare.py").write_text(
        "import pytest\n\n\n@pytest.mark.skip\ndef test_bare() -> None:\n    pass\n"
    )
    (tmp_path / "empty.py").write_text('import pytest\n\npytest.skip("")\n')
    (tmp_path / "why.py").write_text(
        'import pytest\n\n\n@pytest.mark.skip(\n    reason="loads the model"\n)\n'
        'def test_why() -> None:\n    pass\n\n\npytest.skip("no extract at 2026-10-01")\n'
    )
    assert unexplained_skips(tmp_path) == ["bare.py:4", "empty.py:3"]
    found = unexplained_skips(ROOT)
    assert not found, (
        "Rule: every skip and xfail mark gives a non-empty reason. Missing at: "
        + ", ".join(found)
        + '. Fix: pass the reason, for example `@pytest.mark.skip(reason="loads the model")`.'
    )
