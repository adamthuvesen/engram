"""No file turns strict type checking off, and no file lifts an import ban it does not own."""

import ast
import os
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNGRADE = re.compile(
    r"^\s*#\s*pyright:\s*(?:[^#\n]*,\s*)?(?:basic|standard|off|recommended|all|strict\s*=\s*false)\b"
)
# An inline suppression must end with a reason after its rule codes.
UNEXPLAINED = re.compile(r"#\s*(?:noqa:\s*[A-Z0-9, ]+|pyright:\s*ignore(?:\[[^\]]*\])?)\s*$")
BAN_SUPPRESSION = re.compile(r"#\s*(?:ruff:\s*)?noqa:[^#\n]*\bTID251\b", re.IGNORECASE)
# Each per-file TID251 ignore in pyproject.toml, and the banned modules that owner needs.
OWNED_BANS = {
    "src/engram/llm/client.py": {"openai"},
    "src/engram/tooling/smoke.py": {"socket"},
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
    assert downgrades(tmp_path) == ["loose.py:1"]


def test_no_file_downgrades_type_checking_mode() -> None:
    found = downgrades(ROOT)
    assert not found, "file-level `# pyright:` comment turns strict mode off: " + ", ".join(found)


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
    assert not found, "inline suppression without a reason: " + ", ".join(found)


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
    return modules


def unowned_bans(root: Path, owned: dict[str, set[str]]) -> list[str]:
    lint = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["ruff"]["lint"]
    banned = set(lint["flake8-tidy-imports"]["banned-api"])
    owners = {path for path, codes in lint["per-file-ignores"].items() if "TID251" in codes}
    problems = [f"{path}: TID251 ignore has no OWNED_BANS entry" for path in owners - owned.keys()]
    for path in sorted(owners & owned.keys()):
        for module in sorted(imported_modules((root / path).read_text())):
            ban = next((b for b in banned if module == b or module.startswith(b + ".")), None)
            if ban and ban not in owned[path]:
                problems.append(f"{path}: imports {module}, but owns only {sorted(owned[path])}")
    return problems


def test_no_file_suppresses_an_import_ban() -> None:
    found = ban_suppressions(ROOT)
    assert not found, "noqa on TID251 lifts an import ban: " + ", ".join(found)


def test_ban_owners_import_only_the_modules_they_own() -> None:
    assert unowned_bans(ROOT, OWNED_BANS) == []


def test_ban_checks_find_a_suppression_and_a_stray_owner_import(tmp_path: Path) -> None:
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, env=env, check=True)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ruff.lint.per-file-ignores]\n"smoke.py" = ["TID251"]\n"extra.py" = ["TID251"]\n'
        '[tool.ruff.lint.flake8-tidy-imports.banned-api]\n"socket".msg = "x"\n"openai".msg = "x"\n',
        encoding="utf-8",
    )
    (tmp_path / "smoke.py").write_text("import socket\nfrom openai.types import X\n")
    (tmp_path / "quiet.py").write_text("import openai  # " + "noqa: TID251\n")
    assert ban_suppressions(tmp_path) == ["quiet.py:1"]
    assert unowned_bans(tmp_path, {"smoke.py": {"socket"}}) == [
        "extra.py: TID251 ignore has no OWNED_BANS entry",
        "smoke.py: imports openai.types, but owns only ['socket']",
    ]
