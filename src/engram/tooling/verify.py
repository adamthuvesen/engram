"""Run the same locked, offline verification profiles locally and in CI."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import BaseModel, Field

from engram.tooling.boundaries import check_boundaries

Profile = Literal["check", "verify", "sync-transport", "package"]
OFFLINE_SELECTION = "not live_provider and not sync_transport"
FAST_SELECTION = f"{OFFLINE_SELECTION} and not subprocess_smoke"
FOCUSED_TESTS = [
    "tests/test_models.py",
    "tests/test_event_log.py",
    "tests/test_interfaces.py",
    "tests/test_retriever.py",
    "tests/test_llm.py",
    "tests/test_tooling.py",
]


class CheckResult(BaseModel):
    name: str
    status: Literal["passed", "failed", "skipped"]
    reason: str = ""


class VerificationReport(BaseModel):
    profile: Profile
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(check.status == "failed" for check in self.checks)

    @property
    def status(self) -> str:
        if self.failed:
            return "failed"
        return (
            "passed_with_limits"
            if any(check.status == "skipped" for check in self.checks)
            else "passed"
        )


def commands(profile: Profile) -> list[list[str]]:
    if profile == "sync-transport":
        return [
            [
                "uv",
                "run",
                "--locked",
                "--extra",
                "dev",
                "pytest",
                "tests/test_sync.py",
                "-q",
                "-m",
                "sync_transport",
                "--run-sync-transport",
            ]
        ]
    if profile == "package":
        return []
    prefix = ["uv", "run", "--locked", "--extra", "dev"]
    tests, selection = (
        (FOCUSED_TESTS, FAST_SELECTION)
        if profile == "check"
        else (["tests/"], OFFLINE_SELECTION)
    )
    return [
        ["uv", "lock", "--check"],
        [*prefix, "ruff", "check", "."],
        [*prefix, "ruff", "format", "--check", "."],
        [*prefix, "mypy"],
        [*prefix, "python", "-m", "engram.tooling.reference", "--check"],
        [*prefix, "pytest", *tests, "-q", "-m", selection],
    ]


def _run(
    command: list[str],
    root: Path,
    report: VerificationReport,
    *,
    env: dict[str, str] | None = None,
) -> bool:
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=root, env=env, check=False)
    report.checks.append(
        CheckResult(
            name=" ".join(command),
            status="passed" if result.returncode == 0 else "failed",
            reason="" if result.returncode == 0 else f"exit {result.returncode}",
        )
    )
    return result.returncode == 0


def package_smoke(root: Path, report: VerificationReport) -> None:
    """Install a wheel and locked dependencies outside the source checkout."""
    with TemporaryDirectory(prefix="engram-wheel-") as temporary:
        work = Path(temporary)
        dist = work / "dist"
        environment = work / "environment"
        for command in (
            ["uv", "lock", "--check"],
            ["uv", "build", "--no-build-isolation", "--wheel", "--out-dir", str(dist)],
        ):
            if not _run(command, root, report):
                return
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("PYTHONPATH", "VIRTUAL_ENV")
        }
        if not _run(
            [
                "uv",
                "sync",
                "--locked",
                "--offline",
                "--no-dev",
                "--no-install-project",
                "--python",
                sys.executable,
            ],
            root,
            report,
            env={**env, "UV_PROJECT_ENVIRONMENT": str(environment)},
        ):
            return
        wheels = list(dist.glob("*.whl"))
        if len(wheels) != 1:
            report.checks.append(
                CheckResult(
                    name="wheel discovery",
                    status="failed",
                    reason="Expected exactly one built wheel.",
                )
            )
            return
        if not _run(
            [
                "uv",
                "pip",
                "install",
                "--offline",
                "--no-deps",
                "--python",
                str(environment / "bin" / "python"),
                str(wheels[0]),
            ],
            root,
            report,
        ):
            return
        python = str(environment / "bin" / "python")
        origin = subprocess.run(
            [python, "-c", "import engram; print(engram.__file__)"],
            cwd=work,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if origin.returncode != 0 or Path(origin.stdout.strip()).is_relative_to(root):
            report.checks.append(
                CheckResult(
                    name="installed import origin",
                    status="failed",
                    reason=origin.stderr or origin.stdout,
                )
            )
            return
        report.checks.append(
            CheckResult(
                name="installed import origin",
                status="passed",
                reason=origin.stdout.strip(),
            )
        )
        command = [
            python,
            "-m",
            "engram.tooling.smoke",
            "run",
            "--assets",
            "--entrypoints",
        ]
        print("+ " + " ".join(command), flush=True)
        result = subprocess.run(command, cwd=work, env=env, check=False)
        report.checks.append(
            CheckResult(
                name="installed wheel transports and assets",
                status="passed" if result.returncode == 0 else "failed",
                reason=f"exit {result.returncode}",
            )
        )


def verify(root: Path, profile: Profile) -> VerificationReport:
    report = VerificationReport(profile=profile)
    for executable in ("uv", "git"):
        if shutil.which(executable) is None:
            report.checks.append(
                CheckResult(
                    name=executable,
                    status="failed",
                    reason=f"Required executable {executable!r} is missing.",
                )
            )
    if sys.version_info < (3, 13):
        report.checks.append(
            CheckResult(
                name="Python version",
                status="failed",
                reason="Python 3.13 or newer is required.",
            )
        )
    if report.failed:
        return report
    if profile in ("check", "verify"):
        violations = check_boundaries(root)
        report.checks.append(
            CheckResult(
                name="storage and async import boundaries",
                status="failed" if violations else "passed",
                reason="\n".join(
                    f"{item.path}:{item.line}: {item.message}" for item in violations
                ),
            )
        )
        if violations:
            return report
    for command in commands(profile):
        if not _run(command, root, report):
            return report
    if profile in ("verify", "package"):
        package_smoke(root, report)
    if profile != "sync-transport":
        report.checks.extend(
            [
                CheckResult(
                    name="sync transport integration",
                    status="skipped",
                    reason="Required additional profile: just verify-sync. It performs Git pushes to temporary local remotes. The local push safeguard can prohibit it.",
                ),
                CheckResult(
                    name="live provider evaluation",
                    status="skipped",
                    reason="Requires explicit ENGRAM_EVAL_PROVIDER=1 and credentials. Offline checks do not prove live model quality.",
                ),
            ]
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile",
        choices=("check", "verify", "sync-transport", "package"),
        nargs="?",
        default="check",
    )
    args = parser.parse_args()
    report = verify(Path.cwd(), args.profile)
    print(f"Verification profile: {report.profile}. Status: {report.status}")
    for check in report.checks:
        print(
            f"{check.status}: {check.name}"
            + (f". {check.reason}" if check.reason else "")
        )
    return int(report.failed)


if __name__ == "__main__":
    raise SystemExit(main())
