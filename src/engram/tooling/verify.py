"""Run the same locked, offline verification profiles locally and in CI.

``check`` runs static checks, then the offline suite on parallel pytest workers.
``verify`` adds only what ``check`` leaves out: subprocess smokes and the installed wheel.
"""

from __future__ import annotations

import argparse
import json
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
SYNC_TRANSPORT = "sync transport: just verify-sync. It pushes to temporary local Git remotes, which a local push freeze can refuse."
LIVE_PROVIDER = "live provider evaluation: needs ENGRAM_EVAL_PROVIDER=1 and credentials. Offline checks do not prove live model quality."
EXCLUDED: dict[Profile, list[str]] = {
    "check": [
        "subprocess smokes and installed wheel: just smoke.",
        SYNC_TRANSPORT,
        LIVE_PROVIDER,
    ],
    "verify": [SYNC_TRANSPORT, LIVE_PROVIDER],
    "sync-transport": [LIVE_PROVIDER],
    "package": [SYNC_TRANSPORT, LIVE_PROVIDER],
}


BASELINE = Path(".basedpyright/baseline.json")
MAIN_REF = "origin/main"


def baseline_error_count(text: str) -> int:
    """Count the errors a basedpyright baseline file records."""
    files: dict[str, list[object]] = json.loads(text)["files"]
    return sum(len(errors) for errors in files.values())


def baseline_ratchet(root: Path) -> CheckResult:
    """Fail when the type-error baseline holds more errors than on origin/main."""
    name = f"basedpyright baseline does not grow past {MAIN_REF}"
    path = root / BASELINE
    count = baseline_error_count(path.read_text()) if path.exists() else 0
    shown = subprocess.run(
        ["git", "show", f"{MAIN_REF}:{BASELINE.as_posix()}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if shown.returncode != 0:
        known = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", MAIN_REF],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if known.returncode != 0:
            return CheckResult(
                name=name,
                status="failed",
                reason=f"{MAIN_REF} is missing. Run git fetch origin main.",
            )
        return CheckResult(name=name, status="passed", reason="main has no baseline yet")
    allowed = baseline_error_count(shown.stdout)
    if count > allowed:
        return CheckResult(
            name=name, status="failed", reason=f"{count} baselined errors, {allowed} on main"
        )
    return CheckResult(name=name, status="passed", reason=f"{count} baselined, {allowed} on main")


class CheckResult(BaseModel):
    name: str
    status: Literal["passed", "failed"]
    reason: str = ""


class VerificationReport(BaseModel):
    profile: Profile
    checks: list[CheckResult] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(check.status == "failed" for check in self.checks)

    @property
    def status(self) -> str:
        return "failed" if self.failed else "passed"


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
    if profile == "verify":
        smoke = f"{OFFLINE_SELECTION} and subprocess_smoke"
        return [[*prefix, "pytest", "tests/", "-q", "-m", smoke]]
    return [
        ["uv", "lock", "--check"],
        [*prefix, "ruff", "check", "."],
        [*prefix, "ruff", "format", "--check", "."],
        [*prefix, "basedpyright"],
        [*prefix, "python", "-m", "engram.tooling.reference", "--check"],
        [*prefix, "pytest", "tests/", "-q", "-n", "auto", "-m", FAST_SELECTION],
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
    report = VerificationReport(profile=profile, excluded=EXCLUDED[profile])
    for executable in ("uv", "git"):
        if shutil.which(executable) is None:
            report.checks.append(
                CheckResult(
                    name=executable,
                    status="failed",
                    reason=f"Required executable {executable!r} is missing.",
                )
            )
    if report.failed:
        return report
    if profile == "check":
        violations = check_boundaries(root)
        report.checks.append(
            CheckResult(
                name="storage and async import boundaries",
                status="failed" if violations else "passed",
                reason="\n".join(f"{item.path}:{item.line}: {item.message}" for item in violations),
            )
        )
        if violations:
            return report
        ratchet = baseline_ratchet(root)
        report.checks.append(ratchet)
        if ratchet.status == "failed":
            return report
    for command in commands(profile):
        if not _run(command, root, report):
            return report
    if profile in ("verify", "package"):
        package_smoke(root, report)
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
        print(f"{check.status}: {check.name}" + (f". {check.reason}" if check.reason else ""))
    for item in report.excluded:
        print(f"excluded: {item}")
    return int(report.failed)


if __name__ == "__main__":
    raise SystemExit(main())
