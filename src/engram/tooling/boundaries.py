"""Check the storage and async application boundaries."""

import ast
from pathlib import Path

from pydantic import BaseModel


class BoundaryViolation(BaseModel):
    path: str
    line: int
    message: str


def check_source(source: str, path: str) -> list[BoundaryViolation]:
    tree = ast.parse(source, filename=path)
    violations: list[BoundaryViolation] = []
    storage_modules: set[str] = set()
    fact_store_names: set[str] = set()
    is_storage = path.startswith("src/engram/storage/")
    is_dashboard = path.startswith("src/engram/dashboard/")

    def report(node: ast.AST, message: str) -> None:
        violations.append(
            BoundaryViolation(
                path=path, line=getattr(node, "lineno", 1), message=message
            )
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "engram.storage.store":
            for alias in node.names:
                if alias.name.startswith("_") and not is_storage:
                    report(node, "Use the public lifecycle policy or storage API.")
                if alias.name == "FactStore":
                    fact_store_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "engram.storage.store":
                    storage_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "engram.storage":
            storage_modules.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "store"
            )
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "sync_store"
            and is_dashboard
        ):
            report(node, "Dashboard code must use the public store API.")

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("_")
            and not is_storage
        ):
            if ast.unparse(node.value) in storage_modules:
                report(node, "Use the public lifecycle policy or storage API.")
        if not isinstance(node, ast.AsyncFunctionDef) or is_storage:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                direct_constructor = (
                    isinstance(child.func, ast.Name)
                    and child.func.id in fact_store_names
                )
                storage_class = (
                    isinstance(child.func, ast.Attribute)
                    and child.func.attr == "FactStore"
                    and ast.unparse(child.func.value) in storage_modules
                )
                if direct_constructor or storage_class:
                    report(
                        child,
                        "Construct FactStore through run_sync or asyncio.to_thread in async code.",
                    )
            if isinstance(child, ast.Attribute) and child.attr == "sync_store":
                report(child, "Use AsyncFactStore in async application code.")
    return violations


def check_boundaries(root: Path) -> list[BoundaryViolation]:
    return [
        violation
        for path in sorted((root / "src" / "engram").rglob("*.py"))
        for violation in check_source(path.read_text(), str(path.relative_to(root)))
    ]
