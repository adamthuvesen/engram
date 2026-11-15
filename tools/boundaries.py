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

    def report(node: ast.AST, message: str) -> None:
        violations.append(
            BoundaryViolation(path=path, line=getattr(node, "lineno", 1), message=message)
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "engram.storage.store":
            for alias in node.names:
                if alias.name.startswith("_") and not is_storage:
                    report(
                        node,
                        (
                            "Rule: only src/engram/storage/ may use underscore names from engram.storage.store. "
                            "Fix: use the public lifecycle policy in engram.core.fact_state or a public "
                            "FactStore method."
                        ),
                    )
                if alias.name == "FactStore":
                    fact_store_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "engram.storage.store":
                    storage_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "engram.storage":
            storage_modules.update(
                alias.asname or alias.name for alias in node.names if alias.name == "store"
            )

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("_")
            and not is_storage
            and ast.unparse(node.value) in storage_modules
        ):
            report(
                node,
                (
                    "Rule: only src/engram/storage/ may use underscore names from engram.storage.store. "
                    "Fix: use the public lifecycle policy in engram.core.fact_state or a public "
                    "FactStore method."
                ),
            )
        if not isinstance(node, ast.AsyncFunctionDef) or is_storage:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                direct_constructor = (
                    isinstance(child.func, ast.Name) and child.func.id in fact_store_names
                )
                storage_class = (
                    isinstance(child.func, ast.Attribute)
                    and child.func.attr == "FactStore"
                    and ast.unparse(child.func.value) in storage_modules
                )
                classmethod_constructor = (
                    isinstance(child.func, ast.Attribute)
                    and child.func.attr == "open_readonly"
                    and (
                        ast.unparse(child.func.value) in fact_store_names
                        or ast.unparse(child.func.value)
                        in {f"{module}.FactStore" for module in storage_modules}
                    )
                )
                if direct_constructor or storage_class or classmethod_constructor:
                    report(
                        child,
                        (
                            "Rule: async code never builds FactStore on the event loop. "
                            "Fix: construct it inside run_sync(...) or asyncio.to_thread(...), "
                            "or use AsyncFactStore."
                        ),
                    )
            if isinstance(child, ast.Attribute) and child.attr == "sync_store":
                report(
                    child,
                    (
                        "Rule: async code never touches .sync_store. "
                        "Fix: await the matching AsyncFactStore method."
                    ),
                )
    return violations


def check_boundaries(root: Path) -> list[BoundaryViolation]:
    return [
        violation
        for package in ("src/engram", "evals", "tools")
        for path in sorted((root / package).rglob("*.py"))
        for violation in check_source(path.read_text(), str(path.relative_to(root)))
    ]
