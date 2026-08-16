#!/usr/bin/env python3
"""Show exact test definitions and pytest items for every test module.

Run from the project root with: ``python tests/test_breakdown.py``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TESTS_PATH = Path("tests")


@dataclass(frozen=True)
class ReportFunction:
    name: str
    line: int
    docstring: str
    item_count: int


@dataclass(frozen=True)
class ReportFile:
    path: Path
    functions: tuple[ReportFunction, ...]

    @property
    def definition_count(self) -> int:
        return len(self.functions)

    @property
    def item_count(self) -> int:
        return sum(function.item_count for function in self.functions)


def collect_pytest_items() -> list[str]:
    """Return node IDs directly from pytest, the authority on executable items."""

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("pytest collection failed")

    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    ]


def count_items_by_function(node_ids: list[str]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for node_id in node_ids:
        file_path, *test_parts = node_id.split("::")
        relative_path = str(Path(file_path).relative_to(TESTS_PATH))
        function_name = test_parts[-1].split("[", 1)[0]
        counts[(relative_path, function_name)] += 1
    return counts


def inspect_test_files(item_counts: Counter[tuple[str, str]]) -> list[ReportFile]:
    files: list[ReportFile] = []
    for path in sorted(TESTS_PATH.rglob("test_*.py")):
        if "__pycache__" in path.parts:
            continue

        relative_path = path.relative_to(TESTS_PATH)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        nodes = sorted(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            ),
            key=lambda node: node.lineno,
        )
        functions = tuple(
            ReportFunction(
                name=node.name,
                line=node.lineno,
                docstring=ast.get_docstring(node) or "",
                item_count=item_counts[(str(relative_path), node.name)],
            )
            for node in nodes
        )
        files.append(ReportFile(path=relative_path, functions=functions))
    return files


def print_report(files: list[ReportFile], node_ids: list[str]) -> None:
    total_definitions = sum(file.definition_count for file in files)
    total_items = sum(file.item_count for file in files)

    print("COMPLETE TEST BREAKDOWN")
    print("=" * 80)
    for file in files:
        print(f"\n{file.path}")
        if not file.functions:
            print("  (reporting utility; no test functions)")
        for function in file.functions:
            suffix = (
                f"; {function.item_count} parametrized items" if function.item_count > 1 else ""
            )
            print(f"  - {function.name} (line {function.line}{suffix})")
            if function.docstring:
                print(f"    {function.docstring}")
        print(f"  definitions={file.definition_count}, pytest_items={file.item_count}")

    print("\nSUMMARY")
    print("=" * 80)
    print(f"Test function definitions: {total_definitions}")
    print(f"Test items from breakdown: {total_items}")
    print(f"Test items collected by pytest: {len(node_ids)}")
    print(f"Additional parametrized items: {total_items - total_definitions}")

    if total_items != len(node_ids):
        raise RuntimeError("breakdown does not match pytest collection")


def main() -> None:
    node_ids = collect_pytest_items()
    counts = count_items_by_function(node_ids)
    files = inspect_test_files(counts)
    print_report(files, node_ids)


if __name__ == "__main__":
    main()
