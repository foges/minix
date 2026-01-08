#!/usr/bin/env python3
"""
Concatenate a Python/Rust/Swift codebase into one text file with two indexes.

Output format:

# (header comments)

INDEX
<line_in_FILE_INDEX> <relative/path>

FILE_INDEX
<file_start_line> <relative/path>
  <symbol_line> <kind> <name>

CONTENT
----
=== relative/path/to/file.ext ===
<file contents>
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


OUTPUT_HEADER_LINES: tuple[str, ...] = (
    "# This file is an auto-generated concatenation of a codebase.",
    "#",
    "# Structure:",
    "#",
    "# INDEX",
    "#   '<line_in_FILE_INDEX> <path>'",
    "#   Jump to that line number to find the corresponding FILE_INDEX entry for the file.",
    "#",
    "# FILE_INDEX",
    "#   '<file_start_line> <path>'",
    "#     '  <symbol_line> <kind> <name>'",
    "#   The first line for a file tells you where the file's content starts in this combined output.",
    "#   The indented lines list classes/functions (and similar symbols) with their line numbers.",
    "#",
    "# CONTENT",
    "#   After the '----' separator, each file is emitted as:",
    "#     '=== <path> ==='",
    "#     <original file contents>",
    "#",
    "# Notes:",
    "# - All line numbers refer to THIS combined file (not the original repository files).",
    "# - Symbol extraction: Python uses AST; Rust/Swift use regex heuristics (best-effort).",
    "# - Common generated/dependency/VCS artifacts are excluded; pass --include-tests to include tests.",
)


@dataclass(frozen=True)
class Symbol:
    kind: str
    name: str
    line_in_file: int


@dataclass
class FileBundle:
    relative_path: str
    content_lines: list[str]
    symbols: list[Symbol]
    file_start_line: int = 0
    file_index_entry_line: int = 0


ALWAYS_EXCLUDE_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        ".nox",
        ".venv",
        "venv",
        "env",
        "site-packages",
        "node_modules",
        "dist",
        "build",
        "target",
        ".build",
        "DerivedData",
        ".swiftpm",
        "Pods",
        "Carthage",
        ".gradle",
    }
)

ALWAYS_EXCLUDE_FILE_NAMES: frozenset[str] = frozenset(
    {
        ".DS_Store",
        ".gitignore",
        ".gitattributes",
        ".gitmodules",
        ".gitkeep",
        "Cargo.lock",
        "Pipfile.lock",
        "poetry.lock",
        "uv.lock",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
    }
)

TEST_DIR_NAMES: frozenset[str] = frozenset(
    {
        "test",
        "tests",
        "__tests__",
        "spec",
        "specs",
        "Specs",
        "Testing",
        "Tests",
        "UITests",
    }
)

GENERATED_FILE_NAME_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r".*_pb2\.pyi?$"),
    re.compile(r".*_pb2_grpc\.py$"),
    re.compile(r".*_grpc\.py$"),
    re.compile(r".*\.pb\.rs$"),
    re.compile(r".*\.pb\.swift$"),
    re.compile(r".*\.generated\.swift$"),
    re.compile(r".*\.g\.swift$"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="concat_codebase.py")
    parser.add_argument("root", type=Path, help="Path to the repository/codebase root")
    parser.add_argument(
        "--language",
        choices=("auto", "python", "rust", "swift", "all"),
        default="auto",
        help="Which language to include (default: auto).",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include tests (otherwise common test dirs/files are excluded).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="Output path, or '-' for stdout (default: '-').",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Text encoding for source files (default: utf-8).",
    )
    return parser.parse_args()


def allowed_suffixes(language: str) -> frozenset[str]:
    if language == "python":
        return frozenset({".py", ".pyi"})
    if language == "rust":
        return frozenset({".rs"})
    if language == "swift":
        return frozenset({".swift"})
    return frozenset({".py", ".pyi", ".rs", ".swift"})


def is_generated_file_name(file_name: str) -> bool:
    for pattern in GENERATED_FILE_NAME_REGEXES:
        if pattern.fullmatch(file_name):
            return True
    return False


def looks_like_test_file_name(file_name: str) -> bool:
    if file_name.startswith("test_") and file_name.endswith(".py"):
        return True
    if file_name.endswith("_test.py"):
        return True
    if file_name.endswith("Tests.swift"):
        return True
    if file_name.endswith("_test.rs"):
        return True
    return False


def should_exclude_dir_name(dir_name: str) -> bool:
    if dir_name in ALWAYS_EXCLUDE_DIR_NAMES:
        return True
    if dir_name.startswith("."):
        return True
    return False


def should_exclude_file_name(file_name: str) -> bool:
    if file_name in ALWAYS_EXCLUDE_FILE_NAMES:
        return True
    if file_name.startswith("."):
        return True
    if is_generated_file_name(file_name):
        return True
    return False


def path_has_test_segment(relative_parts: tuple[str, ...]) -> bool:
    return any(part in TEST_DIR_NAMES for part in relative_parts)


def iter_source_files(root: Path, *, suffixes: frozenset[str], include_tests: bool) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(str(root))
    if not root.is_dir():
        raise NotADirectoryError(str(root))

    collected: list[Path] = []
    root_str = str(root)

    for dirpath, dirnames, filenames in os.walk(root_str, topdown=True):
        dirnames[:] = [
            d
            for d in dirnames
            if not should_exclude_dir_name(d) and (include_tests or d not in TEST_DIR_NAMES)
        ]

        for file_name in filenames:
            if should_exclude_file_name(file_name):
                continue

            file_path = Path(dirpath) / file_name
            if file_path.suffix not in suffixes:
                continue

            relative_path = file_path.relative_to(root)
            if not include_tests:
                if path_has_test_segment(relative_path.parts) or looks_like_test_file_name(file_name):
                    continue

            collected.append(file_path)

    collected.sort(key=lambda p: p.relative_to(root).as_posix())
    return collected


class _PythonSymbolCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self._scope: list[str] = []
        self.symbols: list[Symbol] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualified_name = ".".join([*self._scope, node.name]) if self._scope else node.name
        self.symbols.append(Symbol(kind="class", name=qualified_name, line_in_file=node.lineno))
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        qualified_name = ".".join([*self._scope, node.name]) if self._scope else node.name
        self.symbols.append(Symbol(kind="def", name=qualified_name, line_in_file=node.lineno))
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        qualified_name = ".".join([*self._scope, node.name]) if self._scope else node.name
        self.symbols.append(Symbol(kind="async def", name=qualified_name, line_in_file=node.lineno))
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()


def extract_python_symbols(source_text: str) -> list[Symbol]:
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        raise ValueError(f"Failed to parse Python file for symbols: {exc}") from exc
    collector = _PythonSymbolCollector()
    collector.visit(tree)
    collector.symbols.sort(key=lambda s: (s.line_in_file, s.kind, s.name))
    return collector.symbols


_RUST_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fn", re.compile(r"^\s*(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
    ("struct", re.compile(r"^\s*(?:pub(?:\([^)]+\))?\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
    ("enum", re.compile(r"^\s*(?:pub(?:\([^)]+\))?\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
    ("trait", re.compile(r"^\s*(?:pub(?:\([^)]+\))?\s+)?trait\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
    ("mod", re.compile(r"^\s*(?:pub(?:\([^)]+\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
    ("macro", re.compile(r"^\s*macro_rules!\s*([A-Za-z_][A-Za-z0-9_]*)\b")),
    ("impl", re.compile(r"^\s*impl(?:\s*<[^>]*>)?\s+(.+?)\s*\{?\s*$")),
)


def extract_rust_symbols(content_lines: list[str]) -> list[Symbol]:
    symbols: list[Symbol] = []
    for index, line in enumerate(content_lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        for kind, pattern in _RUST_PATTERNS:
            match = pattern.match(line)
            if not match:
                continue
            name = match.group(1).strip()
            if name:
                symbols.append(Symbol(kind=kind, name=name, line_in_file=index))
            break
    symbols.sort(key=lambda s: (s.line_in_file, s.kind, s.name))
    return symbols


_SWIFT_TYPE_PATTERN = re.compile(
    r"^\s*(?:(?:public|internal|private|fileprivate|open)\s+)?(?:(?:final|indirect)\s+)?"
    r"(class|struct|enum|protocol|extension|actor)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)
_SWIFT_FUNC_PATTERN = re.compile(
    r"^\s*(?:(?:public|internal|private|fileprivate|open)\s+)?(?:(?:static|class)\s+)?"
    r"(?:(?:mutating|nonmutating)\s+)?func\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)


def extract_swift_symbols(content_lines: list[str]) -> list[Symbol]:
    symbols: list[Symbol] = []
    for index, line in enumerate(content_lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        type_match = _SWIFT_TYPE_PATTERN.match(line)
        if type_match:
            kind, name = type_match.group(1), type_match.group(2)
            symbols.append(Symbol(kind=kind, name=name, line_in_file=index))
            continue
        func_match = _SWIFT_FUNC_PATTERN.match(line)
        if func_match:
            symbols.append(Symbol(kind="func", name=func_match.group(1), line_in_file=index))
    symbols.sort(key=lambda s: (s.line_in_file, s.kind, s.name))
    return symbols


def extract_symbols_for_file(path: Path, content_lines: list[str], source_text: str) -> list[Symbol]:
    suffix = path.suffix
    if suffix in {".py", ".pyi"}:
        return extract_python_symbols(source_text)
    if suffix == ".rs":
        return extract_rust_symbols(content_lines)
    if suffix == ".swift":
        return extract_swift_symbols(content_lines)
    return []


def build_bundles(files: list[Path], root: Path, encoding: str) -> list[FileBundle]:
    bundles: list[FileBundle] = []
    for path in files:
        relative_path = path.relative_to(root).as_posix()
        source_text = path.read_text(encoding=encoding)
        content_lines = source_text.splitlines()
        symbols = extract_symbols_for_file(path, content_lines, source_text)
        bundles.append(FileBundle(relative_path=relative_path, content_lines=content_lines, symbols=symbols))
    return bundles


def assign_file_index_entry_lines(bundles: list[FileBundle]) -> None:
    header_line_count = len(OUTPUT_HEADER_LINES)
    file_count = len(bundles)

    index_header_line = header_line_count + 1
    index_entry_first_line = index_header_line + 1

    file_index_header_line = index_entry_first_line + file_count
    file_index_entry_first_line = file_index_header_line + 1

    current_line = file_index_entry_first_line
    for bundle in bundles:
        bundle.file_index_entry_line = current_line
        current_line += 1 + len(bundle.symbols)


def assign_file_start_lines(bundles: list[FileBundle]) -> None:
    header_line_count = len(OUTPUT_HEADER_LINES)
    file_count = len(bundles)
    symbol_count = sum(len(b.symbols) for b in bundles)

    pre_content_lines = (
        header_line_count
        + 1  # INDEX
        + file_count  # INDEX entries
        + 1  # FILE_INDEX
        + (file_count + symbol_count)  # FILE_INDEX entries (file lines + symbol lines)
        + 1  # CONTENT
        + 1  # ----
    )

    current_line = pre_content_lines + 1
    for bundle in bundles:
        bundle.file_start_line = current_line
        current_line += 1 + len(bundle.content_lines) + 1


def render_output(bundles: list[FileBundle]) -> str:
    assign_file_index_entry_lines(bundles)
    assign_file_start_lines(bundles)

    output_lines: list[str] = list(OUTPUT_HEADER_LINES)

    output_lines.append("INDEX")
    for bundle in bundles:
        output_lines.append(f"{bundle.file_index_entry_line} {bundle.relative_path}")

    output_lines.append("FILE_INDEX")
    for bundle in bundles:
        output_lines.append(f"{bundle.file_start_line} {bundle.relative_path}")
        for symbol in bundle.symbols:
            symbol_line = bundle.file_start_line + symbol.line_in_file
            output_lines.append(f"  {symbol_line} {symbol.kind} {symbol.name}")

    output_lines.append("CONTENT")
    output_lines.append("----")

    for bundle in bundles:
        output_lines.append(f"=== {bundle.relative_path} ===")
        output_lines.extend(bundle.content_lines)
        output_lines.append("")

    return "\n".join(output_lines)


def main() -> int:
    args = parse_args()
    root: Path = args.root.resolve()
    suffixes = allowed_suffixes(args.language)

    source_files = iter_source_files(root, suffixes=suffixes, include_tests=args.include_tests)
    bundles = build_bundles(source_files, root=root, encoding=args.encoding)
    combined_text = render_output(bundles)

    if args.output == "-":
        sys.stdout.write(combined_text)
        if not combined_text.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    Path(args.output).write_text(combined_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())