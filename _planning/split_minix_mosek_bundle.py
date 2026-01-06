#!/usr/bin/env python3
"""
Split a Minix "bundle" text file into multiple files.

Usage:
  python split_minix_mosek_bundle.py --bundle minix_mosek_code_bundle.txt --root /path/to/repo

The bundle format is:

  === relative/path/to/file.rs ===
  <file contents...>

Repeated for each file.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def split_bundle(bundle_path: Path) -> dict[str, str]:
    text = bundle_path.read_text(encoding="utf-8").splitlines(keepends=True)

    files: dict[str, list[str]] = {}
    current_path: str | None = None

    for line in text:
        if line.startswith("=== ") and line.rstrip().endswith(" ==="):
            path = line.strip()[4:-4].strip()
            if not path:
                raise ValueError("Empty path marker in bundle.")
            current_path = path
            files.setdefault(current_path, [])
            continue

        if current_path is None:
            continue

        files[current_path].append(line)

    return {p: "".join(lines) for p, lines in files.items()}


def write_files(files: dict[str, str], root: Path, *, dry_run: bool) -> None:
    for rel_path, content in files.items():
        out_path = (root / rel_path).resolve()
        if dry_run:
            print(f"[dry-run] would write: {out_path} ({len(content.encode('utf-8'))} bytes)")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        print(f"wrote: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.bundle.exists():
        raise FileNotFoundError(args.bundle)

    files = split_bundle(args.bundle)
    if not files:
        raise ValueError("No files found in bundle. Check marker formatting.")

    write_files(files, args.root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
