#!/usr/bin/env python3
"""
Split a text bundle into files.

Usage:
  python split_mosek_standalone_linalg_bundle.py mosek_standalone_linalg_bundle.txt ./outdir

Bundle format:

===== FILE: relative/path =====
<file contents>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HEADER_RE = re.compile(r"^===== FILE: (.+?) =====\s*$")


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python split_mosek_standalone_linalg_bundle.py <bundle.txt> <outdir>")
        return 2

    bundle_path = Path(sys.argv[1]).expanduser().resolve()
    outdir = Path(sys.argv[2]).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    lines = bundle_path.read_text(encoding="utf-8").splitlines(keepends=True)

    current_path: Path | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_path, current_lines
        if current_path is None:
            return
        target = outdir / current_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(current_lines), encoding="utf-8")
        current_path = None
        current_lines = []

    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            flush()
            current_path = Path(m.group(1).strip())
            current_lines = []
        else:
            if current_path is not None:
                current_lines.append(line)

    flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
