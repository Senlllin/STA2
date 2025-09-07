"""Explain why an empty file is acceptable or requires content."""

from __future__ import annotations

import sys
from pathlib import Path

from check_empty_files import classify, is_empty, ROOT


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python justify_empty.py <path>")
        return 1
    p = (ROOT / sys.argv[1]).resolve()
    if not p.exists():
        print(f"{p} does not exist")
        return 1
    if not is_empty(p):
        print(f"{p} is not empty")
        return 0
    status, why = classify(p)
    print(f"{p}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())