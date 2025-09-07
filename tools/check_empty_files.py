from __future__ import annotations
import sys, os, pathlib, textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOW_EMPTY = {"__init__.py", ".gitkeep", ".keep", "__init__.pyi"}
MUST_HAVE_CONTENT_DIRS = {"models", "losses", "tools", "configs"}
REPORT = ROOT / "reports" / "empty_files_report.md"
REPORT.parent.mkdir(parents=True, exist_ok=True)

def is_empty(p: pathlib.Path) -> bool:
    if not p.is_file():
        return False
    try:
        data = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    return len(data.strip()) == 0

def classify(p: pathlib.Path) -> tuple[str, str]:
    name = p.name
    if name in ALLOW_EMPTY:
        return ("OK empty", "Package/VCS marker; no import-time side effects.")
    if any(seg in MUST_HAVE_CONTENT_DIRS for seg in p.parts) or p.suffix in {".py", ".yaml", ".md"}:
        return ("Should contain content", "Code/config/doc file should not be empty.")
    return ("Unknown", "Review manually.")

def main() -> int:
    bad = []
    lines = ["| path | status | action | why |", "|---|---|---|---|"]
    for p in ROOT.rglob("*"):
        if p.is_file() and is_empty(p):
            status, why = classify(p)
            action = "none" if status == "OK empty" else "populate or delete"
            rel = p.relative_to(ROOT)
            lines.append(f"| {rel} | {status} | {action} | {why} |")
            if status != "OK empty":
                bad.append(p)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())