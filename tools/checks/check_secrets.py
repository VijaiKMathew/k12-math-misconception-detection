"""Refuse credential material in tracked text files (sandbox-safe traversal)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qaggle.repo import repo_root

PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AQ\.Ab8RN6[A-Za-z0-9_\-]{20,}"),
    re.compile(r"AIzaSy[A-Za-z0-9_\-]{33}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|kaggle_key)\s*[:=]\s*['\"][A-Za-z0-9_\-]{20,}"),
)
SKIP = {".venv", ".venv-soup", "data", "scratch", "logs", ".git", "__pycache__"}
SUFFIXES = {".py", ".md", ".json", ".toml", ".txt", ".yml", ".yaml"}


def main() -> int:
    root = repo_root()
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        for fname in filenames:
            path = Path(dirpath) / fname
            if path.suffix not in SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in PATTERNS:
                if pattern.search(text):
                    hits.append(f"{path.relative_to(root)} matches {pattern.pattern}")
    if hits:
        print("secrets scan failed:")
        print("\n".join(hits))
        return 1
    print("secrets scan OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
