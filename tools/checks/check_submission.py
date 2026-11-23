"""Validate a submission CSV file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qaggle.package import validate_submission_file
from qaggle.repo import repo_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True, help="Submission CSV to validate")
    args = parser.parse_args(argv)

    target = args.file if args.file.is_absolute() else repo_root() / args.file
    problems = validate_submission_file(target)

    if problems:
        print(f"Validation FAILED for {target}:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"Validation OK: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
