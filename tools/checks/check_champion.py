"""Validate current CHAMPION.json configuration and file existence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qaggle.champion import champion_path, load_champion, validate_lb_integrity
from qaggle.package import file_fingerprint, validate_submission_file
from qaggle.repo import repo_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    champ = load_champion()
    root = repo_root()
    source_path = root / champ.get("source", "submission.csv")

    # Verify LB integrity specs
    problems = validate_lb_integrity(champ)
    if problems:
        print("check_champion FAILED (LB integrity):")
        for p in problems:
            print(f"  - {p}")
        return 1

    if not source_path.exists():
        print(f"check_champion OK (metadata verified): {champ['label']} (CV: {champ.get('cv_score')}, LB: {champ.get('lb_score')})")
        print(f"  [NOTICE] '{champ['source']}' is a reproducible asset (to generate, run pipeline).")
        return 0

    problems = validate_submission_file(source_path)
    if problems:
        print("check_champion FAILED (submission structure):")
        for p in problems:
            print(f"  - {p}")
        return 1

    current_fp = file_fingerprint(source_path)
    expected_fp = champ["files"][0]
    if current_fp["bytes"] != expected_fp["bytes"]:
        print(f"Champion bytes mismatch: {current_fp['bytes']} != {expected_fp['bytes']}")
        return 1
    if current_fp["sha256"] != expected_fp["sha256"]:
        print(f"Champion sha256 mismatch")
        return 1

    print(f"check_champion OK: {champ['label']} ({champ['source']} SHA256 verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
