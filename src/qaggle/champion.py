"""CHAMPION.json: submission tracking, LB verification integrity, and baseline management."""

from __future__ import annotations

import json
from pathlib import Path

from qaggle.package import file_fingerprint, validate_submission_file
from qaggle.repo import repo_root

INVALID_LEAKY_SUBMISSION_REFS = {57761499}


def champion_path() -> Path:
    return repo_root() / "CHAMPION.json"


def load_champion() -> dict:
    return json.loads(champion_path().read_text(encoding="utf-8"))


def _normalize_lb_ref(
    lb_ref: int | str | float | None,
) -> int | str | float | None:
    """Normalize numeric string refs so ``"57761499"`` cannot bypass the leak guard."""
    if isinstance(lb_ref, str):
        candidate = lb_ref.strip()
        if candidate.lstrip("+-").isdigit():
            return int(candidate)
    return lb_ref


def validate_lb_integrity(champ: dict) -> list[str]:
    """Pure LB integrity checks: score requires a verified ref and must be numeric."""
    problems: list[str] = []
    lb_score = champ.get("lb_score")
    lb_ref = _normalize_lb_ref(champ.get("lb_ref"))
    if lb_score is not None and not lb_ref:
        problems.append("Champion claims a non-null lb_score without a verified lb_ref")
    if lb_ref in INVALID_LEAKY_SUBMISSION_REFS and lb_score is not None:
        problems.append(
            f"Submission ref {lb_ref} is flagged as an invalid oracle-leak run; "
            "lb_score must be null until active champion is submitted"
        )
    if lb_score is not None and (
        isinstance(lb_score, bool) or not isinstance(lb_score, (int, float))
    ):
        problems.append(
            f"Champion lb_score must be numeric or null, got {type(lb_score).__name__}: {lb_score!r}"
        )
    return problems


def check_champion(path: Path | None = None) -> list[str]:
    root = repo_root()
    champion_file = Path(path) if path is not None else champion_path()
    champ = json.loads(champion_file.read_text(encoding="utf-8"))
    source_path = root / champ["source"]
    problems = validate_submission_file(source_path)
    if problems:
        return problems

    current_fp = file_fingerprint(source_path)
    expected_fp = champ["files"][0]
    if current_fp["bytes"] != expected_fp["bytes"]:
        problems.append(f"Champion bytes mismatch: {current_fp['bytes']} != {expected_fp['bytes']}")
    if current_fp["sha256"] != expected_fp["sha256"]:
        problems.append("Champion sha256 changed")

    problems.extend(validate_lb_integrity(champ))

    return problems


def write_champion(payload: dict) -> None:
    champion_path().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
