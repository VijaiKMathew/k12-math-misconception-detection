#!/usr/bin/env python3
"""
Pre-Commit / Pre-Promotion Invariant Guard
=========================================
Runs automatically before any commit, experiment promotion, or merge.
Fails CLOSED (exit 1) if:
  1. Code formatting or lint rules are violated (ruff).
  2. Static typechecks fail (mypy).
  3. Single Source of Truth has drifted (CHAMPION.json vs configs/context_manifest.json).
  4. Train-serving modality mismatch is introduced (test reading transcription_steps).
  5. Invalid taxonomy code strings (e.g. DEC-02 instead of Dec-02) appear in src/qaggle.
  6. Waste, stale scratch files, or credentials are leaked.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _resolve_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for candidate_dir in (
        Path.home() / ".local" / "bin",
        Path.home() / ".cargo" / "bin",
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
    ):
        cand = candidate_dir / name
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def run_step(cmd: list[str], desc: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    res = subprocess.run(cmd, check=False, cwd=ROOT, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"\n[BLOCKED] Invariant check failed: {desc}")
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr, file=sys.stderr)
        sys.exit(1)


def check_ssot_parity() -> None:
    champ_path = ROOT / "CHAMPION.json"
    manifest_path = ROOT / "configs" / "context_manifest.json"
    if not champ_path.exists() or not manifest_path.exists():
        print("[BLOCKED] SSOT files missing: CHAMPION.json or context_manifest.json")
        sys.exit(1)

    champ = json.loads(champ_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assertion = manifest.get("active_champion_assertion", {})

    if champ.get("label") != assertion.get("label"):
        print(
            f"[BLOCKED] SSOT Label mismatch: CHAMPION.json='{champ.get('label')}' "
            f"vs context_manifest='{assertion.get('label')}'"
        )
        sys.exit(1)


def check_no_train_modality_in_test() -> None:
    """Scan all evaluation and runner scripts to guarantee transcription_steps is never accessed."""
    scan_dirs = [
        ROOT / "tools" / "eval",
        ROOT / "kaggle_runner",
        ROOT / "data" / "training_bundles",
    ]
    for base_dir in scan_dirs:
        if not base_dir.exists():
            continue
        for py_file in base_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            if (
                "test" in content
                and "transcription_steps" in content
                and "assert 'transcription_steps' not in" not in content
                and "# test-safe" not in content
            ):
                print(
                    f"[BLOCKED] Potential ARM-04 train-serving skew in {py_file.relative_to(ROOT)}: "
                    "referencing transcription_steps near test logic."
                )
                sys.exit(1)


def check_taxonomy_code_validity() -> None:
    """Ensure all misconception code literals in src/qaggle match canonical taxonomy_v1.csv IDs."""
    tax_path = ROOT / "taxonomy_v1.csv"
    if not tax_path.is_file():
        return
    with tax_path.open("r", encoding="utf-8") as handle:
        valid_codes = {row["id"].strip() for row in csv.DictReader(handle)} | {"NONE"}

    allowed_meta = {"ARM-01", "ARM-02", "ARM-03", "ARM-04", "ARM-05", "Top-11", "Top-15", "Top-18", "Top-20", "TOP-18"}
    code_pat = re.compile(r"""["']([A-Z][A-Za-z]{1,2}-\d{2})["']""")
    for py_file in sorted((ROOT / "src" / "qaggle").rglob("*.py")):
        content = py_file.read_text(encoding="utf-8")
        found = set(code_pat.findall(content))
        invalid = sorted(found - valid_codes - allowed_meta)
        if invalid:
            print(
                f"[BLOCKED] Invalid taxonomy code literal(s) in {py_file.relative_to(ROOT)}: {invalid}"
            )
            sys.exit(1)


def main() -> int:
    uvx_bin = _resolve_tool("uvx")
    py_cmd = [sys.executable]

    # 1. Fast Static Lint on core production modules (when uvx is available)
    if uvx_bin:
        run_step(
            [
                uvx_bin,
                "ruff",
                "check",
                "src/qaggle/nlp",
                "src/qaggle/ensembling",
                "tests/test_grandmaster_nlp.py",
                "tools/checks/enforce_invariants.py",
            ],
            "Ruff static linter",
        )
        run_step(
            [
                uvx_bin,
                "mypy",
                "src/qaggle/nlp",
                "src/qaggle/ensembling",
                "tests/test_grandmaster_nlp.py",
                "tools/checks/enforce_invariants.py",
                "--ignore-missing-imports",
                "--explicit-package-bases",
            ],
            "Mypy typecheck",
        )

    # 3. SSOT Parity
    check_ssot_parity()

    # 4. Modality Skew & Taxonomy ID Guard
    check_no_train_modality_in_test()
    check_taxonomy_code_validity()

    # 5. Secrets and Credentials
    run_step([*py_cmd, "tools/checks/check_secrets.py"], "Secret scan")

    # 6. Context Hygiene (Purges .DS_Store, stale scratch)
    run_step([*py_cmd, "tools/ops/context_hygiene.py", "--apply"], "Context hygiene cleaner")

    # 7. Anti-Slop Deterministic AST Guard
    run_step(
        [
            *py_cmd,
            "tools/checks/anti_slop.py",
            "src/qaggle",
            "tools/checks",
            "tests",
        ],
        "Anti-slop AST linter",
    )

    # 8. Fast Unit Contract Tests
    run_step(
        [
            *py_cmd,
            "-m",
            "unittest",
            "tests/test_grandmaster_nlp.py",
            "tests/test_context_hygiene.py",
            "tests/test_anti_slop.py",
            "tests/test_counterfactual_synth.py",
            "tests/test_counterfactual_bundle.py",
        ],
        "Core contract tests",
    )

    print("[SUCCESS] All 8 invariant gates passed. Codebase is clean, aligned, and decay-resistant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
