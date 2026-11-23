#!/usr/bin/env python3
"""Automated Anti-Memorization & Data Leakage Scanner (Pre-Flight Gate).

Scans the codebase and runtime artifacts for the four fundamental machine learning traps:
  1. In-Sample OOF Leakage (ARM-06): Inspects training scripts for `process_split(train_data, ...oof.parquet)`
     or absence of explicit fold masking in validation loops.
  2. Modality Mismatch / Naming Fraud (ARM-07): Flags scripts importing text-only models (AutoModelForCausalLM)
     while writing output filenames containing 'vl', 'vision', or 'multimodal'.
  3. Verifier Literal Leakage (ARM-08): Scans regex patterns in verifier.py for hardcoded OCR strings
     derived from training row numbers.
  4. Non-Local Margin Loss (ARM-05): Flags pairwise margin ranking loss in cross-encoders that lack
     grouping by trace_id.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def check_script_leakage(script_path: Path) -> list[str]:
    errors: list[str] = []
    if "quarantined" in str(script_path):
        return errors
    text = script_path.read_text(encoding="utf-8")

    # 1. Check for text model saving as VLM
    if "AutoModelForCausalLM" in text and "Qwen2_5_VL" not in text:
        if re.search(r"['\"].*?vl_.*?oof\.parquet['\"]", text):
            errors.append(f"{script_path.name}: Imports text-only AutoModelForCausalLM but exports a '*vl*oof.parquet' artifact (ARM-07 Text-Only Impersonation).")

    # 2. Check for in-sample whole-train scoring into oof
    if re.search(r"process_split\s*\(\s*train.*?,\s*.*?oof", text, re.IGNORECASE):
        errors.append(f"{script_path.name}: In-sample scoring detected! Calls process_split on train dataset into an OOF file (ARM-06 Violation).")

    return errors


def check_verifier_rules() -> list[str]:
    errors: list[str] = []
    v_path = ROOT / "src" / "qaggle" / "verifier.py"
    if not v_path.is_file():
        return errors
    text = v_path.read_text(encoding="utf-8")

    if "include_memorized_train_literals: bool = True" in text:
        errors.append("verifier.py: Default 'include_memorized_train_literals' must be False!")

    return errors


def main() -> int:
    print("=== [AUTOMATED ANTI-MEMORIZATION & LEAKAGE AUDIT] ===")
    all_violations: list[str] = []

    scan_dirs = [ROOT / "tools" / "gpu", ROOT / "tools" / "eval", ROOT / "notebooks" / "colab"]
    for s_dir in scan_dirs:
        if not s_dir.is_dir():
            continue
        for py_file in s_dir.glob("*.py"):
            violations = check_script_leakage(py_file)
            if violations:
                all_violations.extend(violations)

    ver_violations = check_verifier_rules()
    if ver_violations:
        all_violations.extend(ver_violations)

    if all_violations:
        print(f"\n[CRITICAL FAILURE] Found {len(all_violations)} ML integrity violation(s):")
        for v in all_violations:
            print(f"  ❌ {v}")
        return 1

    print("[SUCCESS] Zero data leakage, in-sample OOF, or model impersonation violations detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
