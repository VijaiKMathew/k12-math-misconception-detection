"""Global Optima Orchestrator, Paradigm Registry & Anti-Local-Optima Stagnation Circuit Breaker.

Synthesizes the harness governance rules from `kaggle/playground-series-s6e10` (Multi-Paradigm
Orthogonal Logit Stacking) and `kaggle/quantum-circuit` (Verifier-Driven Horizon Expansion &
Zero-Regression Subspace Promotion), plus static Oracle-Leak Auditing across `tools/eval/`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from qaggle.repo import repo_root

STAGNATION_DELTA_THRESHOLD = 0.0050
MAX_CONSECUTIVE_MICRO_TWEAKS = 2
MIN_UNCLEAN_HORIZON_RECALL = 0.9000

ORACLE_LEAK_PATTERNS = (
    re.compile(r"clean_map\s*=\s*\{[^}]*row\[['\"]clean['\"]\]"),
    re.compile(r"assemble_two_stage_predictions\([^)]*row\[['\"]clean['\"]\]"),
)


def paradigm_registry_path() -> Path:
    return repo_root() / "configs" / "paradigm_registry.json"


def load_paradigm_registry() -> dict:
    path = paradigm_registry_path()
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def scan_eval_scripts_for_oracle_leak(eval_dir: Path | None = None) -> list[str]:
    """Scan eval scripts to ensure none inject ground-truth `clean` into OOF predictions."""
    root = repo_root()
    scan_dir = root / "tools" / "eval" if eval_dir is None else Path(eval_dir)
    violations: list[str] = []
    if not scan_dir.is_dir():
        return violations
    for py_file in sorted(scan_dir.glob("*.py")):
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        try:
            display_path = py_file.relative_to(root)
        except ValueError:
            display_path = py_file
        for pat in ORACLE_LEAK_PATTERNS:
            if pat.search(text):
                violations.append(f"{display_path} matches oracle leak pattern: {pat.pattern}")
    return violations


def check_promotion_gates(
    candidate_metrics: dict,
    baseline_metrics: dict,
    horizon_audit: dict | None = None,
    diversity_audit: dict | None = None,
    tolerance: float = -0.0005,
) -> dict:
    """Evaluate all 6 Global Optima Promotion Gates before updating CHAMPION.json."""
    gates = {
        "G1_Overall_CV_Improvement": candidate_metrics["overall_map3"] > baseline_metrics["overall_map3"],
        "G2_Unclean_Cohort_Gain": candidate_metrics["unclean_map3"] >= baseline_metrics["unclean_map3"],
        "G3_Clean_Gate_NonRegression": (
            candidate_metrics["clean_map3"] - baseline_metrics["clean_map3"]
        ) >= tolerance,
        "G4_HighSchool_NonRegression": (
            candidate_metrics["high_school_map3"] - baseline_metrics["high_school_map3"]
        ) >= tolerance,
        "G5_Horizon_Recall_Expanded": (
            False
            if horizon_audit is None
            else horizon_audit.get("unclean_recall", 0.0) >= MIN_UNCLEAN_HORIZON_RECALL
        ),
        "G6_Orthogonal_Paradigm_Diversity": (
            False
            if diversity_audit is None
            else bool(diversity_audit.get("passes_orthogonality_gate", False))
        ),
    }
    deltas = {
        "overall_map3_delta": round(candidate_metrics["overall_map3"] - baseline_metrics["overall_map3"], 5),
        "unclean_map3_delta": round(candidate_metrics["unclean_map3"] - baseline_metrics["unclean_map3"], 5),
        "clean_map3_delta": round(candidate_metrics["clean_map3"] - baseline_metrics["clean_map3"], 5),
        "high_school_map3_delta": round(
            candidate_metrics["high_school_map3"] - baseline_metrics["high_school_map3"], 5
        ),
        "bottleneck_5_map3_delta": round(
            candidate_metrics["bottleneck_5_map3"] - baseline_metrics["bottleneck_5_map3"], 5
        ),
    }
    return {
        "all_passed": all(gates.values()),
        "gates": gates,
        "deltas": deltas,
    }
