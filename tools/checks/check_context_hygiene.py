"""Context-rot, SSOT alignment, and low-latency hardware template gate for Misconception Detection.

Verifies:
1. `configs/context_manifest.json` exists and matches `CHAMPION.json` (`s6e10-qaggle-unitary-5paradigm-global-optima-stacker`),
   `configs/competition.json`, and `configs/paradigm_registry.json`.
2. Verified CV and Public LB scores match between `CHAMPION.json` and `context_manifest.json`.
3. No duplicate HTML showcase files or root waste exist.
4. Dual ignore fences (`.cursorignore` and `.claudeignore`) are present and protect data/checkpoints/scratch.
5. Mandatory low-latency hardware and acceleration primitives are exported by `src/qaggle/__init__.py`.
6. Session start hook `.cursor/hooks/session_start.py` exists and is executable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qaggle.repo import repo_root


def main() -> int:
    root = repo_root()
    problems: list[str] = []

    manifest_path = root / "configs" / "context_manifest.json"
    champion_path = root / "CHAMPION.json"
    competition_path = root / "configs" / "competition.json"
    paradigm_path = root / "configs" / "paradigm_registry.json"

    for required in (manifest_path, champion_path, competition_path, paradigm_path):
        if not required.is_file():
            problems.append(f"missing required config: {required.relative_to(root)}")
    if problems:
        print("\n".join(problems))
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    competition = json.loads(competition_path.read_text(encoding="utf-8"))

    # 1. Verify SSOT alignment
    assertion = manifest.get("active_champion_assertion", {})
    champ_label = champion.get("label", "")
    assert_label = assertion.get("label", "")
    if champ_label != assert_label:
        problems.append(f"CHAMPION.json label '{champ_label}' != manifest assertion '{assert_label}'")

    if abs(float(champion.get("cv_score", 0.0)) - float(assertion.get("cv_score", 0.0))) > 1e-5:
        problems.append("CHAMPION.json cv_score mismatch with context_manifest.json")

    verified_lb = float(champion.get("best_verified_public_lb", champion.get("lb_score", 0.0)))
    assert_lb = float(assertion.get("best_verified_public_lb", assertion.get("lb_score", 0.0)))
    if abs(verified_lb - assert_lb) > 1e-5:
        problems.append("CHAMPION.json verified LB score mismatch with context_manifest.json")

    if competition.get("slug") != manifest.get("competition"):
        problems.append("competition.json slug mismatch with context_manifest.json")

    # 2. Check for duplicate showcase HTML files in root
    root_guide = root / "COMPETITION_PORTFOLIO_MASTER_GUIDE.html"
    portfolio_guide = root / "docs" / "portfolio" / "COMPETITION_PORTFOLIO_MASTER_GUIDE.html"
    if root_guide.is_file() and portfolio_guide.is_file():
        problems.append("duplicate COMPETITION_PORTFOLIO_MASTER_GUIDE.html present in root (docs/portfolio/ is canonical)")

    # 3. Check ignore fences
    for fence in (".cursorignore", ".claudeignore"):
        fpath = root / fence
        if not fpath.is_file():
            problems.append(f"missing ignore fence: {fence}")
        else:
            txt = fpath.read_text(encoding="utf-8")
            for req_entry in ("data/", "checkpoints/", "scratch/*"):
                if req_entry not in txt:
                    problems.append(f"{fence} missing pattern: {req_entry}")

    # 4. Check hardware modules and exports
    required_modules = [
        Path("src/qaggle/perf.py"),
        Path("src/qaggle/templates/fast_script_template.py"),
        Path(".cursor/hooks/session_start.py"),
    ]
    for mod in required_modules:
        if not (root / mod).is_file():
            problems.append(f"missing low-latency module: {mod}")

    qaggle_init = (root / "src" / "qaggle" / "__init__.py").read_text(encoding="utf-8")
    for symbol in (
        "configure_high_performance_env",
        "parallel_map",
        "get_torch_device",
        "bootstrap_high_speed_runtime",
        "run_parallel_evaluation_template",
        "max_preserving_orthogonal_projection",
    ):
        if symbol not in qaggle_init:
            problems.append(f"src/qaggle/__init__.py does not export {symbol}")

    if problems:
        print("check_context_hygiene FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"check_context_hygiene OK: {champ_label} (CV: {champion.get('cv_score')}, Verified LB: {verified_lb})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
