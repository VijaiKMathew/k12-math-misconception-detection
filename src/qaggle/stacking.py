"""Playground S6E10 Orthogonal Logit Stacker, Diversity Gate & Max-Preserving Projection.

Implements the mathematical ensembling laws from `kaggle/playground-series-s6e10` and
`kaggle/quantum-circuit`:
  1. Orthogonal Diversity Auditing (Pearson logit correlation & Top-1 rank disagreement)
  2. Unbounded Log-Odds / Simplex Stacking across diverse model paradigms
  3. Max-Preserving & Order-Statistic Orthogonal Subspace Projection (decoupling Clean-Gate from Error Ranking)
  4. Clean Hard Anchor & False-Clean Evidence Demotion (`NONE` Rank Insertion)
"""

from __future__ import annotations

from collections.abc import Sequence


def max_preserving_orthogonal_projection(
    error_scores: dict[str, float],
    none_score: float,
    anchor_max_error: float,
) -> dict[str, float]:
    """Project error-candidate adjustments onto the subspace preserving max_{c != NONE} s(i, c).

    Guarantees that relative reranking among error candidates never perturbs the calibrated
    Clean-Gate decision boundary between NONE and max_{c != NONE} s(i, c).
    """
    if not error_scores:
        return {"NONE": none_score}
    new_max = max(error_scores.values())
    shift = new_max - anchor_max_error
    projected = {"NONE": none_score}
    for code, val in error_scores.items():
        if code != "NONE":
            projected[code] = val - shift
    return projected


def order_statistic_orthogonal_projection(
    error_scores: dict[str, float],
    anchor_map: dict[str, float],
) -> dict[str, float]:
    """Map reranked error candidates onto the exact sorted order-statistic values of anchor_map.

    Guarantees both max_{c != NONE} s(i, c) == max_{c != NONE} anchor_map[c] (ARM-02)
    and exact preservation of every rank slot relative to NONE (Rank 1, Rank 2, Rank 3).
    """
    none_score = anchor_map.get("NONE", 0.0)
    anchor_err_vals = sorted(
        (v for c, v in anchor_map.items() if c != "NONE"), reverse=True
    )
    if not anchor_err_vals or not error_scores:
        return dict(anchor_map)

    ranked_errs = [
        c
        for c, _ in sorted(error_scores.items(), key=lambda item: item[1], reverse=True)
        if c != "NONE"
    ]
    projected: dict[str, float] = {"NONE": none_score}
    last_val = anchor_err_vals[-1]
    for idx, code in enumerate(ranked_errs):
        if idx < len(anchor_err_vals):
            projected[code] = anchor_err_vals[idx]
        else:
            projected[code] = last_val - 0.05 * (idx - len(anchor_err_vals) + 1)
    return projected


def apply_clean_hard_anchor(
    score_map: dict[str, float],
    is_clean_anchor: bool,
    margin: float = 3.0,
) -> dict[str, float]:
    """Apply Clean Hard Anchor: when verified clean or answer matches, ensure NONE dominates.

    s(i, NONE) >= max_{c != NONE} s(i, c) + margin
    """
    if not is_clean_anchor or not score_map:
        return dict(score_map)
    out = dict(score_map)
    err_vals = [v for c, v in out.items() if c != "NONE"]
    if not err_vals:
        out["NONE"] = max(out.get("NONE", 0.0), margin)
        return out
    max_err = max(err_vals)
    target_none = max_err + margin
    if out.get("NONE", float("-inf")) < target_none:
        out["NONE"] = target_none
    return out


def apply_false_clean_demotion(
    score_map: dict[str, float],
    has_verifier_evidence: bool = False,
    has_exclusive_error_fline: bool = False,
    verifier_penalty: float = 0.25,
    fline_penalty: float = 0.20,
) -> dict[str, float]:
    """Demote `NONE` when answer equivalence fails and high-precision error evidence fires.

    Resolves Roadblock 1 (the False-Clean Trap) where `NONE` occupies Rank 1 on unclean
    traces despite deterministic symbolic transitions or classmate error-line matches.
    """
    if not score_map or "NONE" not in score_map:
        return dict(score_map)
    out = dict(score_map)
    shift = 0.0
    if has_verifier_evidence:
        shift += verifier_penalty
    if has_exclusive_error_fline:
        shift += fline_penalty
    if shift > 0.0:
        out["NONE"] = out["NONE"] - shift
    return out


def compute_diversity_matrix(
    paradigm_predictions: dict[str, Sequence[Sequence[str] | str]],
) -> dict:
    """Compute pairwise Top-1 Rank Agreement and Diversity across model paradigms."""
    names = list(paradigm_predictions.keys())
    matrix: dict[str, dict[str, float]] = {n: {} for n in names}
    disagreements: list[float] = []

    top1_lists: dict[str, list[str]] = {}
    for name, preds in paradigm_predictions.items():
        t1 = []
        for p in preds:
            tokens = p.strip().split() if isinstance(p, str) else list(p)
            t1.append(tokens[0] if tokens else "NONE")
        top1_lists[name] = t1

    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            l1, l2 = top1_lists[n1], top1_lists[n2]
            n_tot = max(min(l1_len := len(l1), len(l2)), 1)
            agree = sum(1 for a, b in zip(l1[:l1_len], l2) if a == b) / n_tot
            matrix[n1][n2] = round(agree, 4)
            if i < j:
                disagreements.append(1.0 - agree)

    mean_disagreement = sum(disagreements) / len(disagreements) if disagreements else 0.0
    return {
        "agreement_matrix": matrix,
        "mean_top1_disagreement": round(mean_disagreement, 4),
        "passes_orthogonality_gate": mean_disagreement >= 0.08,
    }


def orthogonal_logit_blend(
    paradigm_score_maps: Sequence[dict[str, float]],
    weights: Sequence[float],
    none_anchor_idx: int = 0,
) -> dict[str, float]:
    """Combine multiple orthogonal paradigm score maps with Max-Preserving Projection."""
    if not paradigm_score_maps:
        return {"NONE": 0.0}

    anchor_map = paradigm_score_maps[none_anchor_idx]
    none_score = anchor_map.get("NONE", 0.0)
    anchor_err_vals = [v for c, v in anchor_map.items() if c != "NONE"]
    anchor_max_err = max(anchor_err_vals) if anchor_err_vals else -1.0

    all_err_codes = set()
    for sm in paradigm_score_maps:
        for c in sm:
            if c != "NONE":
                all_err_codes.add(c)

    blended_err: dict[str, float] = {}
    for c in all_err_codes:
        val = 0.0
        for sm, w in zip(paradigm_score_maps, weights):
            default_floor = min(sm.values()) - 0.35 if sm else -2.0
            val += w * sm.get(c, default_floor)
        blended_err[c] = val

    return max_preserving_orthogonal_projection(blended_err, none_score, anchor_max_err)
