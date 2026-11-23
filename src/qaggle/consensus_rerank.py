"""Wave 1 CPU consensus-grounded rerank (score-side, no GPU/TPU required).

Blends Directive 1 first-divergent-step diagnosis scores into the champion anchor
score maps under ``max_preserving_orthogonal_projection`` (ARM-02) and optionally
applies the Directive 2 root-over-slip referee demotion (ARM-05, structural
evidence only). Zero-weight fusion is a true no-op so the anchor floor is always in
the search space (learned rule from ``reports/wave34/SUMMARY.md``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from qaggle.sibling_referee import (
    SECONDARY_SLIP_CODES,
    get_sibling_pairs_list,
)
from qaggle.stacking import max_preserving_orthogonal_projection
from qaggle.wave34 import evaluate_predictions, rank_top3

NONE = "NONE"
DEFAULT_OUT_BIAS = -0.05
DEFAULT_REFEREE_MARGIN = 0.35

DEFAULT_WEIGHT_GRID: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)

Match = tuple[str, float, str]
ScoreMap = dict[str, float]


def diagnosis_score_map(
    matches: Sequence[Match], valid_taxonomy_codes: set[str] | None
) -> ScoreMap:
    """Convert symbolic-error-predicate matches into a normalized per-trace score map.

    Non-taxonomy codes and ``NONE`` are dropped, strengths are normalized against
    the strongest surviving match (ceiling 1.0), and per-code strength collapses to
    the maximum. An empty taxonomy set falls back to raw matched codes.
    """
    if valid_taxonomy_codes is None or not valid_taxonomy_codes:
        allowed = None
    else:
        allowed = {c for c in valid_taxonomy_codes if c != NONE}

    strongest: ScoreMap = {}
    for code, strength, _rationale in matches:
        code = str(code)
        if allowed is not None and code not in allowed:
            continue
        value = float(strength)
        current = strongest.get(code)
        if current is None or value > current:
            strongest[code] = value

    peak = max(strongest.values(), default=0.0)
    if peak <= 0.0:
        return {}
    return {code: value / peak for code, value in strongest.items()}


def fused_rerank(
    anchor: ScoreMap,
    diag: ScoreMap,
    w_diag: float,
    *,
    out_bias: float = DEFAULT_OUT_BIAS,
) -> ScoreMap:
    """Add weighted consensus-diagnosis scores to an anchor map under ARM-02.

    Behaviour contract (mirrors :func:`qaggle.wave34.fuse_trace`):
    - ``w_diag <= 0`` is a true no-op: diag candidates are never injected.
    - A clean anchor (``NONE`` ranks first) is returned untouched so the Clean-Gate
      boundary can never degrade.
    - Diag codes absent from the anchor error map are injected at
      ``second_error + out_bias`` so extravagant matches cannot leapfrog.
    - The result is projected so ``max_{c != NONE}`` equals the anchor maximum.
    """
    if w_diag <= 0.0 or not diag:
        return dict(anchor)

    if not anchor or NONE not in anchor:
        return dict(anchor)

    errors = {code: value for code, value in anchor.items() if code != NONE}
    if not errors:
        return dict(anchor)

    anchor_none = anchor[NONE]
    anchor_max = max(errors.values())
    if anchor_none >= anchor_max:
        return dict(anchor)

    ordered = sorted(errors.values(), reverse=True)
    second_err = ordered[1] if len(ordered) >= 2 else anchor_max - 0.2

    codes = set(errors)
    codes.update(code for code in diag if code != NONE)

    blended: ScoreMap = {}
    for code in sorted(codes):
        base = errors[code] if code in errors else second_err + out_bias
        blended[code] = base + w_diag * float(diag.get(code, 0.0))

    return max_preserving_orthogonal_projection(blended, anchor_none, anchor_max)


def apply_referee_demotion(
    anchor: ScoreMap,
    diag: ScoreMap,
    *,
    margin: float = DEFAULT_REFEREE_MARGIN,
) -> ScoreMap:
    """Demote secondary slips inside the anchor when structural evidence exists.

    ARM-05-compliant: demotion happens only for this specific trace-slip pairing
    (both sides come from the same trace) and only when the diagnosis matched a
    structural misconception code that is already in the anchor error map —
    ``CAL-01``/``ANS-01``-only evidence never demotes anything. A clean gate
    (``NONE`` first) is never touched.
    """
    result = dict(anchor)
    if not anchor or NONE not in anchor:
        return result

    errors = {code: value for code, value in anchor.items() if code != NONE}
    if not errors:
        return result

    if anchor[NONE] >= max(errors.values()):
        return result

    structural = {
        code
        for code, value in diag.items()
        if code not in SECONDARY_SLIP_CODES
        and code != NONE
        and code in errors
        and value > 0.0
    }
    if not structural:
        return result

    best_structural = max(errors[code] for code in structural)

    for slip in SECONDARY_SLIP_CODES:
        if slip in errors and errors[slip] > best_structural:
            result[slip] = best_structural - margin

    return result


def predict_top3_trace(
    anchor: ScoreMap,
    diag: ScoreMap,
    w_diag: float,
    *,
    apply_referee: bool = False,
    apply_sibling_swap: bool = False,
    out_bias: float = DEFAULT_OUT_BIAS,
) -> list[str]:
    """Produce the top-three predicted codes for one trace (wave34 insertion-order ties)."""
    fused = fused_rerank(anchor, diag, w_diag, out_bias=out_bias)
    if apply_referee and w_diag >= 0.0:
        fused = apply_referee_demotion(fused, diag)
    if apply_sibling_swap and w_diag >= 0.0:
        fused = sibling_swap_projection(fused, diag)
    return rank_top3(fused)


def _top2_error_codes(score_map: Mapping[str, float]) -> list[str]:
    ranked = rank_top3(
        {c: v for c, v in score_map.items() if c != NONE}
    )
    return ranked[:2]


def sibling_swap_projection(
    anchor: ScoreMap,
    diag: ScoreMap,
    sibling_pairs: Sequence[tuple[str, str]] | None = None,
) -> ScoreMap:
    """Conflict-aware Directive 2 projection: swap a diagnosed sibling ahead.

    Fires only when (a) the anchor Clean-Gate is not active (``NONE`` below the
    error maximum, never touched either way), (b) the diagnosis matched a
    structural code ``C`` (a member of a sibling pair, not a secondary slip), and
    (c) the anchor's top-2 error codes contain C's sibling partner ``P`` that was
    NOT diagnosed. The swap then only transposes C and P inside the top-2, so the
    multiset of error scores is unchanged and the max-preserving invariant holds
    by construction (no projection needed for a pure permutation).
    """
    result = dict(anchor)
    if not anchor or NONE not in anchor:
        return result

    errors = {code: value for code, value in anchor.items() if code != NONE}
    if not errors:
        return result

    if anchor[NONE] >= max(errors.values()):
        return result

    pairs = list(get_sibling_pairs_list()) if sibling_pairs is None else list(
        sibling_pairs
    )
    partner_of: dict[str, str] = {}
    for code_a, code_b in pairs:
        partner_of.setdefault(code_a, code_b)
        partner_of.setdefault(code_b, code_a)

    diagnosed = {code for code, value in diag.items() if value > 0.0}
    top2 = _top2_error_codes(result)

    for code in (top2[1], top2[0]) if len(top2) >= 2 else top2:
        partner = partner_of.get(code)
        if (
            code in diagnosed
            and code not in SECONDARY_SLIP_CODES
            and partner is not None
            and partner in top2
            and partner not in diagnosed
            and partner not in SECONDARY_SLIP_CODES
        ):
            code_score = result[code]
            partner_score = result[partner]
            result[code] = partner_score
            result[partner] = code_score
            break
    return {**result, "NONE": anchor.get(NONE, 0.0)}


def fuse_predictions(
    oof_anchor: Mapping[str, ScoreMap],
    oof_diag: Mapping[str, ScoreMap],
    rows: Sequence[dict],
    w_diag: float,
    apply_referee: bool = False,
    apply_sibling_swap: bool = False,
    *,
    out_bias: float = DEFAULT_OUT_BIAS,
) -> dict[str, str]:
    """Build a ``trace_id -> "C1 C2 C3"`` prediction map for cohort evaluation."""
    predictions: dict[str, str] = {}
    for row in rows:
        trace_id = str(row.get("trace_id", ""))
        anchor = oof_anchor.get(trace_id)
        if not anchor:
            predictions[trace_id] = NONE
            continue
        tokens = predict_top3_trace(
            anchor,
            oof_diag.get(trace_id, {}),
            w_diag,
            apply_referee=apply_referee,
            apply_sibling_swap=apply_sibling_swap,
            out_bias=out_bias,
        )
        predictions[trace_id] = " ".join(tokens) if tokens else NONE
    return predictions


__all__ = [
    "DEFAULT_WEIGHT_GRID",
    "apply_referee_demotion",
    "diagnosis_score_map",
    "evaluate_predictions",
    "fuse_predictions",
    "fused_rerank",
    "predict_top3_trace",
    "sibling_swap_projection",
]
