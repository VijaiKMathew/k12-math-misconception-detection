"""Pure Wave 3/4 score fusion, classmate consensus, and promotion audits.

This module is intentionally dependency-free beyond the standard library so it can be
unit-tested and reused by both the OOF selection loop and test-time inference. It leans
on the existing pure helpers in :mod:`qaggle.stacking` and :mod:`qaggle.cv`.

Calibration contract
--------------------
``fuse_trace`` does **not** rescale its inputs. Callers must pass per-trace calibrated
score maps for ``listwise``, ``p2``, and ``p3`` (typically via :func:`trace_zscores` or
``load_trace_zscores``). Double scaling inside the fusion would distort the weights.

Clean-gate invariant
--------------------
Every non-anchor contribution is routed through
:func:`qaggle.stacking.max_preserving_orthogonal_projection`, so
``max_{c != NONE} s(i, c)`` is unchanged by fusion. A trace whose anchor already ranks
``NONE`` first is returned untouched, guaranteeing the champion Clean-Gate boundary is
never degraded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from qaggle.cv import evaluate_cohorts
from qaggle.stacking import (
    compute_diversity_matrix,
    max_preserving_orthogonal_projection,
)

ScoreMap = dict[str, float]
Maps = dict[str, ScoreMap]

NONE = "NONE"
MIN_UNCLEAN_HORIZON_RECALL = 0.90

DEFAULT_SIBLING_PAIRS: tuple[tuple[str, str], ...] = (
    ("EXP-04", "EXP-07"),
    ("ALG-04", "ALG-06"),
    ("FRA-01", "FRA-06"),
    ("KPF-01", "FRM-01"),
    ("KPF-01", "KPF-02"),
    ("WOP-01", "WOP-02"),
    ("INT-03", "INT-04"),
    ("EXP-01", "EXP-02"),
    ("EXP-03", "EXP-04"),
    ("FRA-03", "FRA-04"),
    ("ALG-02", "ALG-03"),
    ("ANG-01", "ANG-02"),
    ("MEA-01", "MEA-04"),
    ("MEA-04", "MEA-05"),
    ("WOP-01", "ANS-01"),
    ("INT-02", "INT-03"),
    ("Dec-01", "Dec-02"),
    ("SEQ-01", "SEQ-02"),
    ("PRO-01", "PRO-04"),
    ("ANS-01", "ANS-02"),
    ("ALG-01", "ALG-02"),
    ("ALG-05", "ALG-06"),
    ("LIN-01", "SIM-01"),
)


def _load_jsonl_maps(path: Path, field: str) -> Maps:
    maps: Maps = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            raw = record.get(field) or {}
            maps[str(record["trace_id"])] = {
                str(code): float(value) for code, value in raw.items()
            }
    return maps


def load_score_maps(path: Path) -> Maps:
    """Load champion-style JSONL records shaped ``{"trace_id", "scores"}``."""
    return _load_jsonl_maps(path, "scores")


def load_candidate_scores(path: Path) -> Maps:
    """Load listwise JSONL records shaped ``{"trace_id", "candidate_scores"}``."""
    return _load_jsonl_maps(path, "candidate_scores")


def load_horizon_maps(path: Path) -> dict[str, list[str]]:
    """Load horizon JSONL records shaped ``{"trace_id", "candidates"}``."""
    maps: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            maps[str(record["trace_id"])] = [
                str(code) for code in (record.get("candidates") or [])
            ]
    return maps


def trace_zscores(maps: Maps) -> Maps:
    """Standardize each trace's error-candidate scores to zero mean and unit std.

    ``NONE`` is excluded from the statistics and from the output, matching the champion's
    calibration convention. Traces whose error scores have zero variance map to ``0.0``.
    """
    zscored: Maps = {}
    for trace_id, scores in maps.items():
        errors = {code: value for code, value in scores.items() if code != NONE}
        if not errors:
            zscored[trace_id] = {}
            continue
        values = list(errors.values())
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        if std < 1e-12:
            zscored[trace_id] = {code: 0.0 for code in errors}
        else:
            zscored[trace_id] = {
                code: (value - mean) / std for code, value in errors.items()
            }
    return zscored


def rank_top3(scores: ScoreMap) -> list[str]:
    """Return up to three codes ordered by score descending.

    Ties are deliberately resolved by the input mapping's insertion order, matching the
    producer's ``sorted(..., reverse=True)`` behaviour so the reconstructed anchor is
    byte-for-byte faithful to the champion baseline.
    """
    return [
        code
        for code, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            :3
        ]
    ]


def fuse_trace(
    anchor: ScoreMap,
    listwise: ScoreMap | None = None,
    p2: ScoreMap | None = None,
    p3: ScoreMap | None = None,
    w_listwise: float = 0.0,
    w_p2: float = 0.0,
    w_p3: float = 0.0,
    out_bias: float = -0.05,
) -> ScoreMap:
    """Fuse calibrated auxiliary scores into an anchor map while preserving its maximum.

    New candidates start at ``second_err + out_bias`` while anchor candidates keep their
    anchor score; weighted auxiliary contributions are then added. The result is projected
    with :func:`max_preserving_orthogonal_projection`, so both ``NONE`` and
    ``max_{c != NONE}`` remain exactly as they were in ``anchor``.

    When every auxiliary weight is zero or negative this is a true no-op: auxiliary
    candidates are not injected and the anchor map is returned unchanged.
    """
    if w_listwise <= 0.0 and w_p2 <= 0.0 and w_p3 <= 0.0:
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

    ordered_errors = sorted(errors.values(), reverse=True)
    second_err = (
        ordered_errors[1] if len(ordered_errors) >= 2 else anchor_max - 0.2
    )

    listwise_map = listwise or {}
    p2_map = p2 or {}
    p3_map = p3 or {}

    codes = set(errors)
    if w_listwise > 0.0:
        codes.update(code for code in listwise_map if code != NONE)
    if w_p2 > 0.0:
        codes.update(code for code in p2_map if code != NONE)
    if w_p3 > 0.0:
        codes.update(code for code in p3_map if code != NONE)

    blended: ScoreMap = {}
    for code in sorted(codes):
        base = errors[code] if code in errors else second_err + out_bias
        blended[code] = (
            base
            + w_listwise * listwise_map.get(code, 0.0)
            + w_p2 * p2_map.get(code, 0.0)
            + w_p3 * p3_map.get(code, 0.0)
        )

    return max_preserving_orthogonal_projection(blended, anchor_none, anchor_max)


def build_consensus_counts(
    rows: list[dict],
    fold_of: dict[str, int] | None,
    exclude_fold: int | None,
) -> dict[str, dict[str, int]]:
    """Count empirical root codes per ``question_id``, optionally excluding one fold.

    ``NONE`` roots are never counted. When ``fold_of`` is ``None`` every row is used,
    regardless of ``exclude_fold`` (the test-time inference case).
    """
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        trace_id = str(row.get("trace_id", ""))
        if (
            fold_of is not None
            and exclude_fold is not None
            and fold_of.get(trace_id) == exclude_fold
        ):
            continue
        question_id = row.get("question_id")
        root_code = row.get("root_code")
        if not question_id or not root_code or root_code == NONE:
            continue
        bucket = counts.setdefault(str(question_id), {})
        code = str(root_code)
        bucket[code] = bucket.get(code, 0) + 1
    return counts


def _counts_for_trace(
    trace_id: str,
    question_id: str,
    fold_of: dict[str, int] | None,
    all_counts: dict[str, dict[str, int]],
    counts_by_fold: dict[int, dict[str, dict[str, int]]],
) -> dict[str, int]:
    if fold_of is None:
        return all_counts.get(question_id, {})
    fold = fold_of.get(trace_id)
    if fold is None:
        return {}
    fold_counts = counts_by_fold.get(int(fold))
    if fold_counts is None:
        return {}
    return fold_counts.get(question_id, {})


def apply_classmate_consensus(
    predictions: dict[str, str],
    rows: list[dict],
    fold_of: dict[str, int] | None = None,
    sibling_pairs: list[tuple[str, str]] | None = None,
    min_count: int = 1,
) -> dict[str, str]:
    """Rerank top-3 predictions using fold-strict classmate evidence for the same question.

    Sibling misconceptions are swapped only when their evidence comes from other folds;
    a frequently observed tail code is promoted to top-1 when the current top-1 has no
    empirical support. Predictions containing ``NONE`` are left untouched.
    """
    pairs = DEFAULT_SIBLING_PAIRS if sibling_pairs is None else sibling_pairs
    sibling_set = {frozenset(pair) for pair in pairs if len(pair) == 2}

    question_of: dict[str, str] = {}
    for row in rows:
        trace_id = row.get("trace_id")
        question_id = row.get("question_id")
        if trace_id is not None and question_id:
            question_of[str(trace_id)] = str(question_id)

    all_counts = build_consensus_counts(rows, None, None)
    counts_by_fold: dict[int, dict[str, dict[str, int]]] = {}
    if fold_of is not None:
        for fold in set(fold_of.values()):
            counts_by_fold[int(fold)] = build_consensus_counts(
                rows, fold_of, int(fold)
            )

    updated = dict(predictions)
    for trace_id, raw in predictions.items():
        tokens = raw.strip().split() if isinstance(raw, str) else [str(t) for t in raw]
        tokens = tokens[:3]
        if not tokens or NONE in tokens:
            continue

        question_id = question_of.get(str(trace_id))
        if question_id is None:
            continue

        counts = _counts_for_trace(
            str(trace_id), question_id, fold_of, all_counts, counts_by_fold
        )

        if (
            len(tokens) >= 2
            and frozenset((tokens[0], tokens[1])) in sibling_set
            and counts.get(tokens[0], 0) == 0
            and counts.get(tokens[1], 0) >= min_count
        ):
            tokens = [tokens[1], tokens[0], *tokens[2:]]

        if len(tokens) >= 2 and counts:
            top_code, top_count = min(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
            if (
                top_count >= 2
                and counts.get(tokens[0], 0) == 0
                and top_code in tokens[1:]
            ):
                tail_index = tokens.index(top_code, 1)
                tokens = [
                    top_code,
                    *[token for i, token in enumerate(tokens) if i != tail_index],
                ]

        updated[str(trace_id)] = " ".join(tokens[:3])

    return updated


def horizon_audit(
    anchor_maps: Maps,
    truth: dict[str, str],
    unclean_tids: set[str],
    horizon_maps: dict[str, list[str] | dict] | None = None,
    source: str | None = None,
) -> dict:
    """Audit whether the candidate pool contains each true root code.

    ``horizon_maps`` should carry the true unitary horizon (``champion_oof_horizon`` for
    OOF or ``champion_test_horizon`` for test). When omitted, the anchor shortlist is used
    as a fallback, which measures anchor recall rather than expanded horizon recall.
    """
    if horizon_maps is not None:
        def candidates(trace_id: str) -> set[str]:
            raw = horizon_maps.get(trace_id)
            if raw is None:
                return set()
            if isinstance(raw, dict):
                return set(raw.keys())
            return set(raw)

        audit_source = source or "horizon_maps"
    else:
        def candidates(trace_id: str) -> set[str]:
            return set(anchor_maps.get(trace_id, {}))

        audit_source = source or "anchor"

    unclean_hits = sum(
        1 for trace_id in unclean_tids if truth.get(trace_id) in candidates(trace_id)
    )
    unclean_recall = unclean_hits / len(unclean_tids) if unclean_tids else 0.0

    present = [trace_id for trace_id in truth if trace_id in anchor_maps]
    overall_hits = sum(
        1 for trace_id in present if truth[trace_id] in candidates(trace_id)
    )
    overall_recall = overall_hits / len(present) if present else 0.0

    return {
        "unclean_recall": unclean_recall,
        "overall_recall": overall_recall,
        "source": audit_source,
        "passes": (
            audit_source != "anchor_fallback"
            and unclean_recall >= MIN_UNCLEAN_HORIZON_RECALL
        ),
    }


def _error_top1(scores: ScoreMap) -> str:
    errors = {code: value for code, value in scores.items() if code != NONE}
    ranked = rank_top3(errors)
    return ranked[0] if ranked else NONE


def diversity_audit(
    prediction_maps: dict[str, Maps],
    unclean_tids: set[str] | None = None,
) -> dict:
    """Audit pairwise Top-1 disagreement across paradigms.

    When ``unclean_tids`` is provided, only those traces are compared and ``NONE`` is
    excluded from the ranking (error-only top-1), so the gate reflects disagreement on the
    cohort that can actually move the unclean score. When it is omitted, the audit keeps
    the original behaviour: full top-1 over the union of all trace ids.

    Any empty paradigm map or incomplete coverage of the audited trace set forces the
    orthogonality gate to fail closed.
    """
    if unclean_tids is not None:
        trace_ids = sorted(unclean_tids)
        cohort = "unclean"
    else:
        trace_ids = sorted(
            {trace_id for maps in prediction_maps.values() for trace_id in maps}
        )
        cohort = "all"

    complete = bool(trace_ids) and all(
        bool(maps) and all(trace_id in maps and bool(maps[trace_id]) for trace_id in trace_ids)
        for maps in prediction_maps.values()
    )

    top1: dict[str, list[str]] = {}
    for name, maps in prediction_maps.items():
        top1[name] = []
        for trace_id in trace_ids:
            scores = maps.get(trace_id, {})
            if unclean_tids is not None:
                top1[name].append(_error_top1(scores))
            else:
                ranked = rank_top3(scores)
                top1[name].append(ranked[0] if ranked else NONE)

    audit = compute_diversity_matrix(top1)
    audit["cohort"] = cohort
    audit["complete_coverage"] = complete
    if not complete:
        audit["passes_orthogonality_gate"] = False
    return audit


def evaluate_predictions(
    y_true: list[str],
    predictions: dict[str, str],
    folds: list[int],
    rows: list[dict],
) -> dict:
    """Evaluate a trace-keyed prediction map against the full set of cohort metrics."""
    ordered = [
        predictions.get(str(row.get("trace_id", "")), NONE) for row in rows
    ]
    return evaluate_cohorts(y_true, ordered, folds, rows)
