"""Quantum-Circuit Unitary Candidate Horizon Expansion & Recall Ceiling Auditor.

Inspired by `kaggle/quantum-circuit` (`qaggle`), where restricting gate optimization to a
truncated subspace traps the solver in barren plateaus / local optima. In Misconception
Detection, restricting reranking to Qwen3.5-9B's frozen Top-8 candidates caps unclean
workings recall at 79.16%. This module audits candidate recall ceilings and expands the
candidate subspace via multi-channel unitary union (P1 VLM + P2 Conditional Priors +
P3 Step/Span Retrieval + P4 Symbolic Verifier).
"""

from __future__ import annotations

import ast
import struct
from collections import Counter
from collections.abc import Sequence
from pathlib import Path


def load_npy_float_matrix(path: Path) -> tuple[list[list[float]], tuple[int, ...]]:
    """Pure-Python reader for 2D .npy float32/float64 matrices (zero external dependencies)."""
    raw = path.read_bytes()
    if raw[:6] != b"\x93NUMPY":
        raise ValueError(f"Invalid .npy file: {path}")
    major = raw[6]
    if major == 1:
        header_len = struct.unpack("<H", raw[8:10])[0]
        data_offset = 10 + header_len
    else:
        header_len = struct.unpack("<I", raw[8:12])[0]
        data_offset = 12 + header_len
    header_str = raw[data_offset - header_len : data_offset].decode("latin1").strip()
    meta = ast.literal_eval(header_str)
    descr = meta["descr"]
    shape = tuple(meta["shape"])
    rows, cols = shape[0], shape[1]
    body = raw[data_offset:]
    fmt_char = "f" if "f4" in descr else "d"
    elem_size = 4 if fmt_char == "f" else 8
    matrix: list[list[float]] = []
    for r in range(rows):
        start = r * cols * elem_size
        row_vals = list(struct.unpack_from(f"<{cols}{fmt_char}", body, start))
        matrix.append(row_vals)
    return matrix, shape


def compute_horizon_recall(
    candidate_lists: Sequence[Sequence[str] | str],
    y_true: Sequence[str],
    clean_flags: Sequence[bool | str] | None = None,
) -> dict:
    """Audit candidate pool recall ceiling before training or tuning any reranker."""
    n_total = max(len(y_true), 1)
    hits_all = 0
    hits_unclean = 0
    n_unclean = 0
    k_sizes = []

    for idx, (cands_raw, target) in enumerate(zip(candidate_lists, y_true)):
        cands = cands_raw.strip().split() if isinstance(cands_raw, str) else list(cands_raw)
        k_sizes.append(len(cands))
        in_pool = target in cands
        if in_pool:
            hits_all += 1
        if clean_flags is not None:
            is_clean = str(clean_flags[idx]) == "True"
            if not is_clean:
                n_unclean += 1
                if in_pool:
                    hits_unclean += 1

    overall_recall = hits_all / n_total
    unclean_recall = hits_unclean / max(n_unclean, 1) if clean_flags is not None else overall_recall
    avg_k = sum(k_sizes) / len(k_sizes) if k_sizes else 0.0

    return {
        "avg_horizon_k": avg_k,
        "overall_recall": overall_recall,
        "unclean_recall": unclean_recall,
        "ceiling_map3": overall_recall,
        "unclean_ceiling_map3": unclean_recall,
        "is_subspace_trapped": unclean_recall < 0.90,
    }


def expand_unitary_horizon(
    vlm_top_cands: Sequence[str],
    full_vlm_probs: dict[str, float],
    question_error_counts: Counter,
    topic_error_counts: Counter,
    verifier_bonuses: dict[str, float],
    span_matches: Counter,
    fline_matches: Counter,
    max_horizon: int = 24,
) -> list[str]:
    """Expand a trace's candidate subspace to a high-recall Unitary Horizon (H18 -> H24)."""
    ordered: list[str] = ["NONE"]
    seen = {"NONE"}

    def _add_many(codes: Sequence[str], limit: int) -> None:
        added = 0
        for code in codes:
            if code and code not in seen and len(ordered) < max_horizon:
                seen.add(code)
                ordered.append(code)
                added += 1
                if added >= limit:
                    break

    # 1. Primary VLM Top-8 candidates
    _add_many(list(vlm_top_cands), limit=8)

    # 2. High-precision span & final-line signature matches
    _add_many([c for c, _ in fline_matches.most_common()], limit=4)
    _add_many([c for c, _ in span_matches.most_common()], limit=4)

    # 3. Symbolic step-transition verifier candidates
    _add_many([c for c, _ in sorted(verifier_bonuses.items(), key=lambda x: x[1], reverse=True)], limit=4)

    # 4. Question-conditioned non-NONE empirical error modes
    _add_many([c for c, _ in question_error_counts.most_common()], limit=6)

    # 5. Topic-conditioned non-NONE empirical error modes
    _add_many([c for c, _ in topic_error_counts.most_common()], limit=5)

    # 6. Full 75-class VLM probability modes (outside top-8)
    _add_many([c for c, _ in sorted(full_vlm_probs.items(), key=lambda x: x[1], reverse=True)], limit=8)

    return ordered

