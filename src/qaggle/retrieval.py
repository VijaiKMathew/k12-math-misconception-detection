"""P2 retrieval utilities: candidate recall and contrastive hard-negative mining."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _as_codes(candidates: Sequence[str] | str) -> list[str]:
    if isinstance(candidates, str):
        return candidates.split()
    return list(candidates)


def _dedupe(codes: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            unique.append(code)
    return unique


def _is_clean(flag: bool | str) -> bool:
    if isinstance(flag, bool):
        return flag
    return str(flag).strip().lower() in {"true", "1", "yes"}


def _is_root(flag: bool | str | int | None) -> bool:
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        return flag.strip().lower() in {"true", "1", "yes"}
    return False


def compute_recall_at_k(
    candidate_lists: Sequence[Sequence[str] | str],
    y_true: Sequence[str],
    clean_flags: Sequence[bool | str] | None = None,
    k_values: Sequence[int] = (8, 16, 18, 25),
) -> dict[str, float | int]:
    """Fraction of traces whose true code appears in the first ``k`` candidates.

    Candidate lists shorter than ``k`` are used as-is, so a trace can still be a
    hit when its list has fewer entries than the requested cutoff.
    """
    n_total = len(y_true)
    if len(candidate_lists) != n_total:
        raise ValueError(
            "candidate_lists and y_true must have the same length: "
            f"{len(candidate_lists)} != {n_total}"
        )
    if clean_flags is not None and len(clean_flags) != n_total:
        raise ValueError(
            "clean_flags and y_true must have the same length: "
            f"{len(clean_flags)} != {n_total}"
        )

    n_unclean = 0
    if clean_flags is not None:
        n_unclean = sum(1 for flag in clean_flags if not _is_clean(flag))

    result: dict[str, float | int] = {"n_unclean": n_unclean}
    for k in k_values:
        hits_all = 0
        hits_unclean = 0
        for idx, (cands_raw, target) in enumerate(zip(candidate_lists, y_true)):
            cands = _dedupe(_as_codes(cands_raw))
            hit = target in cands[:k]
            if hit:
                hits_all += 1
            if clean_flags is not None and not _is_clean(clean_flags[idx]) and hit:
                hits_unclean += 1
        overall = hits_all / n_total if n_total else 0.0
        if clean_flags is None:
            unclean = overall
        else:
            unclean = hits_unclean / n_unclean if n_unclean else 0.0
        result[f"overall_recall@{k}"] = overall
        result[f"unclean_recall@{k}"] = unclean
    return result


def _parse_misconceptions(
    raw: str | Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [m for m in raw if isinstance(m, dict)]
    if not raw or not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [m for m in parsed if isinstance(m, dict)]


def _topic_index(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    topic_codes: dict[str, set[str]] = {}
    topic_roots: dict[str, set[str]] = {}
    for row in rows:
        topic = str(row.get("topic_name") or "")
        codes = topic_codes.setdefault(topic, set())
        for item in _parse_misconceptions(row.get("misconceptions")):
            category = item.get("category")
            if category:
                codes.add(str(category))
        root = row.get("root_code")
        if root and root != "NONE":
            topic_roots.setdefault(topic, set()).add(str(root))
    return topic_codes, topic_roots


def build_hard_negatives_from_champion(
    rows: Iterable[dict[str, Any]],
    champion_maps: Mapping[str, Mapping[str, float]],
    k: int = 7,
) -> list[dict[str, Any]]:
    """Mine hard negatives for each unclean trace from champion OOF false positives.

    Negatives are the champion's top-scoring error codes on *this specific trace*
    that are not the root and not ``NONE`` — the strongest false positives are the
    hardest confusions for a bi-encoder. Traces whose champion map lacks enough
    such candidates fall back to the deterministic sibling/topic channels of
    :func:`build_hard_negatives` (for the missing slots only), preserving the
    original pair schema ``{"trace_id", "positive", "negatives"}``.
    """
    fallback_rows: list[dict[str, Any]] = []
    for row in rows:
        root = row.get("root_code")
        if not root or root == "NONE":
            continue
        enriched = dict(row)
        misconceptions = _parse_misconceptions(row.get("misconceptions"))
        has_root_annotation = any(_is_root(item.get("root")) for item in misconceptions)
        if not has_root_annotation:
            enriched["misconceptions"] = json.dumps(
                [{"step": "L1", "span": "", "category": str(root), "root": True, "text": ""}]
            )
        fallback_rows.append(enriched)
    fallback_pairs = build_hard_negatives(fallback_rows, k=4096)
    fallback_by_tid = {str(pair.get("trace_id")): pair for pair in fallback_pairs}

    pairs: list[dict[str, Any]] = []
    for row in rows:
        root = row.get("root_code")
        trace_id = str(row.get("trace_id"))
        if not root or root == "NONE":
            continue
        positive = str(root)

        scores = champion_maps.get(trace_id) or {}
        champion_negatives = [
            code
            for code, _score in sorted(
                scores.items(), key=lambda item: (-float(item[1]), 0.0)
            )
            if code and code != positive and code != "NONE"
        ]
        negatives: list[str] = []
        seen: set[str] = set()
        for code in champion_negatives:
            if code in seen:
                continue
            seen.add(code)
            negatives.append(code)
            if len(negatives) >= k:
                break
        if len(negatives) < k:
            fallback = fallback_by_tid.get(trace_id, {})
            for code in fallback.get("negatives") or []:
                if code in seen or code == positive:
                    continue
                seen.add(code)
                negatives.append(code)
                if len(negatives) >= k:
                    break
        pairs.append({"trace_id": row.get("trace_id"), "positive": positive, "negatives": negatives})
    return pairs


def build_hard_negatives(
    rows: Iterable[dict[str, Any]],
    k: int = 7,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Mine deterministic hard negatives for each non-clean trace with a root code.

    Negatives come from three channels: non-root misconceptions on the trace,
    sibling codes sharing the positive's code prefix, and other root codes seen
    for the same ``topic_name``.
    """
    row_list = list(rows)
    topic_codes, topic_roots = _topic_index(row_list)
    all_codes = {code for codes in topic_codes.values() for code in codes}

    pairs: list[dict[str, Any]] = []
    for index, row in enumerate(row_list):
        root = row.get("root_code")
        if not root or root == "NONE":
            continue

        misconceptions = _parse_misconceptions(row.get("misconceptions"))
        root_categories = {
            str(item["category"])
            for item in misconceptions
            if _is_root(item.get("root")) and item.get("category")
        }
        roots = sorted(root_categories)
        if not roots:
            continue
        positive = str(root) if str(root) in roots else roots[0]

        topic = str(row.get("topic_name") or "")
        prefix = positive.split("-")[0]

        candidate_codes: set[str] = set()
        for item in misconceptions:
            category = item.get("category")
            if category and not _is_root(item.get("root")):
                candidate_codes.add(str(category))
        candidate_codes.update(
            code for code in all_codes if code.split("-")[0] == prefix
        )
        candidate_codes.update(topic_roots.get(topic, set()))

        candidate_codes.difference_update(root_categories)
        candidate_codes.discard("NONE")
        candidates = sorted(candidate_codes)

        rng = random.Random(seed + index)
        negatives = rng.sample(candidates, min(k, len(candidates)))

        pairs.append(
            {
                "trace_id": row.get("trace_id"),
                "positive": positive,
                "negatives": negatives,
            }
        )
    return pairs


def summarize_hard_negative_coverage(pairs: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    """Summarise how many traces and positive codes the mined pairs cover."""
    n_pairs = len(pairs)
    n_with_negatives = sum(1 for pair in pairs if pair.get("negatives"))
    total_negatives = sum(len(pair.get("negatives") or []) for pair in pairs)
    distinct_positives = {pair.get("positive") for pair in pairs if pair.get("positive")}

    return {
        "n_pairs": n_pairs,
        "n_with_negatives": n_with_negatives,
        "mean_negatives": total_negatives / n_pairs if n_pairs else 0.0,
        "n_distinct_positives": len(distinct_positives),
    }
