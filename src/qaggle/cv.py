"""High-performance, zero-dependency-compatible cross-validation and cohort diagnostics for Misconception Detection."""

from __future__ import annotations

import csv
import math
import struct
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from qaggle.metric import compute_map3
from qaggle.repo import repo_root, test_parquet_path, train_parquet_path


def _snappy_decompress(data: bytes) -> bytes:
    pos = 0
    n = len(data)
    uncompressed_len = 0
    shift = 0
    while pos < n:
        b = data[pos]
        pos += 1
        uncompressed_len |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    out = bytearray()
    while pos < n:
        tag = data[pos]
        pos += 1
        tag_type = tag & 0x03
        if tag_type == 0:
            length = (tag >> 2) + 1
            if length > 60:
                extra_bytes = length - 60
                length = int.from_bytes(data[pos : pos + extra_bytes], "little") + 1
                pos += extra_bytes
            out.extend(data[pos : pos + length])
            pos += length
        elif tag_type == 1:
            length = ((tag >> 2) & 0x07) + 4
            offset = ((tag >> 5) << 8) | data[pos]
            pos += 1
            for _ in range(length):
                out.append(out[-offset])
        elif tag_type == 2:
            length = (tag >> 2) + 1
            offset = int.from_bytes(data[pos : pos + 2], "little")
            pos += 2
            for _ in range(length):
                out.append(out[-offset])
        elif tag_type == 3:
            length = (tag >> 2) + 1
            offset = int.from_bytes(data[pos : pos + 4], "little")
            pos += 4
            for _ in range(length):
                out.append(out[-offset])
    return bytes(out)


def _decode_rle_indices(data: bytes, bit_width: int, num_values: int) -> list[int]:
    pos = 0
    indices: list[int] = []
    while pos < len(data) and len(indices) < num_values:
        header = 0
        shift = 0
        while True:
            b = data[pos]
            pos += 1
            header |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        if header & 1:
            num_groups = header >> 1
            num_vals = num_groups * 8
            bytes_needed = (num_vals * bit_width + 7) // 8
            chunk = data[pos : pos + bytes_needed]
            pos += bytes_needed
            val_int = int.from_bytes(chunk, "little")
            mask = (1 << bit_width) - 1
            for _ in range(num_vals):
                if len(indices) < num_values:
                    indices.append(val_int & mask)
                    val_int >>= bit_width
        else:
            run_len = header >> 1
            bytes_per_val = (bit_width + 7) // 8
            val = int.from_bytes(data[pos : pos + bytes_per_val], "little")
            pos += bytes_per_val
            indices.extend([val] * min(run_len, num_values - len(indices)))
    return indices


def _read_uvarint(buf: bytes, pos: int) -> tuple[int, int]:
    res = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            return res, pos
        shift += 7


def _read_zigzag(buf: bytes, pos: int) -> tuple[int, int]:
    v, pos = _read_uvarint(buf, pos)
    return (v >> 1) ^ -(v & 1), pos


def _skip_thrift(buf: bytes, pos: int, ttype: int) -> int:
    if ttype in (1, 2):
        return pos
    if ttype == 3:
        return pos + 1
    if ttype in (4, 5, 6):
        _, pos = _read_uvarint(buf, pos)
        return pos
    if ttype == 7:
        return pos + 8
    if ttype == 8:
        ln, pos = _read_uvarint(buf, pos)
        return pos + ln
    if ttype in (9, 10):
        hdr = buf[pos]
        pos += 1
        sz = hdr >> 4
        et = hdr & 0x0F
        if sz == 15:
            sz, pos = _read_uvarint(buf, pos)
        for _ in range(sz):
            pos = _skip_thrift(buf, pos, et)
        return pos
    if ttype == 11:
        sz, pos = _read_uvarint(buf, pos)
        if sz == 0:
            return pos
        kv = buf[pos]
        pos += 1
        kt, vt = kv >> 4, kv & 0x0F
        for _ in range(sz):
            pos = _skip_thrift(buf, pos, kt)
            pos = _skip_thrift(buf, pos, vt)
        return pos
    if ttype == 12:
        while True:
            b = buf[pos]
            pos += 1
            if b == 0:
                return pos
            ft = b & 0x0F
            if (b >> 4) == 0:
                _, pos = _read_uvarint(buf, pos)
            pos = _skip_thrift(buf, pos, ft)
    raise ValueError(f"Unsupported Thrift type: {ttype}")


def _parse_parquet_page_header(buf: bytes, pos: int) -> tuple[int | None, int, int, int, int, int]:
    fid = 0
    page_type: int | None = None
    uncomp_sz = 0
    comp_sz = 0
    num_vals = 0
    enc = 0
    while True:
        b = buf[pos]
        pos += 1
        if b == 0:
            break
        delta = b >> 4
        ft = b & 0x0F
        if delta == 0:
            fid, pos = _read_zigzag(buf, pos)
        else:
            fid += delta
        if fid == 1 and ft == 5:
            page_type, pos = _read_zigzag(buf, pos)
        elif fid == 2 and ft == 5:
            uncomp_sz, pos = _read_zigzag(buf, pos)
        elif fid == 3 and ft == 5:
            comp_sz, pos = _read_zigzag(buf, pos)
        elif fid in (5, 7, 8) and ft == 12:
            sub_fid = 0
            while True:
                sb = buf[pos]
                pos += 1
                if sb == 0:
                    break
                sdelta = sb >> 4
                sft = sb & 0x0F
                if sdelta == 0:
                    sub_fid, pos = _read_zigzag(buf, pos)
                else:
                    sub_fid += sdelta
                if sub_fid == 1 and sft == 5:
                    num_vals, pos = _read_zigzag(buf, pos)
                elif sub_fid == 2 and sft == 5:
                    enc, pos = _read_zigzag(buf, pos)
                else:
                    pos = _skip_thrift(buf, pos, sft)
        else:
            pos = _skip_thrift(buf, pos, ft)
    return page_type, uncomp_sz, comp_sz, num_vals, enc, pos


def read_parquet_score_triplets(parquet_path: Path) -> list[tuple[str, str, float]]:
    """Zero-dependency pure-Python Parquet reader for (trace_id, cand_code, score) tables."""
    b = Path(parquet_path).read_bytes()
    footer_len = struct.unpack("<I", b[-8:-4])[0]
    end = len(b) - 8 - footer_len
    pos = 4
    cols: list[tuple[tuple[bytes, int], list[tuple[bytes, int, int]]]] = []
    cur_dict: tuple[bytes, int] | None = None
    cur_vals: list[tuple[bytes, int, int]] = []
    while pos < end:
        pt, _usz, csz, nvals, enc, hdr_end = _parse_parquet_page_header(b, pos)
        raw = _snappy_decompress(b[hdr_end : hdr_end + csz])
        pos = hdr_end + csz
        if pt == 2:
            if cur_dict is not None:
                cols.append((cur_dict, cur_vals))
                cur_vals = []
            cur_dict = (raw, nvals)
        elif pt == 0:
            cur_vals.append((raw, nvals, enc))
    if cur_dict is not None:
        cols.append((cur_dict, cur_vals))
    if len(cols) < 3:
        return []

    def _decode_str_col(col_t: tuple[tuple[bytes, int], list[tuple[bytes, int, int]]]) -> list[str]:
        (d_raw, d_n), pages = col_t
        d_list: list[str] = []
        p = 0
        for _ in range(d_n):
            ln = struct.unpack_from("<I", d_raw, p)[0]
            p += 4
            d_list.append(d_raw[p : p + ln].decode("utf-8"))
            p += ln
        out: list[str] = []
        for raw, nv, _ in pages:
            dl = struct.unpack_from("<I", raw, 0)[0]
            bw = raw[4 + dl]
            out.extend(d_list[i] for i in _decode_rle_indices(raw[5 + dl :], bw, nv))
        return out

    def _decode_flt_col(
        col_t: tuple[tuple[bytes, int], list[tuple[bytes, int, int]]],
    ) -> list[float]:
        (d_raw, d_n), pages = col_t
        d_list = list(struct.unpack(f"<{d_n}d", d_raw))
        out: list[float] = []
        for raw, nv, enc in pages:
            dl = struct.unpack_from("<I", raw, 0)[0]
            pay = raw[4 + dl :]
            if enc in (2, 8) or len(pay) < nv * 8:
                bw = pay[0]
                out.extend(d_list[i] for i in _decode_rle_indices(pay[1:], bw, nv))
            else:
                out.extend(struct.unpack(f"<{nv}d", pay[: nv * 8]))
        return out

    return list(
        zip(
            _decode_str_col(cols[0]),
            _decode_str_col(cols[1]),
            _decode_flt_col(cols[-1]),
        )
    )


def load_exact_folds() -> list[int]:
    """Decode deterministic 5-fold assignment directly from data/processed/train.parquet."""
    tp_path = train_parquet_path()
    tp_bytes = tp_path.read_bytes()
    dict_raw = _snappy_decompress(tp_bytes[580139:580161])
    dict_folds = list(struct.unpack("<5i", dict_raw))
    data_raw = _snappy_decompress(tp_bytes[580211 : 580211 + 580])
    def_len = struct.unpack("<I", data_raw[:4])[0]
    bw = data_raw[4 + def_len]
    return [dict_folds[idx] for idx in _decode_rle_indices(data_raw[5 + def_len :], bw, 1500)]


def load_canonical_folds() -> dict[str, int]:
    """Return the canonical ``trace_id -> fold`` partition from ``data/processed/train.parquet``.

    Falls back to the dependency-free parquet decoder paired with the train CSV
    when pandas is unavailable, so the P2 trainer uses the exact same folds as
    P1/P3 instead of an independent hash partition.
    """
    seen: set[Path] = set()
    candidates = [train_parquet_path(), repo_root() / "data" / "processed" / "train.parquet"]
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            import pandas as pd
        except ImportError:
            break
        try:
            frame = pd.read_parquet(path, columns=["trace_id", "fold"])
        except (OSError, ValueError, ImportError) as exc:
            print(f"warning: could not read folds from {path}: {exc}", file=sys.stderr)
            continue
        return {
            str(trace_id): int(fold)
            for trace_id, fold in zip(frame["trace_id"].tolist(), frame["fold"].tolist())
            if not pd.isna(trace_id) and not pd.isna(fold)
        }

    return {
        str(row["trace_id"]): int(row["fold"])
        for row in load_train_rows()
    }


def load_train_rows() -> list[dict]:
    """Load raw train rows enriched with deterministic fold IDs."""
    root = repo_root()
    csv_path = root / "train.csv"
    if not csv_path.is_file():
        csv_path = root / "data" / "competition" / "train.csv"
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    folds = load_exact_folds()
    for r, fold in zip(rows, folds):
        r["fold"] = fold
    return rows


def load_test_rows() -> list[dict]:
    """Load raw test rows."""
    root = repo_root()
    csv_path = root / "test.csv"
    if not csv_path.is_file():
        csv_path = root / "data" / "competition" / "test.csv"
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_train_folds():
    """Load train dataset as pandas DataFrame when pandas is installed, else list of dicts."""
    try:
        import pandas as pd

        return pd.read_parquet(train_parquet_path())
    except (OSError, ValueError, ImportError):
        return load_train_rows()


def load_test():
    """Load test dataset as pandas DataFrame when pandas is installed, else list of dicts."""
    try:
        import pandas as pd

        return pd.read_parquet(test_parquet_path())
    except (OSError, ValueError, ImportError):
        return load_test_rows()


def evaluate_oof(
    y_true: Sequence[str],
    y_pred: Sequence[Sequence[str] | str],
    folds: Sequence[int],
) -> dict:
    """Compute overall OOF MAP@3 and fold-wise mean/std without requiring numpy."""
    overall_score = float(compute_map3(y_true, y_pred))
    unique_folds = sorted({int(f) for f in folds})
    fold_scores: list[float] = []

    for f in unique_folds:
        sub_true = [t for t, fl in zip(y_true, folds) if int(fl) == f]
        sub_pred = [p for p, fl in zip(y_pred, folds) if int(fl) == f]
        fold_scores.append(float(compute_map3(sub_true, sub_pred)))

    mean_val = sum(fold_scores) / len(fold_scores) if fold_scores else 0.0
    var_val = sum((s - mean_val) ** 2 for s in fold_scores) / len(fold_scores) if fold_scores else 0.0

    return {
        "overall_map3": overall_score,
        "fold_mean": float(mean_val),
        "fold_std": float(math.sqrt(var_val)),
        "fold_scores": fold_scores,
    }


BOTTLENECK_TOPICS = (
    "Linear Equation in One Variable",
    "Rationalizing the Denominator",
    "Combination",
    "Exponents",
    "Operations of Exponents",
)


def evaluate_cohorts(
    y_true: Sequence[str],
    y_pred: Sequence[Sequence[str] | str],
    folds: Sequence[int],
    train_rows: Sequence[dict] | None = None,
) -> dict:
    """Compute comprehensive multi-cohort diagnostic metrics for Global Optima gating."""
    if train_rows is None:
        train_rows = load_train_rows()

    base = evaluate_oof(y_true, y_pred, folds)
    ap_clean, ap_unclean, ap_hs, ap_diag, ap_text = [], [], [], [], []
    top1_all, top3_all, top1_unclean, top3_unclean = 0, 0, 0, 0
    topic_aps: dict[str, list[float]] = defaultdict(list)

    for r, target, pred in zip(train_rows, y_true, y_pred):
        cands = pred.strip().split()[:3] if isinstance(pred, str) else list(pred)[:3]
        score = 0.0
        for idx, c in enumerate(cands):
            if c == target:
                score = 1.0 / (idx + 1)
                break
        if cands and cands[0] == target:
            top1_all += 1
        if target in cands:
            top3_all += 1

        is_clean = str(r.get("clean", "")) == "True"
        if is_clean:
            ap_clean.append(score)
        else:
            ap_unclean.append(score)
            if cands and cands[0] == target:
                top1_unclean += 1
            if target in cands:
                top3_unclean += 1

        if str(r.get("grade", "")) in ("10", "11", "12"):
            ap_hs.append(score)
        if str(r.get("question_image_file", "")).strip():
            ap_diag.append(score)
        else:
            ap_text.append(score)

        topic_aps[str(r.get("topic_name", ""))].append(score)

    n_all = max(len(y_true), 1)
    n_unclean = max(len(ap_unclean), 1)
    bottleneck_scores = {
        t: (sum(topic_aps[t]) / len(topic_aps[t]) if topic_aps[t] else 0.0)
        for t in BOTTLENECK_TOPICS
    }
    bottleneck_all = [s for t in BOTTLENECK_TOPICS for s in topic_aps[t]]

    return {
        **base,
        "clean_map3": sum(ap_clean) / max(len(ap_clean), 1),
        "unclean_map3": sum(ap_unclean) / n_unclean,
        "high_school_map3": sum(ap_hs) / max(len(ap_hs), 1),
        "diagram_map3": sum(ap_diag) / max(len(ap_diag), 1),
        "text_only_map3": sum(ap_text) / max(len(ap_text), 1),
        "bottleneck_5_map3": sum(bottleneck_all) / max(len(bottleneck_all), 1),
        "bottleneck_topics": bottleneck_scores,
        "top1_acc": top1_all / n_all,
        "top3_recall": top3_all / n_all,
        "unclean_top1_acc": top1_unclean / n_unclean,
        "unclean_top3_recall": top3_unclean / n_unclean,
    }
