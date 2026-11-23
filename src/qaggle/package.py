"""Submission file validation for Misconception Detection."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from qaggle.repo import competition


def file_fingerprint(path: Path) -> dict:
    path = Path(path)
    size = path.stat().st_size
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            hasher.update(chunk)
    return {"bytes": size, "sha256": hasher.hexdigest()}


def validate_submission_file(path: Path) -> list[str]:
    """Validate submission CSV file schema, row count, and prediction format."""
    problems: list[str] = []
    if not path.is_file():
        return [f"File not found: {path}"]

    cfg = competition()
    expected_rows = cfg.get("expected_test_rows", 3498)
    id_col = cfg.get("id_column", "trace_id")
    target_col = cfg.get("target_column", "root_code")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return ["Submission file is empty"]
        if header != [id_col, target_col]:
            problems.append(f"Header mismatch: got {header}, expected {[id_col, target_col]}")

        row_count = 0
        has_empty_pred = False
        seen_ids = set()

        for row in reader:
            row_count += 1
            if len(row) != 2:
                problems.append(f"Row {row_count} has {len(row)} columns instead of 2")
                break
            
            trace_id, preds = row[0].strip(), row[1].strip()
            if not trace_id:
                problems.append(f"Row {row_count} has empty trace_id")
            if trace_id in seen_ids:
                problems.append(f"Row {row_count} has duplicate trace_id: {trace_id}")
            seen_ids.add(trace_id)

            if not preds:
                has_empty_pred = True
            
            # Check candidate tokens
            tokens = preds.split()
            if len(tokens) > 3:
                problems.append(f"Row {row_count} ({trace_id}) has {len(tokens)} predictions (max allowed 3)")

        if has_empty_pred:
            problems.append("Found empty prediction cells in submission")

        if row_count != expected_rows:
            problems.append(f"Row count mismatch: expected {expected_rows}, got {row_count}")

    return problems
