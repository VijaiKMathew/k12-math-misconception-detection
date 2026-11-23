"""MAP@3 Metric evaluation for Misconception Detection."""

from __future__ import annotations

from collections.abc import Sequence


def compute_map3(y_true: Sequence[str], y_pred: Sequence[Sequence[str] | str]) -> float:
    """Compute Mean Average Precision at 3 (MAP@3).
    
    Each prediction row can be:
      - a list/tuple of up to 3 strings (e.g. ['ALG-06', 'CAL-01', 'NONE'])
      - a space-delimited string (e.g. 'ALG-06 CAL-01 NONE')
    """
    total = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for true_val, pred_row in zip(y_true, y_pred):
        if isinstance(pred_row, str):
            cands = pred_row.strip().split()
        else:
            cands = list(pred_row)
        
        row_score = 0.0
        seen = set()
        k = 0
        for code in cands:
            if code in seen:
                continue
            seen.add(code)
            if code == true_val:
                row_score = 1.0 / (k + 1)
                break
            k += 1
            if k >= 3:
                break
        total += row_score

    return total / n


def selftest() -> bool:
    y_true = ["ALG-06", "NONE", "WOP-01"]
    
    # 1. First rank match (1.0), second rank match (0.5), missed (0.0) -> (1.0 + 0.5 + 0.0) / 3 = 0.5
    y_pred = [
        "ALG-06 CAL-01 NONE",
        "CAL-01 NONE WOP-01",
        "FRM-01 CAL-01 ANS-01",
    ]
    score = compute_map3(y_true, y_pred)
    assert abs(score - 0.5) < 1e-6, f"Expected 0.5, got {score}"

    # 2. Perfect score: all rank 1
    perfect = ["ALG-06", "NONE", "WOP-01"]
    assert abs(compute_map3(y_true, perfect) - 1.0) < 1e-6

    # 3. Third rank match -> 1/3
    rank3 = [["CAL-01", "ANS-01", "ALG-06"]]
    assert abs(compute_map3(["ALG-06"], rank3) - 1.0 / 3.0) < 1e-6

    return True


if __name__ == "__main__":
    if selftest():
        print("MAP@3 metric self-test PASSED")
