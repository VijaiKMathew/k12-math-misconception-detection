"""
Runtime and Score Jump Tracker Module (src/qaggle/tracker.py)
Tracks total execution time, hardware accelerators, start/end timestamps,
and score deltas (MAP@3 jump, CV, LB) across runs.
Persists records to data/processed/runs_ledger.json and provides structured reporting.
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_LEDGER_PATH = REPO_ROOT / "data" / "processed" / "runs_ledger.json"


def load_runs_ledger() -> list[dict[str, Any]]:
    if RUNS_LEDGER_PATH.is_file():
        try:
            return json.loads(RUNS_LEDGER_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_runs_ledger(ledger: list[dict[str, Any]]) -> None:
    RUNS_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNS_LEDGER_PATH.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def record_run_entry(
    run_id: str,
    run_type: str,  # 'kaggle_gpu', 'kaggle_tpu', 'colab_gpu', 'local_cpu', 'local_metal'
    platform: str,
    paradigm: str,
    duration_seconds: float,
    start_time: str,
    end_time: str,
    baseline_cv: float,
    new_cv: float,
    cv_delta: float,
    baseline_lb: float | None = None,
    new_lb: float | None = None,
    lb_delta: float | None = None,
    notes: str = "",
    status: str = "COMPLETED",
) -> dict[str, Any]:
    """Appends an execution record with timing and score delta into the persistent ledger."""
    entry: dict[str, Any] = {
        "run_id": run_id,
        "run_type": run_type,
        "platform": platform,
        "paradigm": paradigm,
        "duration_seconds": round(duration_seconds, 2),
        "duration_human": f"{int(duration_seconds // 3600)}h {int((duration_seconds % 3600) // 60)}m {int(duration_seconds % 60)}s" if duration_seconds >= 3600 else (f"{int(duration_seconds // 60)}m {int(duration_seconds % 60)}s" if duration_seconds >= 60 else f"{round(duration_seconds, 2)}s"),
        "start_time": start_time,
        "end_time": end_time,
        "baseline_cv": baseline_cv,
        "new_cv": new_cv,
        "cv_delta": round(cv_delta, 6),
        "baseline_lb": baseline_lb,
        "new_lb": new_lb,
        "lb_delta": round(lb_delta, 6) if lb_delta is not None else None,
        "notes": notes,
        "status": status,
    }
    ledger = load_runs_ledger()
    ledger.append(entry)
    save_runs_ledger(ledger)
    return entry


@contextmanager
def track_execution_run(
    run_id: str,
    run_type: str,
    platform: str,
    paradigm: str,
    baseline_cv: float,
    baseline_lb: float | None = None,
    notes: str = "",
) -> Generator[dict[str, Any], None, None]:
    """Context manager to measure runtime and log score jump upon exit."""
    start_wall = time.time()
    start_iso = datetime.now(timezone.utc).isoformat()
    state: dict[str, Any] = {
        "new_cv": baseline_cv,
        "new_lb": baseline_lb,
        "status": "COMPLETED",
        "notes": notes,
    }
    try:
        yield state
    except Exception as exc:
        state["status"] = f"FAILED: {exc}"
        raise
    finally:
        end_wall = time.time()
        end_iso = datetime.now(timezone.utc).isoformat()
        dur = end_wall - start_wall
        b_cv = baseline_cv
        n_cv = state.get("new_cv", b_cv)
        cv_jump = n_cv - b_cv
        b_lb = baseline_lb
        n_lb = state.get("new_lb", b_lb)
        lb_jump = (n_lb - b_lb) if (n_lb is not None and b_lb is not None) else None

        record_run_entry(
            run_id=run_id,
            run_type=run_type,
            platform=platform,
            paradigm=paradigm,
            duration_seconds=dur,
            start_time=start_iso,
            end_time=end_iso,
            baseline_cv=b_cv,
            new_cv=n_cv,
            cv_delta=cv_jump,
            baseline_lb=b_lb,
            new_lb=n_lb,
            lb_delta=lb_jump,
            notes=state.get("notes", notes),
            status=state.get("status", "COMPLETED"),
        )
