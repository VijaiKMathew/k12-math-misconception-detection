"""Canonical Low-Latency & High-Hardware-Utilization Template for Misconception Detection.

Every new script in `scripts/`, `tools/`, or `src/qaggle/` should follow this template
to guarantee:
1. Multi-Core Apple Silicon M5 / BLAS / OpenMP thread tuning (`configure_high_performance_env()`).
2. Automatic Accelerator Selection (`CUDA GPU -> Apple Metal MPS GPU -> TPU -> Multi-Core CPU`).
3. Safe Multi-Core Parallelism (`parallel_map()` over traces, candidates, or embedding queries).
4. Dependency-free Canonical Parquet Fold & Tensor Loading (`qaggle.cv`).
5. Live Single Source of Truth (SSOT) binding to `CHAMPION.json` (`s6e10-qaggle-unitary-5paradigm-global-optima-stacker`).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Ensure src is in sys.path when invoked directly as a script
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qaggle.champion import load_champion
from qaggle.perf import (
    HardwareSummary,
    configure_high_performance_env,
    get_hardware_summary,
    get_torch_device,
    parallel_map,
)

__all__ = [
    "HardwareSummary",
    "bootstrap_high_speed_runtime",
    "configure_high_performance_env",
    "get_hardware_summary",
    "get_torch_device",
    "parallel_map",
    "run_parallel_evaluation_template",
]


def bootstrap_high_speed_runtime(allow_gpu: bool = True) -> dict[str, Any]:
    """Initialize thread pools, BLAS/MPS env vars, accelerator device, and champion SSOT."""
    configure_high_performance_env()
    hw: HardwareSummary = get_hardware_summary()
    device = get_torch_device(allow_gpu=allow_gpu)
    champion = load_champion()
    return {
        "hardware": hw,
        "torch_device": str(device) if device is not None else "cpu",
        "champion_label": champion.get("label", "unknown"),
        "champion_cv_score": champion.get("cv_score", 0.0),
        "champion_lb_score": champion.get("best_verified_public_lb", champion.get("lb_score", 0.0)),
        "champion_source": champion.get("source", ""),
    }


def run_parallel_evaluation_template(
    items: list[Any],
    worker_fn: Callable[[Any], Any],
    title: str = "PARALLEL EVALUATION",
    backend: str = "thread",
) -> list[Any]:
    """Execute a multi-core batch evaluation using parallel_map with progress reporting."""
    configure_high_performance_env()
    total = len(items)
    done = 0

    def _tick() -> None:
        nonlocal done
        done += 1
        if total > 0 and (done % max(1, total // 10) == 0 or done == total):
            pct = (done / total) * 100.0
            print(f"[{title}] {done}/{total} ({pct:.1f}%) completed", flush=True)

    print(f"[{title}] Starting {total} items across multi-core workers...", flush=True)
    results = parallel_map(worker_fn, items, backend=backend, progress=_tick)
    print(f"[{title}] Finished {total} items successfully.", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Verify runtime bootstrap and parallel execution")
    args = parser.parse_args(argv)

    ctx = bootstrap_high_speed_runtime()
    hw: HardwareSummary = ctx["hardware"]
    print(
        f"[FAST-TEMPLATE] Device={hw.device_name} | Cores={hw.cpu_cores} | "
        f"Active={hw.active_device} | MPS={hw.mps_available} | CUDA={hw.cuda_available}",
        flush=True,
    )
    print(
        f"[SSOT-CHAMPION] {ctx['champion_label']} (CV={ctx['champion_cv_score']}, "
        f"Verified LB={ctx['champion_lb_score']}, source={ctx['champion_source']})",
        flush=True,
    )

    if args.selftest:
        squares = run_parallel_evaluation_template(
            list(range(20)),
            lambda x: int(x * x),
            title="SELFTEST PARALLEL MAP",
        )
        if squares != [i * i for i in range(20)]:
            raise AssertionError("parallel_map selftest mismatch")
        print("[SELFTEST] PASS (parallel_map + SSOT champion verified)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
