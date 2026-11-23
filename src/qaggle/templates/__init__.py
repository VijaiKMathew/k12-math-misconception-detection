"""Templates for high-speed, low-latency misconception detection scripts and workflows."""

from __future__ import annotations

from qaggle.templates.fast_script_template import (
    HardwareSummary,
    bootstrap_high_speed_runtime,
    configure_high_performance_env,
    get_hardware_summary,
    get_torch_device,
    parallel_map,
    run_parallel_evaluation_template,
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
