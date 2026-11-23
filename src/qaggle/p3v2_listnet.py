"""Pure, device-agnostic Wave-2 P3 ListNet loss and trace-grouping helpers.

Embedded verbatim into ``notebooks/tpu/p3_deberta_tpu_v3_8.ipynb`` so the TPU
trainer and the local unit tests exercise the identical math. Device tensor
math (torch) only; no qaggle imports to keep the notebook self-contained.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

SECONDARY_SLIP_CODES = frozenset({"CAL-01", "ANS-01", "NOT-01"})


def listnet_ce(
    logits: torch.Tensor, root_index: int, valid_mask: torch.Tensor
) -> torch.Tensor:
    """ListNet cross-entropy over the trace's valid candidates (softmax at root).

    ``logits``: 1-D torch tensor; ``root_index``: position of the true root;
    ``valid_mask``: bool tensor over candidates (padding positions excluded).
    Padding candidates are forced to ``-inf`` before the softmax so they cannot
    receive probability mass.
    """
    import torch as _torch

    masked = _torch.where(
        valid_mask.to(_torch.bool),
        logits,
        _torch.full_like(logits, -1e4),
    )
    log_probs = _torch.log_softmax(masked, dim=-1)
    return -log_probs[root_index]


def slip_margin_hinge(
    logits: torch.Tensor,
    root_index: int,
    slip_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """ARM-05 root-vs-secondary-slip hinge, strictly within this trace group.

    ``slip_mask`` marks candidates that are secondary-slip negatives; the hinge is
    averaged over the valid slips and is exactly zero when none are present.
    """
    import torch as _torch

    active = slip_mask.to(_torch.bool) & valid_mask.to(_torch.bool)
    if not bool(active.any()):
        return _torch.zeros((), dtype=logits.dtype, device=logits.device)
    s_root = logits[root_index]
    s_slips = logits[active]
    return _torch.relu(gamma - s_root + s_slips).mean()


def p3v2_total_loss(
    logits: torch.Tensor,
    root_index: int,
    valid_mask: torch.Tensor,
    slip_mask: torch.Tensor,
    gamma: float = 2.0,
    lambda_slip: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wave-2 P3v2 objective: ListNet CE + lambda * intra-trace slip hinge."""
    ce = listnet_ce(logits, root_index, valid_mask)
    hinge = slip_margin_hinge(logits, root_index, slip_mask, valid_mask, gamma)
    return ce + lambda_slip * hinge, ce, hinge


def pad_group_indices(
    trace_size: int, group_pad: int
) -> list[int]:
    """Return candidate-slot plan for a trace group padded to ``group_pad``."""
    if trace_size > group_pad:
        raise ValueError(f"trace group {trace_size} exceeds padded capacity {group_pad}")
    return list(range(trace_size))


def group_pad_from_groups(sizes: list[int]) -> int:
    """Pick the static candidate-slot capacity for the run (max group size)."""
    return max(sizes)
