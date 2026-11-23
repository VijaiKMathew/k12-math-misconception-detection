"""
Cutting-Edge Model Soups & Weight-Space Ensembling Primitives (`src/qaggle/ensembling/model_soup.py`)
====================================================================================================
Implements linear mode connectivity ensembling techniques for zero-overhead inference:
- `average_state_dicts`: Uniform or weighted parameter-space averaging.
- `greedy_model_soup`: Progressively includes checkpoints that improve validation MAP@3.
- `dare_ties_merge`: TIES-merging (Trim, Elect Sign, Disjoint Mean) for task vectors.
- `SwitchModelEMA`: Switch Exponential Moving Average for flat-minima regularization.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeAlias

import torch
from torch import nn

StateDict: TypeAlias = dict[str, torch.Tensor]


def average_state_dicts(
    state_dicts: Sequence[StateDict],
    weights: Sequence[float] | None = None,
) -> StateDict:
    """Compute the weighted or uniform average across multiple PyTorch state dicts in-place."""
    if not state_dicts:
        raise ValueError("Cannot average an empty list of state dicts.")
    if len(state_dicts) == 1:
        return {k: v.clone() for k, v in state_dicts[0].items()}

    n = len(state_dicts)
    if weights is None:
        w_tensors = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(f"Length of weights ({len(weights)}) does not match state_dicts ({n}).")
        total_w = sum(weights)
        if total_w <= 0.0:
            raise ValueError(f"Sum of weights must be positive, got {total_w}.")
        w_tensors = [w / total_w for w in weights]

    ref_sd = state_dicts[0]
    averaged_sd: StateDict = {}

    for key, val in ref_sd.items():
        if not val.is_floating_point():
            # Non-floating point tensors (e.g. position_ids, int counts) take values from the first checkpoint
            averaged_sd[key] = val.clone()
            continue

        accum = torch.zeros_like(val, dtype=torch.float32)
        for sd, weight in zip(state_dicts, w_tensors):
            accum.add_(sd[key].to(dtype=torch.float32), alpha=weight)
        averaged_sd[key] = accum.to(dtype=val.dtype)

    return averaged_sd


def greedy_model_soup(
    checkpoint_paths: Sequence[Path | str],
    model: nn.Module,
    eval_map3_fn: Callable[[nn.Module], float],
    tolerance: float = 0.0,
) -> tuple[StateDict, float, list[Path | str]]:
    """Greedy Model Soup (Wortsman et al., ICML 2022).

    Ranks candidate checkpoints on validation MAP@3, starts with the single best checkpoint,
    and sequentially averages candidate checkpoints, keeping only candidates that do not
    degrade validation score below (best_score + tolerance).

    Returns:
        (final_soup_state_dict, best_score, selected_checkpoint_paths)
    """
    if not checkpoint_paths:
        raise ValueError("Must provide at least one checkpoint path for greedy model soup.")

    # Step 1: Score individual checkpoints
    ranked_candidates: list[tuple[float, Path | str, StateDict]] = []
    for path in checkpoint_paths:
        p = Path(path)
        sd: StateDict = torch.load(p, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
        score = float(eval_map3_fn(model))
        ranked_candidates.append((score, path, sd))

    # Sort descending by validation score
    ranked_candidates.sort(key=lambda x: x[0], reverse=True)

    best_score, first_path, first_sd = ranked_candidates[0]
    running_sds: list[StateDict] = [first_sd]
    selected_paths: list[Path | str] = [first_path]

    # Step 2: Progressively add checkpoints if the running soup score is >= best_score + tolerance
    for cand_score, cand_path, cand_sd in ranked_candidates[1:]:
        trial_sds = [*running_sds, cand_sd]
        trial_sd = average_state_dicts(trial_sds)
        model.load_state_dict(trial_sd, strict=True)
        trial_score = float(eval_map3_fn(model))

        if trial_score >= best_score + tolerance:
            best_score = trial_score
            running_sds.append(cand_sd)
            selected_paths.append(cand_path)

    final_soup_sd = average_state_dicts(running_sds)
    model.load_state_dict(final_soup_sd, strict=True)
    return final_soup_sd, best_score, selected_paths


def dare_ties_merge(
    pretrained_sd: StateDict,
    finetuned_sds: Sequence[StateDict],
    scaling_coef: float = 0.85,
    trim_quantile: float = 0.20,
) -> StateDict:
    """Merge fine-tuned checkpoints via TIES-Merging (Trim, Elect Sign, Disjoint Merge).

    Reference: Yadav et al. (NeurIPS 2023) / DARE (Yu et al., 2024).
    """
    if not finetuned_sds:
        raise ValueError("finetuned_sds cannot be empty.")

    merged: StateDict = {}
    for key, theta_0 in pretrained_sd.items():
        if not theta_0.is_floating_point():
            merged[key] = theta_0.clone()
            continue

        base_f32 = theta_0.to(dtype=torch.float32)
        deltas = [sd[key].to(dtype=torch.float32) - base_f32 for sd in finetuned_sds]
        stacked = torch.stack(deltas, dim=0)  # [K, ...]

        # Step 1 (Trim): Zero out small delta parameters below trim_quantile
        if trim_quantile > 0.0:
            abs_deltas = stacked.abs().flatten(start_dim=1)
            thresh = torch.quantile(abs_deltas, q=trim_quantile, dim=1)
            while thresh.dim() < stacked.dim():
                thresh = thresh.unsqueeze(-1)
            stacked = torch.where(stacked.abs() >= thresh, stacked, torch.zeros_like(stacked))

        # Step 2 (Elect Sign): Resolve sign conflict by total magnitude sum
        elected_sign = torch.sign(stacked.sum(dim=0))

        # Step 3 (Disjoint Mean): Average only deltas whose sign matches elected_sign
        sign_match = (torch.sign(stacked) == elected_sign) & (elected_sign != 0)
        match_counts = sign_match.sum(dim=0).clamp(min=1).to(dtype=torch.float32)
        disjoint_mean = (stacked * sign_match).sum(dim=0) / match_counts

        merged[key] = (base_f32 + scaling_coef * disjoint_mean).to(dtype=theta_0.dtype)

    return merged


class SwitchModelEMA:
    """Switch Exponential Moving Average (SEMA, arXiv:2402.09240) for flat-minima tracking.

    Maintains a shadow EMA copy of trainable parameters during optimization and
    switches the main model weights to the EMA shadow weights at epoch boundaries.
    """

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone().to(dtype=torch.float32)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights with current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.detach().to(dtype=torch.float32), alpha=1.0 - self.decay
                )

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite live model parameters with EMA shadow parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.copy_(self.shadow[name].to(dtype=param.dtype))
