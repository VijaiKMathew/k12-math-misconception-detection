"""
Loss Formulations for Cross-Encoders and Candidate Rerankers
============================================================
Kaggle Grandmaster Best Practice (Yauhen Babakhin / Team Hydrogen / Eedi 1st Place).
Provides:
  - TraceGroupedMarginLoss: Pairwise margin loss evaluated strictly within trace candidates.
  - RootVsSecondaryMarginLoss: Asymmetric penalty punishing secondary trap candidates ranked above true roots.
  - StepDiscountedIntraTraceMarginLoss: Step-discounted margin punishing late secondary slips over early root errors.
  - InfoNCELoss: Temperature-scaled contrastive loss over candidate logits.
  - intra_trace_step_discounted_penalty_python: Pure-Python stdlib reference implementation.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment,misc]


class TraceGroupedMarginLoss(nn.Module if torch is not None else object):  # type: ignore[misc]
    """Pairwise margin ranking loss strictly evaluated within trace candidates."""

    def __init__(self, margin: float = 0.5, reduction: str = "mean") -> None:
        if torch is not None:
            super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, root_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
        """
        root_logits: [N] tensor of scores for true root misconception
        neg_logits: [N, K] tensor of scores for negative candidate misconceptions
        """
        if root_logits.dim() == 1:
            root_expanded = root_logits.unsqueeze(1)
        else:
            root_expanded = root_logits

        # Loss: max(0, margin - (root_score - neg_score))
        losses = F.relu(self.margin - (root_expanded - neg_logits))
        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        return losses


class RootVsSecondaryMarginLoss(nn.Module if torch is not None else object):  # type: ignore[misc]
    """Asymmetric margin loss explicitly punishing secondary misconception traps."""

    def __init__(self, root_margin: float = 0.5, secondary_penalty: float = 1.0) -> None:
        if torch is not None:
            super().__init__()
        self.root_margin = root_margin
        self.secondary_penalty = secondary_penalty

    def forward(
        self,
        root_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        is_secondary_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        root_logits: [N, 1] scores for true root code
        neg_logits: [N, K] scores for other candidates
        is_secondary_mask: [N, K] boolean tensor indicating which negatives are secondary errors (root: false)
        """
        raw_diff = self.root_margin - (root_logits - neg_logits)
        base_loss = F.relu(raw_diff)

        # Scale penalty for secondary traps
        weights = 1.0 + (self.secondary_penalty - 1.0) * is_secondary_mask.float()
        weighted_loss = base_loss * weights
        return weighted_loss.mean()


class StepDiscountedIntraTraceMarginLoss(nn.Module if torch is not None else object):  # type: ignore[misc]
    """Step-discounted intra-trace margin loss punishing late secondary slips over early root errors.

    Penalizes candidates on Step k_neg being ranked above Step k_root when k_neg > k_root:
    L = max(0, gamma * (1 + beta * (k_neg - k_root)) - s_root + s_neg) * (1 + (penalty - 1) * is_sec)
    """

    def __init__(
        self,
        root_margin: float = 0.5,
        secondary_penalty: float = 1.5,
        step_discount_beta: float = 0.25,
    ) -> None:
        if torch is not None:
            super().__init__()
        self.root_margin = root_margin
        self.secondary_penalty = secondary_penalty
        self.step_discount_beta = step_discount_beta

    def forward(
        self,
        root_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        is_secondary_mask: torch.Tensor,
        root_steps: torch.Tensor,
        neg_steps: torch.Tensor,
    ) -> torch.Tensor:
        """
        root_logits: [N, 1] scores for true root misconception
        neg_logits: [N, K] scores for negative candidates
        is_secondary_mask: [N, K] boolean tensor indicating secondary slip candidates
        root_steps: [N, 1] integer/float step index where root misconception occurs (e.g. 1.0 for L1)
        neg_steps: [N, K] integer/float step index where candidate error occurs
        """
        step_delta = F.relu(neg_steps - root_steps)
        dynamic_margin = self.root_margin * (1.0 + self.step_discount_beta * step_delta)
        raw_diff = dynamic_margin - (root_logits - neg_logits)
        base_loss = F.relu(raw_diff)

        weights = 1.0 + (self.secondary_penalty - 1.0) * is_secondary_mask.float()
        weighted_loss = base_loss * weights
        return weighted_loss.mean()


class InfoNCELoss(nn.Module if torch is not None else object):  # type: ignore[misc]
    """Temperature-scaled InfoNCE loss for listwise candidate reranking."""

    def __init__(self, temperature: float = 0.07) -> None:
        if torch is not None:
            super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        """
        logits: [N, K] scores for K candidates per trace
        target_indices: [N] index of true root candidate in each trace
        """
        scaled_logits = logits / self.temperature
        return F.cross_entropy(scaled_logits, target_indices)


def intra_trace_step_discounted_penalty_python(
    root_score: float,
    neg_score: float,
    is_secondary: bool,
    root_step: int = 1,
    neg_step: int = 4,
    gamma: float = 0.5,
    secondary_penalty: float = 1.5,
    step_discount_beta: float = 0.25,
) -> float:
    """Pure-Python stdlib reference implementation of step-discounted intra-trace penalty."""
    step_diff = max(0, neg_step - root_step)
    dyn_margin = gamma * (1.0 + step_discount_beta * float(step_diff))
    diff = dyn_margin - (root_score - neg_score)
    loss = max(0.0, diff)
    weight = secondary_penalty if is_secondary else 1.0
    return loss * weight
