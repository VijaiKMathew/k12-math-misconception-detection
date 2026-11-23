"""
Hill-Climbing Greedy Ensembling Optimizer
=========================================
Kaggle Grandmaster Best Practice (Chris Deotte / cdeotte / NVIDIA Team).
Iteratively performs forward selection of model predictions on Out-of-Fold (OOF) arrays
to strictly maximize the evaluation metric without overfitting.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class HillClimbingOptimizer:
    """Greedy forward-selection model ensembling optimizer."""

    def __init__(
        self,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        max_iterations: int = 50,
        weights_to_try: list[float] | None = None,
        tolerance: float = 1e-5,
    ) -> None:
        self.metric_fn = metric_fn
        self.max_iterations = max_iterations
        self.weights_to_try = weights_to_try or [0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
        self.tolerance = tolerance

        self.selected_indices_: list[int] = []
        self.weights_: list[float] = []
        self.best_score_: float = 0.0

    def fit(self, oofs: list[np.ndarray], y_true: np.ndarray) -> dict[str, Any]:
        """
        oofs: list of [N, ...] array predictions from each candidate model
        y_true: ground truth targets
        """
        if not oofs:
            raise ValueError("Candidate OOF list cannot be empty.")

        num_models = len(oofs)
        single_scores = [float(self.metric_fn(y_true, pred)) for pred in oofs]
        best_single_idx = int(np.argmax(single_scores))

        self.selected_indices_ = [best_single_idx]
        self.weights_ = [1.0]
        self.best_score_ = single_scores[best_single_idx]

        current_ensemble = oofs[best_single_idx].astype(np.float64).copy()
        current_sum_weights = 1.0

        for _ in range(self.max_iterations):
            best_candidate_idx = -1
            best_candidate_score = self.best_score_
            best_candidate_w = 0.0

            for m_idx in range(num_models):
                for w in self.weights_to_try:
                    trial_ensemble = current_ensemble + w * oofs[m_idx]
                    trial_pred = trial_ensemble / (current_sum_weights + w)
                    trial_score = float(self.metric_fn(y_true, trial_pred))

                    if trial_score > best_candidate_score + self.tolerance:
                        best_candidate_score = trial_score
                        best_candidate_idx = m_idx
                        best_candidate_w = w

            if best_candidate_idx == -1:
                break  # Convergence reached: no addition improves CV

            self.selected_indices_.append(best_candidate_idx)
            self.weights_.append(best_candidate_w)
            current_ensemble += best_candidate_w * oofs[best_candidate_idx]
            current_sum_weights += best_candidate_w
            self.best_score_ = best_candidate_score

        norm_weights = [w / current_sum_weights for w in self.weights_]
        return {
            "selected_indices": self.selected_indices_,
            "weights": norm_weights,
            "raw_weights": self.weights_,
            "best_score": self.best_score_,
            "single_scores": single_scores,
        }

    def predict(self, test_preds: list[np.ndarray]) -> np.ndarray:
        """Blend test set predictions using the learned optimal weights."""
        if not self.selected_indices_:
            raise RuntimeError("Optimizer must be fit() before predict().")

        total_weight = sum(self.weights_)
        blend = np.zeros_like(test_preds[self.selected_indices_[0]], dtype=np.float64)

        for idx, w in zip(self.selected_indices_, self.weights_):
            blend += w * test_preds[idx]

        return (blend / total_weight).astype(test_preds[0].dtype)
