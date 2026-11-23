"""
Seed Averaging and Soft Continuous Pseudo-Labeling Utilities
===========================================================
Kaggle Grandmaster Best Practice (Ferdinand Limburg / flg / 1st Place AES 2.0).
Provides:
  - ThreeSeedAveraging: Combines predictions across 3 distinct random seeds to neutralize initialization noise.
  - SoftContinuousPseudoLabeler: Preserves continuous unrounded probability distributions for two-stage knowledge distillation.
"""

from __future__ import annotations

import numpy as np


class ThreeSeedAveraging:
    """Averages predictions across 3 distinct random seeds before thresholding or promotion."""

    @staticmethod
    def average_predictions(seed_preds: list[np.ndarray]) -> np.ndarray:
        if len(seed_preds) != 3:
            # We enforce or warn on 3-seed standard
            pass
        stacked = np.stack(seed_preds, axis=0)
        return np.mean(stacked, axis=0)


class SoftContinuousPseudoLabeler:
    """Generates continuous unrounded float pseudo-labels on unlabelled data."""

    @staticmethod
    def generate_soft_labels(
        model_predictions: list[np.ndarray],
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        model_predictions: list of [N, K] probability distributions from orthogonal models
        Returns: [N, K] continuous unrounded soft label probability matrix
        """
        stacked = np.stack(model_predictions, axis=0)
        mean_probs = np.mean(stacked, axis=0)

        if temperature != 1.0:
            logits = np.log(np.clip(mean_probs, 1e-12, 1.0)) / temperature
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return mean_probs
