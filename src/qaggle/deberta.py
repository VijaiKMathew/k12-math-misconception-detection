"""Pure schema and feature helpers for the P3 DeBERTa step cross-encoder."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None  # type: ignore[assignment]  # stdlib fallback when pandas unavailable

if TYPE_CHECKING:
    import pandas as _pd

SECONDARY_TRAP_CODES = {"CAL-01", "ANS-01", "NOT-01", "NOT-02", "INT-04"}

REQUIRED_COLUMNS = ["trace_id", "fold", "cand_code", "query", "candidate_text", "label"]


def is_secondary_trap(cand_code: str) -> bool:
    """Return True when a candidate is a known root-vs-secondary trap code."""
    return str(cand_code) in SECONDARY_TRAP_CODES


def build_trace_grouped_indices(
    trace_ids: Sequence[str],
    seed: int = 42,
) -> list[int]:
    """Order row indices so all candidates of each ``trace_id`` stay contiguous (ARM-05).

    Trace groups are shuffled deterministically using ``seed``, while intra-trace
    candidate ordering is preserved.
    """
    groups: dict[str, list[int]] = {}
    for idx, raw_tid in enumerate(trace_ids):
        tid = str(raw_tid)
        groups.setdefault(tid, []).append(idx)
    ordered_tids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(ordered_tids)
    indices: list[int] = []
    for tid in ordered_tids:
        indices.extend(groups[tid])
    return indices


def validate_pairs(
    df: _pd.DataFrame,
    expected_folds: int | None = None,
    allow_zero_positive: bool = False,
) -> list[str]:
    """Return human-readable schema and label problems for a cross-encoder pair table."""
    assert pd is not None, "pandas is required for validate_pairs"
    problems: list[str] = []

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        return [f"missing required columns: {', '.join(missing)}"]

    if len(df) == 0:
        return ["pair table is empty"]

    folds = pd.to_numeric(df["fold"], errors="coerce")
    if folds.isna().any():
        problems.append("fold column contains non-numeric values")
    else:
        unique_folds = sorted(int(value) for value in folds.unique())
        if any(value < 0 for value in unique_folds):
            problems.append("fold values must be non-negative")
        if expected_folds is not None and len(unique_folds) != expected_folds:
            problems.append(
                f"expected {expected_folds} folds, found {len(unique_folds)}: {unique_folds}"
            )

    labels = pd.to_numeric(df["label"], errors="coerce")
    if labels.isna().any():
        problems.append("label column contains non-numeric values")
    else:
        invalid_labels = sorted(set(labels.unique()) - {0, 1})
        if invalid_labels:
            problems.append(f"labels must be in {{0, 1}}, found {invalid_labels}")

    duplicate_mask = df.duplicated(subset=["trace_id", "cand_code"], keep=False)
    if duplicate_mask.any():
        first_duplicate = df.loc[duplicate_mask, ["trace_id", "cand_code"]].iloc[0]
        problems.append(
            "duplicate (trace_id, cand_code) rows: "
            f"{int(duplicate_mask.sum())} "
            f"(e.g. trace_id={first_duplicate['trace_id']!r}, "
            f"cand_code={first_duplicate['cand_code']!r})"
        )

    empty_query = df["query"].fillna("").astype(str).str.strip().eq("")
    if empty_query.any():
        problems.append(
            f"empty query rows: {int(empty_query.sum())} "
            f"(e.g. trace_id={df.loc[empty_query, 'trace_id'].iloc[0]!r})"
        )

    empty_candidate = df["candidate_text"].fillna("").astype(str).str.strip().eq("")
    if empty_candidate.any():
        problems.append(
            f"empty candidate_text rows: {int(empty_candidate.sum())} "
            f"(e.g. trace_id={df.loc[empty_candidate, 'trace_id'].iloc[0]!r})"
        )

    if not labels.isna().any():
        positives = labels.groupby(df["trace_id"]).sum()
        for trace_id, count in positives.items():
            count = int(count)
            if count == 1:
                continue
            if count == 0 and allow_zero_positive:
                continue
            problems.append(
                f"trace_id={trace_id!r} has {count} positives (expected exactly 1)"
            )

    return problems


def zero_positive_trace_ids(df: _pd.DataFrame) -> set[str]:
    """Return trace IDs whose pair rows contain no positive label."""
    assert pd is not None, "pandas is required for zero_positive_trace_ids"
    if "trace_id" not in df.columns or "label" not in df.columns:
        return set()

    labels = pd.to_numeric(df["label"], errors="coerce")
    all_traces = {str(value) for value in df["trace_id"].dropna().tolist()}
    positive_traces = {
        str(value) for value in df.loc[labels == 1, "trace_id"].dropna().tolist()
    }
    return all_traces - positive_traces


def training_arguments_kwargs(
    output_dir: str,
    *,
    learning_rate: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    num_train_epochs: float,
    weight_decay: float = 0.01,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    save_steps: int = 500,
    load_best_model_at_end: bool = True,
    fp16: bool = True,
    report_to: str = "none",
    seed: int = 42,
    save_total_limit: int = 1,
    bf16: bool = False,
    dataloader_pin_memory: bool = True,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    optim: str | None = None,
    metric_for_best_model: str | None = None,
    greater_is_better: bool | None = None,
    max_grad_norm: float | None = None,
    warmup_ratio: float | None = None,
    lr_scheduler_type: str | None = None,
) -> dict[str, object]:
    """Return TrainingArguments kwargs that preserve the custom trap column and enable high-speed GPU execution."""
    res: dict[str, object] = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "num_train_epochs": num_train_epochs,
        "weight_decay": weight_decay,
        "eval_strategy": eval_strategy,
        "save_strategy": save_strategy,
        "load_best_model_at_end": load_best_model_at_end,
        "fp16": fp16 and not bf16,
        "bf16": bf16,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "dataloader_pin_memory": dataloader_pin_memory,
        "report_to": report_to,
        "seed": seed,
        "save_total_limit": save_total_limit,
        # The HF Trainer defaults remove_unused_columns=True, which drops
        # is_secondary_trap and trace_id_idx before compute_loss can apply the margin penalty.
        "remove_unused_columns": False,
    }
    if save_strategy == "steps":
        res["save_steps"] = save_steps
    if optim is not None:
        res["optim"] = optim
    if metric_for_best_model is not None:
        res["metric_for_best_model"] = metric_for_best_model
    if greater_is_better is not None:
        res["greater_is_better"] = greater_is_better
    if max_grad_norm is not None:
        res["max_grad_norm"] = max_grad_norm
    if warmup_ratio is not None:
        res["warmup_ratio"] = warmup_ratio
    if lr_scheduler_type is not None:
        res["lr_scheduler_type"] = lr_scheduler_type
    return res


def to_trainer_records(
    df: _pd.DataFrame,
    include_prior_columns: bool = False,
) -> _pd.DataFrame:
    """Map parquet pair columns to the Trainer's expected labels, trap flag, and optional prior logits."""
    assert pd is not None, "pandas is required for to_trainer_records"
    cols: dict[str, object] = {
        "trace_id": df["trace_id"].to_numpy(),
        "fold": df["fold"].to_numpy(),
        "labels": df["label"].astype(int).to_numpy(),
        "is_secondary_trap": [
            int(is_secondary_trap(code)) for code in df["cand_code"].tolist()
        ],
        "query": df["query"].to_numpy(),
        "candidate_text": df["candidate_text"].to_numpy(),
    }
    if include_prior_columns or "prior_logit" in df.columns:
        cols["is_none"] = [
            int(str(code) == "NONE") for code in df["cand_code"].tolist()
        ]
        cols["prior_logit"] = (
            df["prior_logit"].astype(float).to_numpy()
            if "prior_logit" in df.columns
            else [0.0] * len(df)
        )
    return pd.DataFrame(cols)


def assign_oof_logits(
    df: _pd.DataFrame,
    fold_predictions: Mapping[int, list[float]],
) -> list[float]:
    """Assign each fold's predictions back to the input row order."""
    assert pd is not None, "pandas is required for assign_oof_logits"
    row_folds = [int(fold) for fold in df["fold"].tolist()]
    logits = [float("nan")] * len(row_folds)

    for fold, predictions in fold_predictions.items():
        fold = int(fold)
        positions = [index for index, row_fold in enumerate(row_folds) if row_fold == fold]
        if len(positions) != len(predictions):
            raise ValueError(
                f"fold {fold} has {len(positions)} rows but "
                f"{len(predictions)} predictions were supplied"
            )
        for position, prediction in zip(positions, predictions):
            logits[position] = float(prediction)

    missing_folds = sorted({fold for fold, value in zip(row_folds, logits) if pd.isna(value)})
    if missing_folds:
        raise ValueError(f"missing predictions for folds: {missing_folds}")

    return logits
