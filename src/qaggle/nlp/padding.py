"""
Dynamic Padding and Batch Collator Utilities
============================================
Kaggle Grandmaster Best Practice (Dieter / Chris Deotte / NVIDIA Team).
Dynamically pads each batch to the longest sequence in that batch (instead of a global max_len),
padded to a multiple of 8 for optimal tensor core efficiency on Ampere/Ada/Hopper GPUs and Apple Silicon.
"""

from __future__ import annotations

from typing import Any

import torch


class DynamicPaddingCollator:
    """Collates a list of tokenized dicts, dynamically padding to batch max length multiple of 8."""

    def __init__(self, pad_token_id: int = 0, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            return {}

        raw_lens = [len(item["input_ids"]) for item in batch]
        max_len = max(raw_lens)

        if self.pad_to_multiple_of > 1:
            remainder = max_len % self.pad_to_multiple_of
            if remainder != 0:
                max_len += self.pad_to_multiple_of - remainder

        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []

        for item in batch:
            ids = list(item["input_ids"])
            pad_len = max_len - len(ids)
            padded_ids = ids + [self.pad_token_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len
            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(mask)

        output: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }

        # Optional pass-through of labels, token_type_ids, or trace_ids if present
        for key in ["labels", "token_type_ids", "trace_id"]:
            if key in batch[0] and isinstance(batch[0][key], (int, float, list)):
                output[key] = torch.tensor([item[key] for item in batch])

        return output
