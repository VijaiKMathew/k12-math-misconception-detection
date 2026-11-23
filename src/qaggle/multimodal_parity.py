"""Directive 3: Multimodal Vision vs. OCR Parity Proof & Intra-Trace ListNet Loss.

Pure-stdlib implementation (compatible with `/usr/bin/python3` and optional NumPy/PyTorch):
  1. 3-Channel Geometric Ink Feature Extractor (`extract_3channel_geometric_crop`,
     `expand_box_for_radicals_and_scripts`) preserving spatial layout, strikethrough
     energy (Channel 1), and baseline-relative superscript/subscript offset (Channel 2).
  2. Intra-Trace ListNet Cross-Entropy + ARM-05 Sibling & Root-vs-Secondary Margin
     Ranking Loss + ARM-02 Max-Preserving Orthogonal Projection (`intra_trace_listnet_and_margin_loss`,
     `project_probabilities_orthogonal_arm02`).
  3. Canonical Two-Stage Pretrain -> Fine-Tune Hyperparameter Schedule (`TWO_STAGE_VLM_SCHEDULE`)
     for Qwen2.5-VL-7B-Instruct and Gemini 2.5 Flash over 18,000 consensus pseudo-labelled
     traces + 1,500 gold 5-fold traces.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

Grid2D = list[list[float]]
Tensor3D = list[Grid2D]


@dataclass(frozen=True)
class GeometricCropResult:
    """3-channel geometric ink representation and layout descriptors for a handwriting crop."""

    crop_3ch: Tensor3D  # Shape (3, H', W'): [C_ink, C_strike, C_base]
    shape: tuple[int, int, int]
    strike_ratio: float
    grid_thw: tuple[int, int, int]
    spatial_box: tuple[float, float, float, float, float, float]


TWO_STAGE_VLM_SCHEDULE: dict[str, dict[str, Any]] = {
    "stage1_consensus_pretrain": {
        "dataset_rows": 18000,
        "confidence_threshold": 0.85,
        "backbone": "Qwen/Qwen2.5-VL-7B-Instruct",
        "precision": "bf16",
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1280 * 28 * 28,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "use_dora": True,
        "micro_batch_traces": 2,
        "candidates_per_trace": 24,
        "grad_accum_steps": 16,
        "epochs": 2,
        "lr_llm_lora": 1.5e-4,
        "lr_vision_merger": 3.0e-5,
        "lr_fusion_head": 3.0e-4,
        "warmup_ratio": 0.08,
    },
    "stage2_gold_listnet_finetune": {
        "dataset_rows": 1500,
        "folds": 5,
        "backbone": "Qwen/Qwen2.5-VL-7B-Instruct",
        "precision": "bf16",
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.08,
        "micro_batch_traces": 1,
        "candidates_per_trace": 24,
        "grad_accum_steps": 16,
        "epochs": 4,
        "lr_llm_lora": 4.0e-5,
        "lr_vision_merger": 8.0e-6,
        "lr_fusion_head": 1.0e-4,
        "warmup_ratio": 0.10,
        "weight_decay": 0.02,
        "tau_listnet": 0.70,
        "gamma_sec": 0.50,
        "gamma_sib": 0.35,
        "lambda_sec": 0.60,
        "lambda_sib": 0.45,
        "lambda_orth": 0.30,
    },
}


def expand_box_for_radicals_and_scripts(
    box: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Asymmetric box expansion preserving superscripts (+35% top), subscripts (+25% bottom), and sqrt bars."""
    x1, y1, x2, y2 = box
    h = max(4.0, y2 - y1)
    nx1 = max(0, math.floor(x1 - 0.20 * h))
    ny1 = max(0, math.floor(y1 - 0.35 * h))
    nx2 = min(img_w, math.ceil(x2 + 0.15 * h))
    ny2 = min(img_h, math.ceil(y2 + 0.25 * h))
    return nx1, ny1, nx2, ny2


def extract_3channel_geometric_crop(
    gray_image: Sequence[Sequence[float]],
    box: tuple[float, float, float, float] | None = None,
    *,
    patch_multiple: int = 28,
    target_line_height: int = 84,
    max_line_width: int = 784,
    step_idx: int = 0,
    total_steps: int = 1,
    margin_x_threshold: float = 0.68,
) -> GeometricCropResult:
    """Compute the 3-channel geometric ink tensor [C_ink, C_strike, C_base] from a 2D grayscale grid."""
    img_h = len(gray_image)
    img_w = len(gray_image[0]) if img_h > 0 else 0
    if img_h == 0 or img_w == 0:
        raise ValueError("gray_image must be non-empty 2D grid")

    if box is None:
        box = (0.0, 0.0, float(img_w), float(img_h))

    ex1, ey1, ex2, ey2 = expand_box_for_radicals_and_scripts(box, img_w, img_h)
    sub_rows = [
        [float(gray_image[r][c]) for c in range(ex1, ex2)]
        for r in range(ey1, ey2)
    ]
    h_sub = len(sub_rows)
    w_sub = len(sub_rows[0]) if h_sub > 0 else 0
    if h_sub == 0 or w_sub == 0:
        sub_rows = [[255.0] * patch_multiple for _ in range(target_line_height)]
        h_sub, w_sub = target_line_height, patch_multiple

    max_val = max(max(row) for row in sub_rows)
    scale_div = 255.0 if max_val > 1.5 else 1.0

    scale = target_line_height / float(max(h_sub, 1))
    new_h = target_line_height
    raw_w = min(max_line_width, max(patch_multiple, round(w_sub * scale)))
    pad_w = math.ceil(raw_w / float(patch_multiple)) * patch_multiple

    resized: Grid2D = []
    for r in range(new_h):
        src_r = min(h_sub - 1, round(r * (h_sub - 1) / max(new_h - 1, 1)))
        row_vals: list[float] = []
        for c in range(raw_w):
            src_c = min(w_sub - 1, round(c * (w_sub - 1) / max(raw_w - 1, 1)))
            row_vals.append(sub_rows[src_r][src_c] / scale_div)
        resized.append(row_vals)

    # Column-wise 85th percentile background estimation
    bg_cols: list[float] = []
    for c in range(raw_w):
        col_sorted = sorted(resized[r][c] for r in range(new_h))
        idx85 = min(new_h - 1, int(0.85 * (new_h - 1)))
        bg_cols.append(max(0.15, col_sorted[idx85]))

    ink: Grid2D = []
    for r in range(new_h):
        row_ink = [
            max(0.0, min(1.0, (bg_cols[c] - resized[r][c]) / bg_cols[c]))
            for c in range(raw_w)
        ]
        if pad_w > raw_w:
            row_ink.extend([0.0] * (pad_w - raw_w))
        ink.append(row_ink)

    # Channel 1: Horizontal strikethrough energy
    half_k = 7
    strike_map: Grid2D = [[0.0] * pad_w for _ in range(new_h)]
    active_ink_count = 0.0
    strike_pixel_count = 0.0

    for r in range(new_h):
        r_up = max(0, r - 3)
        r_dn = min(new_h - 1, r + 3)
        for c in range(pad_w):
            val = ink[r][c]
            if val > 0.20:
                active_ink_count += 1.0
            if val > 0.15:
                c0 = max(0, c - half_k)
                c1 = min(pad_w, c + half_k + 1)
                h_mean = sum(ink[r][ c0:c1 ]) / float(c1 - c0)
                v_sur = 0.5 * (ink[r_up][c] + ink[r_dn][c])
                s_val = max(0.0, min(1.0, (h_mean - 0.35 * v_sur) * 2.2))
                strike_map[r][c] = s_val
                if s_val > 0.35:
                    strike_pixel_count += 1.0

    # Channel 2: Baseline-relative vertical offset C_base in [-1, 1]
    y_base_cols: list[float] = []
    sigma_cols: list[float] = []
    for c in range(pad_w):
        mass = sum(ink[r][c] for r in range(new_h)) + 1e-4
        yb = sum(ink[r][c] * float(r) for r in range(new_h)) / mass
        y_base_cols.append(yb)
        var = sum(ink[r][c] * ((float(r) - yb) ** 2) for r in range(new_h)) / mass
        sigma_cols.append(max(6.0, min(float(new_h) * 0.35, math.sqrt(var))))

    rel_y: Grid2D = [[0.0] * pad_w for _ in range(new_h)]
    for r in range(new_h):
        for c in range(pad_w):
            if ink[r][c] > 0.12:
                rel_y[r][c] = math.tanh(
                    (y_base_cols[c] - float(r)) / (1.5 * sigma_cols[c])
                )

    strike_ratio = max(
        0.0, min(1.0, strike_pixel_count / (active_ink_count + 1.0))
    )

    xc = 0.5 * (box[0] + box[2]) / max(img_w, 1)
    yc = 0.5 * (box[1] + box[3]) / max(img_h, 1)
    bw = (box[2] - box[0]) / max(img_w, 1)
    bh = (box[3] - box[1]) / max(img_h, 1)
    is_margin = 1.0 if xc > margin_x_threshold else 0.0
    step_pos = float(step_idx) / float(max(total_steps - 1, 1))

    crop_3ch: Tensor3D = [ink, strike_map, rel_y]
    return GeometricCropResult(
        crop_3ch=crop_3ch,
        shape=(3, new_h, pad_w),
        strike_ratio=strike_ratio,
        grid_thw=(1, new_h // 14, pad_w // 14),
        spatial_box=(xc, yc, bw, bh, step_pos, is_margin),
    )


def project_probabilities_orthogonal_arm02(
    p_base: Sequence[Sequence[float]],
    s_reranker: Sequence[Sequence[float]],
    *,
    alpha: float = 0.40,
    tau_proj: float = 0.85,
    none_index: int = 0,
    clean_margin_eps: float = 1e-5,
    eps: float = 1e-8,
) -> list[list[float]]:
    """ARM-02 Max-Preserving Orthogonal Projection over batched probability matrices (B, K)."""
    bsz = len(p_base)
    if bsz != len(s_reranker) or bsz == 0:
        raise ValueError("p_base and s_reranker must have identical non-empty batch size")
    k_cands = len(p_base[0])

    out: list[list[float]] = []
    for b in range(bsz):
        pb_row = [float(x) for x in p_base[b]]
        sr_row = [float(x) for x in s_reranker[b]]
        if len(pb_row) != k_cands or len(sr_row) != k_cands:
            raise ValueError("Row length mismatch in p_base / s_reranker")

        p_none = pb_row[none_index]
        err_indices = [idx for idx in range(k_cands) if idx != none_index]
        p_err_base = [pb_row[idx] for idx in err_indices]
        s_err = [sr_row[idx] for idx in err_indices]

        mu_i = sum(s_err) / max(len(s_err), 1)
        var_i = sum((x - mu_i) ** 2 for x in s_err) / max(len(s_err), 1)
        sigma_i = math.sqrt(var_i) + eps
        z_tilde = [(x - mu_i) / sigma_i for x in s_err]

        scaled = [z / tau_proj for z in z_tilde]
        max_sc = max(scaled) if scaled else 0.0
        exp_z = [math.exp(sc - max_sc) for sc in scaled]
        sum_exp = sum(exp_z) + eps
        q_err_rerank = [ez / sum_exp for ez in exp_z]

        err_mass_base = sum(p_err_base)
        p_err_cond_base = [pe / (err_mass_base + eps) for pe in p_err_base]

        residual_budget = max(0.0, min(1.0, 1.0 - p_none))
        p_err_fused = [
            ((1.0 - alpha) * p_cond + alpha * q_rerank) * residual_budget
            for p_cond, q_rerank in zip(p_err_cond_base, q_err_rerank)
        ]

        base_top1 = max(range(k_cands), key=lambda idx: pb_row[idx])
        max_err_fused = max(p_err_fused) if p_err_fused else 0.0
        cap_threshold = max(0.0, p_none - clean_margin_eps)

        if base_top1 == none_index and max_err_fused >= cap_threshold:
            scale_factor = cap_threshold / (max_err_fused + eps)
            p_err_safe = [val * scale_factor for val in p_err_fused]
        else:
            p_err_safe = p_err_fused

        proj_row = [0.0] * k_cands
        proj_row[none_index] = p_none
        for idx, val in zip(err_indices, p_err_safe):
            proj_row[idx] = val
        out.append(proj_row)

    return out


def intra_trace_listnet_and_margin_loss(
    logits: Sequence[Sequence[float]],
    root_idx: Sequence[int],
    secondary_mask: Sequence[Sequence[bool]],
    sibling_mask: Sequence[Sequence[bool]],
    *,
    p_base: Sequence[Sequence[float]] | None = None,
    tau_listnet: float = 1.0,
    gamma_sec: float = 0.50,
    gamma_sib: float = 0.35,
    gamma_neg: float = 0.25,
    lambda_sec: float = 0.60,
    lambda_sib: float = 0.45,
    lambda_neg: float = 0.20,
    lambda_orth: float = 0.30,
    secondary_soft_mass: float = 0.08,
    label_smoothing: float = 0.02,
    none_index: int = 0,
) -> tuple[float, dict[str, float]]:
    """Compute Intra-Trace ListNet + ARM-05 Sibling/Secondary Margin + ARM-02 Orthogonal Loss."""
    bsz = len(logits)
    if bsz == 0:
        return 0.0, {}
    k_cands = len(logits[0])

    listnet_losses: list[float] = []
    root_sec_losses: list[float] = []
    sib_losses: list[float] = []
    hard_neg_losses: list[float] = []

    for b in range(bsz):
        row_logits = [float(x) for x in logits[b]]
        r_idx = int(root_idx[b])
        sec_m = [bool(secondary_mask[b][j]) and (j != r_idx) for j in range(k_cands)]
        sib_m = [
            bool(sibling_mask[b][j]) and (j != r_idx) and not sec_m[j]
            for j in range(k_cands)
        ]

        scaled = [x / tau_listnet for x in row_logits]
        m_sc = max(scaled)
        log_denom = m_sc + math.log(sum(math.exp(x - m_sc) for x in scaled))
        log_probs = [x - log_denom for x in scaled]

        sec_cnt = sum(1.0 for flag in sec_m if flag)
        has_sec = 1.0 if sec_cnt > 0 else 0.0

        target_dist = [0.0] * k_cands
        target_dist[r_idx] = 1.0 - has_sec * secondary_soft_mass
        for j in range(k_cands):
            if sec_m[j]:
                target_dist[j] += secondary_soft_mass / (sec_cnt + 1e-8)
            target_dist[j] = (1.0 - label_smoothing) * target_dist[j] + (
                label_smoothing / float(k_cands)
            )

        listnet_losses.append(
            -sum(t * lp for t, lp in zip(target_dist, log_probs))
        )

        s_root = row_logits[r_idx]

        # ARM-05 Root-vs-Secondary Margin within trace b
        sec_viol = sum(
            max(0.0, gamma_sec - (s_root - row_logits[j])) ** 2
            for j in range(k_cands)
            if sec_m[j]
        )
        root_sec_losses.append(sec_viol / max(sec_cnt, 1.0))

        # ARM-05 Root-vs-Sibling Margin within trace b
        sib_cnt = sum(1.0 for flag in sib_m if flag)
        sib_viol = sum(
            max(0.0, gamma_sib - (s_root - row_logits[j]))
            for j in range(k_cands)
            if sib_m[j]
        )
        sib_losses.append(sib_viol / max(sib_cnt, 1.0))

        # Hardest non-sibling negative margin within trace b
        other_vals = [
            row_logits[j]
            for j in range(k_cands)
            if j != r_idx and not sec_m[j] and not sib_m[j]
        ]
        if other_vals:
            hardest_neg = max(other_vals)
            hard_neg_losses.append(max(0.0, gamma_neg - (s_root - hardest_neg)))
        else:
            hard_neg_losses.append(0.0)

    loss_listnet = sum(listnet_losses) / bsz
    loss_root_sec = sum(root_sec_losses) / bsz
    loss_sib = sum(sib_losses) / bsz
    loss_hard_neg = sum(hard_neg_losses) / bsz

    loss_orth = 0.0
    if p_base is not None:
        p_proj = project_probabilities_orthogonal_arm02(
            p_base, logits, none_index=none_index
        )
        unclean_nlls = [
            -math.log(max(1e-8, min(1.0, p_proj[b][int(root_idx[b])])))
            for b in range(bsz)
            if int(root_idx[b]) != none_index
        ]
        if unclean_nlls:
            loss_orth = sum(unclean_nlls) / len(unclean_nlls)

    total_loss = (
        loss_listnet
        + lambda_sec * loss_root_sec
        + lambda_sib * loss_sib
        + lambda_neg * loss_hard_neg
        + lambda_orth * loss_orth
    )
    metrics = {
        "loss_total": total_loss,
        "loss_listnet": loss_listnet,
        "loss_root_sec": loss_root_sec,
        "loss_sibling": loss_sib,
        "loss_hard_neg": loss_hard_neg,
        "loss_orth_arm02": loss_orth,
    }
    return total_loss, metrics
