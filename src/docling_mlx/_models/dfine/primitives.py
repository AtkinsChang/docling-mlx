# Implemented after Hugging Face Transformers (transformers/models/d_fine); module
# structure, parameter names, and forward-pass order follow it so published
# checkpoints load unchanged.
# SPDX-License-Identifier: Apache-2.0
"""D-FINE numerical primitives and exact neutral detector re-exports."""

from __future__ import annotations

import mlx.core as mx

from docling_mlx._models.detector_primitives import (
    MLP,
    SelfAttention,
    generate_anchors,
    grid_sample_bilinear_zeros_align_corners_false,
    inverse_sigmoid,
)

__all__ = [
    "distance2bbox",
    "generate_anchors",
    "grid_sample_bilinear_zeros_align_corners_false",
    "inverse_sigmoid",
    "MLP",
    "SelfAttention",
    "weighting_function",
]


def weighting_function(max_num_bins: int, up: mx.array, reg_scale: mx.array) -> mx.array:
    upper_bound = mx.abs(up[0]) * mx.abs(reg_scale)
    outer_bound = upper_bound * 2.0
    step = (upper_bound + 1.0) ** (2.0 / (max_num_bins - 2))
    left = [-(step**index) + 1.0 for index in range(max_num_bins // 2 - 1, 0, -1)]
    right = [step**index - 1.0 for index in range(1, max_num_bins // 2)]
    return mx.concatenate(
        [-outer_bound, *left, mx.zeros_like(up), *right, outer_bound],
        axis=0,
    )


def distance2bbox(points: mx.array, distance: mx.array, reg_scale: mx.array) -> mx.array:
    scale = mx.abs(reg_scale)
    top_left_x = points[..., 0] - (0.5 * scale + distance[..., 0]) * (points[..., 2] / scale)
    top_left_y = points[..., 1] - (0.5 * scale + distance[..., 1]) * (points[..., 3] / scale)
    bottom_right_x = points[..., 0] + (0.5 * scale + distance[..., 2]) * (points[..., 2] / scale)
    bottom_right_y = points[..., 1] + (0.5 * scale + distance[..., 3]) * (points[..., 3] / scale)
    top_left = mx.stack([top_left_x, top_left_y], axis=-1)
    bottom_right = mx.stack([bottom_right_x, bottom_right_y], axis=-1)
    return mx.concatenate([(top_left + bottom_right) * 0.5, bottom_right - top_left], axis=-1)
