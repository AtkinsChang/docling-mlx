# Adapted from mlx-vlm (mlx_vlm/models/rt_detr_v2).
# SPDX-License-Identifier: MIT
"""Exact numerical primitives shared by the native detector decoders.

The implementation has the same semantics as ``torch.nn.functional.grid_sample``
for its only supported contract: channel-last FP32 tensors, bilinear sampling,
zero padding, and ``align_corners=False``.  It is intentionally not a general
image-resampling API.

The readable composed implementation is retained as a differential oracle for
the Metal kernel.  The production kernel checks every neighbour is in bounds
*before* calculating or dereferencing its source address.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def inverse_sigmoid(x: mx.array, eps: float = 1e-5) -> mx.array:
    """Return the stable logit transform used for detector reference boxes."""
    if not isinstance(x, mx.array):
        raise TypeError("x must be an MLX array")
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be greater than zero and less than 0.5")
    clipped = mx.clip(x, 0.0, 1.0)
    numerator = mx.clip(clipped, eps, 1.0)
    denominator = mx.clip(1.0 - clipped, eps, 1.0)
    return mx.log(numerator / denominator)


def _validate_spatial_shapes(
    spatial_shapes: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    return tuple((int(height), int(width)) for height, width in spatial_shapes)


def generate_anchors(
    spatial_shapes: Sequence[tuple[int, int]],
    grid_size: float = 0.05,
    dtype: mx.Dtype = mx.float32,
) -> tuple[mx.array, mx.array]:
    """Generate encoder-query anchor logits and their valid mask."""
    shapes = _validate_spatial_shapes(spatial_shapes)

    anchors_per_level = []
    for level, (height, width) in enumerate(shapes):
        grid_y, grid_x = mx.meshgrid(
            mx.arange(height, dtype=dtype), mx.arange(width, dtype=dtype), indexing="ij"
        )
        grid_xy = mx.stack([grid_x, grid_y], axis=-1)[None, ...] + 0.5
        grid_xy = grid_xy / mx.array([width, height], dtype=dtype)[None, None, None, :]
        box_size = mx.ones_like(grid_xy) * grid_size * (2.0**level)
        anchors_per_level.append(
            mx.concatenate([grid_xy, box_size], axis=-1).reshape(1, height * width, 4)
        )

    anchors = mx.concatenate(anchors_per_level, axis=1)
    eps = 1e-2
    valid_mask = ((anchors > eps) & (anchors < 1.0 - eps)).all(axis=-1, keepdims=True)
    anchor_logits = mx.log(anchors / (1.0 - anchors))
    invalid_logit = mx.array(mx.finfo(dtype).max, dtype=dtype)
    return mx.where(valid_mask, anchor_logits, invalid_logit), valid_mask


def select_encoder_queries(
    memory: mx.array,
    scores: mx.array,
    box_logits: mx.array,
    num_queries: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Select detector proposals and gather their decoder inputs."""
    scores_max = scores.max(axis=-1)
    indices = mx.argpartition(-scores_max, num_queries - 1, axis=1)[:, :num_queries]
    selected_scores = mx.take_along_axis(scores_max, indices, axis=1)
    indices = mx.take_along_axis(indices, mx.argsort(-selected_scores, axis=1), axis=1)

    def gather(values: mx.array) -> mx.array:
        expanded = mx.broadcast_to(indices[:, :, None], (*indices.shape, values.shape[-1]))
        return mx.take_along_axis(values, expanded, axis=1)

    reference_logits = gather(box_logits)
    return (
        gather(scores),
        mx.sigmoid(reference_logits),
        mx.stop_gradient(reference_logits),
        mx.stop_gradient(gather(memory)),
    )


class MLP(nn.Module):
    """Linear stack with ReLU between layers and checkpoint-compatible keys."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [input_dim, *([hidden_dim] * (num_layers - 1)), output_dim]
        self.layers = [nn.Linear(dims[index], dims[index + 1]) for index in range(num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = nn.relu(x)
        return x


class SelfAttention(nn.Module):
    """Positional q/k self-attention with shared detector state keys."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        use_fast_attention: bool = False,
    ) -> None:
        super().__init__()
        self._use_fast_attention = use_fast_attention
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: mx.array | None = None,
    ) -> mx.array:
        batch, length, hidden_dim = hidden_states.shape
        query_key_input = (
            hidden_states + position_embeddings
            if position_embeddings is not None
            else hidden_states
        )
        query = (
            self.q_proj(query_key_input)
            .reshape(batch, length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        key = (
            self.k_proj(query_key_input)
            .reshape(batch, length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        value = (
            self.v_proj(hidden_states)
            .reshape(batch, length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        if self._use_fast_attention:
            output = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.scale)
        else:
            attention = mx.softmax(
                (query @ key.transpose(0, 1, 3, 2)) * self.scale,
                axis=-1,
            )
            output = attention @ value
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, hidden_dim)
        return self.out_proj(output)


_METAL_SOURCE = """
    const size_t element = thread_position_in_grid.x;
    const size_t batch_size = size_t(grid_shape[0]);
    const size_t output_height = size_t(grid_shape[1]);
    const size_t output_width = size_t(grid_shape[2]);
    const size_t channels = size_t(x_shape[3]);
    const size_t output_count = batch_size * output_height * output_width * channels;
    if (element >= output_count) return;

    const int input_height = int(x_shape[1]);
    const int input_width = int(x_shape[2]);
    const size_t channel = element % channels;
    const size_t sample = element / channels;
    const size_t grid_offset = sample * 2;
    const size_t batch = sample / (output_height * output_width);

    // align_corners=False maps normalized coordinates to pixel centres.
    // Keep the scale and shift as separate FP32 operations.  Collapsing them
    // lets Metal reassociate the expression and diverge from Torch/MLX.
    const float x_scaled = (grid[grid_offset] + 1.0f) * float(input_width);
    const float y_scaled = (grid[grid_offset + 1] + 1.0f) * float(input_height);
    const float x_coord = (x_scaled - 1.0f) * 0.5f;
    const float y_coord = (y_scaled - 1.0f) * 0.5f;
    const int x0 = int(floor(x_coord));
    const int y0 = int(floor(y_coord));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const T w00 = T((float(x1) - x_coord) * (float(y1) - y_coord));
    const T w01 = T((x_coord - float(x0)) * (float(y1) - y_coord));
    const T w10 = T((float(x1) - x_coord) * (y_coord - float(y0)));
    const T w11 = T((x_coord - float(x0)) * (y_coord - float(y0)));

    const size_t batch_offset = batch * size_t(input_height) * size_t(input_width) * channels;
    T value00 = T(0);
    T value01 = T(0);
    T value10 = T(0);
    T value11 = T(0);

    // Do not move these reads into conditional expressions: a source address
    // is formed and dereferenced only after its coordinate has been checked.
    if (y0 >= 0 && y0 < input_height && x0 >= 0 && x0 < input_width) {
        const size_t source = batch_offset
            + (size_t(y0) * size_t(input_width) + size_t(x0)) * channels + channel;
        value00 = x[source];
    }
    if (y0 >= 0 && y0 < input_height && x1 >= 0 && x1 < input_width) {
        const size_t source = batch_offset
            + (size_t(y0) * size_t(input_width) + size_t(x1)) * channels + channel;
        value01 = x[source];
    }
    if (y1 >= 0 && y1 < input_height && x0 >= 0 && x0 < input_width) {
        const size_t source = batch_offset
            + (size_t(y1) * size_t(input_width) + size_t(x0)) * channels + channel;
        value10 = x[source];
    }
    if (y1 >= 0 && y1 < input_height && x1 >= 0 && x1 < input_width) {
        const size_t source = batch_offset
            + (size_t(y1) * size_t(input_width) + size_t(x1)) * channels + channel;
        value11 = x[source];
    }

    out[element] = w00 * value00 + w01 * value01 + w10 * value10 + w11 * value11;
"""


def _validate_inputs(x: mx.array, grid: mx.array) -> tuple[int, int, int, int, int, int]:
    if not isinstance(x, mx.array) or not isinstance(grid, mx.array):
        raise TypeError("x and grid must be MLX arrays")
    if x.ndim != 4:
        raise ValueError("x must have shape [B, H, W, C]")
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError("grid must have shape [B, out_h, out_w, 2]")
    batch, height, width, channels = (int(dim) for dim in x.shape)
    grid_batch, output_height, output_width, _ = (int(dim) for dim in grid.shape)
    if grid_batch != batch:
        raise ValueError("x and grid must have the same batch size")
    if height < 1 or width < 1 or channels < 1:
        raise ValueError("x spatial dimensions and channels must be nonempty")
    if x.dtype != mx.float32 or grid.dtype != mx.float32:
        raise TypeError("detector grid sampling requires FP32 x and grid arrays")
    return batch, height, width, channels, output_height, output_width


def grid_sample_bilinear_zeros_align_corners_false_reference(
    x: mx.array, grid: mx.array
) -> mx.array:
    """Return the composed-MLX reference implementation for the narrow contract."""
    batch, height, width, channels, output_height, output_width = _validate_inputs(x, grid)
    if batch == 0 or output_height == 0 or output_width == 0:
        return mx.zeros((batch, output_height, output_width, channels), dtype=mx.float32)

    x_coord = ((grid[..., 0] + 1.0) * width - 1.0) * 0.5
    y_coord = ((grid[..., 1] + 1.0) * height - 1.0) * 0.5
    x0 = mx.floor(x_coord).astype(mx.int32)
    y0 = mx.floor(y_coord).astype(mx.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    w00 = ((x1.astype(mx.float32) - x_coord) * (y1.astype(mx.float32) - y_coord))[..., None]
    w01 = ((x_coord - x0.astype(mx.float32)) * (y1.astype(mx.float32) - y_coord))[..., None]
    w10 = ((x1.astype(mx.float32) - x_coord) * (y_coord - y0.astype(mx.float32)))[..., None]
    w11 = ((x_coord - x0.astype(mx.float32)) * (y_coord - y0.astype(mx.float32)))[..., None]
    flattened = x.reshape(batch, height * width, channels)
    batch_indices = mx.arange(batch)[:, None]

    def gather(y_indices: mx.array, x_indices: mx.array) -> mx.array:
        valid = (y_indices >= 0) & (y_indices < height) & (x_indices >= 0) & (x_indices < width)
        clipped_y = mx.clip(y_indices, 0, height - 1)
        clipped_x = mx.clip(x_indices, 0, width - 1)
        indices = (clipped_y * width + clipped_x).reshape(batch, -1)
        values = flattened[batch_indices, indices].reshape(
            batch, output_height, output_width, channels
        )
        return values * valid[..., None]

    return w00 * gather(y0, x0) + w01 * gather(y0, x1) + w10 * gather(y1, x0) + w11 * gather(y1, x1)


@lru_cache(maxsize=1)
def _grid_sample_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="docling_rt_detr_v2_grid_sample_bilinear_zeros_ac_false",
        input_names=["x", "grid"],
        output_names=["out"],
        source=_METAL_SOURCE,
        compile_options={"math_mode": "safe"},
    )


def grid_sample_bilinear_zeros_align_corners_false(x: mx.array, grid: mx.array) -> mx.array:
    """Sample ``x[B,H,W,C]`` at normalized ``grid[B,out_h,out_w,2]`` coordinates.

    Only the detectors' FP32 bilinear/zeros/``align_corners=False`` operation is
    supported.  The result has shape ``[B, out_h, out_w, C]``.
    """
    batch, _, _, channels, output_height, output_width = _validate_inputs(x, grid)
    output_shape = (batch, output_height, output_width, channels)
    if batch == 0 or output_height == 0 or output_width == 0:
        return mx.zeros(output_shape, dtype=mx.float32)
    kernel = _grid_sample_kernel()
    return kernel(
        inputs=[mx.contiguous(x), mx.contiguous(grid)],
        template=[("T", mx.float32)],
        output_shapes=[output_shape],
        output_dtypes=[mx.float32],
        grid=(batch * output_height * output_width * channels, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]
