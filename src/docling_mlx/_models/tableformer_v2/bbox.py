# Implemented after docling-ibm-models (docling_ibm_models/tableformer_v2); module
# structure, parameter names, and forward-pass order follow it so the published
# checkpoint loads unchanged.
# SPDX-License-Identifier: Apache-2.0
"""Attention-based cell bounding-box head for TableFormerV2.

The upstream checkpoint uses PyTorch ``nn.MultiheadAttention`` modules.  Its
query/key/value weights are therefore kept as the fused
``in_proj_{weight,bias}`` tensors rather than split into three native linear
modules.  This keeps strict artifact conversion one-to-one with the source
state dictionary.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def cxcywh_to_xyxy(boxes: mx.array) -> mx.array:
    """Convert normalized center-width-height boxes to clipped xyxy boxes."""
    if boxes.ndim < 1 or boxes.shape[-1] != 4:
        raise ValueError("Expected bounding boxes with a final dimension of four")
    cx, cy, width, height = mx.split(boxes, 4, axis=-1)
    return mx.concatenate(
        [
            mx.clip(cx - 0.5 * width, 0.0, 1.0),
            mx.clip(cy - 0.5 * height, 0.0, 1.0),
            mx.clip(cx + 0.5 * width, 0.0, 1.0),
            mx.clip(cy + 0.5 * height, 0.0, 1.0),
        ],
        axis=-1,
    )


class _FusedMultiheadAttention(nn.Module):
    """FP32 batch-first MHA with PyTorch's fused in-projection state layout."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.in_proj_weight = mx.zeros((3 * embed_dim, embed_dim), dtype=mx.float32)
        self.in_proj_bias = mx.zeros((3 * embed_dim,), dtype=mx.float32)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _project(self, x: mx.array, offset: int) -> mx.array:
        start = offset * self.embed_dim
        stop = start + self.embed_dim
        return x @ self.in_proj_weight[start:stop].T + self.in_proj_bias[start:stop]

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        *,
        attn_mask: mx.array | None = None,
    ) -> mx.array:
        """Apply attention; a true boolean mask entry excludes that key."""
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("Multihead attention inputs must have shape [batch, length, embed]")
        if (
            query.shape[0] != key.shape[0]
            or key.shape != value.shape
            or query.shape[-1] != self.embed_dim
            or key.shape[-1] != self.embed_dim
        ):
            raise ValueError("Multihead attention input shapes are incompatible")
        if query.dtype != mx.float32 or key.dtype != mx.float32 or value.dtype != mx.float32:
            raise TypeError("TableFormerV2 bbox attention requires float32 inputs")

        batch_size, target_length, _ = query.shape
        source_length = key.shape[1]
        query_projected = self._project(query, 0)
        key_projected = self._project(key, 1)
        value_projected = self._project(value, 2)
        query_heads = query_projected.reshape(
            batch_size, target_length, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_heads = key_projected.reshape(
            batch_size, source_length, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_heads = value_projected.reshape(
            batch_size, source_length, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        scores = (query_heads @ key_heads.transpose(0, 1, 3, 2)) * self.scale

        if attn_mask is not None:
            if attn_mask.dtype != mx.bool_ or tuple(attn_mask.shape) != (
                target_length,
                source_length,
            ):
                raise ValueError("Attention mask must be boolean with shape [target, source]")
            excluded = mx.broadcast_to(
                attn_mask[None, None, :, :],
                (batch_size, self.num_heads, target_length, source_length),
            )
            scores = mx.where(excluded, mx.array(-mx.inf, dtype=scores.dtype), scores)

        attention = mx.softmax(scores, axis=-1)
        attended = attention @ value_heads
        attended = attended.transpose(0, 2, 1, 3).reshape(batch_size, target_length, self.embed_dim)
        return self.out_proj(attended)


class BboxDecoderLayer(nn.Module):
    """One self-attention, cross-attention, and GELU feed-forward bbox block."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.self_attn = _FusedMultiheadAttention(embed_dim, num_heads)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = _FusedMultiheadAttention(embed_dim, num_heads)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        self.ffn = _BboxFeedForward(embed_dim, ff_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def __call__(self, x: mx.array, memory: mx.array, batch_mask: mx.array) -> mx.array:
        if x.ndim != 2 or memory.ndim != 3:
            raise ValueError("Bbox decoder expects cells [N, D] and memory [N, S, D]")
        if x.shape[0] != memory.shape[0] or x.shape[-1] != memory.shape[-1]:
            raise ValueError("Bbox decoder cell and memory shapes are incompatible")
        if batch_mask.dtype != mx.bool_ or tuple(batch_mask.shape) != (x.shape[0], x.shape[0]):
            raise ValueError("Bbox decoder batch mask must be boolean with shape [N, N]")

        self_attended = self.self_attn(
            x[None, :, :],
            x[None, :, :],
            x[None, :, :],
            attn_mask=~batch_mask,
        )
        x = self.self_attn_norm(x + self_attended[0])
        cross_attended = self.cross_attn(x[:, None, :], memory, memory)
        x = self.cross_attn_norm(x + cross_attended[:, 0, :])
        return self.ffn_norm(x + self.ffn(x))


class _BboxFeedForward(nn.Module):
    """Source-compatible ``Linear -> GELU -> Dropout -> Linear -> Dropout`` FFN."""

    def __init__(self, embed_dim: int, ff_dim: int) -> None:
        super().__init__()
        self.layers = [
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(0.1),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class _BboxMlp(nn.Module):
    """Source-compatible box MLP with GELU activations and inference dropout."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.layers = [
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 4),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class BboxHead(nn.Module):
    """Predict one normalized xyxy box for each generated table-cell token."""

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError("TableFormerV2 bbox head requires exactly two decoder layers")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)
        self.kv_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.layers = [
            BboxDecoderLayer(embed_dim, num_heads, embed_dim * 4) for _ in range(num_layers)
        ]
        self.bbox_mlp = _BboxMlp(embed_dim)

    def __call__(
        self,
        cell_embeddings: mx.array,
        encoder_hidden: mx.array,
        cell_batch_indices: mx.array,
    ) -> mx.array:
        """Return one clipped, normalized xyxy box for every cell embedding."""
        if cell_embeddings.ndim != 2 or encoder_hidden.ndim != 3 or cell_batch_indices.ndim != 1:
            raise ValueError(
                "Expected cell embeddings [N, D], encoder hidden [B, S, D], and indices [N]"
            )
        if (
            cell_embeddings.shape[1] != self.embed_dim
            or encoder_hidden.shape[2] != self.embed_dim
            or cell_embeddings.shape[0] != cell_batch_indices.shape[0]
        ):
            raise ValueError("TableFormerV2 bbox inputs do not match the closed profile")
        if cell_embeddings.dtype != mx.float32 or encoder_hidden.dtype != mx.float32:
            raise TypeError("TableFormerV2 bbox head requires float32 inputs")
        if cell_embeddings.shape[0] == 0:
            return mx.zeros((0, 4), dtype=cell_embeddings.dtype)

        batch_mask = cell_batch_indices[:, None] == cell_batch_indices[None, :]
        encoder_for_cells = encoder_hidden[cell_batch_indices]
        x = self.input_norm(self.input_proj(cell_embeddings))
        memory = self.kv_norm(self.kv_proj(encoder_for_cells))
        for layer in self.layers:
            x = layer(x, memory, batch_mask)
        return cxcywh_to_xyxy(mx.sigmoid(self.bbox_mlp(x)))


__all__ = ["BboxDecoderLayer", "BboxHead", "cxcywh_to_xyxy"]
