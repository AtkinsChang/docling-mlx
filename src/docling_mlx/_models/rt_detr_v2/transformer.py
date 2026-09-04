# Adapted from mlx-vlm (mlx_vlm/models/rt_detr_v2).
# SPDX-License-Identifier: MIT
"""RT-DETRv2 transformer: deformable-attention decoder and helpers.

Multi-scale deformable attention runs once per feature level via the
shared `grid_sample` Metal kernel and sums across levels with the
softmaxed attention weights. `n_points_scale` is carried as a
non-learnable `mx.array` buffer per layer to remain faithful to
checkpoints that use non-uniform `n_points_list`.
"""

import mlx.core as mx
import mlx.nn as nn

from docling_mlx._models.detector_primitives import (
    MLP,
    SelfAttention,
    grid_sample_bilinear_zeros_align_corners_false,
    inverse_sigmoid,
)

from .config import RtDetrV2TransformerConfig

# ─── Multi-Scale Deformable Attention ───


class MSDeformableAttention(nn.Module):
    """Multi-scale deformable attention for RT-DETRv2.

    Reference points are 4D `(cx, cy, w, h)` normalized to `[0, 1]`.
    Sampling offsets are predicted per `(n_heads, n_levels, n_points)`
    and scaled by `1/n_points * ref_wh * offset_scale` before being added
    to the reference center. Sampling itself is done by `grid_sample`
    (bilinear, padding=zeros, align_corners=False), once per level, and
    the outputs are concatenated and weighted-summed by the softmaxed
    attention weights.
    """

    def __init__(self, config: RtDetrV2TransformerConfig) -> None:
        super().__init__()
        d = config.d_model
        n_heads = config.decoder_attention_heads
        if d % n_heads != 0:
            raise ValueError(f"d_model ({d}) must be divisible by num_heads ({n_heads})")

        self.d_model = d
        self.n_heads = n_heads
        self.n_levels = config.decoder_n_levels
        self.points_per_level = config.points_per_level
        self.total_points = sum(config.points_per_level)
        self.head_dim = d // n_heads
        self.offset_scale = config.decoder_offset_scale
        self.method = config.decoder_method

        self.sampling_offsets = nn.Linear(d, n_heads * self.total_points * 2)
        self.attention_weights = nn.Linear(d, n_heads * self.total_points)
        self.value_proj = nn.Linear(d, d)
        self.output_proj = nn.Linear(d, d)

        # Per-(level, point) scale = 1/n_points. Stored as a non-learnable
        # buffer so non-uniform n_points_list configurations (where each
        # level can have a different number of sampling points) also work.
        self.n_points_scale = mx.array(
            [1.0 / count for count in self.points_per_level for _ in range(count)],
            dtype=mx.float32,
        )

    def __call__(
        self,
        query: mx.array,
        reference_points: mx.array,
        value: mx.array,
        spatial_shapes: tuple[tuple[int, int], ...],
        position_embeddings: mx.array | None = None,
    ) -> mx.array:
        """
        Args:
            query: (B, Q, D) decoder hidden states.
            reference_points: (B, Q, 1|L, 2|4) normalized reference points.
            value: (B, sum_HW, D) flattened multi-scale encoder features.
            spatial_shapes: tuple of (H, W) per level.
            position_embeddings: optional (B, Q, D) added to query.
        Returns:
            (B, Q, D)
        """
        if position_embeddings is not None:
            query = query + position_embeddings

        B, Q, D = query.shape
        n_heads = self.n_heads
        head_dim = self.head_dim

        v = self.value_proj(value).reshape(B, value.shape[1], n_heads, head_dim)
        offsets = self.sampling_offsets(query).reshape(B, Q, n_heads, self.total_points, 2)
        attn = self.attention_weights(query).reshape(B, Q, n_heads, self.total_points)
        attn = mx.softmax(attn, axis=-1)

        if reference_points.shape[-1] not in {2, 4}:
            raise ValueError(
                f"Expected 2D or 4D reference points, got last dim {reference_points.shape[-1]}"
            )

        level_sizes = [H * W for H, W in spatial_shapes]
        offsets_running = 0
        v_levels = []
        for s in level_sizes:
            v_levels.append(v[:, offsets_running : offsets_running + s, :, :])
            offsets_running += s

        sampled_per_level = []
        points_running = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            v_l = v_levels[lvl].reshape(B, H, W, n_heads, head_dim)
            v_l = v_l.transpose(0, 3, 1, 2, 4).reshape(B * n_heads, H, W, head_dim)
            point_count = self.points_per_level[lvl]
            offsets_l = offsets[:, :, :, points_running : points_running + point_count, :]
            ref_level = 0 if reference_points.shape[-2] == 1 else lvl
            ref_l = reference_points[:, :, ref_level, :]
            if ref_l.shape[-1] == 4:
                scale = self.n_points_scale[points_running : points_running + point_count].astype(
                    query.dtype
                )
                locations = ref_l[:, :, None, None, :2] + (
                    offsets_l
                    * scale[None, None, None, :, None]
                    * ref_l[:, :, None, None, 2:]
                    * self.offset_scale
                )
            else:
                locations = ref_l[:, :, None, None, :] + offsets_l / mx.array(
                    [W, H], dtype=query.dtype
                )
            samp = locations.transpose(0, 2, 1, 3, 4).reshape(B * n_heads, Q, point_count, 2)
            if self.method == "default":
                out_l = grid_sample_bilinear_zeros_align_corners_false(
                    v_l.astype(mx.float32), (2.0 * samp - 1.0).astype(mx.float32)
                ).astype(query.dtype)
            else:
                coordinates = (samp * mx.array([W, H], dtype=query.dtype) + 0.5).astype(mx.int32)
                x = mx.clip(coordinates[..., 0], 0, W - 1)
                y = mx.clip(coordinates[..., 1], 0, H - 1)
                indices = (y * W + x).reshape(B * n_heads, Q * point_count)
                out_l = v_l.reshape(B * n_heads, H * W, head_dim)[
                    mx.arange(B * n_heads)[:, None], indices
                ].reshape(B * n_heads, Q, point_count, head_dim)
            sampled_per_level.append(out_l)
            points_running += point_count

        sampled = mx.concatenate(sampled_per_level, axis=-2)
        w = attn.transpose(0, 2, 1, 3).reshape(B * n_heads, Q, self.total_points)
        out = (sampled * w[..., None]).sum(axis=-2)
        out = out.reshape(B, n_heads, Q, head_dim).transpose(0, 2, 1, 3).reshape(B, Q, D)
        return self.output_proj(out.astype(query.dtype))


# ─── Decoder ───


class DecoderLayer(nn.Module):
    """One decoder block: self-attn -> norm -> deformable cross-attn -> norm
    -> FFN -> norm. Field names match the saved state-dict."""

    def __init__(self, config: RtDetrV2TransformerConfig) -> None:
        super().__init__()
        d = config.d_model
        self.self_attn = SelfAttention(
            d,
            config.decoder_attention_heads,
            use_fast_attention=True,
        )
        self.self_attn_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.encoder_attn = MSDeformableAttention(config)
        self.encoder_attn_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(d, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, d)
        self.final_layer_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.activation = _resolve_activation(config.decoder_activation_function)

    def __call__(
        self,
        x: mx.array,
        object_queries_position_embeddings: mx.array,
        encoder_hidden_states: mx.array,
        reference_points: mx.array,
        spatial_shapes: tuple[tuple[int, int], ...],
    ) -> mx.array:
        residual = x
        x = self.self_attn(x, object_queries_position_embeddings)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.encoder_attn(
            query=x,
            reference_points=reference_points,
            value=encoder_hidden_states,
            spatial_shapes=spatial_shapes,
            position_embeddings=object_queries_position_embeddings,
        )
        x = residual + x
        x = self.encoder_attn_layer_norm(x)

        residual = x
        x = self.fc2(self.activation(self.fc1(x)))
        x = residual + x
        x = self.final_layer_norm(x)
        return x


class Decoder(nn.Module):
    """Decoder stack with iterative bbox refinement.

    Per-layer `bbox_embed` (3-layer MLP) and `class_embed` (Linear) heads
    are attached here, matching the saved keys
    `decoder.bbox_embed.{L}.layers.{i}.{weight,bias}` and
    `decoder.class_embed.{L}.{weight,bias}`.
    """

    def __init__(self, config: RtDetrV2TransformerConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.layers = [DecoderLayer(config) for _ in range(config.decoder_layers)]

        # 2-layer MLP that converts 4D reference points to D-dim position
        # embeddings added to queries in each decoder layer.
        self.query_pos_head = MLP(4, 2 * d, d, num_layers=2)

        self.bbox_embed = [MLP(d, d, 4, num_layers=3) for _ in range(config.decoder_layers)]
        self.class_embed = [nn.Linear(d, config.num_labels) for _ in range(config.decoder_layers)]

    def __call__(
        self,
        target: mx.array,
        reference_points_unact: mx.array,
        encoder_hidden_states: mx.array,
        spatial_shapes: tuple[tuple[int, int], ...],
    ) -> dict[str, mx.array]:
        """
        Args:
            target: (B, Q, D) initial query content.
            reference_points_unact: (B, Q, 4) initial reference boxes in logit
                space (pre-sigmoid). Decoder takes sigmoid as first step.
            encoder_hidden_states: (B, sum_HW, D) flattened encoder features.
            spatial_shapes: per-level (H, W) tuples.
        Returns:
            dict with `last_hidden_state`, `intermediate_hidden_states`,
            `intermediate_reference_points`, `intermediate_logits`.
        """
        hidden = target
        ref_points = mx.sigmoid(reference_points_unact)

        all_hidden = []
        all_refs = []
        all_logits = []

        for idx, layer in enumerate(self.layers):
            # (B, Q, 1, 4) broadcasts across feature levels in cross-attn.
            ref_input = ref_points[:, :, None, :]
            pos_embed = self.query_pos_head(ref_points)
            hidden = layer(
                x=hidden,
                object_queries_position_embeddings=pos_embed,
                encoder_hidden_states=encoder_hidden_states,
                reference_points=ref_input,
                spatial_shapes=spatial_shapes,
            )

            predicted_corners = self.bbox_embed[idx](hidden)
            new_refs = mx.sigmoid(predicted_corners + inverse_sigmoid(ref_points))
            ref_points = mx.stop_gradient(new_refs)

            all_hidden.append(hidden)
            all_refs.append(new_refs)
            all_logits.append(self.class_embed[idx](hidden))

        return {
            "last_hidden_state": hidden,
            "intermediate_hidden_states": mx.stack(all_hidden, axis=1),
            "intermediate_reference_points": mx.stack(all_refs, axis=1),
            "intermediate_logits": mx.stack(all_logits, axis=1),
        }


def _resolve_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")
