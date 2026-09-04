# Implemented after Hugging Face Transformers (transformers/models/d_fine); module
# structure, parameter names, and forward-pass order follow it so published
# checkpoints load unchanged.
# SPDX-License-Identifier: Apache-2.0
"""Native MLX D-FINE query decoder and Fine-grained Distribution Refinement."""

from __future__ import annotations

from typing import TypedDict

import mlx.core as mx
import mlx.nn as nn

from docling_mlx._models.dfine.config import DFineDecoderConfig
from docling_mlx._models.dfine.primitives import (
    MLP,
    SelfAttention,
    distance2bbox,
    grid_sample_bilinear_zeros_align_corners_false,
    inverse_sigmoid,
    weighting_function,
)


class DFineDecoderOutput(TypedDict):
    last_hidden_state: mx.array
    intermediate_hidden_states: mx.array
    intermediate_logits: mx.array
    intermediate_reference_points: mx.array
    intermediate_predicted_corners: mx.array
    initial_reference_points: mx.array


class DFineGate(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def __call__(self, residual: mx.array, hidden_states: mx.array) -> mx.array:
        gates = mx.sigmoid(self.gate(mx.concatenate([residual, hidden_states], axis=-1)))
        residual_gate, hidden_gate = mx.split(gates, 2, axis=-1)
        return self.norm(residual_gate * residual + hidden_gate * hidden_states)


class DFineMultiscaleDeformableAttention(nn.Module):
    """D-FINE's unequal-point deformable attention with exact checkpoint state."""

    def __init__(self, config: DFineDecoderConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.attention_heads
        self.num_levels = config.num_feature_levels
        self.points_per_level = config.points_per_level
        self.method = config.method
        self.head_dim = config.hidden_dim // config.attention_heads
        self.offset_scale = config.offset_scale
        self.num_points_scale = mx.array(
            [1.0 / count for count in self.points_per_level for _ in range(count)],
            dtype=mx.float32,
        )
        total_points = sum(self.points_per_level)
        self.sampling_offsets = nn.Linear(
            config.hidden_dim, config.attention_heads * total_points * 2
        )
        self.attention_weights = nn.Linear(config.hidden_dim, config.attention_heads * total_points)

    def __call__(
        self,
        hidden_states: mx.array,
        reference_points: mx.array,
        encoder_hidden_states: mx.array,
        spatial_shapes: tuple[tuple[int, int], ...],
    ) -> mx.array:
        batch, queries, _ = hidden_states.shape
        sequence_length = encoder_hidden_states.shape[1]
        if sum(height * width for height, width in spatial_shapes) != sequence_length:
            raise ValueError("spatial_shapes must align with the encoder sequence length")
        if len(spatial_shapes) != self.num_levels:
            raise ValueError("spatial_shapes must contain one shape per feature level")
        if reference_points.shape[-1] != 4:
            raise ValueError("D-FINE decoder reference points must be 4D boxes")

        total_points = sum(self.points_per_level)
        value = encoder_hidden_states.reshape(batch, sequence_length, self.num_heads, self.head_dim)
        offsets = self.sampling_offsets(hidden_states).reshape(
            batch, queries, self.num_heads, total_points, 2
        )
        weights = mx.softmax(
            self.attention_weights(hidden_states).reshape(
                batch, queries, self.num_heads, total_points
            ),
            axis=-1,
        )
        scale = self.num_points_scale.astype(hidden_states.dtype)[None, None, None, :, None]
        locations = (
            reference_points[:, :, None, :, :2]
            + offsets * scale * reference_points[:, :, None, :, 2:] * self.offset_scale
        )

        sampled_levels = []
        value_offset = point_offset = 0
        for (height, width), point_count in zip(spatial_shapes, self.points_per_level, strict=True):
            level_size = height * width
            level_value = value[:, value_offset : value_offset + level_size].reshape(
                batch, height, width, self.num_heads, self.head_dim
            )
            level_value = level_value.transpose(0, 3, 1, 2, 4).reshape(
                batch * self.num_heads, height, width, self.head_dim
            )
            level_grid = locations[:, :, :, point_offset : point_offset + point_count].transpose(
                0, 2, 1, 3, 4
            )
            level_grid = level_grid.reshape(batch * self.num_heads, queries, point_count, 2)
            if self.method == "default":
                sampled_levels.append(
                    grid_sample_bilinear_zeros_align_corners_false(
                        level_value.astype(mx.float32),
                        (2.0 * level_grid - 1.0).astype(mx.float32),
                    ).astype(hidden_states.dtype)
                )
            else:
                coordinates = (
                    level_grid * mx.array([width, height], dtype=hidden_states.dtype) + 0.5
                ).astype(mx.int32)
                x = mx.clip(coordinates[..., 0], 0, width - 1)
                y = mx.clip(coordinates[..., 1], 0, height - 1)
                indices = (y * width + x).reshape(batch * self.num_heads, queries * point_count)
                sampled_levels.append(
                    level_value.reshape(batch * self.num_heads, height * width, self.head_dim)[
                        mx.arange(batch * self.num_heads)[:, None], indices
                    ].reshape(batch * self.num_heads, queries, point_count, self.head_dim)
                )
            value_offset += level_size
            point_offset += point_count

        sampled = mx.concatenate(sampled_levels, axis=-2)
        weights = weights.transpose(0, 2, 1, 3).reshape(
            batch * self.num_heads, queries, total_points
        )
        output = (sampled * weights[..., None]).sum(axis=-2)
        return (
            output.reshape(batch, self.num_heads, queries, self.head_dim)
            .transpose(0, 2, 1, 3)
            .reshape(batch, queries, self.hidden_dim)
        )


class DFineDecoderLayer(nn.Module):
    def __init__(self, config: DFineDecoderConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        self.self_attn = SelfAttention(hidden_dim, config.attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.encoder_attn = DFineMultiscaleDeformableAttention(config)
        self.fc1 = nn.Linear(hidden_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, hidden_dim)
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.gateway = DFineGate(hidden_dim)
        self.activation = _activation(config.activation_function)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: mx.array,
        reference_points: mx.array,
        encoder_hidden_states: mx.array,
        spatial_shapes: tuple[tuple[int, int], ...],
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.encoder_attn(
            hidden_states + position_embeddings,
            reference_points,
            encoder_hidden_states,
            spatial_shapes,
        )
        hidden_states = self.gateway(residual, hidden_states)

        residual = hidden_states
        hidden_states = residual + self.fc2(self.activation(self.fc1(hidden_states)))
        return self.final_layer_norm(mx.clip(hidden_states, -65504.0, 65504.0))


class DFineIntegral(nn.Module):
    def __init__(self, max_num_bins: int) -> None:
        super().__init__()
        self.max_num_bins = max_num_bins

    def __call__(self, pred_corners: mx.array, project: mx.array) -> mx.array:
        batch, queries, _ = pred_corners.shape
        probabilities = mx.softmax(pred_corners.reshape(-1, self.max_num_bins + 1), axis=1)
        distances = (probabilities * project[None, :]).sum(axis=1)
        return distances.reshape(batch, queries, 4)


class DFineLQE(nn.Module):
    def __init__(self, config: DFineDecoderConfig) -> None:
        super().__init__()
        self.top_prob_values = config.top_prob_values
        self.max_num_bins = config.max_num_bins
        self.reg_conf = MLP(
            4 * (config.top_prob_values + 1),
            config.lqe_hidden_dim,
            1,
            config.lqe_layers,
        )

    def __call__(self, scores: mx.array, pred_corners: mx.array) -> mx.array:
        batch, length, _ = pred_corners.shape
        probabilities = mx.softmax(
            pred_corners.reshape(batch, length, 4, self.max_num_bins + 1), axis=-1
        )
        top = -mx.sort(-probabilities, axis=-1)[..., : self.top_prob_values]
        statistics = mx.concatenate([top, top.mean(axis=-1, keepdims=True)], axis=-1)
        return scores + self.reg_conf(statistics.reshape(batch, length, -1))


class DFineDecoder(nn.Module):
    """D-FINE decoder with Transformers-compatible state keys."""

    def __init__(self, config: DFineDecoderConfig, *, num_labels: int) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        self.eval_idx = config.eval_idx if config.eval_idx >= 0 else config.layers + config.eval_idx
        self.layers = [DFineDecoderLayer(config) for _ in range(config.layers)]
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)
        self.reg_scale = mx.array([config.reg_scale], dtype=mx.float32)
        self.max_num_bins = config.max_num_bins
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.integral = DFineIntegral(config.max_num_bins)
        self.up = mx.array([config.up], dtype=mx.float32)
        self.lqe_layers = [DFineLQE(config) for _ in range(config.layers)]
        self.class_embed = [nn.Linear(hidden_dim, num_labels) for _ in range(config.layers)]
        self.bbox_embed = [
            MLP(hidden_dim, hidden_dim, 4 * (config.max_num_bins + 1), 3)
            for _ in range(config.layers)
        ]

    def __call__(
        self,
        *,
        encoder_hidden_states: mx.array,
        reference_points: mx.array,
        inputs_embeds: mx.array,
        spatial_shapes: tuple[tuple[int, int], ...],
    ) -> DFineDecoderOutput:
        hidden_states = inputs_embeds
        intermediate_hidden = []
        intermediate_logits = []
        intermediate_references = []
        intermediate_corners = []
        initial_references = []
        output_detached = None
        previous_corners = None

        project = weighting_function(self.max_num_bins, self.up, self.reg_scale)
        detached_references = mx.sigmoid(reference_points)

        for index, layer in enumerate(self.layers):
            reference_input = detached_references[:, :, None, :]
            position = mx.clip(self.query_pos_head(detached_references), -10.0, 10.0)
            hidden_states = layer(
                hidden_states,
                position,
                reference_input,
                encoder_hidden_states,
                spatial_shapes,
            )

            if index == 0:
                new_references = mx.sigmoid(
                    self.pre_bbox_head(hidden_states) + inverse_sigmoid(detached_references)
                )
                initial_reference = mx.stop_gradient(new_references)

            corner_input = hidden_states
            if output_detached is not None:
                corner_input = corner_input + output_detached
            predicted_corners = self.bbox_embed[index](corner_input)
            if previous_corners is not None:
                predicted_corners = predicted_corners + previous_corners
            refined_reference = distance2bbox(
                initial_reference,
                self.integral(predicted_corners, project),
                self.reg_scale,
            )
            previous_corners = predicted_corners
            detached_references = mx.stop_gradient(refined_reference)
            output_detached = mx.stop_gradient(hidden_states)
            intermediate_hidden.append(hidden_states)

            if index == self.eval_idx:
                scores = self.lqe_layers[index](
                    self.class_embed[index](hidden_states), predicted_corners
                )
                intermediate_logits.append(scores)
                intermediate_references.append(refined_reference)
                intermediate_corners.append(predicted_corners)
                initial_references.append(initial_reference)

        return {
            "last_hidden_state": hidden_states,
            "intermediate_hidden_states": mx.stack(intermediate_hidden, axis=0),
            "intermediate_logits": mx.stack(intermediate_logits, axis=1),
            "intermediate_reference_points": mx.stack(intermediate_references, axis=1),
            "intermediate_predicted_corners": mx.stack(intermediate_corners, axis=1),
            "initial_reference_points": mx.stack(initial_references, axis=1),
        }


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported D-FINE decoder activation: {name!r}")
