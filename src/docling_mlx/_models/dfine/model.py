# Implemented after Hugging Face Transformers (transformers/models/d_fine); module
# structure, parameter names, and forward-pass order follow it so published
# checkpoints load unchanged.
# SPDX-License-Identifier: Apache-2.0
"""Native evaluation-only D-FINE composition."""

from __future__ import annotations

from typing import TypedDict

import mlx.core as mx
import mlx.nn as nn

from docling_mlx._models.detector_primitives import generate_anchors, select_encoder_queries

from .config import DFineConfig
from .decoder import DFineDecoder
from .primitives import MLP
from .vision import DFineVisionTower


class _DFineOutput(TypedDict):
    pred_logits: mx.array
    pred_boxes: mx.array
    last_hidden_state: mx.array
    intermediate_hidden_states: mx.array
    intermediate_logits: mx.array
    intermediate_reference_points: mx.array
    intermediate_predicted_corners: mx.array
    initial_reference_points: mx.array
    init_reference_points: mx.array
    enc_topk_logits: mx.array
    enc_topk_bboxes: mx.array
    enc_outputs_class: mx.array
    enc_outputs_coord_logits: mx.array


class _Identity(nn.Module):
    def __call__(self, value: mx.array) -> mx.array:
        return value


class _DecoderInputProjection(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        eps: float,
        *,
        kernel_size: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.norm = nn.BatchNorm(out_channels, eps=eps)

    def __call__(self, value: mx.array) -> mx.array:
        return self.norm(self.conv(value))


class _EncoderOutput(nn.Module):
    def __init__(self, hidden_dim: int, eps: float) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)

    def __call__(self, value: mx.array) -> mx.array:
        return self.norm(self.fc(value))


class DFine(nn.Module):
    """Compose a config-driven D-FINE inference graph."""

    def __init__(self, config: DFineConfig) -> None:
        super().__init__()
        if not isinstance(config, DFineConfig):
            raise TypeError("config must be a DFineConfig")
        self.config = config
        self.vision = DFineVisionTower(config)
        base_projections: list[nn.Module] = [
            _Identity()
            if config.decoder.hidden_dim == in_channels
            else _DecoderInputProjection(
                in_channels,
                config.decoder.hidden_dim,
                config.encoder.batch_norm_eps,
            )
            for in_channels in config.decoder.in_channels
        ]
        extra_projections: list[nn.Module] = []
        in_channels = config.decoder.in_channels[-1]
        for _ in range(config.decoder.num_feature_levels - len(base_projections)):
            extra_projections.append(
                _DecoderInputProjection(
                    in_channels,
                    config.decoder.hidden_dim,
                    config.encoder.batch_norm_eps,
                    kernel_size=3,
                    stride=2,
                )
            )
            in_channels = config.decoder.hidden_dim
        self.decoder_input_proj = [*base_projections, *extra_projections]
        hidden_dim = config.decoder.hidden_dim
        self.enc_output = _EncoderOutput(hidden_dim, config.encoder.layer_norm_eps)
        self.enc_score_head = nn.Linear(hidden_dim, config.num_labels)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.decoder = DFineDecoder(config.decoder, num_labels=config.num_labels)
        self.eval()

    def compile_backbone(self) -> None:
        self.vision.compile_backbone()

    def __call__(self, pixel_values: mx.array) -> _DFineOutput:
        if self.training:
            raise ValueError("DFine supports eval mode only; call model.eval()")
        if pixel_values.dtype not in {mx.float16, mx.float32}:
            raise ValueError("DFine inputs must be float16 or float32")
        if pixel_values.ndim != 4 or pixel_values.shape[-1] != self.config.backbone.num_channels:
            raise ValueError(
                f"Expected NHWC input with {self.config.backbone.num_channels} channels"
            )
        if pixel_values.shape[0] <= 0:
            raise ValueError("DFine requires a nonempty batch")

        features = self.vision(pixel_values)
        projected: list[mx.array] = [
            projection(feature)
            for projection, feature in zip(self.decoder_input_proj, features, strict=False)
        ]
        while len(projected) < len(self.decoder_input_proj):
            projected.append(self.decoder_input_proj[len(projected)](projected[-1]))
        projected_features = tuple(projected)
        spatial_shapes = tuple(
            (feature.shape[1], feature.shape[2]) for feature in projected_features
        )
        flat = mx.concatenate(
            [
                feature.reshape(
                    feature.shape[0], feature.shape[1] * feature.shape[2], feature.shape[3]
                )
                for feature in projected_features
            ],
            axis=1,
        )

        anchors, valid_mask = generate_anchors(spatial_shapes, dtype=flat.dtype)
        output_memory = self.enc_output(flat * valid_mask.astype(flat.dtype))
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors
        enc_topk_logits, enc_topk_bboxes, init_reference_points, target = select_encoder_queries(
            output_memory,
            enc_outputs_class,
            enc_outputs_coord_logits,
            self.config.decoder.num_queries,
        )

        decoder_output = self.decoder(
            encoder_hidden_states=flat,
            reference_points=init_reference_points,
            inputs_embeds=target,
            spatial_shapes=spatial_shapes,
        )
        intermediate_logits = decoder_output["intermediate_logits"]
        intermediate_reference_points = decoder_output["intermediate_reference_points"]
        return {
            "pred_logits": intermediate_logits[:, -1],
            "pred_boxes": intermediate_reference_points[:, -1],
            "last_hidden_state": decoder_output["last_hidden_state"],
            "intermediate_hidden_states": decoder_output["intermediate_hidden_states"],
            "intermediate_logits": intermediate_logits,
            "intermediate_reference_points": intermediate_reference_points,
            "intermediate_predicted_corners": decoder_output["intermediate_predicted_corners"],
            "initial_reference_points": decoder_output["initial_reference_points"],
            "init_reference_points": init_reference_points,
            "enc_topk_logits": enc_topk_logits,
            "enc_topk_bboxes": enc_topk_bboxes,
            "enc_outputs_class": enc_outputs_class,
            "enc_outputs_coord_logits": enc_outputs_coord_logits,
        }
