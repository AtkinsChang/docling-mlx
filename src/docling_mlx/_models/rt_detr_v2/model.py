# Adapted from mlx-vlm (mlx_vlm/models/rt_detr_v2).
# SPDX-License-Identifier: MIT
"""Config-driven NHWC RT-DETR-v2 model assembly."""

from typing import TypedDict

import mlx.core as mx
import mlx.nn as nn

from docling_mlx._models.detector_primitives import generate_anchors, select_encoder_queries

from .config import RtDetrV2Config
from .transformer import MLP, Decoder
from .vision import VisionTower


class RtDetrV2Output(TypedDict):
    """Raw detector outputs before Transformers-compatible postprocessing."""

    pred_logits: mx.array
    pred_boxes: mx.array
    intermediate_logits: mx.array
    intermediate_reference_points: mx.array
    last_hidden_state: mx.array


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
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm(out_channels, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.bn(self.conv(x))


class _EncoderOutput(nn.Module):
    def __init__(self, d_model: int, eps: float) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.ln(self.fc(x))


class RtDetrV2(nn.Module):
    """Config-driven, NHWC RT-DETR-v2 model for inference."""

    def __init__(self, config: RtDetrV2Config) -> None:
        super().__init__()
        if not isinstance(config, RtDetrV2Config):
            raise TypeError("config must be an RtDetrV2Config")
        self.config = config
        self.vision = VisionTower(config)

        d_model = config.d_model
        self.decoder_input_proj = [
            _DecoderInputProjection(in_channels, d_model, eps=config.batch_norm_eps)
            for in_channels in config.decoder_in_channels
        ]
        self.decoder_input_proj.extend(
            _DecoderInputProjection(
                d_model,
                d_model,
                eps=config.batch_norm_eps,
                kernel_size=3,
                stride=2,
            )
            for _ in range(config.num_feature_levels - len(config.decoder_in_channels))
        )
        self.enc_output = _EncoderOutput(d_model, eps=config.layer_norm_eps)
        self.enc_score_head = nn.Linear(d_model, config.num_labels)
        self.enc_bbox_head = MLP(d_model, d_model, 4, num_layers=3)

        # Kept because it exists in both source checkpoints and strict loading
        # must account for every inference-artifact tensor.
        self.denoising_class_embed = nn.Embedding(config.num_labels + 1, d_model)
        self.decoder = Decoder(config.transformer_config)
        self.eval()

    def compile_backbone(self) -> None:
        """Compile the backbone once per input shape."""

        self.vision.compile_backbone()

    def __call__(self, pixel_values: mx.array) -> RtDetrV2Output:
        if self.training:
            raise ValueError("RtDetrV2 supports eval mode only; call model.eval()")
        if (
            pixel_values.ndim != 4
            or pixel_values.shape[-1] != self.config.backbone_config.num_channels
        ):
            raise ValueError(
                f"Expected NHWC input shape [B, H, W, {self.config.backbone_config.num_channels}]"
            )
        if pixel_values.shape[0] <= 0:
            raise ValueError("RtDetrV2 requires a nonempty batch")

        encoder_features = self.vision(pixel_values)
        projected = [
            projection(feature)
            for projection, feature in zip(self.decoder_input_proj, encoder_features, strict=False)
        ]
        for projection in self.decoder_input_proj[len(encoder_features) :]:
            projected.append(projection(projected[-1]))
        spatial_shapes = tuple((feature.shape[1], feature.shape[2]) for feature in projected)
        flat = mx.concatenate(
            [
                feature.reshape(
                    feature.shape[0], feature.shape[1] * feature.shape[2], feature.shape[3]
                )
                for feature in projected
            ],
            axis=1,
        )

        anchor_shapes = spatial_shapes
        if (anchor_size := getattr(self.config, "anchor_image_size", None)) is not None:
            height, width = anchor_size
            anchor_shapes = tuple(
                (height // stride, width // stride) for stride in self.config.feat_strides
            )
        anchors, valid_mask = generate_anchors(anchor_shapes, dtype=flat.dtype)
        output_memory = self.enc_output(flat * valid_mask.astype(flat.dtype))
        encoder_scores = self.enc_score_head(output_memory)
        encoder_box_logits = self.enc_bbox_head(output_memory) + anchors

        _, _, reference_points_unact, target = select_encoder_queries(
            output_memory,
            encoder_scores,
            encoder_box_logits,
            self.config.num_queries,
        )

        decoder_output = self.decoder(
            target=target,
            reference_points_unact=reference_points_unact,
            encoder_hidden_states=flat,
            spatial_shapes=spatial_shapes,
        )
        intermediate_logits = decoder_output["intermediate_logits"]
        intermediate_reference_points = decoder_output["intermediate_reference_points"]
        return {
            "pred_logits": intermediate_logits[:, -1],
            "pred_boxes": intermediate_reference_points[:, -1],
            "intermediate_logits": intermediate_logits,
            "intermediate_reference_points": intermediate_reference_points,
            "last_hidden_state": decoder_output["last_hidden_state"],
        }
