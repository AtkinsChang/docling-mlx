# Implemented after Torchvision's EfficientNetV2-S (torchvision/models/efficientnet.py);
# parameter names follow the pinned TableFormerV2 checkpoint.
# SPDX-License-Identifier: Apache-2.0
"""Native NHWC FP32 vision encoder for the pinned TableFormerV2 checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from docling_mlx._models.tableformer_v2.config import TableFormerV2Config

_ConvNorm = list[nn.Module]


def _conv_norm(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
    groups: int = 1,
    bias: bool = False,
) -> _ConvNorm:
    """Match Torchvision's ``Conv2dNormActivation`` parameter namespace."""

    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=bias,
        ),
        # Torchvision EfficientNetV2 constructs BatchNorm2d with eps=1e-3.
        # TableFormer's own mixer uses the PyTorch default 1e-5 separately.
        nn.BatchNorm(out_channels, eps=1e-3),
    ]


def _apply_conv_norm(x: mx.array, layer: _ConvNorm, *, activation: str | None) -> mx.array:
    x = layer[1](layer[0](x))
    if activation == "silu":
        return nn.silu(x)
    if activation is None:
        return x
    raise ValueError(f"Unsupported EfficientNetV2 activation: {activation}")


class _TorchvisionSqueezeExcitation(nn.Module):
    """The EfficientNetV2 MBConv SE block with Torchvision key names."""

    def __init__(self, input_channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        scale = mx.mean(x, axis=(1, 2), keepdims=True)
        scale = nn.silu(self.fc1(scale))
        return x * mx.sigmoid(self.fc2(scale))


class _TableFormerSqueezeExcitation(nn.Module):
    """The ReLU SE module used after the EfficientNetV2 feature head and in the mixer."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        # The ``None`` entries preserve the original Sequential indices: convs
        # are at ``se.1`` and ``se.3`` after parameter flattening.
        self.se: list[nn.Module | None] = [
            None,
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            None,
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            None,
        ]

    def __call__(self, x: mx.array) -> mx.array:
        reduce = cast(nn.Module, self.se[1])
        expand = cast(nn.Module, self.se[3])
        scale = mx.mean(x, axis=(1, 2), keepdims=True)
        return x * mx.sigmoid(expand(nn.relu(reduce(scale))))


class _FusedMBConv(nn.Module):
    """Torchvision EfficientNetV2-S fused inverted residual block in eval mode."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        expanded_channels = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels
        if expanded_channels == in_channels:
            self.block: list[_ConvNorm] = [
                _conv_norm(in_channels, out_channels, 3, stride=stride),
            ]
        else:
            self.block = [
                _conv_norm(in_channels, expanded_channels, 3, stride=stride),
                _conv_norm(expanded_channels, out_channels, 1),
            ]

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        for index, layer in enumerate(self.block):
            x = _apply_conv_norm(x, layer, activation="silu" if index == 0 else None)
        return x + residual if self.use_res_connect else x


class _MBConv(nn.Module):
    """Torchvision EfficientNetV2-S MBConv block in eval mode."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        expanded_channels = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels
        block: list[object] = []
        if expanded_channels != in_channels:
            block.append(_conv_norm(in_channels, expanded_channels, 1))
        block.append(
            _conv_norm(
                expanded_channels,
                expanded_channels,
                3,
                stride=stride,
                groups=expanded_channels,
            )
        )
        block.append(_TorchvisionSqueezeExcitation(expanded_channels, max(1, in_channels // 4)))
        block.append(_conv_norm(expanded_channels, out_channels, 1))
        self.block = block

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        for index, item in enumerate(self.block):
            if isinstance(item, list):
                activation = None if index == len(self.block) - 1 else "silu"
                x = _apply_conv_norm(x, item, activation=activation)
            else:
                x = cast(_TorchvisionSqueezeExcitation, item)(x)
        return x + residual if self.use_res_connect else x


class _EfficientNetV2S(nn.Module):
    """The inference-used ``efficientnet_v2_s().features`` subtree only."""

    _STAGES = (
        (_FusedMBConv, 1, 24, 24, 2, 1),
        (_FusedMBConv, 4, 24, 48, 4, 2),
        (_FusedMBConv, 4, 48, 64, 4, 2),
        (_MBConv, 4, 64, 128, 6, 2),
        (_MBConv, 6, 128, 160, 9, 1),
        (_MBConv, 6, 160, 256, 15, 2),
    )

    def __init__(self) -> None:
        super().__init__()
        features: list[object] = [_conv_norm(3, 24, 3, stride=2)]
        for block_type, ratio, input_channels, output_channels, repeats, stride in self._STAGES:
            stage: list[nn.Module] = []
            for index in range(repeats):
                stage.append(
                    block_type(
                        input_channels if index == 0 else output_channels,
                        output_channels,
                        stride if index == 0 else 1,
                        ratio,
                    )
                )
            features.append(stage)
        features.append(_conv_norm(256, 1280, 1))
        self.features = features

    def __call__(self, x: mx.array, intermediates: dict[str, mx.array] | None = None) -> mx.array:
        x = _apply_conv_norm(x, cast(_ConvNorm, self.features[0]), activation="silu")
        if intermediates is not None:
            intermediates["stem"] = x
        for stage_index, stage in enumerate(self.features[1:-1]):
            for block in cast(list[nn.Module], stage):
                x = block(x)
            if intermediates is not None:
                intermediates[f"backbone.stages.{stage_index}"] = x
        x = _apply_conv_norm(x, cast(_ConvNorm, self.features[-1]), activation="silu")
        if intermediates is not None:
            intermediates["backbone"] = x
        return x


class _DepthwiseSeparableBlock(nn.Module):
    """TableFormer's post-backbone GELU spatial mixer."""

    def __init__(self, channels: int, expansion: float) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        # Preserve the upstream Sequential keys, including stateless GELU slots.
        self.block: list[nn.Module | None] = [
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm(channels),
            None,
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm(hidden),
            None,
            _TableFormerSqueezeExcitation(hidden),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm(channels),
            None,
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for index, operation in enumerate(self.block):
            if operation is None:
                continue
            x = operation(x)
            if index in {1, 4, 8}:
                x = nn.gelu(x)
        return x


class TableFormerV2VisionEncoder(nn.Module):
    """EfficientNetV2-S, TableFormer mixer and 512-dimensional feature projection.

    The public call returns decoder memory in ``(batch, spatial_tokens, embed_dim)``
    NHWC-derived order. ``forward_intermediates`` provides named boundaries for
    the pinned Torch CPU oracle without coupling runtime inference to Torch.
    """

    def __init__(self, config: Mapping[str, Any] | TableFormerV2Config) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, TableFormerV2Config)
            else TableFormerV2Config.from_dict(config)
        )
        self.feature_extractor = _EfficientNetV2S()
        self.se_module = _TableFormerSqueezeExcitation(in_channels=1280)
        self.conv_mixer = _DepthwiseSeparableBlock(1280, expansion=self.config.conv_mixer_expansion)
        self.feature_to_embedding = nn.Linear(1280, self.config.embed_dim)
        self.eval()

    def _forward(
        self, pixels: mx.array, intermediates: dict[str, mx.array] | None
    ) -> tuple[mx.array, tuple[int, int]]:
        if self.training:
            raise ValueError(
                "TableFormerV2 vision encoder supports eval mode only; call model.eval()"
            )
        if pixels.dtype != mx.float32:
            raise ValueError("TableFormerV2 vision encoder inputs must be float32")
        if pixels.ndim != 4 or pixels.shape[-1] != 3:
            raise ValueError("Expected NHWC pixels with three channels")
        if min(pixels.shape[:3]) <= 0:
            raise ValueError("TableFormerV2 vision encoder requires nonempty images")
        features = self.feature_extractor(pixels, intermediates)
        features = self.se_module(features)
        if intermediates is not None:
            intermediates["post_backbone_se"] = features
        features = self.conv_mixer(features)
        if intermediates is not None:
            intermediates["spatial_mixer"] = features
        batch, height, width, channels = features.shape
        encoded = self.feature_to_embedding(mx.reshape(features, (batch, height * width, channels)))
        if intermediates is not None:
            intermediates["encoded"] = encoded
        return encoded, (height, width)

    def __call__(self, pixels: mx.array) -> tuple[mx.array, tuple[int, int]]:
        return self._forward(pixels, None)

    def forward_intermediates(self, pixels: mx.array) -> dict[str, mx.array]:
        intermediates: dict[str, mx.array] = {}
        self._forward(pixels, intermediates)
        return intermediates


def is_ignored_source_key(source_key: str) -> bool:
    """Return whether an upstream key is intentionally outside the inference graph.

    ``efficientnet_v2_s()`` carries an ImageNet classifier that TableFormerV2
    never calls: only ``feature_extractor.features`` appears in ``encode_images``.
    The converter records these keys as explicit ignores instead of creating a
    dead MLX classifier solely to satisfy source state accounting.
    """

    return source_key.startswith("feature_extractor.classifier.")


__all__ = ["TableFormerV2VisionEncoder", "is_ignored_source_key"]
