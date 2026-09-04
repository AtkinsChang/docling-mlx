# Implemented after docling-ibm-models (docling_ibm_models/tableformer/models/table04_rs)
# and Torchvision's ResNet-18 (torchvision/models/resnet.py); module structure,
# parameter names, and forward-pass order follow them so the published checkpoint
# loads unchanged.
# SPDX-License-Identifier: Apache-2.0
"""Native FP32 TableFormer v1 image and tag encoder."""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from docling_mlx._models.tableformer_v1._source import source_parameter_filter
from docling_mlx._models.tableformer_v1.config import TableFormerV1Config


class _ReLU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(x)


class _BasicBlock(nn.Module):
    """Torchvision BasicBlock with its checkpoint parameter names."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(out_channels, eps=1e-5)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(out_channels, eps=1e-5)
        self.downsample = (
            [
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm(out_channels, eps=1e-5),
            ]
            if stride != 1 or in_channels != out_channels
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            residual = self.downsample[1](self.downsample[0](residual))
        return nn.relu(x + residual)


def _adaptive_avg_pool_2d(x: mx.array, output_size: int) -> mx.array:
    height, width = x.shape[1:3]
    if (height, width) == (output_size, output_size):
        return x
    rows = []
    for row in range(output_size):
        start_h = row * height // output_size
        end_h = ((row + 1) * height + output_size - 1) // output_size
        columns = []
        for column in range(output_size):
            start_w = column * width // output_size
            end_w = ((column + 1) * width + output_size - 1) // output_size
            columns.append(mx.mean(x[:, start_h:end_h, start_w:end_w, :], axis=(1, 2)))
        rows.append(mx.stack(columns, axis=1))
    return mx.stack(rows, axis=1)


class TableFormerV1ImageEncoder(nn.Module):
    """ResNet18 through layer3, followed by adaptive 28x28 pooling."""

    def __init__(self, config: TableFormerV1Config) -> None:
        super().__init__()
        self.config = config
        self._resnet: list[Any] = [
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm(64, eps=1e-5),
            _ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            [_BasicBlock(64, 64), _BasicBlock(64, 64)],
            [_BasicBlock(64, 128, stride=2), _BasicBlock(128, 128)],
            [_BasicBlock(128, 256, stride=2), _BasicBlock(256, 256)],
        ]

    valid_parameter_filter = staticmethod(source_parameter_filter)

    def _validate(self, images: mx.array) -> None:
        if self.training:
            raise ValueError("TableFormer v1 native inference requires eval mode")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("TableFormer v1 images must be NHWC with three channels")
        if images.dtype != mx.float32:
            raise ValueError("TableFormer v1 images must be float32")

    def forward_intermediates(self, images: mx.array) -> dict[str, mx.array]:
        self._validate(images)
        x = self._resnet[0](images)
        x = self._resnet[2](self._resnet[1](x))
        intermediates = {"stem": x}
        x = self._resnet[3](x)
        for index in range(4, 7):
            for block in self._resnet[index]:
                x = block(x)
            intermediates[f"resnet.layers.{index - 4}"] = x
        intermediates["image_features"] = _adaptive_avg_pool_2d(x, self.config.encoded_image_size)
        return intermediates

    def __call__(self, images: mx.array) -> mx.array:
        return self.forward_intermediates(images)["image_features"]


class _FusedSelfAttention(nn.Module):
    """Torch MultiheadAttention's fused QKV parameter layout."""

    def __init__(self, dims: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.in_proj_weight = mx.zeros((3 * dims, dims), dtype=mx.float32)
        self.in_proj_bias = mx.zeros((3 * dims,), dtype=mx.float32)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, dims = x.shape
        qkv = x @ self.in_proj_weight.T + self.in_proj_bias
        query, key, value = mx.split(qkv, 3, axis=-1)

        def heads(tensor: mx.array) -> mx.array:
            return tensor.reshape(batch, length, self.num_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )

        query, key, value = heads(query), heads(key), heads(value)
        scores = (query * math.sqrt(1.0 / self.head_dim)) @ key.transpose(0, 1, 3, 2)
        attended = mx.softmax(scores, axis=-1) @ value
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, dims)
        return self.out_proj(attended)


class _PostNormEncoderLayer(nn.Module):
    """The eval-time PyTorch TransformerEncoderLayer computation."""

    def __init__(self, config: TableFormerV1Config) -> None:
        super().__init__()
        self.self_attn = _FusedSelfAttention(config.hidden_dim, config.num_heads)
        self.linear1 = nn.Linear(config.hidden_dim, config.ff_dim)
        self.linear2 = nn.Linear(config.ff_dim, config.hidden_dim)
        self.norm1 = nn.LayerNorm(config.hidden_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(config.hidden_dim, eps=1e-5)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm1(x + self.self_attn(x))
        return self.norm2(x + self.linear2(nn.relu(self.linear1(x))))


class _PostNormEncoder(nn.Module):
    def __init__(self, config: TableFormerV1Config) -> None:
        super().__init__()
        self.layers = [_PostNormEncoderLayer(config) for _ in range(config.encoder_layers)]

    def forward_intermediates(self, x: mx.array) -> dict[str, mx.array]:
        outputs = {}
        for index, layer in enumerate(self.layers):
            x = layer(x)
            outputs[f"tag_encoder.layers.{index}"] = x.transpose(1, 0, 2)
        return outputs

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class TableFormerV1TagEncoder(nn.Module):
    """Two-block 256->512 filter and config-driven post-norm tag encoder."""

    def __init__(self, config: TableFormerV1Config) -> None:
        super().__init__()
        self.config = config
        self._input_filter = [_BasicBlock(256, 512), _BasicBlock(512, 512)]
        self._encoder = _PostNormEncoder(self.config)

    valid_parameter_filter = staticmethod(source_parameter_filter)

    def _validate(self, image_features: mx.array) -> None:
        size = self.config.encoded_image_size
        if self.training:
            raise ValueError("TableFormer v1 native inference requires eval mode")
        if image_features.shape[1:] != (size, size, 256):
            raise ValueError(
                f"TableFormer v1 image features must have shape (B, {size}, {size}, 256)"
            )
        if image_features.dtype != mx.float32:
            raise ValueError("TableFormer v1 image features must be float32")

    def forward_intermediates(self, image_features: mx.array) -> dict[str, mx.array]:
        self._validate(image_features)
        x = image_features
        for block in self._input_filter:
            x = block(x)
        outputs = {"input_filter": x}
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        outputs.update(self._encoder.forward_intermediates(x))
        outputs["memory"] = outputs[f"tag_encoder.layers.{self.config.encoder_layers - 1}"]
        return outputs

    def __call__(self, image_features: mx.array) -> mx.array:
        return self.forward_intermediates(image_features)["memory"]


class TableFormerV1Encoder(nn.Module):
    """Image-to-transformer-memory encoder with source-compatible names."""

    def __init__(self, config: TableFormerV1Config) -> None:
        super().__init__()
        self.config = config
        self._encoder = TableFormerV1ImageEncoder(self.config)
        self._tag_transformer = TableFormerV1TagEncoder(self.config)

    valid_parameter_filter = staticmethod(source_parameter_filter)

    def forward_intermediates(self, images: mx.array) -> dict[str, mx.array]:
        outputs = self._encoder.forward_intermediates(images)
        outputs.update(self._tag_transformer.forward_intermediates(outputs["image_features"]))
        return outputs

    def __call__(self, images: mx.array) -> mx.array:
        return self.forward_intermediates(images)["memory"]


__all__ = [
    "TableFormerV1Encoder",
    "TableFormerV1ImageEncoder",
    "TableFormerV1TagEncoder",
]
