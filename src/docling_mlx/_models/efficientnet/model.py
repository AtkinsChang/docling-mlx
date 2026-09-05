# Implemented after Hugging Face Transformers (transformers/models/efficientnet);
# module structure, parameter names, and forward-pass order follow it so published
# checkpoints load unchanged.
# SPDX-License-Identifier: Apache-2.0
"""FP32 inference for the Hugging Face EfficientNet architecture in native MLX.

Spatial tensors are NHWC. Parameter names mirror the Hugging Face model so
conversion is an auditable layout transform, with no BatchNorm folding. The
topology follows Transformers' ``modeling_efficientnet.py``.
"""

import math
from collections.abc import Callable, Mapping
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .._compile import compile_exclusive
from .config import EfficientNetConfig


def _round_filters(config: EfficientNetConfig, channels: int) -> int:
    return config.round_filters(channels)


def _batchnorm(config: EfficientNetConfig, channels: int) -> nn.BatchNorm:
    return nn.BatchNorm(
        channels,
        eps=config.batch_norm_eps,
        momentum=config.batch_norm_momentum,
    )


class _Embeddings(nn.Module):
    def __init__(self, config: EfficientNetConfig) -> None:
        super().__init__()
        channels = _round_filters(config, 32)
        self.convolution = nn.Conv2d(config.num_channels, channels, 3, stride=2, bias=False)
        self.batchnorm = _batchnorm(config, channels)

    def __call__(self, x: mx.array) -> mx.array:
        # HF's fixed ZeroPad2d((0, 1, 0, 1)), not symmetric SAME padding.
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return nn.silu(self.batchnorm(self.convolution(x)))


class _Expansion(nn.Module):
    def __init__(self, config: EfficientNetConfig, channels: int, expanded: int) -> None:
        super().__init__()
        self.expand_conv = nn.Conv2d(channels, expanded, 1, bias=False)
        self.expand_bn = _batchnorm(config, expanded)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(self.expand_bn(self.expand_conv(x)))


class _Depthwise(nn.Module):
    def __init__(
        self,
        config: EfficientNetConfig,
        channels: int,
        stride: int,
        kernel: int,
        adjust_padding: bool,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.pad_before = kernel // 2 - int(adjust_padding)
        self.pad_after = kernel // 2
        self.depthwise_conv = nn.Conv2d(
            channels,
            channels,
            kernel,
            stride=stride,
            padding=kernel // 2 if stride == 1 else 0,
            groups=channels,
            bias=False,
        )
        self.depthwise_norm = _batchnorm(config, channels)

    def __call__(self, x: mx.array) -> mx.array:
        if self.stride == 2:
            padding = (self.pad_before, self.pad_after)
            x = mx.pad(x, [(0, 0), padding, padding, (0, 0)])
        return nn.silu(self.depthwise_norm(self.depthwise_conv(x)))


class _SqueezeExcite(nn.Module):
    def __init__(self, config: EfficientNetConfig, channels: int, expanded: int) -> None:
        super().__init__()
        reduced = max(1, int(channels * config.squeeze_expansion_ratio))
        self.reduce = nn.Conv2d(expanded, reduced, 1)
        self.expand = nn.Conv2d(reduced, expanded, 1)

    def __call__(self, x: mx.array) -> mx.array:
        scale = mx.mean(x, axis=(1, 2), keepdims=True)
        scale = self.expand(nn.silu(self.reduce(scale)))
        return x * mx.sigmoid(scale)


class _Projection(nn.Module):
    def __init__(
        self,
        config: EfficientNetConfig,
        expanded: int,
        output: int,
        residual: bool,
    ) -> None:
        super().__init__()
        self.project_conv = nn.Conv2d(expanded, output, 1, bias=False)
        self.project_bn = _batchnorm(config, output)
        self.residual = residual

    def __call__(self, x: mx.array, embeddings: mx.array) -> mx.array:
        x = self.project_bn(self.project_conv(x))
        # Drop connect and classifier dropout are identity in HF eval mode.
        return x + embeddings if self.residual else x


class _Block(nn.Module):
    def __init__(
        self,
        config: EfficientNetConfig,
        channels: int,
        output: int,
        stride: int,
        ratio: int,
        kernel: int,
        first: bool,
        index: int,
    ) -> None:
        super().__init__()
        expanded = channels * ratio
        self.has_expansion = ratio != 1
        if self.has_expansion:
            self.expansion = _Expansion(config, channels, expanded)
        self.depthwise_conv = _Depthwise(
            config,
            expanded,
            stride,
            kernel,
            index not in config.depthwise_padding,
        )
        self.squeeze_excite = _SqueezeExcite(config, channels, expanded)
        # HF disables the residual for the first block of EVERY stage.
        self.projection = _Projection(config, expanded, output, stride == 1 and not first)

    def __call__(self, x: mx.array) -> mx.array:
        embeddings = x
        if self.has_expansion:
            x = self.expansion(x)
        x = self.squeeze_excite(self.depthwise_conv(x))
        return self.projection(x, embeddings)


class _Encoder(nn.Module):
    def __init__(self, config: EfficientNetConfig) -> None:
        super().__init__()
        blocks: list[_Block] = []
        output = 0
        for stage, channels in enumerate(config.in_channels):
            channels = _round_filters(config, channels)
            output = _round_filters(config, config.out_channels[stage])
            repeats = math.ceil(config.depth_coefficient * config.num_block_repeats[stage])
            for repeat in range(repeats):
                blocks.append(
                    _Block(
                        config,
                        channels if repeat == 0 else output,
                        output,
                        config.strides[stage] if repeat == 0 else 1,
                        config.expand_ratios[stage],
                        config.kernel_sizes[stage],
                        first=repeat == 0,
                        index=len(blocks),
                    )
                )
        self.blocks = blocks
        self.top_conv = nn.Conv2d(output, _round_filters(config, 1280), 1, bias=False)
        self.top_bn = _batchnorm(config, config.hidden_dim)

    def __call__(self, x: mx.array, intermediates: dict[str, mx.array] | None = None) -> mx.array:
        for index, block in enumerate(self.blocks):
            x = block(x)
            if intermediates is not None:
                intermediates[f"blocks.{index}"] = x
        return nn.silu(self.top_bn(self.top_conv(x)))


class _Backbone(nn.Module):
    def __init__(self, config: EfficientNetConfig) -> None:
        super().__init__()
        self.embeddings = _Embeddings(config)
        self.encoder = _Encoder(config)


class EfficientNet(nn.Module):
    """Hugging Face EfficientNet image classifier, FP32 evaluation only.

    ``__call__(pixels)`` returns ``(N, num_labels)`` logits from NHWC pixels.
    ``forward_intermediates`` exposes stem, each block, top, pooling, and logits
    for parity diagnostics. BatchNorm running statistics remain in parameters,
    even though MLX freezes them in ``trainable_parameters``.
    """

    def __init__(self, config: Mapping[str, Any] | EfficientNetConfig) -> None:
        super().__init__()
        parsed = (
            config
            if isinstance(config, EfficientNetConfig)
            else EfficientNetConfig.from_dict(config)
        )
        self.config = parsed
        self.num_channels = parsed.num_channels
        self.hidden_dim = parsed.hidden_dim
        self.num_labels = parsed.num_labels
        self.efficientnet = _Backbone(parsed)
        self.classifier = nn.Linear(self.hidden_dim, self.num_labels)
        self._compiled_forward: Callable[[mx.array], mx.array] | None = None
        self.eval()

    def _forward(self, pixels: mx.array, intermediates: dict[str, mx.array] | None) -> mx.array:
        if self.training:
            raise ValueError("EfficientNet supports eval mode only; call model.eval()")
        if pixels.dtype not in (mx.float16, mx.float32, mx.bfloat16):
            raise ValueError("EfficientNet inputs must use a floating-point dtype")
        if pixels.ndim != 4 or pixels.shape[-1] != self.num_channels:
            raise ValueError(f"Expected NHWC pixels with {self.num_channels} channels")
        if min(pixels.shape[:3]) <= 0:
            raise ValueError("EfficientNet requires a nonempty batch and nonempty image")
        x = self.efficientnet.embeddings(pixels)
        if intermediates is not None:
            intermediates["stem"] = x
        x = self.efficientnet.encoder(x, intermediates)
        # HF pooling clips its one hidden_dim-sized window to the feature map.
        if x.shape[1] > self.hidden_dim or x.shape[2] > self.hidden_dim:
            raise ValueError("Feature map exceeds the HF EfficientNet pooling window")
        pooled = (
            mx.mean(x, axis=(1, 2))
            if self.config.pooling_type == "mean"
            else mx.max(x, axis=(1, 2))
        )
        logits = self.classifier(pooled)
        if intermediates is not None:
            intermediates.update(top=x, pooled=pooled, logits=logits)
        return logits

    def __call__(self, pixels: mx.array) -> mx.array:
        forward = self._compiled_forward or self._forward_logits
        return forward(pixels)

    def _forward_logits(self, pixels: mx.array) -> mx.array:
        return self._forward(pixels, None)

    def compile_forward(self) -> None:
        if self._compiled_forward is None:
            self._compiled_forward = compile_exclusive(self._forward_logits, inputs=self.state)

    def forward_intermediates(self, pixels: mx.array) -> dict[str, mx.array]:
        intermediates: dict[str, mx.array] = {}
        self._forward(pixels, intermediates)
        return intermediates
