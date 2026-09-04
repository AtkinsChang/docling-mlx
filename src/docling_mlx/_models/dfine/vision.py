# Implemented after Hugging Face Transformers (transformers/models/d_fine and
# transformers/models/hgnet_v2); module structure, parameter names, and forward-pass
# order follow them so published checkpoints load unchanged.
# SPDX-License-Identifier: Apache-2.0
"""Native NHWC HGNetV2 backbone and D-FINE hybrid encoder."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from .config import DFineBackboneConfig, DFineConfig, DFineEncoderConfig


class _Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class _LearnableAffine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = mx.ones((1,))
        self.bias = mx.zeros((1,))

    def __call__(self, x: mx.array) -> mx.array:
        return self.scale * x + self.bias


class _HGNetV2ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        groups: int = 1,
        activation: str | None = "relu",
        use_learnable_affine: bool = False,
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.normalization = nn.BatchNorm(out_channels, eps=1e-5)
        self.activation = _activation(activation)
        self.lab: nn.Module = (
            _LearnableAffine() if activation is not None and use_learnable_affine else _Identity()
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.lab(self.activation(self.normalization(self.convolution(x))))


class _HGNetV2ConvLayerLight(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        activation: str,
        use_learnable_affine: bool,
    ) -> None:
        super().__init__()
        self.conv1 = _HGNetV2ConvLayer(
            in_channels,
            out_channels,
            1,
            activation=None,
            use_learnable_affine=use_learnable_affine,
        )
        self.conv2 = _HGNetV2ConvLayer(
            out_channels,
            out_channels,
            kernel_size,
            groups=out_channels,
            activation=activation,
            use_learnable_affine=use_learnable_affine,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv2(self.conv1(x))


class _HGNetV2Embeddings(nn.Module):
    def __init__(self, config: DFineBackboneConfig) -> None:
        super().__init__()
        stem = config.stem_channels
        affine = config.use_learnable_affine_block
        activation = config.hidden_act
        self.stem1 = _HGNetV2ConvLayer(
            stem[0],
            stem[1],
            3,
            stride=config.stem_strides[0],
            activation=activation,
            use_learnable_affine=affine,
        )
        self.stem2a = _HGNetV2ConvLayer(
            stem[1],
            stem[1] // 2,
            2,
            stride=config.stem_strides[1],
            activation=activation,
            use_learnable_affine=affine,
        )
        self.stem2b = _HGNetV2ConvLayer(
            stem[1] // 2,
            stem[1],
            2,
            stride=config.stem_strides[2],
            activation=activation,
            use_learnable_affine=affine,
        )
        self.stem3 = _HGNetV2ConvLayer(
            stem[1] * 2,
            stem[1],
            3,
            stride=config.stem_strides[3],
            activation=activation,
            use_learnable_affine=affine,
        )
        self.stem4 = _HGNetV2ConvLayer(
            stem[1],
            stem[2],
            1,
            stride=config.stem_strides[4],
            activation=activation,
            use_learnable_affine=affine,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.num_channels = config.num_channels

    def __call__(self, pixel_values: mx.array) -> mx.array:
        if pixel_values.ndim != 4 or pixel_values.shape[-1] != self.num_channels:
            raise ValueError(f"Expected NHWC input with {self.num_channels} channels")
        embedding = self.stem1(pixel_values)
        embedding = mx.pad(embedding, [(0, 0), (0, 1), (0, 1), (0, 0)])
        stem2 = self.stem2a(embedding)
        stem2 = mx.pad(stem2, [(0, 0), (0, 1), (0, 1), (0, 0)])
        stem2 = self.stem2b(stem2)
        embedding = mx.concatenate([self.pool(embedding), stem2], axis=-1)
        return self.stem4(self.stem3(embedding))


class _HGNetV2BasicLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: int,
        *,
        residual: bool,
        light_block: bool,
        activation: str,
        use_learnable_affine: bool,
    ) -> None:
        super().__init__()
        layer_type = _HGNetV2ConvLayerLight if light_block else _HGNetV2ConvLayer
        self.layers = [
            layer_type(
                in_channels if index == 0 else mid_channels,
                mid_channels,
                kernel_size,
                activation=activation,
                use_learnable_affine=use_learnable_affine,
            )
            for index in range(num_layers)
        ]
        total_channels = in_channels + num_layers * mid_channels
        self.aggregation = [
            _HGNetV2ConvLayer(
                total_channels,
                out_channels // 2,
                1,
                activation=activation,
                use_learnable_affine=use_learnable_affine,
            ),
            _HGNetV2ConvLayer(
                out_channels // 2,
                out_channels,
                1,
                activation=activation,
                use_learnable_affine=use_learnable_affine,
            ),
        ]
        self.residual = residual

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        outputs = [x]
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        x = mx.concatenate(outputs, axis=-1)
        for layer in self.aggregation:
            x = layer(x)
        return x + identity if self.residual else x


class _HGNetV2Stage(nn.Module):
    def __init__(self, config: DFineBackboneConfig, index: int) -> None:
        super().__init__()
        in_channels = config.stage_in_channels[index]
        out_channels = config.stage_out_channels[index]
        self.downsample: nn.Module = (
            _HGNetV2ConvLayer(
                in_channels,
                in_channels,
                3,
                stride=config.stage_downsample_strides[index],
                groups=in_channels,
                activation=None,
            )
            if config.stage_downsample[index]
            else _Identity()
        )
        self.blocks = [
            _HGNetV2BasicLayer(
                in_channels if block == 0 else out_channels,
                config.stage_mid_channels[index],
                out_channels,
                config.stage_num_layers[index],
                config.stage_kernel_size[index],
                residual=block != 0,
                light_block=config.stage_light_block[index],
                activation=config.hidden_act,
                use_learnable_affine=config.use_learnable_affine_block,
            )
            for block in range(config.stage_num_blocks[index])
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class _HGNetV2Encoder(nn.Module):
    def __init__(self, config: DFineBackboneConfig) -> None:
        super().__init__()
        self.stages = [_HGNetV2Stage(config, index) for index in range(4)]

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return tuple(outputs)


class HGNetV2Backbone(nn.Module):
    """Configured HGNetV2 backbone returning its selected stages."""

    def __init__(self, config: DFineBackboneConfig) -> None:
        super().__init__()
        self.embedder = _HGNetV2Embeddings(config)
        self.encoder = _HGNetV2Encoder(config)
        self._out_indices = tuple(index - 1 for index in config.out_indices)
        self.out_channels = tuple(config.stage_out_channels[index] for index in self._out_indices)
        stride = 1
        for stem_stride in config.stem_strides:
            stride *= stem_stride
        stage_strides = []
        for downsample, downsample_stride in zip(
            config.stage_downsample, config.stage_downsample_strides, strict=True
        ):
            if downsample:
                stride *= downsample_stride
            stage_strides.append(stride)
        self.out_strides = tuple(stage_strides[index] for index in self._out_indices)

    def __call__(self, pixel_values: mx.array) -> tuple[mx.array, ...]:
        stages = self.encoder(self.embedder(pixel_values))
        return tuple(stages[index] for index in self._out_indices)


class _EncoderInputProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, eps: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm(out_channels, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(self.conv(x))


class _DFineConvNormLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        *,
        groups: int = 1,
        padding: int | None = None,
        activation: str | None = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=False,
        )
        self.norm = nn.BatchNorm(out_channels, eps=eps)
        self.activation = _activation(activation)

    def __call__(self, x: mx.array) -> mx.array:
        return self.activation(self.norm(self.conv(x)))


class _DFineRepVggBlock(nn.Module):
    def __init__(self, channels: int, activation: str, eps: float) -> None:
        super().__init__()
        self.conv1 = _DFineConvNormLayer(channels, channels, 3, 1, padding=1, eps=eps)
        self.conv2 = _DFineConvNormLayer(channels, channels, 1, 1, padding=0, eps=eps)
        self.activation = _activation(activation)

    def __call__(self, x: mx.array) -> mx.array:
        return self.activation(self.conv1(x) + self.conv2(x))


class _DFineCSPRepLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        activation: str,
        eps: float,
    ) -> None:
        super().__init__()
        self.conv1 = _DFineConvNormLayer(
            in_channels, out_channels, 1, 1, activation=activation, eps=eps
        )
        self.conv2 = _DFineConvNormLayer(
            in_channels, out_channels, 1, 1, activation=activation, eps=eps
        )
        self.bottlenecks = [
            _DFineRepVggBlock(out_channels, activation, eps) for _ in range(num_blocks)
        ]
        self.conv3: nn.Module = _Identity()

    def __call__(self, x: mx.array) -> mx.array:
        branch = self.conv1(x)
        for block in self.bottlenecks:
            branch = block(branch)
        return self.conv3(branch + self.conv2(x))


class _DFineRepNCSPELAN4(nn.Module):
    def __init__(self, config: DFineEncoderConfig, num_blocks: int) -> None:
        super().__init__()
        hidden = config.hidden_dim
        inner = round(config.hidden_expansion * hidden // 2)
        activation = config.outer_activation_function
        eps = config.batch_norm_eps
        self.conv_dim = hidden
        self.conv1 = _DFineConvNormLayer(
            hidden * 2, hidden * 2, 1, 1, activation=activation, eps=eps
        )
        self.csp_rep1 = _DFineCSPRepLayer(hidden, inner, num_blocks, activation, eps)
        self.conv2 = _DFineConvNormLayer(inner, inner, 3, 1, activation=activation, eps=eps)
        self.csp_rep2 = _DFineCSPRepLayer(inner, inner, num_blocks, activation, eps)
        self.conv3 = _DFineConvNormLayer(inner, inner, 3, 1, activation=activation, eps=eps)
        self.conv4 = _DFineConvNormLayer(
            hidden * 2 + inner * 2,
            hidden,
            1,
            1,
            activation=activation,
            eps=eps,
        )

    def __call__(self, x: mx.array) -> mx.array:
        left, right = mx.split(self.conv1(x), 2, axis=-1)
        branch1 = self.conv2(self.csp_rep1(right))
        branch2 = self.conv3(self.csp_rep2(branch1))
        return self.conv4(mx.concatenate([left, right, branch1, branch2], axis=-1))


class _DFineSCDown(nn.Module):
    def __init__(self, config: DFineEncoderConfig) -> None:
        super().__init__()
        channels = config.hidden_dim
        self.conv1 = _DFineConvNormLayer(channels, channels, 1, 1, eps=config.batch_norm_eps)
        self.conv2 = _DFineConvNormLayer(
            channels,
            channels,
            3,
            2,
            groups=channels,
            eps=config.batch_norm_eps,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv2(self.conv1(x))


class _DFineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = [nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, output_dim)]
        self.act = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.layers[1](self.act(self.layers[0](x)))


class _DFineSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def __call__(self, x: mx.array, position: mx.array | None) -> mx.array:
        batch, length, hidden = x.shape
        query_key = x + position if position is not None else x
        shape = (batch, length, self.heads, self.head_dim)
        query = self.q_proj(query_key).reshape(shape).transpose(0, 2, 1, 3)
        key = self.k_proj(query_key).reshape(shape).transpose(0, 2, 1, 3)
        value = self.v_proj(x).reshape(shape).transpose(0, 2, 1, 3)
        weights = mx.softmax((query @ key.transpose(0, 1, 3, 2)) * self.scaling, axis=-1)
        output = (weights @ value).transpose(0, 2, 1, 3).reshape(batch, length, hidden)
        return self.o_proj(output)


class _DFineEncoderLayer(nn.Module):
    def __init__(self, config: DFineEncoderConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.normalize_before = config.normalize_before
        self.self_attn = _DFineSelfAttention(hidden, config.attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(hidden, eps=config.layer_norm_eps)
        self.mlp = _DFineMLP(hidden, config.ffn_dim, hidden)
        self.final_layer_norm = nn.LayerNorm(hidden, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, position: mx.array) -> mx.array:
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x = residual + self.self_attn(x, position)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if self.normalize_before:
            x = self.final_layer_norm(x)
        residual = x
        x = residual + self.mlp(x)
        return x if self.normalize_before else self.final_layer_norm(x)


class _DFineSinePositionEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, temperature: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def __call__(self, height: int, width: int, dtype: mx.Dtype) -> mx.array:
        grid_w, grid_h = mx.meshgrid(
            mx.arange(width, dtype=dtype), mx.arange(height, dtype=dtype), indexing="xy"
        )
        position_dim = self.hidden_dim // 4
        omega = 1.0 / (self.temperature ** (mx.arange(position_dim, dtype=dtype) / position_dim))
        out_w = grid_w.flatten()[:, None] * omega[None]
        out_h = grid_h.flatten()[:, None] * omega[None]
        return mx.concatenate(
            [mx.sin(out_h), mx.cos(out_h), mx.sin(out_w), mx.cos(out_w)], axis=-1
        )[None]


class _DFineAIFI(nn.Module):
    def __init__(self, config: DFineEncoderConfig) -> None:
        super().__init__()
        self.position_embedding = _DFineSinePositionEmbedding(
            config.hidden_dim, config.positional_encoding_temperature
        )
        self.layers = [_DFineEncoderLayer(config) for _ in range(config.layers)]

    def __call__(self, x: mx.array) -> mx.array:
        batch, height, width, channels = x.shape
        flattened = x.reshape(batch, height * width, channels)
        position = self.position_embedding(height, width, x.dtype)
        for layer in self.layers:
            flattened = layer(flattened, position)
        return flattened.reshape(batch, height, width, channels)


class DFineHybridEncoder(nn.Module):
    """D-FINE AIFI, top-down FPN, and bottom-up PAN in NHWC layout."""

    def __init__(self, config: DFineEncoderConfig) -> None:
        super().__init__()
        self.encode_proj_layers = config.projection_layers
        self.aifi = [_DFineAIFI(config) for _ in self.encode_proj_layers]
        stages = len(config.in_channels) - 1
        self.lateral_convs = [
            _DFineConvNormLayer(
                config.hidden_dim,
                config.hidden_dim,
                1,
                1,
                eps=config.batch_norm_eps,
            )
            for _ in range(stages)
        ]
        blocks = round(3 * config.depth_mult)
        self.fpn_blocks = [_DFineRepNCSPELAN4(config, blocks) for _ in range(stages)]
        self.downsample_convs = [_DFineSCDown(config) for _ in range(stages)]
        self.pan_blocks = [_DFineRepNCSPELAN4(config, blocks) for _ in range(stages)]
        self.out_channels = (config.hidden_dim,) * len(config.in_channels)
        self.out_strides = config.feature_strides

    def __call__(self, features: tuple[mx.array, ...]) -> tuple[mx.array, ...]:
        if len(features) != len(self.out_channels) or any(
            feature.ndim != 4 or feature.shape[-1] != self.out_channels[index]
            for index, feature in enumerate(features)
        ):
            raise ValueError("Expected configured NHWC feature maps with encoder hidden channels")
        feature_maps = list(features)
        for index, level in enumerate(self.encode_proj_layers):
            feature_maps[level] = self.aifi[index](feature_maps[level])

        fpn = [feature_maps[-1]]
        for index, (lateral, block) in enumerate(
            zip(self.lateral_convs, self.fpn_blocks, strict=True)
        ):
            top = lateral(fpn[-1])
            fpn[-1] = top
            fused = mx.concatenate([_upsample_nearest_2x(top), feature_maps[-2 - index]], axis=-1)
            fpn.append(block(fused))
        fpn.reverse()

        pan = [fpn[0]]
        for index, (downsample, block) in enumerate(
            zip(self.downsample_convs, self.pan_blocks, strict=True)
        ):
            fused = mx.concatenate([downsample(pan[-1]), fpn[index + 1]], axis=-1)
            pan.append(block(fused))
        return tuple(pan)


class DFineVisionTower(nn.Module):
    """HGNetV2, the three source projections, and the D-FINE encoder."""

    def __init__(self, config: DFineConfig) -> None:
        super().__init__()
        self.backbone = HGNetV2Backbone(config.backbone)
        self._backbone_forward: Callable[[mx.array], tuple[mx.array, ...]] | None = None
        self.encoder_input_proj = [
            _EncoderInputProjection(
                in_channels,
                config.encoder.hidden_dim,
                config.encoder.batch_norm_eps,
            )
            for in_channels in config.encoder.in_channels
        ]
        self.encoder = DFineHybridEncoder(config.encoder)
        self.out_channels = self.encoder.out_channels
        self.out_strides = self.encoder.out_strides

    def __call__(self, pixel_values: mx.array) -> tuple[mx.array, ...]:
        backbone = self._backbone_forward or self.backbone
        features = backbone(pixel_values)
        projected = tuple(
            projection(feature)
            for projection, feature in zip(self.encoder_input_proj, features, strict=True)
        )
        return self.encoder(projected)

    def compile_backbone(self) -> None:
        if self._backbone_forward is None:
            self._backbone_forward = mx.compile(self.backbone, inputs=self.backbone.state)


def _upsample_nearest_2x(x: mx.array) -> mx.array:
    batch, height, width, channels = x.shape
    expanded = mx.broadcast_to(x[:, :, None, :, None, :], (batch, height, 2, width, 2, channels))
    return expanded.reshape(batch, height * 2, width * 2, channels)


def _activation(name: str | None) -> nn.Module:
    if name is None:
        return _Identity()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported D-FINE activation: {name!r}")
