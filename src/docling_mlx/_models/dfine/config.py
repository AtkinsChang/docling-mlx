# Implemented after Hugging Face Transformers (transformers/models/d_fine); module
# structure, parameter names, and forward-pass order follow it so published
# checkpoints load unchanged.
# SPDX-License-Identifier: Apache-2.0
"""Runtime subset of Hugging Face D-FINE configuration.

Checkpoint configuration defines the native inference graph. Fields not used
by inference deliberately remain ignorable, matching Transformers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    return float(value)


def _tuple(value: object, name: str, item_type: type) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    if any(isinstance(item, bool) or not isinstance(item, item_type) for item in value):
        raise TypeError(f"{name} contains an invalid value")
    return tuple(value)


def _size(value: object, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, Integral) and not isinstance(value, bool):
        value = (value, value)
    values = _tuple(value, name, int)
    if len(values) != 2 or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain two positive integers")
    return values[0], values[1]


def _labels(
    raw_id2label: object, raw_label2id: object, num_labels: int
) -> tuple[dict[int, str], dict[str, int]]:
    if raw_id2label is None:
        id2label = {index: f"LABEL_{index}" for index in range(num_labels)}
    else:
        if not isinstance(raw_id2label, Mapping):
            raise TypeError("id2label must be a mapping")
        id2label = {}
        for key, value in raw_id2label.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as error:
                raise TypeError("id2label keys must be integer identifiers") from error
            if str(index) != str(key) or not isinstance(value, str):
                raise TypeError("id2label must map canonical integer identifiers to strings")
            id2label[index] = value
    if set(id2label) != set(range(num_labels)):
        raise ValueError("id2label must define exactly one label for every label id")
    inverse = {label: index for index, label in id2label.items()}
    if len(inverse) != len(id2label):
        raise ValueError("id2label names must be unique")
    if raw_label2id is not None:
        if not isinstance(raw_label2id, Mapping):
            raise TypeError("label2id must be a mapping")
        parsed = {
            str(label): _integer(index, "label2id value") for label, index in raw_label2id.items()
        }
        if parsed != inverse:
            raise ValueError("label2id must be the inverse of id2label")
    return id2label, inverse


@dataclass(frozen=True, slots=True)
class DFineBackboneConfig:
    """The HGNetV2 configuration represented by supported D-FINE checkpoints."""

    model_type: str = "hgnet_v2"
    num_channels: int = 3
    embedding_size: int = 64
    depths: tuple[int, ...] = (3, 4, 6, 3)
    hidden_sizes: tuple[int, ...] = (256, 512, 1024, 2048)
    hidden_act: str = "relu"
    out_features: tuple[str, ...] = ("stage2", "stage3", "stage4")
    out_indices: tuple[int, ...] = (2, 3, 4)
    stem_channels: tuple[int, ...] = (3, 32, 48)
    stem_strides: tuple[int, ...] = (2, 1, 1, 2, 1)
    stage_in_channels: tuple[int, ...] = (48, 128, 512, 1024)
    stage_mid_channels: tuple[int, ...] = (48, 96, 192, 384)
    stage_out_channels: tuple[int, ...] = (128, 512, 1024, 2048)
    stage_num_blocks: tuple[int, ...] = (1, 1, 3, 1)
    stage_downsample: tuple[bool, ...] = (False, True, True, True)
    stage_downsample_strides: tuple[int, ...] = (2, 2, 2, 2)
    stage_light_block: tuple[bool, ...] = (False, False, True, True)
    stage_kernel_size: tuple[int, ...] = (3, 3, 5, 5)
    stage_num_layers: tuple[int, ...] = (6, 6, 6, 6)
    use_learnable_affine_block: bool = False

    def __post_init__(self) -> None:
        if self.model_type != "hgnet_v2":
            raise ValueError("backbone_config.model_type must be 'hgnet_v2'")
        if self.num_channels <= 0 or len(self.stem_channels) != 3:
            raise ValueError(
                "backbone_config channels must be positive and stem_channels must have length 3"
            )
        integer_fields = (
            self.depths,
            self.hidden_sizes,
            self.stage_in_channels,
            self.stage_mid_channels,
            self.stage_out_channels,
            self.stage_num_blocks,
            self.stage_downsample_strides,
            self.stage_kernel_size,
            self.stage_num_layers,
        )
        if any(
            len(item) != 4
            for item in (*integer_fields, self.stage_downsample, self.stage_light_block)
        ):
            raise ValueError("HGNetV2 inference requires four configured stages")
        if any(item <= 0 for values in integer_fields for item in values):
            raise ValueError("backbone dimensions and stage counts must be positive")
        if (
            self.stem_channels[0] != self.num_channels
            or self.stage_in_channels[0] != self.stem_channels[-1]
            or any(
                before != after
                for before, after in zip(
                    self.stage_out_channels[:-1], self.stage_in_channels[1:], strict=True
                )
            )
        ):
            raise ValueError("HGNetV2 stage channels must form a continuous backbone")
        if not self.out_indices or any(index < 1 or index > 4 for index in self.out_indices):
            raise ValueError("backbone_config.out_indices must select configured stages")

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> DFineBackboneConfig:
        if not isinstance(raw, Mapping):
            raise TypeError("backbone_config must be a mapping")
        defaults = cls()
        values: dict[str, Any] = {
            "model_type": str(raw.get("model_type", defaults.model_type)),
            "num_channels": _integer(
                raw.get("num_channels", defaults.num_channels), "num_channels"
            ),
            "embedding_size": _integer(
                raw.get("embedding_size", defaults.embedding_size), "embedding_size"
            ),
            "hidden_act": str(raw.get("hidden_act", defaults.hidden_act)),
            "use_learnable_affine_block": bool(
                raw.get("use_learnable_affine_block", defaults.use_learnable_affine_block)
            ),
        }
        for name in (
            "depths",
            "hidden_sizes",
            "out_indices",
            "stem_channels",
            "stem_strides",
            "stage_in_channels",
            "stage_mid_channels",
            "stage_out_channels",
            "stage_num_blocks",
            "stage_downsample_strides",
            "stage_kernel_size",
        ):
            values[name] = _tuple(raw.get(name, getattr(defaults, name)), name, int)
        values["stage_num_layers"] = _tuple(
            raw.get("stage_numb_of_layers", defaults.stage_num_layers),
            "stage_numb_of_layers",
            int,
        )
        values["out_features"] = _tuple(
            raw.get("out_features", defaults.out_features), "out_features", str
        )
        for name in ("stage_downsample", "stage_light_block"):
            value = raw.get(name, getattr(defaults, name))
            if not isinstance(value, (list, tuple)) or any(
                type(item) is not bool for item in value
            ):
                raise TypeError(f"{name} must contain booleans")
            values[name] = tuple(value)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class DFineEncoderConfig:
    hidden_dim: int
    in_channels: tuple[int, ...]
    feature_strides: tuple[int, ...]
    layers: int
    ffn_dim: int
    attention_heads: int
    activation_function: str
    projection_layers: tuple[int, ...]
    positional_encoding_temperature: int
    outer_activation_function: str
    normalize_before: bool
    layer_norm_eps: float
    hidden_expansion: float
    batch_norm_eps: float
    depth_mult: float


@dataclass(frozen=True, slots=True)
class DFineDecoderConfig:
    hidden_dim: int
    layers: int
    attention_heads: int
    ffn_dim: int
    in_channels: tuple[int, ...]
    activation_function: str
    method: str
    points_per_level: tuple[int, ...]
    offset_scale: float
    num_feature_levels: int
    num_queries: int
    max_num_bins: int
    reg_scale: float
    up: float
    top_prob_values: int
    lqe_hidden_dim: int
    lqe_layers: int
    eval_idx: int = -1
    layer_scale: float = 1.0


def _default_encoder() -> DFineEncoderConfig:
    return DFineEncoderConfig(
        256,
        (512, 1024, 2048),
        (8, 16, 32),
        1,
        1024,
        8,
        "gelu",
        (2,),
        10000,
        "silu",
        False,
        1e-5,
        1.0,
        1e-5,
        1.0,
    )


def _default_decoder() -> DFineDecoderConfig:
    return DFineDecoderConfig(
        256,
        6,
        8,
        1024,
        (256, 256, 256),
        "relu",
        "default",
        (4, 4, 4),
        0.5,
        3,
        300,
        32,
        4.0,
        0.5,
        4,
        64,
        2,
        -1,
        1.0,
    )


@dataclass(frozen=True, slots=True)
class DFineConfig:
    """Config-driven native D-FINE evaluation model."""

    model_type: str = "d_fine"
    num_labels: int = 2
    id2label: dict[int, str] = field(default_factory=lambda: {0: "LABEL_0", 1: "LABEL_1"})
    label2id: dict[str, int] = field(default_factory=lambda: {"LABEL_0": 0, "LABEL_1": 1})
    backbone: DFineBackboneConfig = field(default_factory=DFineBackboneConfig)
    encoder: DFineEncoderConfig = field(default_factory=_default_encoder)
    decoder: DFineDecoderConfig = field(default_factory=_default_decoder)
    anchor_image_size: tuple[int, int] | None = None
    eval_size: tuple[int, int] | None = None
    use_focal_loss: bool = True
    reg_max: int | None = None

    def __post_init__(self) -> None:
        if self.model_type != "d_fine":
            raise ValueError("model_type must be 'd_fine'")
        if self.num_labels <= 0:
            raise ValueError("num_labels must be positive")
        id2label, label2id = _labels(self.id2label, self.label2id, self.num_labels)
        object.__setattr__(self, "id2label", id2label)
        object.__setattr__(self, "label2id", label2id)
        encoder, decoder = self.encoder, self.decoder
        if (
            encoder.hidden_dim <= 0
            or decoder.hidden_dim <= 0
            or decoder.layers <= 0
            or decoder.num_queries <= 0
            or decoder.max_num_bins <= 0
        ):
            raise ValueError("D-FINE dimensions, queries, layers, and bins must be positive")
        if (
            len(encoder.in_channels) != len(encoder.feature_strides)
            or len(decoder.in_channels) != len(encoder.in_channels)
            or len(decoder.in_channels) > decoder.num_feature_levels
        ):
            raise ValueError("D-FINE feature level configuration is inconsistent")
        selected = tuple(
            self.backbone.stage_out_channels[index - 1] for index in self.backbone.out_indices
        )
        if selected != encoder.in_channels:
            raise ValueError("encoder_in_channels must match selected HGNetV2 backbone outputs")
        if (
            encoder.hidden_dim % encoder.attention_heads
            or decoder.hidden_dim % decoder.attention_heads
        ):
            raise ValueError("D-FINE hidden dimensions must divide attention heads")
        if any(
            index < 0 or index >= len(encoder.in_channels) for index in encoder.projection_layers
        ):
            raise ValueError("encode_proj_layers contains an invalid feature level")
        if decoder.method not in {"default", "discrete"}:
            raise ValueError("decoder_method must be 'default' or 'discrete'")
        if len(decoder.points_per_level) != decoder.num_feature_levels or any(
            point <= 0 for point in decoder.points_per_level
        ):
            raise ValueError("decoder_n_points must be positive for every feature level")
        eval_idx = decoder.eval_idx if decoder.eval_idx >= 0 else decoder.layers + decoder.eval_idx
        if eval_idx < 0 or eval_idx >= decoder.layers:
            raise ValueError("eval_idx must select a decoder layer")

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.id2label[index] for index in range(self.num_labels))

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> DFineConfig:
        """Parse a Hugging Face-shaped D-FINE config, ignoring unknown keys."""

        if not isinstance(raw, Mapping):
            raise TypeError("D-FINE configuration must be a mapping")
        defaults = cls()
        backbone_raw = raw.get("backbone_config", {})
        if not isinstance(backbone_raw, Mapping):
            raise TypeError("backbone_config must be a mapping")
        raw_labels = raw.get("id2label")
        inferred_labels = (
            len(raw_labels) if isinstance(raw_labels, Mapping) else defaults.num_labels
        )
        num_labels = _integer(raw.get("num_labels", inferred_labels), "num_labels")
        id2label, label2id = _labels(raw.get("id2label"), raw.get("label2id"), num_labels)
        raw_points = raw.get("decoder_n_points", defaults.decoder.points_per_level[0])
        points = (
            _tuple(raw_points, "decoder_n_points", int)
            if isinstance(raw_points, (list, tuple))
            else (_integer(raw_points, "decoder_n_points"),)
        )
        feature_levels = _integer(
            raw.get("num_feature_levels", defaults.decoder.num_feature_levels), "num_feature_levels"
        )
        if len(points) == 1:
            points *= feature_levels
        encoder = DFineEncoderConfig(
            _integer(
                raw.get("encoder_hidden_dim", defaults.encoder.hidden_dim), "encoder_hidden_dim"
            ),
            _tuple(
                raw.get("encoder_in_channels", defaults.encoder.in_channels),
                "encoder_in_channels",
                int,
            ),
            _tuple(raw.get("feat_strides", defaults.encoder.feature_strides), "feat_strides", int),
            _integer(raw.get("encoder_layers", defaults.encoder.layers), "encoder_layers"),
            _integer(raw.get("encoder_ffn_dim", defaults.encoder.ffn_dim), "encoder_ffn_dim"),
            _integer(
                raw.get("encoder_attention_heads", defaults.encoder.attention_heads),
                "encoder_attention_heads",
            ),
            str(raw.get("encoder_activation_function", defaults.encoder.activation_function)),
            _tuple(
                raw.get("encode_proj_layers", defaults.encoder.projection_layers),
                "encode_proj_layers",
                int,
            ),
            _integer(
                raw.get(
                    "positional_encoding_temperature",
                    defaults.encoder.positional_encoding_temperature,
                ),
                "positional_encoding_temperature",
            ),
            str(raw.get("activation_function", defaults.encoder.outer_activation_function)),
            bool(raw.get("normalize_before", defaults.encoder.normalize_before)),
            _real(raw.get("layer_norm_eps", defaults.encoder.layer_norm_eps), "layer_norm_eps"),
            _real(
                raw.get("hidden_expansion", defaults.encoder.hidden_expansion), "hidden_expansion"
            ),
            _real(raw.get("batch_norm_eps", defaults.encoder.batch_norm_eps), "batch_norm_eps"),
            _real(raw.get("depth_mult", defaults.encoder.depth_mult), "depth_mult"),
        )
        decoder = DFineDecoderConfig(
            _integer(raw.get("d_model", defaults.decoder.hidden_dim), "d_model"),
            _integer(raw.get("decoder_layers", defaults.decoder.layers), "decoder_layers"),
            _integer(
                raw.get("decoder_attention_heads", defaults.decoder.attention_heads),
                "decoder_attention_heads",
            ),
            _integer(raw.get("decoder_ffn_dim", defaults.decoder.ffn_dim), "decoder_ffn_dim"),
            _tuple(
                raw.get("decoder_in_channels", defaults.decoder.in_channels),
                "decoder_in_channels",
                int,
            ),
            str(raw.get("decoder_activation_function", defaults.decoder.activation_function)),
            str(raw.get("decoder_method", defaults.decoder.method)),
            points,
            _real(
                raw.get("decoder_offset_scale", defaults.decoder.offset_scale),
                "decoder_offset_scale",
            ),
            feature_levels,
            _integer(raw.get("num_queries", defaults.decoder.num_queries), "num_queries"),
            _integer(
                raw.get("max_num_bins", raw.get("reg_max", defaults.decoder.max_num_bins)),
                "max_num_bins",
            ),
            _real(raw.get("reg_scale", defaults.decoder.reg_scale), "reg_scale"),
            _real(raw.get("up", defaults.decoder.up), "up"),
            _integer(
                raw.get("top_prob_values", defaults.decoder.top_prob_values), "top_prob_values"
            ),
            _integer(raw.get("lqe_hidden_dim", defaults.decoder.lqe_hidden_dim), "lqe_hidden_dim"),
            _integer(raw.get("lqe_layers", defaults.decoder.lqe_layers), "lqe_layers"),
            _integer(raw.get("eval_idx", defaults.decoder.eval_idx), "eval_idx"),
            _real(raw.get("layer_scale", defaults.decoder.layer_scale), "layer_scale"),
        )
        return cls(
            model_type=str(raw.get("model_type", defaults.model_type)),
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            backbone=DFineBackboneConfig.from_dict(backbone_raw),
            encoder=encoder,
            decoder=decoder,
            anchor_image_size=_size(raw.get("anchor_image_size"), "anchor_image_size"),
            eval_size=_size(raw.get("eval_size"), "eval_size"),
            use_focal_loss=bool(raw.get("use_focal_loss", defaults.use_focal_loss)),
            reg_max=(
                _integer(raw["reg_max"], "reg_max")
                if "reg_max" in raw and raw["reg_max"] is not None
                else None
            ),
        )
