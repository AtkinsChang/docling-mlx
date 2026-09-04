# Adapted from mlx-vlm (mlx_vlm/models/rt_detr_v2).
# SPDX-License-Identifier: MIT
"""Runtime subset of Hugging Face RT-DETR-v2 configuration.

Only fields consumed by the native inference model are represented. Unknown
Hugging Face fields deliberately remain ignorable: a checkpoint config, not a
Docling profile, defines the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any


def _tuple(value: object, name: str, item_type: type) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    if any(isinstance(item, bool) or not isinstance(item, item_type) for item in value):
        raise TypeError(f"{name} contains an invalid value")
    return tuple(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    return float(value)


def _size(value: object, name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    items = _tuple(value, name, int)
    if len(items) != 2 or any(item <= 0 for item in items):
        raise ValueError(f"{name} must contain two positive integers")
    return items[0], items[1]


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


@dataclass(frozen=True)
class RtDetrResNetConfig:
    """The ``rt_detr_resnet`` backbone fields used by RT-DETR-v2."""

    model_type: str = "rt_detr_resnet"
    depths: tuple[int, ...] = (3, 4, 6, 3)
    downsample_in_bottleneck: bool = False
    downsample_in_first_stage: bool = False
    embedding_size: int = 64
    hidden_act: str = "relu"
    hidden_sizes: tuple[int, ...] = (256, 512, 1024, 2048)
    layer_type: str = "bottleneck"
    num_channels: int = 3
    out_features: tuple[str, ...] = ("stage2", "stage3", "stage4")

    def __post_init__(self) -> None:
        if self.model_type != "rt_detr_resnet":
            raise ValueError("backbone_config.model_type must be 'rt_detr_resnet'")
        if len(self.depths) != len(self.hidden_sizes) or not self.depths:
            raise ValueError(
                "backbone_config depths and hidden_sizes must have the same nonzero length"
            )
        if any(depth <= 0 for depth in self.depths) or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("backbone_config depths and hidden_sizes must be positive")
        if self.embedding_size <= 0 or self.num_channels <= 0:
            raise ValueError("backbone_config embedding_size and num_channels must be positive")
        if self.layer_type not in {"basic", "bottleneck"}:
            raise ValueError("backbone_config.layer_type must be 'basic' or 'bottleneck'")
        if not self.out_features:
            raise ValueError("backbone_config.out_features must not be empty")
        stages = []
        for name in self.out_features:
            if not name.startswith("stage") or not name.removeprefix("stage").isdigit():
                raise ValueError("backbone_config.out_features must contain stage names")
            stages.append(int(name.removeprefix("stage")))
        if len(set(stages)) != len(stages) or any(
            stage < 1 or stage > len(self.depths) for stage in stages
        ):
            raise ValueError("backbone_config.out_features contains an invalid stage")

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> RtDetrResNetConfig:
        if not isinstance(raw, Mapping):
            raise TypeError("backbone_config must be a mapping")
        defaults = cls()
        return cls(
            model_type=str(raw.get("model_type", defaults.model_type)),
            depths=_tuple(raw.get("depths", defaults.depths), "backbone_config.depths", int),
            downsample_in_bottleneck=bool(
                raw.get("downsample_in_bottleneck", defaults.downsample_in_bottleneck)
            ),
            downsample_in_first_stage=bool(
                raw.get("downsample_in_first_stage", defaults.downsample_in_first_stage)
            ),
            embedding_size=_integer(
                raw.get("embedding_size", defaults.embedding_size), "embedding_size"
            ),
            hidden_act=str(raw.get("hidden_act", defaults.hidden_act)),
            hidden_sizes=_tuple(
                raw.get("hidden_sizes", defaults.hidden_sizes), "backbone_config.hidden_sizes", int
            ),
            layer_type=str(raw.get("layer_type", defaults.layer_type)),
            num_channels=_integer(raw.get("num_channels", defaults.num_channels), "num_channels"),
            out_features=_tuple(
                raw.get("out_features", defaults.out_features), "backbone_config.out_features", str
            ),
        )


@dataclass(frozen=True)
class RtDetrV2HybridEncoderConfig:
    encoder_hidden_dim: int
    encoder_in_channels: tuple[int, ...]
    feat_strides: tuple[int, ...]
    encoder_layers: int
    encoder_ffn_dim: int
    encoder_attention_heads: int
    encoder_activation_function: str
    encode_proj_layers: tuple[int, ...]
    positional_encoding_temperature: int
    activation_function: str
    normalize_before: bool
    layer_norm_eps: float
    batch_norm_eps: float
    hidden_expansion: float
    eval_size: tuple[int, int] | None


@dataclass(frozen=True)
class RtDetrV2TransformerConfig:
    d_model: int
    decoder_layers: int
    decoder_attention_heads: int
    decoder_ffn_dim: int
    decoder_in_channels: tuple[int, ...]
    decoder_activation_function: str
    decoder_method: str
    decoder_n_levels: int
    points_per_level: tuple[int, ...]
    decoder_offset_scale: float
    num_feature_levels: int
    num_queries: int
    num_labels: int
    learn_initial_query: bool
    layer_norm_eps: float
    with_box_refine: bool
    use_focal_loss: bool


@dataclass(frozen=True)
class RtDetrV2Config:
    """Config-driven native RT-DETR-v2 inference model."""

    model_type: str = "rt_detr_v2"
    num_labels: int = 2
    id2label: Mapping[int, str] = field(default_factory=lambda: {0: "LABEL_0", 1: "LABEL_1"})
    label2id: Mapping[str, int] = field(default_factory=lambda: {"LABEL_0": 0, "LABEL_1": 1})
    backbone_config: RtDetrResNetConfig = field(default_factory=RtDetrResNetConfig)
    d_model: int = 256
    encoder_hidden_dim: int = 256
    encoder_in_channels: tuple[int, ...] = (512, 1024, 2048)
    feat_strides: tuple[int, ...] = (8, 16, 32)
    encoder_layers: int = 1
    encoder_ffn_dim: int = 1024
    encoder_attention_heads: int = 8
    encoder_activation_function: str = "gelu"
    encode_proj_layers: tuple[int, ...] = (2,)
    positional_encoding_temperature: int = 10000
    activation_function: str = "silu"
    normalize_before: bool = False
    layer_norm_eps: float = 1e-5
    batch_norm_eps: float = 1e-5
    hidden_expansion: float = 1.0
    eval_size: tuple[int, int] | None = None
    num_queries: int = 300
    decoder_in_channels: tuple[int, ...] = (256, 256, 256)
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    decoder_ffn_dim: int = 1024
    decoder_activation_function: str = "relu"
    decoder_method: str = "default"
    decoder_n_levels: int = 3
    decoder_n_points: int | tuple[int, ...] = 4
    decoder_offset_scale: float = 0.5
    num_feature_levels: int = 3
    learn_initial_query: bool = False
    with_box_refine: bool = True
    use_focal_loss: bool = True
    anchor_image_size: tuple[int, int] | None = None
    hybrid_encoder_config: RtDetrV2HybridEncoderConfig = field(init=False, repr=False)
    transformer_config: RtDetrV2TransformerConfig = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "rt_detr_v2":
            raise ValueError("model_type must be 'rt_detr_v2'")
        if self.num_labels <= 0 or self.d_model <= 0 or self.encoder_hidden_dim <= 0:
            raise ValueError("model dimensions and num_labels must be positive")
        if len(self.encoder_in_channels) != len(self.feat_strides) or not self.encoder_in_channels:
            raise ValueError(
                "encoder_in_channels and feat_strides must have the same nonzero length"
            )
        if any(channel <= 0 for channel in self.encoder_in_channels) or any(
            stride <= 0 for stride in self.feat_strides
        ):
            raise ValueError("encoder_in_channels and feat_strides must be positive")
        if len(self.decoder_in_channels) != len(self.encoder_in_channels):
            raise ValueError("decoder_in_channels must match encoder_in_channels")
        if len(self.decoder_in_channels) > self.num_feature_levels:
            raise ValueError("decoder_in_channels cannot exceed num_feature_levels")
        if self.decoder_n_levels != self.num_feature_levels:
            raise ValueError("decoder_n_levels must equal num_feature_levels")
        if self.decoder_layers <= 0 or self.num_queries <= 0:
            raise ValueError("decoder_layers and num_queries must be positive")
        if (
            self.d_model % self.decoder_attention_heads
            or self.encoder_hidden_dim % self.encoder_attention_heads
        ):
            raise ValueError("model dimensions must be divisible by their attention head counts")
        if self.decoder_method not in {"default", "discrete"}:
            raise ValueError("decoder_method must be 'default' or 'discrete'")
        points = (
            (self.decoder_n_points,) * self.decoder_n_levels
            if isinstance(self.decoder_n_points, int)
            else self.decoder_n_points
        )
        if len(points) != self.decoder_n_levels or any(point <= 0 for point in points):
            raise ValueError("decoder_n_points must be positive for every feature level")
        if any(
            index < 0 or index >= len(self.encoder_in_channels) for index in self.encode_proj_layers
        ):
            raise ValueError("encode_proj_layers contains an invalid feature level")
        id2label, label2id = _labels(self.id2label, self.label2id, self.num_labels)
        object.__setattr__(self, "id2label", id2label)
        object.__setattr__(self, "label2id", label2id)
        object.__setattr__(
            self,
            "hybrid_encoder_config",
            RtDetrV2HybridEncoderConfig(
                encoder_hidden_dim=self.encoder_hidden_dim,
                encoder_in_channels=self.encoder_in_channels,
                feat_strides=self.feat_strides,
                encoder_layers=self.encoder_layers,
                encoder_ffn_dim=self.encoder_ffn_dim,
                encoder_attention_heads=self.encoder_attention_heads,
                encoder_activation_function=self.encoder_activation_function,
                encode_proj_layers=self.encode_proj_layers,
                positional_encoding_temperature=self.positional_encoding_temperature,
                activation_function=self.activation_function,
                normalize_before=self.normalize_before,
                layer_norm_eps=self.layer_norm_eps,
                batch_norm_eps=self.batch_norm_eps,
                hidden_expansion=self.hidden_expansion,
                eval_size=self.eval_size,
            ),
        )
        object.__setattr__(
            self,
            "transformer_config",
            RtDetrV2TransformerConfig(
                d_model=self.d_model,
                decoder_layers=self.decoder_layers,
                decoder_attention_heads=self.decoder_attention_heads,
                decoder_ffn_dim=self.decoder_ffn_dim,
                decoder_in_channels=self.decoder_in_channels,
                decoder_activation_function=self.decoder_activation_function,
                decoder_method=self.decoder_method,
                decoder_n_levels=self.decoder_n_levels,
                points_per_level=points,
                decoder_offset_scale=self.decoder_offset_scale,
                num_feature_levels=self.num_feature_levels,
                num_queries=self.num_queries,
                num_labels=self.num_labels,
                learn_initial_query=self.learn_initial_query,
                layer_norm_eps=self.layer_norm_eps,
                with_box_refine=self.with_box_refine,
                use_focal_loss=self.use_focal_loss,
            ),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> RtDetrV2Config:
        """Parse a Hugging Face-shaped config, ignoring fields this runtime does not use."""
        if not isinstance(raw, Mapping):
            raise TypeError("RT-DETR-v2 configuration must be a mapping")
        defaults = cls()
        backbone_raw = raw.get("backbone_config", {})
        if not isinstance(backbone_raw, Mapping):
            raise TypeError("backbone_config must be a mapping")
        raw_labels = raw.get("id2label")
        inferred_num_labels = (
            len(raw_labels) if isinstance(raw_labels, Mapping) else defaults.num_labels
        )
        num_labels = _integer(raw.get("num_labels", inferred_num_labels), "num_labels")
        id2label, label2id = _labels(raw.get("id2label"), raw.get("label2id"), num_labels)
        raw_points = raw.get("decoder_n_points", defaults.decoder_n_points)
        points: int | tuple[int, ...] = (
            _tuple(raw_points, "decoder_n_points", int)
            if isinstance(raw_points, (list, tuple))
            else _integer(raw_points, "decoder_n_points")
        )
        values: dict[str, Any] = {
            "model_type": str(raw.get("model_type", defaults.model_type)),
            "num_labels": num_labels,
            "id2label": id2label,
            "label2id": label2id,
            "backbone_config": RtDetrResNetConfig.from_dict(backbone_raw),
            "decoder_n_points": points,
            "eval_size": _size(raw.get("eval_size", defaults.eval_size), "eval_size"),
            "anchor_image_size": _size(
                raw.get("anchor_image_size", defaults.anchor_image_size), "anchor_image_size"
            ),
        }
        for name in (
            "encoder_in_channels",
            "feat_strides",
            "encode_proj_layers",
            "decoder_in_channels",
        ):
            values[name] = _tuple(raw.get(name, getattr(defaults, name)), name, int)
        for name in (
            "d_model",
            "encoder_hidden_dim",
            "encoder_layers",
            "encoder_ffn_dim",
            "encoder_attention_heads",
            "positional_encoding_temperature",
            "num_queries",
            "decoder_layers",
            "decoder_attention_heads",
            "decoder_ffn_dim",
            "decoder_n_levels",
            "num_feature_levels",
        ):
            values[name] = _integer(raw.get(name, getattr(defaults, name)), name)
        for name in (
            "layer_norm_eps",
            "batch_norm_eps",
            "hidden_expansion",
            "decoder_offset_scale",
        ):
            values[name] = _real(raw.get(name, getattr(defaults, name)), name)
        for name in (
            "encoder_activation_function",
            "activation_function",
            "decoder_activation_function",
            "decoder_method",
        ):
            values[name] = str(raw.get(name, getattr(defaults, name)))
        for name in (
            "normalize_before",
            "learn_initial_query",
            "with_box_refine",
            "use_focal_loss",
        ):
            values[name] = bool(raw.get(name, getattr(defaults, name)))
        return cls(**values)
