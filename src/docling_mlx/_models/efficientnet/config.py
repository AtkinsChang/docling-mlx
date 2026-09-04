# Implemented after Hugging Face Transformers (transformers/models/efficientnet);
# module structure, parameter names, and forward-pass order follow it so published
# checkpoints load unchanged.
# SPDX-License-Identifier: Apache-2.0
"""The runtime subset of Hugging Face's EfficientNet configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any


def _scaled_filters(width_coefficient: float, depth_divisor: int, channels: int) -> int:
    scaled = channels * width_coefficient
    rounded = max(
        depth_divisor,
        int(scaled + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    return int(rounded + depth_divisor if rounded < 0.9 * scaled else rounded)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"EfficientNet {name} must be an integer")
    return int(value)


def _label_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (Integral, str)):
        raise TypeError("EfficientNet label2id values must be integer identifiers")
    try:
        result = int(value)
    except ValueError as error:
        raise TypeError("EfficientNet label2id values must be integer identifiers") from error
    if str(result) != str(value) and value != result:
        raise TypeError("EfficientNet label2id values must be canonical identifiers")
    return result


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"EfficientNet {name} must be a real number")
    return float(value)


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"EfficientNet {name} must be a list")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
        raise TypeError(f"EfficientNet {name} contains an invalid value")
    return tuple(int(item) for item in value)


def _image_size(value: object, name: str) -> tuple[int, int]:
    if isinstance(value, Integral) and not isinstance(value, bool):
        result = (int(value), int(value))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        result = (_integer(value[0], name), _integer(value[1], name))
    else:
        raise TypeError(f"EfficientNet {name} must be an integer or a two-item list")
    if any(item <= 0 for item in result):
        raise ValueError(f"EfficientNet {name} must be positive")
    return result


def _labels(raw: Mapping[str, Any], num_labels: int) -> tuple[dict[int, str], dict[str, int]]:
    raw_id2label = raw.get("id2label")
    if raw_id2label is None:
        id2label = {index: f"LABEL_{index}" for index in range(num_labels)}
    elif isinstance(raw_id2label, Mapping):
        id2label = {}
        for raw_index, label in raw_id2label.items():
            if isinstance(raw_index, bool):
                raise TypeError("EfficientNet id2label keys must be integer identifiers")
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as error:
                raise TypeError("EfficientNet id2label keys must be integer identifiers") from error
            if str(index) != str(raw_index) and raw_index != index:
                raise TypeError("EfficientNet id2label keys must be canonical identifiers")
            if not isinstance(label, str) or not label:
                raise TypeError("EfficientNet id2label values must be nonempty strings")
            id2label[index] = label
    else:
        raise TypeError("EfficientNet id2label must be a mapping")

    if set(id2label) != set(range(num_labels)):
        raise ValueError("EfficientNet id2label must define every label id")
    label2id = {label: index for index, label in id2label.items()}

    raw_label2id = raw.get("label2id")
    if raw_label2id is not None:
        if not isinstance(raw_label2id, Mapping):
            raise TypeError("EfficientNet label2id must be a mapping")
        parsed = {str(label): _label_id(index) for label, index in raw_label2id.items()}
        if parsed != label2id:
            raise ValueError("EfficientNet label2id must be the inverse of id2label")
    return id2label, label2id


@dataclass(frozen=True, slots=True)
class EfficientNetConfig:
    """Config-driven EfficientNet topology and classification labels."""

    model_type: str
    image_size: tuple[int, int]
    num_channels: int
    width_coefficient: float
    depth_coefficient: float
    depth_divisor: int
    in_channels: tuple[int, ...]
    out_channels: tuple[int, ...]
    kernel_sizes: tuple[int, ...]
    strides: tuple[int, ...]
    expand_ratios: tuple[int, ...]
    num_block_repeats: tuple[int, ...]
    depthwise_padding: tuple[int, ...]
    hidden_dim: int
    hidden_act: str
    batch_norm_eps: float
    batch_norm_momentum: float
    squeeze_expansion_ratio: float
    pooling_type: str
    num_labels: int
    id2label: dict[int, str]
    label2id: dict[str, int]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.model_type != "efficientnet":
            raise ValueError("Expected an efficientnet configuration")
        if self.hidden_act not in {"swish", "silu"}:
            raise ValueError("Only EfficientNet swish/silu activation is supported")
        if self.pooling_type not in {"mean", "max"}:
            raise ValueError("EfficientNet pooling_type must be 'mean' or 'max'")
        if self.num_channels <= 0:
            raise ValueError("EfficientNet num_channels must be positive")
        if self.width_coefficient <= 0 or self.depth_coefficient <= 0:
            raise ValueError("EfficientNet width and depth coefficients must be positive")
        if self.depth_divisor <= 0:
            raise ValueError("EfficientNet depth_divisor must be positive")
        if self.batch_norm_eps <= 0:
            raise ValueError("EfficientNet batch_norm_eps must be positive")
        if not 0 <= self.batch_norm_momentum <= 1:
            raise ValueError("EfficientNet batch_norm_momentum must be between zero and one")
        if not 0 < self.squeeze_expansion_ratio <= 1:
            raise ValueError("EfficientNet squeeze_expansion_ratio must be in (0, 1]")
        if self.num_labels <= 0 or len(self.labels) != self.num_labels:
            raise ValueError("EfficientNet requires at least one class label")
        if len(self.id2label) != self.num_labels or not self.label2id:
            raise ValueError("EfficientNet label mappings must match num_labels")

        stages = len(self.in_channels)
        if stages == 0:
            raise ValueError("EfficientNet requires at least one stage")
        for name in (
            "out_channels",
            "kernel_sizes",
            "strides",
            "expand_ratios",
            "num_block_repeats",
        ):
            if len(getattr(self, name)) != stages:
                raise ValueError(f"Inconsistent EfficientNet stage configuration: {name}")
        for name in (
            "in_channels",
            "out_channels",
            "kernel_sizes",
            "expand_ratios",
            "num_block_repeats",
        ):
            if any(value <= 0 for value in getattr(self, name)):
                raise ValueError(f"EfficientNet {name} values must be positive")
        if any(stride not in {1, 2} for stride in self.strides):
            raise ValueError("EfficientNet strides must contain only 1 or 2")
        if any(kernel % 2 == 0 for kernel in self.kernel_sizes):
            raise ValueError("EfficientNet kernel_sizes must be odd")
        if self.round_filters(1280) != self.hidden_dim:
            raise ValueError("hidden_dim does not match the scaled top convolution")

        block_count = sum(
            math.ceil(self.depth_coefficient * repeats) for repeats in self.num_block_repeats
        )
        if len(set(self.depthwise_padding)) != len(self.depthwise_padding) or any(
            index < 0 or index >= block_count for index in self.depthwise_padding
        ):
            raise ValueError("EfficientNet depthwise_padding contains an invalid block index")

    def round_filters(self, channels: int) -> int:
        return _scaled_filters(self.width_coefficient, self.depth_divisor, channels)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EfficientNetConfig:
        if not isinstance(raw, Mapping):
            raise TypeError("EfficientNet configuration must be a mapping")
        defaults: dict[str, Any] = {
            "model_type": "efficientnet",
            "image_size": 600,
            "num_channels": 3,
            "width_coefficient": 2.0,
            "depth_coefficient": 3.1,
            "depth_divisor": 8,
            "in_channels": [32, 16, 24, 40, 80, 112, 192],
            "out_channels": [16, 24, 40, 80, 112, 192, 320],
            "kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
            "strides": [1, 2, 2, 2, 1, 2, 1],
            "expand_ratios": [1, 6, 6, 6, 6, 6, 6],
            "num_block_repeats": [1, 2, 2, 3, 3, 4, 1],
            "depthwise_padding": [],
            "hidden_dim": None,
            "hidden_act": "swish",
            "batch_norm_eps": 0.001,
            "batch_norm_momentum": 0.99,
            "squeeze_expansion_ratio": 0.25,
            "pooling_type": "mean",
        }
        values = {name: raw.get(name, default) for name, default in defaults.items()}
        width = _real(values["width_coefficient"], "width_coefficient")
        depth = _real(values["depth_coefficient"], "depth_coefficient")
        divisor = _integer(values["depth_divisor"], "depth_divisor")
        hidden_dim = (
            _integer(values["hidden_dim"], "hidden_dim")
            if values["hidden_dim"] is not None
            else _scaled_filters(width, divisor, 1280)
        )
        raw_id2label = raw.get("id2label")
        inferred_labels = len(raw_id2label) if isinstance(raw_id2label, Mapping) else 1000
        raw_num_labels = raw.get("num_labels", inferred_labels)
        num_labels = (
            inferred_labels if raw_num_labels is None else _integer(raw_num_labels, "num_labels")
        )
        id2label, label2id = _labels(raw, num_labels)
        labels = tuple(id2label[index] for index in range(num_labels))
        return cls(
            model_type=str(values["model_type"]),
            image_size=_image_size(values["image_size"], "image_size"),
            num_channels=_integer(values["num_channels"], "num_channels"),
            width_coefficient=width,
            depth_coefficient=depth,
            depth_divisor=divisor,
            in_channels=_integer_tuple(values["in_channels"], "in_channels"),
            out_channels=_integer_tuple(values["out_channels"], "out_channels"),
            kernel_sizes=_integer_tuple(values["kernel_sizes"], "kernel_sizes"),
            strides=_integer_tuple(values["strides"], "strides"),
            expand_ratios=_integer_tuple(values["expand_ratios"], "expand_ratios"),
            num_block_repeats=_integer_tuple(values["num_block_repeats"], "num_block_repeats"),
            depthwise_padding=_integer_tuple(values["depthwise_padding"], "depthwise_padding"),
            hidden_dim=hidden_dim,
            hidden_act=str(values["hidden_act"]),
            batch_norm_eps=_real(values["batch_norm_eps"], "batch_norm_eps"),
            batch_norm_momentum=_real(values["batch_norm_momentum"], "batch_norm_momentum"),
            squeeze_expansion_ratio=_real(
                values["squeeze_expansion_ratio"], "squeeze_expansion_ratio"
            ),
            pooling_type=str(values["pooling_type"]),
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            labels=labels,
        )
