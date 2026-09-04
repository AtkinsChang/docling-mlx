# SPDX-License-Identifier: Apache-2.0

"""Portable EfficientNet configuration and strict-conversion contracts."""

from pathlib import Path

import numpy as np
import pytest

from docling_mlx._models.efficientnet.config import EfficientNetConfig
from tools.document_figure.convert_weights import (
    _verify_source,
    convert_state_dict,
)


def b0_config() -> dict:
    return {
        "model_type": "efficientnet",
        "num_channels": 3,
        "image_size": 224,
        "width_coefficient": 1.0,
        "depth_coefficient": 1.0,
        "depth_divisor": 8,
        "in_channels": [32, 16, 24, 40, 80, 112, 192],
        "out_channels": [16, 24, 40, 80, 112, 192, 320],
        "kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "expand_ratios": [1, 6, 6, 6, 6, 6, 6],
        "num_block_repeats": [1, 2, 2, 3, 3, 4, 1],
        "depthwise_padding": [],
        "hidden_dim": 1280,
        "hidden_act": "swish",
        "batch_norm_eps": 0.001,
        "batch_norm_momentum": 0.99,
        "squeeze_expansion_ratio": 0.25,
        "pooling_type": "mean",
        "id2label": {str(i): str(i) for i in range(26)},
    }


def test_parsed_config_is_immutable_and_owns_structural_validation() -> None:
    parsed = EfficientNetConfig.from_dict(b0_config())
    assert parsed.hidden_act == "swish"
    assert parsed.in_channels == (32, 16, 24, 40, 80, 112, 192)
    assert parsed.num_labels == 26
    with pytest.raises(AttributeError):
        parsed.hidden_dim = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_act", "relu", "activation"),
        ("strides", [1], "stage"),
        ("hidden_dim", 640, "hidden_dim"),
    ],
)
def test_parsed_config_rejects_unsupported_structure(field: str, value, message: str) -> None:
    config = b0_config()
    config[field] = value
    with pytest.raises(ValueError, match=message):
        EfficientNetConfig.from_dict(config)


def test_strict_conversion_transposes_only_convolutions() -> None:
    source = {
        "conv.weight": np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2),
        "linear.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "bn.running_mean": np.zeros(2, dtype=np.float32),
        "bn.num_batches_tracked": np.array(7, dtype=np.int64),
    }
    target = {"conv.weight": (2, 2, 2, 3), "linear.weight": (2, 3), "bn.running_mean": (2,)}
    converted, mappings, ignored = convert_state_dict(source, target)
    np.testing.assert_array_equal(
        converted["conv.weight"], source["conv.weight"].transpose(0, 2, 3, 1)
    )
    np.testing.assert_array_equal(converted["linear.weight"], source["linear.weight"])
    assert ignored == ["bn.num_batches_tracked"]
    assert {entry["transform"] for entry in mappings} == {"identity", "OIHW_to_OHWI"}


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({}, "missing"),
        ({"weight": np.zeros(3, dtype=np.float32)}, "shape"),
        ({"weight": np.zeros(2, dtype=np.float32), "other": np.zeros(1)}, "unexpected"),
        ({"weight": np.array([np.nan, 0], dtype=np.float32)}, "finite"),
    ],
)
def test_conversion_rejects_incomplete_or_invalid_state(source: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        convert_state_dict(source, {"weight": (2,)})


def test_conversion_requires_valid_bn_counter() -> None:
    target = {"bn.running_mean": (2,)}
    source = {"bn.running_mean": np.zeros(2, dtype=np.float32)}
    converted, _, ignored = convert_state_dict(source, target)
    np.testing.assert_array_equal(converted["bn.running_mean"], source["bn.running_mean"])
    assert ignored == []


def test_source_validation_requires_every_converter_input(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="Missing source file"):
        _verify_source(tmp_path)
