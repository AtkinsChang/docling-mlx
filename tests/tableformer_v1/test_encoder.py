# SPDX-License-Identifier: Apache-2.0

"""TableFormer v1 encoder architecture checks."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from docling_mlx._models.tableformer_v1.config import TableFormerV1Config
from tests.tableformer_v1.conftest import accurate_config, fast_config

pytestmark = pytest.mark.mlx


def _required_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as error:
        pytest.fail(f"selected MLX qualification requires {name}: {error}")


def _encoder_type() -> Any:
    return importlib.import_module("docling_mlx._models.tableformer_v1.vision").TableFormerV1Encoder


def test_topology_preserves_frozen_source_namespace() -> None:
    mx = _required_module("mlx.core")
    tree_flatten = _required_module("mlx.utils").tree_flatten
    model = _encoder_type()(TableFormerV1Config.from_dict(accurate_config()))
    parameters = dict(tree_flatten(model.parameters()))

    assert len(parameters) == 172
    assert parameters["_encoder._resnet.0.weight"].shape == (64, 7, 7, 3)
    assert parameters["_encoder._resnet.4.0.conv1.weight"].shape == (64, 3, 3, 64)
    assert parameters["_encoder._resnet.5.0.downsample.0.weight"].shape == (128, 1, 1, 64)
    assert parameters["_encoder._resnet.6.1.bn2.running_var"].dtype == mx.float32
    assert parameters["_tag_transformer._input_filter.0.conv1.weight"].shape == (
        512,
        3,
        3,
        256,
    )
    assert parameters["_tag_transformer._encoder.layers.0.self_attn.in_proj_weight"].shape == (
        1536,
        512,
    )
    assert parameters["_tag_transformer._encoder.layers.5.linear1.weight"].shape == (
        1024,
        512,
    )
    assert parameters["_tag_transformer._encoder.layers.5.norm2.bias"].shape == (512,)


def test_fast_topology_stops_after_the_fourth_tag_encoder_layer() -> None:
    tree_flatten = _required_module("mlx.utils").tree_flatten
    config = TableFormerV1Config.from_dict(fast_config())
    parameters = dict(tree_flatten(_encoder_type()(config).parameters()))

    assert len(parameters) == 148
    assert "_tag_transformer._encoder.layers.3.norm2.bias" in parameters
    assert not any(name.startswith("_tag_transformer._encoder.layers.4.") for name in parameters)


def test_image_encoder_adapts_to_28_by_28() -> None:
    mx = _required_module("mlx.core")
    model = _encoder_type()(TableFormerV1Config.from_dict(accurate_config()))
    model.eval()
    pixels = mx.array(np.random.default_rng(41).random((1, 64, 64, 3), dtype=np.float32))

    outputs = model._encoder.forward_intermediates(pixels)
    mx.eval(outputs)

    assert outputs["stem"].shape == (1, 32, 32, 64)
    assert outputs["resnet.layers.0"].shape == (1, 16, 16, 64)
    assert outputs["resnet.layers.1"].shape == (1, 8, 8, 128)
    assert outputs["resnet.layers.2"].shape == (1, 4, 4, 256)
    assert outputs["image_features"].shape == (1, 28, 28, 256)
    assert np.isfinite(np.asarray(outputs["image_features"])).all()


def test_tag_encoder_is_sequence_first_after_the_two_block_filter() -> None:
    mx = _required_module("mlx.core")
    model = _encoder_type()(TableFormerV1Config.from_dict(accurate_config()))
    model.eval()
    features = mx.zeros((1, 28, 28, 256), dtype=mx.float32)

    filtered = features
    for block in model._tag_transformer._input_filter:
        filtered = block(filtered)
    memory = model._tag_transformer._encoder(filtered.reshape(1, 784, 512))
    mx.eval(filtered, memory)

    assert filtered.shape == (1, 28, 28, 512)
    assert memory.shape == (1, 784, 512)
    assert memory.transpose(1, 0, 2).shape == (784, 1, 512)


def test_encoder_rejects_training_and_non_fp32_inputs() -> None:
    mx = _required_module("mlx.core")
    model = _encoder_type()(TableFormerV1Config.from_dict(accurate_config()))
    with pytest.raises(ValueError, match="eval"):
        model(mx.zeros((1, 64, 64, 3), dtype=mx.float32))

    model.eval()
    with pytest.raises(ValueError, match="float32"):
        model(mx.zeros((1, 64, 64, 3), dtype=mx.float16))
    with pytest.raises(ValueError, match="NHWC"):
        model(mx.zeros((1, 3, 64, 64), dtype=mx.float32))
