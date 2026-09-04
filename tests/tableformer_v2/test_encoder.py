# SPDX-License-Identifier: Apache-2.0

"""TableFormerV2 EfficientNetV2-S vision encoder architecture checks."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.mlx


def _required_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as error:
        pytest.fail(f"selected MLX qualification requires {name}: {error}")


def _config() -> dict[str, object]:
    return {
        "architectures": ["TableFormerV2"],
        "model_type": "TableFormerV2",
        "embed_dim": 512,
        "num_heads": 8,
        "ff_dim": 2048,
        "num_decoder_layers": 4,
        "vocab_size": 13,
        "conv_mixer_expansion": 1.0,
        "data_cells": [4, 5, 10, 11, 12],
        "pad_token_id": 0,
        "eos_token_id": 3,
        "use_fpn": False,
        "dtype": "float32",
    }


def _encoder_type():
    module = importlib.import_module("docling_mlx._models.tableformer_v2.vision")
    return module.TableFormerV2VisionEncoder


def test_closed_topology_preserves_source_state_namespace() -> None:
    mx = _required_module("mlx.core")
    tree_flatten = _required_module("mlx.utils").tree_flatten
    model = _encoder_type()(_config())
    parameters = dict(tree_flatten(model.parameters()))

    assert len(model.feature_extractor.features) == 8
    assert sum(len(stage) for stage in model.feature_extractor.features[1:-1]) == 40
    assert parameters["feature_extractor.features.0.0.weight"].shape == (24, 3, 3, 3)
    assert parameters["feature_extractor.features.3.3.block.1.0.weight"].shape == (64, 1, 1, 256)
    assert parameters["feature_extractor.features.4.0.block.2.fc1.weight"].shape == (16, 1, 1, 256)
    assert parameters["feature_extractor.features.7.1.running_var"].dtype == mx.float32
    assert parameters["se_module.se.1.weight"].shape == (80, 1, 1, 1280)
    assert parameters["conv_mixer.block.6.se.3.weight"].shape == (1280, 1, 1, 80)
    assert parameters["feature_to_embedding.weight"].shape == (512, 1280)
    assert not any(key.startswith("feature_extractor.classifier.") for key in parameters)
    assert model.feature_extractor.features[0][1].eps == 1e-3
    assert model.conv_mixer.block[1].eps == 1e-5


def test_encoder_produces_nhwc_derived_decoder_memory_and_boundaries() -> None:
    mx = _required_module("mlx.core")
    model = _encoder_type()(_config())
    pixels = mx.array(np.random.default_rng(164).random((1, 64, 64, 3), dtype=np.float32))

    encoded, spatial_size = model(pixels)
    intermediates = model.forward_intermediates(pixels)
    mx.eval(encoded, intermediates)

    assert spatial_size == (2, 2)
    assert encoded.shape == (1, 4, 512)
    assert intermediates["stem"].shape == (1, 32, 32, 24)
    assert intermediates["backbone.stages.2"].shape == (1, 8, 8, 64)
    assert intermediates["backbone"].shape == (1, 2, 2, 1280)
    assert intermediates["post_backbone_se"].shape == (1, 2, 2, 1280)
    assert intermediates["spatial_mixer"].shape == (1, 2, 2, 1280)
    assert intermediates["encoded"].shape == (1, 4, 512)
    assert np.isfinite(np.asarray(encoded)).all()


@pytest.mark.parametrize("dtype_name", ["float16", "int32"])
def test_encoder_rejects_non_fp32_inputs(dtype_name: str) -> None:
    mx = _required_module("mlx.core")
    model = _encoder_type()(_config())
    with pytest.raises(ValueError, match="float32"):
        model(mx.zeros((1, 64, 64, 3), dtype=getattr(mx, dtype_name)))


def test_encoder_rejects_non_nhwc_and_training_mode() -> None:
    mx = _required_module("mlx.core")
    model = _encoder_type()(_config())
    with pytest.raises(ValueError, match="NHWC"):
        model(mx.zeros((1, 3, 64, 64)))
    model.train()
    with pytest.raises(ValueError, match="eval"):
        model(mx.zeros((1, 64, 64, 3)))


def test_dead_torchvision_classifier_is_explicitly_ignored_by_conversion() -> None:
    module = importlib.import_module("docling_mlx._models.tableformer_v2.vision")
    assert module.is_ignored_source_key("feature_extractor.classifier.1.weight")
    assert module.is_ignored_source_key("feature_extractor.classifier.1.bias")
    assert not module.is_ignored_source_key("feature_extractor.features.7.0.weight")
