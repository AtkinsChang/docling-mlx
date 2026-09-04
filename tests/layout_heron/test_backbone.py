# SPDX-License-Identifier: Apache-2.0

"""Same-weight differential gates for the adapted ResNet-vd backbone."""

from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.mlx, pytest.mark.parity]

mx: Any
torch: Any
tree_flatten: Any
RTDetrResNetConfig: Any
RTDetrResNetBackbone: Any
MlxResNetConfig: Any
Backbone: Any


@pytest.fixture(scope="module", autouse=True)
def _load_requirements() -> None:
    global mx, torch, tree_flatten
    global RTDetrResNetConfig, RTDetrResNetBackbone, MlxResNetConfig
    global Backbone

    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        pytest.fail(f"selected parity lane requires Metal: {probe.stderr.strip()}")
    try:
        mx = import_module("mlx.core")
        torch = import_module("torch")
        tree_flatten = import_module("mlx.utils").tree_flatten
        config_module = import_module("transformers.models.rt_detr.configuration_rt_detr_resnet")
        model_module = import_module("transformers.models.rt_detr.modeling_rt_detr_resnet")
        mlx_config_module = import_module("docling_mlx._models.rt_detr_v2.config")
        vision_module = import_module("docling_mlx._models.rt_detr_v2.vision")
    except ImportError as error:
        pytest.fail(f"selected parity lane is missing a required dependency: {error}")
    RTDetrResNetConfig = config_module.RTDetrResNetConfig
    RTDetrResNetBackbone = model_module.RTDetrResNetBackbone
    MlxResNetConfig = mlx_config_module.RtDetrResNetConfig
    Backbone = vision_module.Backbone


def _target_key(source_key: str) -> str | None:
    if source_key.endswith(".num_batches_tracked"):
        return None
    key = source_key.replace(".shortcut.1.", ".shortcut.proj.")
    key = key.replace(".convolution.", ".conv.")
    return key.replace(".normalization.", ".bn.")


def _same_weight_models() -> tuple[RTDetrResNetBackbone, Backbone]:
    torch.manual_seed(8231)
    torch_config = RTDetrResNetConfig(
        depths=[3, 4, 6, 3],
        out_features=["stage2", "stage3", "stage4"],
    )
    torch_model = RTDetrResNetBackbone(torch_config).eval()
    mlx_model = Backbone(MlxResNetConfig())

    target_shapes = {key: tuple(value.shape) for key, value in tree_flatten(mlx_model.parameters())}
    converted: list[tuple[str, object]] = []
    mapped: set[str] = set()
    for source_key, tensor in torch_model.state_dict().items():
        target_key = _target_key(source_key)
        if target_key is None:
            continue
        array = tensor.detach().cpu().numpy()
        if target_key.endswith(".conv.weight"):
            array = array.transpose(0, 2, 3, 1)
        assert target_key not in mapped
        assert target_shapes[target_key] == array.shape
        mapped.add(target_key)
        converted.append((target_key, mx.array(array)))
    assert mapped == set(target_shapes)
    mlx_model.load_weights(converted, strict=True)
    mlx_model.eval()
    return torch_model, mlx_model


@pytest.mark.parametrize("seed", [17, 8041])
def test_resnet_vd_named_stages_match_transformers(seed: int) -> None:
    torch_model, mlx_model = _same_weight_models()
    pixels = np.random.default_rng(seed).normal(0.0, 0.25, (1, 3, 64, 64)).astype(np.float32)
    with torch.inference_mode():
        torch_outputs = torch_model(torch.from_numpy(pixels)).feature_maps

    mlx_outputs = mlx_model(mx.array(pixels.transpose(0, 2, 3, 1)))
    mx.eval(mlx_outputs)

    assert len(mlx_outputs) == len(torch_outputs) == 3
    for torch_output, mlx_output in zip(torch_outputs, mlx_outputs, strict=True):
        expected = torch_output.permute(0, 2, 3, 1).numpy()
        actual = np.array(mlx_output)
        assert actual.shape == expected.shape
        error = np.abs(actual - expected)
        assert float(error.mean()) <= 2e-5
        assert float(error.max()) <= 3e-4


def test_r101_uses_the_same_backbone_implementation() -> None:
    model = Backbone(MlxResNetConfig(depths=(3, 4, 23, 3)))
    assert [len(stage.layers) for stage in model.encoder.stages] == [3, 4, 23, 3]
