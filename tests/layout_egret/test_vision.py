# SPDX-License-Identifier: Apache-2.0

"""Same-weight gates for the native Egret HGNetV2 and D-FINE encoder."""

from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from typing import Any

import numpy as np
import pytest

from docling_mlx._models.dfine.config import DFineConfig
from tests.layout_egret.test_config import egret_config

pytestmark = [pytest.mark.mlx, pytest.mark.parity]

mx: Any
torch: Any
tree_flatten: Any
TorchDFineConfig: Any
TorchHGNetV2Backbone: Any
TorchDFineHybridEncoder: Any
HGNetV2Backbone: Any
DFineHybridEncoder: Any


@pytest.fixture(scope="module", autouse=True)
def _load_requirements() -> None:
    global mx, torch, tree_flatten, TorchDFineConfig
    global TorchHGNetV2Backbone, TorchDFineHybridEncoder
    global HGNetV2Backbone, DFineHybridEncoder

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
        config_module = import_module("transformers.models.d_fine.configuration_d_fine")
        dfine_module = import_module("transformers.models.d_fine.modeling_d_fine")
        hgnet_module = import_module("transformers.models.hgnet_v2.modeling_hgnet_v2")
        mlx_module = import_module("docling_mlx._models.dfine.vision")
    except ImportError as error:
        pytest.fail(f"selected parity lane is missing a required dependency: {error}")
    TorchDFineConfig = config_module.DFineConfig
    TorchHGNetV2Backbone = hgnet_module.HGNetV2Backbone
    TorchDFineHybridEncoder = dfine_module.DFineHybridEncoder
    HGNetV2Backbone = mlx_module.HGNetV2Backbone
    DFineHybridEncoder = mlx_module.DFineHybridEncoder


def _configs(profile: str) -> tuple[DFineConfig, Any]:
    raw = egret_config(profile)
    return DFineConfig.from_dict(raw), TorchDFineConfig(**raw)


def _same_weights(torch_model: Any, mlx_model: Any) -> None:
    target_shapes = {key: tuple(value.shape) for key, value in tree_flatten(mlx_model.parameters())}
    converted = []
    mapped = set()
    for source_key, tensor in torch_model.state_dict().items():
        if source_key.endswith(".num_batches_tracked"):
            continue
        array = tensor.detach().cpu().numpy()
        if array.ndim == 4:
            array = array.transpose(0, 2, 3, 1)
        assert source_key not in mapped
        assert target_shapes[source_key] == array.shape
        mapped.add(source_key)
        converted.append((source_key, mx.array(array)))
    assert mapped == set(target_shapes)
    mlx_model.load_weights(converted, strict=True)
    mlx_model.eval()


@pytest.mark.parametrize("profile", ["medium", "large", "xlarge"])
def test_backbone_closed_profiles_construct_exact_stage_shapes(profile: str) -> None:
    mlx_config, _ = _configs(profile)
    model = HGNetV2Backbone(mlx_config.backbone)
    assert [len(stage.blocks) for stage in model.encoder.stages] == list(
        mlx_config.backbone.stage_num_blocks
    )
    assert model.out_channels == mlx_config.encoder.in_channels
    assert model.out_strides == mlx_config.encoder.feature_strides


@pytest.mark.parametrize("profile", ["medium", "large", "xlarge"])
def test_hybrid_encoder_closed_profiles_construct_exact_depth(profile: str) -> None:
    mlx_config, _ = _configs(profile)
    model = DFineHybridEncoder(mlx_config.encoder)
    expected_blocks = round(3 * mlx_config.encoder.depth_mult)
    assert len(model.aifi) == len(mlx_config.encoder.projection_layers) == 1
    assert [len(block.csp_rep1.bottlenecks) for block in model.fpn_blocks] == [
        expected_blocks,
        expected_blocks,
    ]
    assert model.out_channels == (mlx_config.encoder.hidden_dim,) * 3
    assert model.out_strides == mlx_config.encoder.feature_strides


@pytest.mark.parametrize("seed", [17, 8041])
def test_hgnet_v2_medium_named_stages_match_transformers(seed: int) -> None:
    mlx_config, torch_config = _configs("medium")
    torch.manual_seed(8231)
    torch_model = TorchHGNetV2Backbone(torch_config.backbone_config).eval()
    mlx_model = HGNetV2Backbone(mlx_config.backbone)
    _same_weights(torch_model, mlx_model)

    pixels = np.random.default_rng(seed).normal(0.0, 0.25, (1, 3, 64, 64)).astype(np.float32)
    with torch.inference_mode():
        expected_outputs = torch_model(torch.from_numpy(pixels)).feature_maps
    actual_outputs = mlx_model(mx.array(pixels.transpose(0, 2, 3, 1)))
    mx.eval(actual_outputs)

    assert len(actual_outputs) == len(expected_outputs) == 3
    for expected, actual in zip(expected_outputs, actual_outputs, strict=True):
        expected_nhwc = expected.permute(0, 2, 3, 1).numpy()
        error = np.abs(np.array(actual) - expected_nhwc)
        assert float(error.mean()) <= 5e-5
        assert float(error.max()) <= 1e-3


@pytest.mark.parametrize("profile", ["medium", "large", "xlarge"])
def test_hybrid_encoder_named_levels_match_transformers(profile: str) -> None:
    mlx_config, torch_config = _configs(profile)
    torch.manual_seed(2219)
    torch_model = TorchDFineHybridEncoder(torch_config).eval()
    mlx_model = DFineHybridEncoder(mlx_config.encoder)
    _same_weights(torch_model, mlx_model)

    channels = mlx_config.encoder.hidden_dim
    generator = np.random.default_rng(941 + channels)
    inputs = [
        generator.normal(0.0, 0.1, (1, channels, size, size)).astype(np.float32)
        for size in (8, 4, 2)
    ]
    with torch.inference_mode():
        expected_outputs = torch_model(
            inputs_embeds=[torch.from_numpy(value) for value in inputs]
        ).last_hidden_state
    actual_outputs = mlx_model(tuple(mx.array(value.transpose(0, 2, 3, 1)) for value in inputs))
    mx.eval(actual_outputs)

    assert len(actual_outputs) == len(expected_outputs) == 3
    for expected, actual in zip(expected_outputs, actual_outputs, strict=True):
        expected_nhwc = expected.permute(0, 2, 3, 1).numpy()
        error = np.abs(np.array(actual) - expected_nhwc)
        assert float(error.mean()) <= 5e-5
        assert float(error.max()) <= 1e-3
