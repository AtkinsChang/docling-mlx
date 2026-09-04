# SPDX-License-Identifier: Apache-2.0

"""Same-weight differential gates for the adapted RT-DETR-v2 hybrid encoder."""

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
TorchRtDetrV2Config: Any
TorchHybridEncoder: Any
RtDetrV2Config: Any
HybridEncoder: Any


@pytest.fixture(scope="module", autouse=True)
def _load_requirements() -> None:
    global mx, torch, tree_flatten, TorchRtDetrV2Config
    global TorchHybridEncoder, RtDetrV2Config, HybridEncoder

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
        config_module = import_module("transformers.models.rt_detr_v2.configuration_rt_detr_v2")
        model_module = import_module("transformers.models.rt_detr_v2.modeling_rt_detr_v2")
        mlx_module = import_module("docling_mlx._models.rt_detr_v2")
        vision_module = import_module("docling_mlx._models.rt_detr_v2.vision")
    except ImportError as error:
        pytest.fail(f"selected parity lane is missing a required dependency: {error}")
    TorchRtDetrV2Config = config_module.RTDetrV2Config
    TorchHybridEncoder = model_module.RTDetrV2HybridEncoder
    RtDetrV2Config = mlx_module.RtDetrV2Config
    HybridEncoder = vision_module.HybridEncoder


def _target_key(source_key: str) -> str | None:
    if source_key.endswith(".num_batches_tracked"):
        return None
    key = source_key.replace(".self_attn.o_proj.", ".self_attn.out_proj.")
    key = key.replace(".mlp.fc1.", ".fc1.").replace(".mlp.fc2.", ".fc2.")
    return key.replace(".norm.", ".bn.")


def _same_weight_models() -> tuple[TorchHybridEncoder, HybridEncoder]:
    torch.manual_seed(2219)
    torch_config = TorchRtDetrV2Config(
        encoder_hidden_dim=256,
        encoder_ffn_dim=1024,
        encoder_in_channels=[512, 1024, 2048],
        encode_proj_layers=[2],
        encoder_layers=1,
        encoder_attention_heads=8,
        encoder_activation_function="gelu",
        activation_function="silu",
        hidden_expansion=1.0,
    )
    torch_model = TorchHybridEncoder(torch_config).eval()
    mlx_model = HybridEncoder(RtDetrV2Config().hybrid_encoder_config)

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


@pytest.mark.parametrize("seed", [31, 941])
def test_hybrid_encoder_named_levels_match_transformers(seed: int) -> None:
    torch_model, mlx_model = _same_weight_models()
    generator = np.random.default_rng(seed)
    inputs = [
        generator.normal(0.0, 0.1, (1, 256, size, size)).astype(np.float32) for size in (8, 4, 2)
    ]
    with torch.inference_mode():
        torch_outputs = torch_model(
            inputs_embeds=[torch.from_numpy(value) for value in inputs]
        ).last_hidden_state

    mlx_outputs = mlx_model(tuple(mx.array(value.transpose(0, 2, 3, 1)) for value in inputs))
    mx.eval(mlx_outputs)

    assert len(mlx_outputs) == len(torch_outputs) == 3
    for torch_output, mlx_output in zip(torch_outputs, mlx_outputs, strict=True):
        expected = torch_output.permute(0, 2, 3, 1).numpy()
        actual = np.array(mlx_output)
        assert actual.shape == expected.shape
        error = np.abs(actual - expected)
        assert float(error.mean()) <= 3e-6
        assert float(error.max()) <= 5e-5
