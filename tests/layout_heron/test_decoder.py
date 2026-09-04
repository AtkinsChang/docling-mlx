# SPDX-License-Identifier: Apache-2.0

"""Same-weight differential gates for RT-DETR-v2 query decoding and heads."""

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
RTDetrV2ForObjectDetection: Any
RtDetrV2Config: Any
Decoder: Any


@pytest.fixture(scope="module", autouse=True)
def _load_requirements() -> None:
    global mx, torch, tree_flatten, TorchRtDetrV2Config
    global RTDetrV2ForObjectDetection, RtDetrV2Config, Decoder

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
        transformer_module = import_module("docling_mlx._models.rt_detr_v2.transformer")
    except ImportError as error:
        pytest.fail(f"selected parity lane is missing a required dependency: {error}")
    TorchRtDetrV2Config = config_module.RTDetrV2Config
    RTDetrV2ForObjectDetection = model_module.RTDetrV2ForObjectDetection
    RtDetrV2Config = mlx_module.RtDetrV2Config
    Decoder = transformer_module.Decoder


def _target_key(source_key: str) -> str:
    key = source_key.replace(".self_attn.o_proj.", ".self_attn.out_proj.")
    return key.replace(".mlp.fc1.", ".fc1.").replace(".mlp.fc2.", ".fc2.")


def _same_weight_models():
    torch.manual_seed(5107)
    torch_config = TorchRtDetrV2Config(
        num_queries=300,
        num_labels=17,
        d_model=256,
        decoder_layers=6,
        decoder_attention_heads=8,
        decoder_ffn_dim=1024,
        decoder_in_channels=[256, 256, 256],
        decoder_activation_function="relu",
        decoder_method="default",
        decoder_n_levels=3,
        decoder_n_points=4,
        decoder_offset_scale=0.5,
        num_feature_levels=3,
        with_box_refine=True,
        use_focal_loss=True,
    )
    torch_owner = RTDetrV2ForObjectDetection(torch_config).eval()
    torch_model = torch_owner.model.decoder
    mlx_model = Decoder(RtDetrV2Config.from_dict({"num_labels": 17}).transformer_config)

    target_shapes = {key: tuple(value.shape) for key, value in tree_flatten(mlx_model.parameters())}
    converted: list[tuple[str, object]] = []
    mapped: set[str] = set()
    for source_key, tensor in torch_model.state_dict().items():
        target_key = _target_key(source_key)
        array = tensor.detach().cpu().numpy()
        assert target_key not in mapped
        assert target_shapes[target_key] == array.shape
        mapped.add(target_key)
        converted.append((target_key, mx.array(array)))
    assert mapped == set(target_shapes)
    mlx_model.load_weights(converted, strict=True)
    mlx_model.eval()
    return torch_model, mlx_model


@pytest.mark.parametrize("seed", [73, 1861])
def test_six_layer_decoder_and_heads_match_transformers(seed: int) -> None:
    torch_model, mlx_model = _same_weight_models()
    generator = np.random.default_rng(seed)
    target = generator.normal(0.0, 0.1, (1, 5, 256)).astype(np.float32)
    reference = generator.normal(0.0, 0.5, (1, 5, 4)).astype(np.float32)
    memory = generator.normal(0.0, 0.1, (1, 7, 256)).astype(np.float32)
    spatial_shapes = ((2, 2), (1, 2), (1, 1))
    with torch.inference_mode():
        torch_output = torch_model(
            inputs_embeds=torch.from_numpy(target),
            encoder_hidden_states=torch.from_numpy(memory),
            reference_points=torch.from_numpy(reference),
            spatial_shapes=torch.tensor(spatial_shapes, dtype=torch.long),
            spatial_shapes_list=spatial_shapes,
            level_start_index=torch.tensor([0, 4, 6], dtype=torch.long),
        )

    mlx_output = mlx_model(
        target=mx.array(target),
        reference_points_unact=mx.array(reference),
        encoder_hidden_states=mx.array(memory),
        spatial_shapes=spatial_shapes,
    )
    mx.eval(mlx_output)

    comparisons = {
        "last_hidden_state": torch_output.last_hidden_state.numpy(),
        "intermediate_hidden_states": torch_output.intermediate_hidden_states.numpy(),
        "intermediate_reference_points": torch_output.intermediate_reference_points.numpy(),
        "intermediate_logits": torch_output.intermediate_logits.numpy(),
    }
    for name, expected in comparisons.items():
        actual = np.array(mlx_output[name])
        assert actual.shape == expected.shape
        error = np.abs(actual - expected)
        assert float(error.mean()) <= 2e-5
        assert float(error.max()) <= 5e-4
