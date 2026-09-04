# SPDX-License-Identifier: Apache-2.0

"""Same-weight differential gates for the native Egret D-FINE decoder."""

from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.mlx, pytest.mark.parity]

mx: Any
torch: Any
tree_flatten: Any
TorchDFineConfig: Any
TorchDFineDecoder: Any
TorchDFineMLP: Any
DFineDecoderConfig: Any
DFineDecoder: Any
dfine_primitives: Any
neutral_primitives: Any


@pytest.fixture(scope="module", autouse=True)
def _load_requirements() -> None:
    global mx, torch, tree_flatten, TorchDFineConfig, TorchDFineDecoder, TorchDFineMLP
    global DFineDecoderConfig, DFineDecoder, dfine_primitives, neutral_primitives

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
        transformers = import_module("transformers")
        pinned_versions = import_module("tools.pinned_versions")
        tree_flatten = import_module("mlx.utils").tree_flatten
        torch_config = import_module("transformers.models.d_fine.configuration_d_fine")
        torch_model = import_module("transformers.models.d_fine.modeling_d_fine")
        mlx_config = import_module("docling_mlx._models.dfine.config")
        mlx_decoder = import_module("docling_mlx._models.dfine.decoder")
        dfine_primitives = import_module("docling_mlx._models.dfine.primitives")
        neutral_primitives = import_module("docling_mlx._models.detector_primitives")
    except ImportError as error:
        pytest.fail(f"selected parity lane is missing a required dependency: {error}")
    TorchDFineConfig = torch_config.DFineConfig
    TorchDFineDecoder = torch_model.DFineDecoder
    TorchDFineMLP = torch_model.DFineMLP
    DFineDecoderConfig = mlx_config.DFineDecoderConfig
    DFineDecoder = mlx_decoder.DFineDecoder
    assert (
        transformers.__version__
        == pinned_versions.locked_versions(["transformers"])["transformers"]
    )


def _decoder_config() -> Any:
    return DFineDecoderConfig(
        hidden_dim=256,
        layers=4,
        attention_heads=8,
        ffn_dim=1024,
        in_channels=(256, 256, 256),
        activation_function="relu",
        method="default",
        points_per_level=(3, 6, 3),
        offset_scale=0.5,
        num_feature_levels=3,
        num_queries=300,
        max_num_bins=32,
        reg_scale=4.0,
        up=0.5,
        top_prob_values=4,
        lqe_hidden_dim=64,
        lqe_layers=2,
    )


def _target_key(source_key: str) -> str:
    key = source_key.replace(".self_attn.o_proj.", ".self_attn.out_proj.")
    return key.replace(".mlp.layers.0.", ".fc1.").replace(".mlp.layers.1.", ".fc2.")


def _same_weight_models() -> tuple[Any, Any]:
    torch.manual_seed(5107)
    torch_config = TorchDFineConfig(
        num_labels=17,
        d_model=256,
        decoder_layers=4,
        decoder_attention_heads=8,
        decoder_ffn_dim=1024,
        decoder_in_channels=[256, 256, 256],
        decoder_activation_function="relu",
        decoder_method="default",
        decoder_n_points=[3, 6, 3],
        decoder_offset_scale=0.5,
        num_feature_levels=3,
        eval_idx=-1,
        max_num_bins=32,
        reg_scale=4.0,
        up=0.5,
        top_prob_values=4,
        lqe_hidden_dim=64,
        lqe_layers=2,
        layer_scale=1,
        dropout=0.0,
        attention_dropout=0.0,
    )
    source = TorchDFineDecoder(torch_config).eval()
    source.class_embed = torch.nn.ModuleList(
        [torch.nn.Linear(256, 17) for _ in range(torch_config.decoder_layers)]
    )
    source.bbox_embed = torch.nn.ModuleList(
        [
            TorchDFineMLP(256, 256, 4 * (torch_config.max_num_bins + 1), 3)
            for _ in range(torch_config.decoder_layers)
        ]
    )
    target = DFineDecoder(_decoder_config(), num_labels=17)

    target_shapes = {key: tuple(value.shape) for key, value in tree_flatten(target.parameters())}
    converted = []
    mapped = set()
    for key, tensor in source.state_dict().items():
        target_key = _target_key(key)
        array = tensor.detach().cpu().numpy()
        assert target_key not in mapped
        assert target_shapes[target_key] == array.shape
        mapped.add(target_key)
        converted.append((target_key, mx.array(array)))
    assert mapped == set(target_shapes)
    target.load_weights(converted, strict=True)
    target.eval()
    return source, target


def test_dfine_owns_decoder_and_imports_only_neutral_primitives() -> None:
    decoder_source = (
        Path(__file__).resolve().parents[2] / "src/docling_mlx/_models/dfine/decoder.py"
    ).read_text(encoding="utf-8")
    primitive_source = (
        Path(__file__).resolve().parents[2] / "src/docling_mlx/_models/dfine/primitives.py"
    ).read_text(encoding="utf-8")
    assert "rt_detr_v2" not in decoder_source + primitive_source
    assert "detector_primitives" in primitive_source
    for name in (
        "MLP",
        "SelfAttention",
        "generate_anchors",
        "grid_sample_bilinear_zeros_align_corners_false",
        "inverse_sigmoid",
    ):
        assert getattr(dfine_primitives, name) is getattr(neutral_primitives, name)


@pytest.mark.parametrize("seed", [73, 1861])
def test_four_layer_decoder_and_fdr_heads_match_transformers(seed: int) -> None:
    source, target = _same_weight_models()
    generator = np.random.default_rng(seed)
    inputs = generator.normal(0.0, 0.1, (1, 3, 256)).astype(np.float32)
    references = generator.normal(0.0, 0.5, (1, 3, 4)).astype(np.float32)
    memory = generator.normal(0.0, 0.1, (1, 7, 256)).astype(np.float32)
    spatial_shapes = ((2, 2), (1, 2), (1, 1))

    with torch.inference_mode():
        expected = source(
            encoder_hidden_states=torch.from_numpy(memory),
            reference_points=torch.from_numpy(references),
            inputs_embeds=torch.from_numpy(inputs),
            spatial_shapes=torch.tensor(spatial_shapes, dtype=torch.long),
            spatial_shapes_list=spatial_shapes,
        )
    actual = target(
        encoder_hidden_states=mx.array(memory),
        reference_points=mx.array(references),
        inputs_embeds=mx.array(inputs),
        spatial_shapes=spatial_shapes,
    )
    mx.eval(actual)

    for name in (
        "last_hidden_state",
        "intermediate_hidden_states",
        "intermediate_logits",
        "intermediate_reference_points",
        "intermediate_predicted_corners",
        "initial_reference_points",
    ):
        expected_array = getattr(expected, name).numpy()
        actual_array = np.array(actual[name])
        assert actual_array.shape == expected_array.shape
        error = np.abs(actual_array - expected_array)
        assert float(error.mean()) <= 2e-5
        assert float(error.max()) <= 5e-4
