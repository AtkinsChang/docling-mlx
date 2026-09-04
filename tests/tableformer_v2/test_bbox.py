# SPDX-License-Identifier: Apache-2.0

"""Same-weight differential gates for the TableFormerV2 bounding-box head."""

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
TorchBboxHead: Any
BboxHead: Any
cxcywh_to_xyxy: Any


@pytest.fixture(scope="module", autouse=True)
def _load_requirements() -> None:
    global mx, torch, tree_flatten, TorchBboxHead, BboxHead, cxcywh_to_xyxy
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
        TorchBboxHead = import_module("docling_ibm_models.tableformer_v2.model").BboxHead
        bbox_module = import_module("docling_mlx._models.tableformer_v2.bbox")
    except ImportError as error:
        pytest.fail(f"selected parity lane is missing a required dependency: {error}")
    BboxHead = bbox_module.BboxHead
    cxcywh_to_xyxy = bbox_module.cxcywh_to_xyxy


def _target_key(source_key: str) -> str:
    key = source_key.replace(".ffn.0.", ".ffn.layers.0.")
    key = key.replace(".ffn.3.", ".ffn.layers.3.")
    return (
        key.replace("bbox_mlp.0.", "bbox_mlp.layers.0.")
        .replace("bbox_mlp.3.", "bbox_mlp.layers.3.")
        .replace("bbox_mlp.6.", "bbox_mlp.layers.6.")
    )


def _same_weight_models() -> tuple[Any, Any]:
    torch.manual_seed(20260830)
    torch_model = TorchBboxHead(embed_dim=8, num_heads=2, num_layers=2).eval()
    mlx_model = BboxHead(embed_dim=8, num_heads=2, num_layers=2)
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


@pytest.mark.parametrize("seed", [127, 1861])
def test_bbox_head_matches_adapted_torch_oracle_for_mixed_page_cells(seed: int) -> None:
    torch_model, mlx_model = _same_weight_models()
    generator = np.random.default_rng(seed)
    cell_embeddings = generator.normal(0.0, 0.2, (5, 8)).astype(np.float32)
    encoder_hidden = generator.normal(0.0, 0.2, (2, 7, 8)).astype(np.float32)
    batch_indices = np.array([0, 1, 0, 1, 1], dtype=np.int64)

    with torch.inference_mode():
        expected = torch_model(
            torch.from_numpy(cell_embeddings),
            torch.from_numpy(encoder_hidden),
            torch.from_numpy(batch_indices),
        ).numpy()
    actual = mlx_model(
        mx.array(cell_embeddings),
        mx.array(encoder_hidden),
        mx.array(batch_indices),
    )
    mx.eval(actual)

    error = np.abs(np.array(actual) - expected)
    assert float(error.mean()) <= 1e-5
    assert float(error.max()) <= 1e-4


def test_empty_cells_keep_float32_empty_box_contract() -> None:
    _, mlx_model = _same_weight_models()
    actual = mlx_model(
        mx.zeros((0, 8), dtype=mx.float32),
        mx.zeros((2, 7, 8), dtype=mx.float32),
        mx.zeros((0,), dtype=mx.int32),
    )
    mx.eval(actual)
    assert actual.shape == (0, 4)
    assert actual.dtype == mx.float32


def test_cxcywh_conversion_clips_to_normalized_xyxy() -> None:
    converted = cxcywh_to_xyxy(
        mx.array([[0.1, 0.9, 0.8, 0.6], [0.5, 0.5, 0.2, 0.4]], dtype=mx.float32)
    )
    mx.eval(converted)
    np.testing.assert_allclose(
        np.array(converted),
        np.array([[0.0, 0.6, 0.5, 1.0], [0.4, 0.3, 0.6, 0.7]], dtype=np.float32),
        rtol=0.0,
        atol=1e-7,
    )
