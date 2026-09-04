# SPDX-License-Identifier: Apache-2.0

"""Independent numerical checks for RT-DETR-v2 transformer helper functions."""

from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.mlx

mx: Any
torch: Any
generate_anchors: Any
inverse_sigmoid: Any
select_encoder_queries: Any


@pytest.fixture(scope="module", autouse=True)
def _load_requirements() -> None:
    global mx, torch, generate_anchors, inverse_sigmoid, select_encoder_queries

    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        pytest.fail(f"selected MLX lane requires Metal: {probe.stderr.strip()}")
    try:
        mx = import_module("mlx.core")
        torch = import_module("torch")
        module = import_module("docling_mlx._models.detector_primitives")
    except ImportError as error:
        pytest.fail(f"selected MLX lane is missing a required dependency: {error}")
    generate_anchors = module.generate_anchors
    inverse_sigmoid = module.inverse_sigmoid
    select_encoder_queries = module.select_encoder_queries


def _torch_anchors(
    spatial_shapes: tuple[tuple[int, int], ...], grid_size: float
) -> tuple[np.ndarray, np.ndarray]:
    per_level = []
    for level, (height, width) in enumerate(spatial_shapes):
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1)[None, ...] + 0.5
        grid_xy = grid_xy / torch.tensor([width, height], dtype=torch.float32)[None, None, None, :]
        box_size = torch.ones_like(grid_xy) * grid_size * (2.0**level)
        per_level.append(torch.cat([grid_xy, box_size], dim=-1).reshape(1, height * width, 4))
    anchors = torch.cat(per_level, dim=1)
    valid_mask = ((anchors > 1e-2) & (anchors < 1.0 - 1e-2)).all(dim=-1, keepdim=True)
    logits = torch.log(anchors / (1.0 - anchors))
    logits = torch.where(valid_mask, logits, torch.tensor(torch.finfo(torch.float32).max))
    return logits.numpy(), valid_mask.numpy()


@pytest.mark.parity
def test_inverse_sigmoid_matches_transformers_formula_at_extrema() -> None:
    values = np.array([-2.0, 0.0, 1e-8, 0.25, 0.5, 0.75, 1.0 - 1e-8, 1.0, 3.0], dtype=np.float32)
    expected_values = torch.from_numpy(values)
    clipped = torch.clamp(expected_values, 0.0, 1.0)
    expected = torch.log(torch.clamp(clipped, 1e-5, 1.0) / torch.clamp(1.0 - clipped, 1e-5, 1.0))

    actual = inverse_sigmoid(mx.array(values))
    mx.eval(actual)
    np.testing.assert_allclose(np.array(actual), expected.numpy(), rtol=2e-6, atol=2e-6)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("spatial_shapes", "grid_size"),
    [(((1, 1), (2, 3), (5, 7)), 0.05), (((3, 5), (7, 11)), 0.005)],
)
def test_generate_anchors_matches_torch(
    spatial_shapes: tuple[tuple[int, int], ...], grid_size: float
) -> None:
    expected_logits, expected_mask = _torch_anchors(spatial_shapes, grid_size)
    actual_logits, actual_mask = generate_anchors(spatial_shapes, grid_size)
    mx.eval(actual_logits, actual_mask)

    np.testing.assert_array_equal(np.array(actual_mask), expected_mask)
    np.testing.assert_allclose(np.array(actual_logits), expected_logits, rtol=2e-6, atol=2e-6)


def test_select_encoder_queries_matches_transformers_topk_and_gathers() -> None:
    memory = mx.array([[[1.0, 10.0], [4.0, 40.0], [2.0, 20.0], [3.0, 30.0]]])
    scores = mx.array([[[0.1, 0.2], [0.8, 0.7], [0.3, 0.4], [0.6, 0.5]]])
    box_logits = mx.arange(16, dtype=mx.float32).reshape(1, 4, 4) / 10.0

    selected_logits, selected_boxes, reference_logits, target = select_encoder_queries(
        memory, scores, box_logits, 2
    )
    mx.eval(selected_logits, selected_boxes, reference_logits, target)

    indices = torch.topk(torch.from_numpy(np.array(scores)).max(-1).values, 2, dim=1).indices
    expected_logits = torch.from_numpy(np.array(scores)).gather(
        1, indices[:, :, None].repeat(1, 1, 2)
    )
    expected_reference = torch.from_numpy(np.array(box_logits)).gather(
        1, indices[:, :, None].repeat(1, 1, 4)
    )
    expected_target = torch.from_numpy(np.array(memory)).gather(
        1, indices[:, :, None].repeat(1, 1, 2)
    )
    np.testing.assert_array_equal(np.array(selected_logits), expected_logits.numpy())
    np.testing.assert_allclose(
        np.array(selected_boxes), torch.sigmoid(expected_reference).numpy(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(np.array(reference_logits), expected_reference.numpy())
    np.testing.assert_array_equal(np.array(target), expected_target.numpy())


def test_inverse_sigmoid_rejects_invalid_epsilon() -> None:
    with pytest.raises(ValueError, match="less than 0.5"):
        inverse_sigmoid(mx.array([0.5], dtype=mx.float32), eps=0.5)
