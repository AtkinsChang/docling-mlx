# SPDX-License-Identifier: Apache-2.0

"""Torch-oracle checks for RT-DETR-v2's deliberately narrow grid sampler."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from importlib import import_module
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.mlx

mx: Any
torch: Any
functional: Any
grid_sample_bilinear_zeros_align_corners_false: Any
grid_sample_bilinear_zeros_align_corners_false_reference: Any


@pytest.fixture(scope="module", autouse=True)
def _load_requirements() -> None:
    global mx, torch, functional
    global grid_sample_bilinear_zeros_align_corners_false
    global grid_sample_bilinear_zeros_align_corners_false_reference

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
        functional = import_module("torch.nn.functional")
        module = import_module("docling_mlx._models.detector_primitives")
    except ImportError as error:
        pytest.fail(f"selected MLX lane is missing a required dependency: {error}")
    grid_sample_bilinear_zeros_align_corners_false = (
        module.grid_sample_bilinear_zeros_align_corners_false
    )
    grid_sample_bilinear_zeros_align_corners_false_reference = (
        module.grid_sample_bilinear_zeros_align_corners_false_reference
    )


def _torch_grid_sample(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return (
        functional.grid_sample(
            torch.from_numpy(x).permute(0, 3, 1, 2),
            torch.from_numpy(grid),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        .permute(0, 2, 3, 1)
        .numpy()
    )


def _mlx_call(function: Callable, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    result = function(mx.array(x), mx.array(grid))
    mx.eval(result)
    return np.array(result)


def _grid(batch: int) -> np.ndarray:
    epsilon = np.float32(1e-6)
    coordinates = np.array(
        [
            [-3.0, -3.0],
            [-1.0, -1.0],
            [-1.0 + epsilon, -1.0 + epsilon],
            [0.0, 0.0],
            [1.0 - epsilon, 1.0 - epsilon],
            [1.0, 1.0],
            [3.0, 3.0],
            [-3.0, 0.0],
            [3.0, 0.0],
            [0.0, -3.0],
            [0.0, 3.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [-1.0 - epsilon, 1.0 + epsilon],
            [1.0 + epsilon, -1.0 - epsilon],
            [-0.5, 0.5],
            [0.5, -0.5],
            [-0.25, -0.75],
            [0.75, 0.25],
            [0.125, -0.125],
        ],
        dtype=np.float32,
    ).reshape(1, 4, 5, 2)
    return np.repeat(coordinates, batch, axis=0)


def _input(pattern: str, batch: int, height: int, width: int, channels: int = 3) -> np.ndarray:
    if pattern == "constant":
        return np.full((batch, height, width, channels), 0.75, dtype=np.float32)
    if pattern == "impulse":
        result = np.zeros((batch, height, width, channels), dtype=np.float32)
        result[:, height // 2, width // 2] = np.linspace(1.0, -0.5, channels)
        return result
    if pattern == "gradient":
        rows, columns = np.indices((height, width), dtype=np.float32)
        base = (rows + 2 * columns)[..., None]
        offsets = np.arange(channels, dtype=np.float32)[None, None, :]
        single = base + offsets
        return np.broadcast_to(single, (batch, height, width, channels)).copy()
    return (
        np.random.default_rng(80401)
        .standard_normal((batch, height, width, channels))
        .astype(np.float32)
    )


@pytest.mark.parity
@pytest.mark.parametrize(
    "implementation",
    ["reference", "metal"],
)
@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 5), (5, 7)])
@pytest.mark.parametrize("pattern", ["constant", "impulse", "gradient", "random"])
def test_grid_sample_matches_torch_for_boundary_and_out_of_range_coordinates(
    implementation: str, shape: tuple[int, int], pattern: str
) -> None:
    height, width = shape
    x = _input(pattern, batch=2, height=height, width=width)
    grid = _grid(batch=2)

    expected = _torch_grid_sample(x, grid)
    function = (
        grid_sample_bilinear_zeros_align_corners_false_reference
        if implementation == "reference"
        else grid_sample_bilinear_zeros_align_corners_false
    )
    actual = _mlx_call(function, x, grid)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_safe_metal_kernel_matches_composed_reference() -> None:
    x = _input("random", batch=2, height=5, width=7)
    grid = _grid(batch=2)
    composed = _mlx_call(grid_sample_bilinear_zeros_align_corners_false_reference, x, grid)
    metal = _mlx_call(grid_sample_bilinear_zeros_align_corners_false, x, grid)
    np.testing.assert_allclose(metal, composed, rtol=2e-6, atol=2e-6)


@pytest.mark.parity
def test_grid_sample_matches_real_heron_level_zero_values() -> None:
    """Regress the worst R101 decoder layer-1, feature-level-0 sample."""
    x = np.zeros((1, 80, 80, 1), dtype=np.float32)
    x[0, 63:65, 2:4, 0] = np.array(
        [[0.47857875, 0.062522665], [-0.52816236, -0.5087993]],
        dtype=np.float32,
    )
    grid = np.array([[[[-0.9363847, 0.60730124]]]], dtype=np.float32)

    expected = _torch_grid_sample(x, grid)
    reference = _mlx_call(grid_sample_bilinear_zeros_align_corners_false_reference, x, grid)
    metal = _mlx_call(grid_sample_bilinear_zeros_align_corners_false, x, grid)

    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_allclose(metal, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parity
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("channels", [1, 32])
def test_grid_sample_matches_torch_for_random_out_of_range_grids(batch: int, channels: int) -> None:
    generator = np.random.default_rng(9137 + batch + channels)
    x = _input("random", batch=batch, height=7, width=11, channels=channels)
    grid = generator.uniform(-3.0, 3.0, size=(batch, 5, 13, 2)).astype(np.float32)

    expected = _torch_grid_sample(x, grid)
    reference = _mlx_call(grid_sample_bilinear_zeros_align_corners_false_reference, x, grid)
    metal = _mlx_call(grid_sample_bilinear_zeros_align_corners_false, x, grid)

    np.testing.assert_allclose(reference, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(metal, expected, rtol=2e-6, atol=2e-6)


def test_grid_sample_rejects_out_of_contract_inputs() -> None:
    x = mx.zeros((1, 2, 3, 4), dtype=mx.float32)
    grid = mx.zeros((1, 2, 3, 2), dtype=mx.float32)

    with pytest.raises(ValueError, match="same batch"):
        grid_sample_bilinear_zeros_align_corners_false(x, mx.zeros((2, 2, 3, 2)))
    with pytest.raises(ValueError, match="shape"):
        grid_sample_bilinear_zeros_align_corners_false(x, mx.zeros((1, 2, 3, 3)))
    with pytest.raises(TypeError, match="FP32"):
        grid_sample_bilinear_zeros_align_corners_false(x.astype(mx.float16), grid)
