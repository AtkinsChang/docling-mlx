# SPDX-License-Identifier: Apache-2.0

"""TableFormerV2 Pillow resize and MLX normalization parity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from docling_mlx.engines.table_structure.tableformer_v2.artifact import (
    validate_tableformer_v2_artifact,
)
from docling_mlx.engines.table_structure.tableformer_v2.preprocessing import (
    preprocess_images,
    resize_pil_rgb_uint8,
)
from tests.tableformer_v2.test_artifact import _artifact
from tools.tableformer_v2.reference_preprocessing import preprocess_with_torchvision


def _spec(tmp_path: Path):
    return validate_tableformer_v2_artifact(_artifact(tmp_path)).preprocessing


def _pattern(width: int, height: int, mode: str = "RGB") -> Image.Image:
    y, x = np.indices((height, width))
    rgb = np.stack(((x * 17 + y) % 256, (x + y * 13) % 256, (x * 7 + y * 5) % 256), axis=-1)
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    return image.convert(mode)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("size", "mode"),
    [
        ((448, 448), "RGB"),
        ((1301, 57), "L"),
        ((57, 1301), "RGBA"),
        ((1, 1), "RGB"),
    ],
)
def test_pillow_resize_matches_torchvision_pixels(
    tmp_path: Path, size: tuple[int, int], mode: str
) -> None:
    image = _pattern(*size, mode=mode)
    expected, _ = preprocess_with_torchvision(image)
    actual = resize_pil_rgb_uint8(image, _spec(tmp_path))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.mlx
@pytest.mark.parity
@pytest.mark.parametrize(
    "image",
    [
        pytest.param(_pattern(448, 448), id="identity"),
        pytest.param(_pattern(1301, 57), id="wide-downscale"),
        pytest.param(_pattern(57, 1301), id="tall-downscale"),
    ],
)
def test_mlx_normalization_matches_torch_cpu(tmp_path: Path, image: Image.Image) -> None:
    import mlx.core as mx

    expected_pixels, expected = preprocess_with_torchvision(image)
    spec = _spec(tmp_path)
    actual = preprocess_images([image], spec)
    mx.eval(actual)
    actual_array = np.array(actual[0], copy=True)
    np.testing.assert_array_equal(resize_pil_rgb_uint8(image, spec), expected_pixels)
    assert actual_array.dtype == np.float32
    assert actual_array.shape == (448, 448, 3)
    assert np.max(np.abs(actual_array - expected)) <= 1e-6


def test_preprocessing_config_rejects_reciprocal_semantic_drift(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    path = artifact / "preprocessor_config.json"
    value = json.loads(path.read_text())
    value["image_mean"] = [0.0, 0.0, 0.0]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="ImageNet"):
        validate_tableformer_v2_artifact(artifact)
